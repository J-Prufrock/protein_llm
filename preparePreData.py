from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from pretrain import is_esm_compatible_sequence, normalize_sequence


SEQUENCE_TAG_PATTERN = re.compile(r"<([A-Za-z0-9_]+)>(.*?)</\1>", re.DOTALL)
CHAIN_ORDER = ("H", "L")
@dataclass
class PreparedPretrainSample:
    sample_id: str
    heavy_sequence: str
    light_sequence: str
    text: str


@dataclass
class InvalidPretrainSample:
    sample_id: str
    reason: str
    instruction: str
    answer: str
    heavy_sequence: str
    light_sequence: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare caption-based Stage 1 pretraining data.")
    parser.add_argument("--caption-path", type=Path, default=Path("data/AFDData/caption.parquet"))
    parser.add_argument("--output-path", type=Path, default=Path("data/cache/afd_pretrain_caption.json"))
    parser.add_argument("--summary-path", type=Path, default=Path("data/cache/afd_pretrain_caption_summary.json"))
    parser.add_argument("--invalid-output-path", type=Path, default=Path("data/cache/afd_pretrain_caption_invalid.json"))
    parser.add_argument("--max-seq-length", type=int, default=256)
    return parser.parse_args()


def extract_tagged_sequences(text: str) -> dict[str, str]:
    extracted: dict[str, str] = {}
    for tag, raw_sequence in SEQUENCE_TAG_PATTERN.findall(text):
        normalized_tag = tag.upper()
        if normalized_tag in CHAIN_ORDER and normalized_tag not in extracted:
            extracted[normalized_tag] = normalize_sequence(raw_sequence)
    return extracted


def is_valid_chain_sequence(sequence: str) -> bool:
    return is_esm_compatible_sequence(sequence)


def strip_sequences_from_text(text: str) -> str:
    stripped = SEQUENCE_TAG_PATTERN.sub(" ", text)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    stripped = re.sub(r"\s+([,.;:?!])", r"\1", stripped)
    stripped = re.sub(r"([(:])\s+", r"\1", stripped)
    stripped = re.sub(r"\s+([)])", r"\1", stripped)
    return stripped


def extract_conversation(record: dict[str, Any]) -> tuple[str, str]:
    messages = list(record.get("messages", []))
    instruction = next((msg.get("value", "") for msg in messages if str(msg.get("from", "")).lower() == "human"), "")
    answer = next((msg.get("value", "") for msg in messages if str(msg.get("from", "")).lower() != "human"), "")
    instruction = str(instruction).strip()
    answer = str(answer).strip()
    if not instruction or not answer:
        raise ValueError("record does not contain a usable instruction/answer pair")
    return instruction, answer


def build_text(instruction: str, answer: str) -> str:
    # Stage 1 text target is built by removing inline sequence tags and concatenating question + answer.
    instruction_without_sequences = strip_sequences_from_text(instruction)
    parts = [part for part in (instruction_without_sequences, answer.strip()) if part]
    return "\n".join(parts).strip()


def build_invalid_sample(
    sample_id: str,
    reason: str,
    instruction: str = "",
    answer: str = "",
    heavy_sequence: str = "",
    light_sequence: str = "",
) -> InvalidPretrainSample:
    return InvalidPretrainSample(
        sample_id=sample_id,
        reason=reason,
        instruction=instruction,
        answer=answer,
        heavy_sequence=heavy_sequence,
        light_sequence=light_sequence,
    )


def load_samples(
    caption_path: Path,
    max_seq_length: int,
) -> tuple[int, int, list[PreparedPretrainSample], list[InvalidPretrainSample], Counter[str]]:
    if not caption_path.exists():
        raise FileNotFoundError(f"Caption parquet not found: {caption_path}")

    total = 0
    kept = 0
    skip_counts: Counter[str] = Counter()
    samples: list[PreparedPretrainSample] = []
    invalid_samples: list[InvalidPretrainSample] = []
    frame = pd.read_parquet(caption_path)

    records = frame.to_dict(orient="records")
    progress = tqdm(records, desc="Preparing Stage 1 data", unit="sample")
    for row_index, record in enumerate(progress):
        total += 1
        sample_id = f"caption-{record.get('pdb_id', row_index)}"
        try:
            instruction, answer = extract_conversation(record)
        except ValueError:
            skip_counts["invalid_conversation"] += 1
            invalid_samples.append(build_invalid_sample(sample_id=sample_id, reason="invalid_conversation"))
            progress.set_postfix(kept=kept, skipped=total - kept, invalid=len(invalid_samples))
            continue

        sequences = extract_tagged_sequences(instruction)
        heavy_sequence = sequences.get("H", "")
        light_sequence = sequences.get("L", "")
        if not heavy_sequence or not light_sequence:
            skip_counts["missing_chain"] += 1
            progress.set_postfix(kept=kept, skipped=total - kept, invalid=len(invalid_samples))
            continue
        if not is_valid_chain_sequence(heavy_sequence) or not is_valid_chain_sequence(light_sequence):
            # Accept X directly and convert '-' to X during normalization; reject only truly malformed strings.
            skip_counts["invalid_sequence_format"] += 1
            invalid_samples.append(
                build_invalid_sample(
                    sample_id=sample_id,
                    reason="invalid_sequence_format",
                    instruction=instruction,
                    answer=answer,
                    heavy_sequence=heavy_sequence,
                    light_sequence=light_sequence,
                )
            )
            progress.set_postfix(kept=kept, skipped=total - kept, invalid=len(invalid_samples))
            continue
        if len(heavy_sequence) > max_seq_length or len(light_sequence) > max_seq_length:
            skip_counts["sequence_too_long"] += 1
            progress.set_postfix(kept=kept, skipped=total - kept, invalid=len(invalid_samples))
            continue

        text = build_text(instruction, answer)
        if not text:
            skip_counts["empty_text"] += 1
            progress.set_postfix(kept=kept, skipped=total - kept, invalid=len(invalid_samples))
            continue

        samples.append(
            PreparedPretrainSample(
                sample_id=sample_id,
                heavy_sequence=heavy_sequence,
                light_sequence=light_sequence,
                text=text,
            )
        )
        kept += 1
        progress.set_postfix(kept=kept, skipped=total - kept, invalid=len(invalid_samples))

    if not samples:
        raise ValueError("No usable pretraining samples were extracted from caption.parquet.")
    return total, kept, samples, invalid_samples, skip_counts


def main() -> None:
    args = parse_args()
    total, kept, samples, invalid_samples, skip_counts = load_samples(args.caption_path, args.max_seq_length)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(sample) for sample in samples], handle, ensure_ascii=False, indent=2)

    args.invalid_output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.invalid_output_path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(sample) for sample in invalid_samples], handle, ensure_ascii=False, indent=2)

    summary = {
        "caption_path": str(args.caption_path),
        "output_path": str(args.output_path),
        "invalid_output_path": str(args.invalid_output_path),
        "total_records": total,
        "kept_records": kept,
        "num_samples": len(samples),
        "invalid_samples": len(invalid_samples),
        "chain_order": list(CHAIN_ORDER),
        "max_seq_length": args.max_seq_length,
        "skip_counts": dict(skip_counts),
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Saved {len(samples)} Stage 1 samples to {args.output_path}")
    print(f"Saved {len(invalid_samples)} invalid Stage 1 samples to {args.invalid_output_path}")
    print(f"Saved summary to {args.summary_path}")
    print(f"Skip counts: {dict(skip_counts)}")


if __name__ == "__main__":
    main()
