from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.distributed as dist
from tqdm import tqdm

from pretrain import ProteinEncoder, is_esm_compatible_sequence, normalize_sequence


SEQUENCE_TAG_PATTERN = re.compile(r"<([A-Za-z0-9_]+)>(.*?)</\1>", re.DOTALL)
CHAIN_ORDER = ("H", "L")
@dataclass
class AFDPreparedSample:
    sample_id: str
    task_name: str
    instruction: str
    answer: str
    sequences: dict[str, str]


@dataclass
class InvalidAFDSample:
    sample_id: str
    task_name: str
    reason: str
    instruction: str
    answer: str
    heavy_sequence: str
    light_sequence: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare AFD cache for Stage 2 LLM training.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/AFDData"))
    parser.add_argument("--output-path", type=Path, default=Path("data/cache/afd_stage2_cache.pt"))
    parser.add_argument("--summary-path", type=Path, default=Path("data/cache/afd_stage2_summary.json"))
    parser.add_argument("--invalid-output-path", type=Path, default=Path("data/cache/afd_stage2_invalid.json"))
    parser.add_argument("--protein-model-name", type=str, default="esm2_t36_3B_UR50D")
    parser.add_argument("--tasks", nargs="+", default=["caption", "cdr", "qa"])
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--esm-batch-size", type=int, default=2)
    parser.add_argument("--cache-dtype", choices=["float16", "float32"], default="float16")
    return parser.parse_args()


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed() -> tuple[torch.device, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        backend = "nccl" if torch.cuda.is_available() and os.name != "nt" else "gloo"
        dist.init_process_group(backend=backend)
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
        return device, local_rank, dist.get_world_size()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, 0, 1


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def extract_tagged_sequences(text: str) -> dict[str, str]:
    extracted: dict[str, str] = {}
    for tag, raw_sequence in SEQUENCE_TAG_PATTERN.findall(text):
        normalized_tag = tag.upper()
        if normalized_tag in CHAIN_ORDER and normalized_tag not in extracted:
            extracted[normalized_tag] = normalize_sequence(raw_sequence)
    return extracted


def is_valid_chain_sequence(sequence: str) -> bool:
    return is_esm_compatible_sequence(sequence)


def extract_conversation(record: dict[str, Any]) -> tuple[str, str]:
    messages = list(record.get("messages", []))
    instruction = next((msg.get("value", "") for msg in messages if str(msg.get("from", "")).lower() == "human"), "")
    answer = next((msg.get("value", "") for msg in messages if str(msg.get("from", "")).lower() != "human"), "")
    instruction = str(instruction).strip()
    answer = str(answer).strip()
    if not instruction or not answer:
        raise ValueError("record does not contain a usable instruction/answer pair")
    return instruction, answer


def build_invalid_sample(
    sample_id: str,
    task_name: str,
    reason: str,
    instruction: str = "",
    answer: str = "",
    heavy_sequence: str = "",
    light_sequence: str = "",
) -> InvalidAFDSample:
    return InvalidAFDSample(
        sample_id=sample_id,
        task_name=task_name,
        reason=reason,
        instruction=instruction,
        answer=answer,
        heavy_sequence=heavy_sequence,
        light_sequence=light_sequence,
    )


def load_afd_samples(
    data_dir: Path,
    tasks: list[str],
    max_seq_length: int,
) -> tuple[int, int, list[AFDPreparedSample], list[InvalidAFDSample], Counter[str]]:
    total = 0
    kept = 0
    skip_counts: Counter[str] = Counter()
    samples: list[AFDPreparedSample] = []
    invalid_samples: list[InvalidAFDSample] = []

    for task_name in tasks:
        parquet_path = data_dir / f"{task_name}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Missing parquet file: {parquet_path}")

        frame = pd.read_parquet(parquet_path)
        records = frame.to_dict(orient="records")
        progress = tqdm(records, desc=f"Preparing {task_name}", unit="sample", disable=not is_main_process())
        for row_index, record in enumerate(progress):
            total += 1
            sample_id = f"{task_name}-{record.get('pdb_id', row_index)}"
            try:
                instruction, answer = extract_conversation(record)
            except ValueError:
                skip_counts["invalid_conversation"] += 1
                invalid_samples.append(build_invalid_sample(sample_id=sample_id, task_name=task_name, reason="invalid_conversation"))
                progress.set_postfix(kept=kept, skipped=total - kept, invalid=len(invalid_samples))
                continue

            sequences = extract_tagged_sequences(instruction)
            heavy_sequence = sequences.get("H", "")
            light_sequence = sequences.get("L", "")
            # Stage 2 keeps only complete H/L antibodies and preserves the original instruction text as-is.
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
                        task_name=task_name,
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

            samples.append(
                AFDPreparedSample(
                    sample_id=sample_id,
                    task_name=task_name,
                    instruction=instruction,
                    answer=answer,
                    sequences={"H": heavy_sequence, "L": light_sequence},
                )
            )
            kept += 1
            progress.set_postfix(kept=kept, skipped=total - kept, invalid=len(invalid_samples))

    if not samples:
        raise ValueError("No usable AFD samples were loaded.")
    return total, kept, samples, invalid_samples, skip_counts


def gather_variable_tensors(local_tensor: torch.Tensor, local_indices: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    world_size = get_world_size()
    if world_size == 1:
        return [local_tensor], [local_indices]

    backend = dist.get_backend()
    comm_device = torch.device("cuda", torch.cuda.current_device()) if backend == "nccl" else torch.device("cpu")

    count_tensor = torch.tensor([local_tensor.size(0)], dtype=torch.long, device=comm_device)
    gathered_counts = [torch.zeros_like(count_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_counts, count_tensor)
    max_count = max(int(item.item()) for item in gathered_counts)

    padded_tensor = local_tensor.to(comm_device)
    padded_indices = local_indices.to(comm_device)
    if local_tensor.size(0) < max_count:
        pad_rows = max_count - local_tensor.size(0)
        tensor_padding = torch.zeros((pad_rows, local_tensor.size(1)), dtype=local_tensor.dtype, device=comm_device)
        index_padding = torch.full((pad_rows,), -1, dtype=torch.long, device=comm_device)
        padded_tensor = torch.cat([padded_tensor, tensor_padding], dim=0)
        padded_indices = torch.cat([padded_indices, index_padding], dim=0)

    gathered_tensors = [torch.zeros_like(padded_tensor) for _ in range(world_size)]
    gathered_indices = [torch.zeros_like(padded_indices) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, padded_tensor)
    dist.all_gather(gathered_indices, padded_indices)

    trimmed_tensors: list[torch.Tensor] = []
    trimmed_indices: list[torch.Tensor] = []
    for count, tensor_chunk, index_chunk in zip(gathered_counts, gathered_tensors, gathered_indices):
        valid_count = int(count.item())
        trimmed_tensors.append(tensor_chunk[:valid_count].cpu())
        trimmed_indices.append(index_chunk[:valid_count].cpu())
    return trimmed_tensors, trimmed_indices


def cache_antibody_embeddings(
    samples: list[AFDPreparedSample],
    protein_encoder: ProteinEncoder,
    esm_batch_size: int,
    target_dtype: torch.dtype,
) -> torch.Tensor | None:
    if esm_batch_size < 1:
        raise ValueError("esm_batch_size must be at least 1.")

    rank = get_rank()
    world_size = get_world_size()
    local_indices = list(range(rank, len(samples), world_size))
    local_antibody_embeddings = torch.zeros(
        (len(local_indices), protein_encoder.output_dim * 2),
        dtype=target_dtype,
    )

    heavy_sequences = [samples[index].sequences["H"] for index in local_indices]
    light_sequences = [samples[index].sequences["L"] for index in local_indices]
    local_iterator = range(0, len(local_indices), esm_batch_size)
    progress = tqdm(
        local_iterator,
        desc=f"GPU caching rank {rank}",
        unit="batch",
        position=rank if world_size > 1 else 0,
        leave=rank == 0,
        dynamic_ncols=True,
    )

    with torch.no_grad():
        for local_start in progress:
            local_end = local_start + esm_batch_size
            batch_heavy = heavy_sequences[local_start:local_end]
            batch_light = light_sequences[local_start:local_end]
            if not batch_heavy:
                continue

            heavy_embeddings = protein_encoder(batch_heavy)
            light_embeddings = protein_encoder(batch_light)

            # Cache one antibody vector formed by concatenating pooled H and pooled L ESM embeddings.
            batch_antibody_embeddings = torch.cat([heavy_embeddings, light_embeddings], dim=1)
            local_antibody_embeddings[local_start:local_end] = batch_antibody_embeddings.to(device="cpu", dtype=target_dtype)
            progress.set_postfix(samples=min(local_end, len(local_indices)), total=len(local_indices))

    progress.close()

    gathered_tensors, gathered_indices = gather_variable_tensors(
        local_antibody_embeddings,
        torch.tensor(local_indices, dtype=torch.long),
    )

    if not is_main_process():
        return None

    antibody_embeddings = torch.zeros(
        (len(samples), protein_encoder.output_dim * 2),
        dtype=target_dtype,
    )
    for tensor_chunk, index_chunk in zip(gathered_tensors, gathered_indices):
        for row_index, sample_index in enumerate(index_chunk.tolist()):
            if sample_index >= 0:
                antibody_embeddings[sample_index] = tensor_chunk[row_index]
    return antibody_embeddings


def main() -> None:
    args = parse_args()
    device, _, world_size = setup_distributed()
    try:
        total, kept, samples, invalid_samples, skip_counts = load_afd_samples(args.data_dir, args.tasks, args.max_seq_length)
        if is_main_process():
            print(f"Using device: {device}, world_size: {world_size}")

        protein_encoder = ProteinEncoder(args.protein_model_name).to(device)
        protein_encoder.eval()

        cache_dtype = torch.float16 if args.cache_dtype == "float16" else torch.float32
        antibody_embeddings = cache_antibody_embeddings(samples, protein_encoder, args.esm_batch_size, cache_dtype)

        if not is_main_process():
            return

        payload = {
            "config": {
                "data_dir": str(args.data_dir),
                "protein_model_name": args.protein_model_name,
                "tasks": list(args.tasks),
                "max_seq_length": args.max_seq_length,
                "esm_batch_size": args.esm_batch_size,
                "cache_dtype": args.cache_dtype,
                "world_size": world_size,
                "skip_counts": dict(skip_counts),
            },
            "chain_order": list(CHAIN_ORDER),
            "single_chain_embedding_dim": protein_encoder.output_dim,
            "antibody_embedding_dim": protein_encoder.output_dim * 2,
            "samples": [asdict(sample) for sample in samples],
            "antibody_embeddings": antibody_embeddings,
        }

        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, args.output_path)

        args.invalid_output_path.parent.mkdir(parents=True, exist_ok=True)
        with args.invalid_output_path.open("w", encoding="utf-8") as handle:
            json.dump([asdict(sample) for sample in invalid_samples], handle, ensure_ascii=False, indent=2)

        summary = {
            "total_records": total,
            "kept_records": kept,
            "num_samples": len(samples),
            "invalid_samples": len(invalid_samples),
            "chain_order": list(CHAIN_ORDER),
            "single_chain_embedding_dim": protein_encoder.output_dim,
            "antibody_embedding_dim": protein_encoder.output_dim * 2,
            "cache_path": str(args.output_path),
            "invalid_output_path": str(args.invalid_output_path),
            "summary_generated_from": str(args.data_dir),
            "world_size": world_size,
            "skip_counts": dict(skip_counts),
        }
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

        print(f"Saved {len(samples)} prepared samples to {args.output_path}")
        print(f"Saved {len(invalid_samples)} invalid Stage 2 samples to {args.invalid_output_path}")
        print(f"Saved cache summary to {args.summary_path}")
        print(f"Skip counts: {dict(skip_counts)}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
