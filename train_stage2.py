from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from train import normalize_sequence


SEQUENCE_TAG_PATTERN = re.compile(r"<([A-Za-z0-9_]+)>(.*?)</\1>", re.DOTALL)
SLOT_ORDER = ("ANTI", "H", "L")
SLOT_TAGS = {
    "ANTI": "Anti",
    "H": "H",
    "L": "L",
}


@dataclass
class Stage2Config:
    data_dir: Path
    output_dir: Path
    dataset_cache_path: Path | None
    save_dataset_cache: bool
    llm_model_name: str
    llm_cache_dir: Path | None
    use_pretoken: bool
    tasks: tuple[str, ...]
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    validation_split: float
    max_length: int
    max_seq_length: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: tuple[str, ...]
    num_workers: int
    seed: int
    local_files_only: bool
    use_gradient_checkpointing: bool
    save_every_epoch: bool
    log_file_name: str


@dataclass
class AFDStage2Sample:
    sample_id: str
    task_name: str
    instruction: str
    answer: str
    sequences: dict[str, str]
    pretoken_slot_order: tuple[str, ...]
    cached_pretokens: dict[str, list[float]] | None = None


class AFDStage2Dataset(Dataset):
    """Dataset wrapper for instruction-answer pairs with optional sequence slots."""

    def __init__(self, samples: list[AFDStage2Sample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        return {
            "sample_id": sample.sample_id,
            "task_name": sample.task_name,
            "instruction": sample.instruction,
            "answer": sample.answer,
            "sequences": sample.sequences,
            "pretoken_slot_order": sample.pretoken_slot_order,
            "cached_pretokens": sample.cached_pretokens,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 training on AFD with sequence pretokens.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/AFDData"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage2"))
    parser.add_argument("--dataset-cache-path", type=Path, default=Path("data/cache/afd_stage2_samples.json"))
    parser.add_argument("--save-dataset-cache", type=bool, default=True)
    parser.add_argument("--llm-model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--llm-cache-dir", type=Path, default=Path("models/"))
    parser.add_argument("--use-pretoken", action="store_true")
    parser.add_argument("--tasks", nargs="+", default=["caption", "cdr", "qa"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-seq-length", type=int, default=115)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--use-gradient-checkpointing", action="store_true")
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--log-file-name", type=str, default="train_log.jsonl")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Stage2Config:
    return Stage2Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dataset_cache_path=args.dataset_cache_path,
        save_dataset_cache=args.save_dataset_cache,
        llm_model_name=args.llm_model_name,
        llm_cache_dir=args.llm_cache_dir,
        use_pretoken=args.use_pretoken,
        tasks=tuple(args.tasks),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_split=args.validation_split,
        max_length=args.max_length,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=tuple(args.lora_target_modules),
        num_workers=args.num_workers,
        seed=args.seed,
        local_files_only=args.local_files_only,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        save_every_epoch=args.save_every_epoch,
        log_file_name=args.log_file_name,
    )


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible splitting and training."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_distributed() -> bool:
    """Return whether torch.distributed has been initialized."""

    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Return the current distributed rank."""

    return dist.get_rank() if is_distributed() else 0


def is_main_process() -> bool:
    """Return True only on rank 0."""

    return get_rank() == 0


def ddp_barrier() -> None:
    """Synchronize all ranks when running DDP."""

    if is_distributed():
        dist.barrier()


def setup_distributed() -> tuple[torch.device, int, int]:
    """Initialize DDP from torchrun environment variables when available."""

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return torch.device("cuda", local_rank), local_rank, dist.get_world_size()

    return torch.device("cuda" if torch.cuda.is_available() else "cpu"), 0, 1


def cleanup_distributed() -> None:
    """Shut down the process group cleanly."""

    if is_distributed():
        dist.destroy_process_group()

def extract_tagged_sequences(text: str) -> dict[str, str]:
    """Extract at most the antigen/heavy/light sequences from one instruction."""

    extracted: dict[str, str] = {}
    for tag, raw_sequence in SEQUENCE_TAG_PATTERN.findall(text):
        normalized_tag = tag.upper()
        if normalized_tag in SLOT_ORDER and normalized_tag not in extracted:
            extracted[normalized_tag] = normalize_sequence(raw_sequence)
    return extracted


def extract_conversation(record: dict[str, Any]) -> tuple[str, str]:
    """Read one human instruction and one assistant answer from the record."""

    messages = list(record.get("messages", [])) 
    # messages = record.get("messages", [])
    # if not isinstance(messages, list):
    #     raise ValueError("messages must be a list")

    instruction = next((msg.get("value", "") for msg in messages if str(msg.get("from", "")).lower() == "human"), "")
    answer = next((msg.get("value", "") for msg in messages if str(msg.get("from", "")).lower() != "human"), "")
    instruction = str(instruction).strip()
    answer = str(answer).strip()
    if not instruction or not answer:
        raise ValueError("record does not contain a usable instruction/answer pair")
    return instruction, answer


def load_afd_samples(
    data_dir: Path,
    tasks: tuple[str, ...],
    max_seq_length: int,
) -> tuple[int, int, list[AFDStage2Sample]]:
    """Load parquet files, extract sequences, and filter overlong protein inputs."""

    total = 0
    kept = 0
    samples: list[AFDStage2Sample] = []
    for task_name in tasks:
        parquet_path = data_dir / f"{task_name}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Missing parquet file: {parquet_path}")

        frame = pd.read_parquet(parquet_path)
        for row_index, record in enumerate(frame.to_dict(orient="records")):
            total += 1
            try:
                instruction, answer = extract_conversation(record)
            except ValueError:
                continue
            sequences = extract_tagged_sequences(instruction)
            if any(len(sequence) > max_seq_length for sequence in sequences.values()):
                continue
            kept += 1

            samples.append(
                AFDStage2Sample(
                    sample_id=f"{task_name}-{record.get('pdb_id', row_index)}",
                    task_name=task_name,
                    instruction=instruction,
                    answer=answer,
                    sequences=sequences,
                    pretoken_slot_order=tuple(slot for slot in SLOT_ORDER if slot in sequences),
                )
            )

    if not samples:
        raise ValueError("No usable AFD samples were loaded.")
    return total, kept, samples


def save_cached_samples(cache_path: Path, samples: list[AFDStage2Sample]) -> None:
    """Save processed Stage 2 samples so later runs can reuse them directly."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(sample) for sample in samples]
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_cached_samples(cache_path: Path) -> list[AFDStage2Sample]:
    """Load previously processed Stage 2 samples from a JSON cache file."""

    with cache_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Cached dataset at {cache_path} is not a list.")

    samples = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        samples.append(
            AFDStage2Sample(
                sample_id=str(item["sample_id"]),
                task_name=str(item["task_name"]),
                instruction=str(item["instruction"]),
                answer=str(item["answer"]),
                sequences={str(key): str(value) for key, value in dict(item["sequences"]).items()},
                pretoken_slot_order=tuple(str(slot) for slot in item["pretoken_slot_order"]),
                cached_pretokens=(
                    {str(key): [float(value) for value in values] for key, values in dict(item["cached_pretokens"]).items()}
                    if item.get("cached_pretokens") is not None
                    else None
                ),
            )
        )
    if not samples:
        raise ValueError(f"No usable cached samples were found in {cache_path}.")
    return samples


def split_samples(
    samples: list[AFDStage2Sample],
    validation_split: float,
    seed: int,
) -> tuple[list[AFDStage2Sample], list[AFDStage2Sample]]:
    """Split AFD samples into train and validation subsets."""

    if not 0.0 <= validation_split < 1.0:
        raise ValueError("validation_split must be in [0.0, 1.0).")

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    validation_size = int(len(shuffled) * validation_split)
    if validation_split > 0 and len(shuffled) > 1:
        validation_size = max(1, validation_size)

    validation_samples = shuffled[:validation_size]
    train_samples = shuffled[validation_size:]
    if not train_samples:
        raise ValueError("Training split is empty after applying validation_split.")
    return train_samples, validation_samples


class Stage2Collator:
    """Keep raw instruction/answer text so pretokens can be serialized first."""

    def __init__(self, tokenizer: AutoTokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        slot_sequences = {slot: [] for slot in SLOT_ORDER}
        pretoken_slot_orders = []
        cached_pretokens = []
        instructions = []
        answers = []

        for item in batch:
            instructions.append(item["instruction"])
            answers.append(item["answer"])
            pretoken_slot_orders.append(item["pretoken_slot_order"])
            cached_pretokens.append(item["cached_pretokens"])
            for slot in SLOT_ORDER:
                slot_sequences[slot].append(item["sequences"].get(slot))

        return {
            "instructions": instructions,
            "answers": answers,
            "slot_sequences": slot_sequences,
            "pretoken_slot_orders": pretoken_slot_orders,
            "cached_pretokens": cached_pretokens,
        }


def build_dataloaders(
    config: Stage2Config,
    tokenizer: AutoTokenizer,
    distributed: bool,
) -> tuple[DataLoader, DataLoader | None, DistributedSampler | None, DistributedSampler | None]:
    """Create train/validation dataloaders from AFD parquet files."""

    if config.dataset_cache_path is not None and config.dataset_cache_path.exists():
        samples = load_cached_samples(config.dataset_cache_path)
        print(f"Loaded {len(samples)} cached samples from {config.dataset_cache_path}.")
    else:
        total, kept, samples = load_afd_samples(config.data_dir, config.tasks, config.max_seq_length)
        print(f"Kept {kept}/{total} samples after max_seq_length={config.max_seq_length} filtering.")
        if config.dataset_cache_path is not None and config.save_dataset_cache:
            save_cached_samples(config.dataset_cache_path, samples)
            print(f"Saved processed dataset cache to {config.dataset_cache_path}.")

    if config.use_pretoken:
        missing_pretoken_ids = [
            sample.sample_id
            for sample in samples
            if sample.pretoken_slot_order and not sample.cached_pretokens
        ]
        if missing_pretoken_ids:
            preview = ", ".join(missing_pretoken_ids[:5])
            raise ValueError(
                "Pretoken training requires cached pretokens, but the current dataset cache is incomplete. "
                f"Missing cached_pretokens for {len(missing_pretoken_ids)} samples, such as: {preview}"
            )

    train_samples, validation_samples = split_samples(samples, config.validation_split, config.seed)
    collator = Stage2Collator(tokenizer=tokenizer, max_length=config.max_length)
    train_dataset = AFDStage2Dataset(train_samples)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=config.num_workers,
        collate_fn=collator,
    )

    validation_loader = None
    validation_sampler = None
    if validation_samples:
        validation_dataset = AFDStage2Dataset(validation_samples)
        validation_sampler = DistributedSampler(validation_dataset, shuffle=False) if distributed else None
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            sampler=validation_sampler,
            num_workers=config.num_workers,
            collate_fn=collator,
        )
    return train_loader, validation_loader, train_sampler, validation_sampler


def normalize_pretoken_to_token_ids(pretoken: torch.Tensor, tokenizer: AutoTokenizer) -> list[int]:
    """Normalize one pretoken vector and map it into the tokenizer's id space."""

    values = pretoken.detach().to(device="cpu", dtype=torch.float32)
    if values.numel() == 0:
        return []

    min_value = values.min()
    max_value = values.max()
    if torch.isclose(max_value, min_value):
        normalized = torch.zeros_like(values)
    else:
        normalized = (values - min_value) / (max_value - min_value)

    token_upper_bound = max(int(tokenizer.vocab_size) - 1, 0)
    mapped = torch.round(normalized * token_upper_bound).to(dtype=torch.long)
    return mapped.tolist()


def build_tag_prefixed_token_ids(
    tokenizer: AutoTokenizer,
    sample_pretokens: list[tuple[str, torch.Tensor]],
) -> list[int]:
    """Convert tagged pretokens into token ids that can be concatenated with prompt ids."""

    prefix_ids: list[int] = []
    for slot_name, pretoken in sample_pretokens:
        open_tag_ids = tokenizer(f"<{SLOT_TAGS[slot_name]}>", add_special_tokens=False)["input_ids"]
        close_tag_ids = tokenizer(f"</{SLOT_TAGS[slot_name]}>", add_special_tokens=False)["input_ids"]
        prefix_ids.extend(open_tag_ids)
        prefix_ids.extend(normalize_pretoken_to_token_ids(pretoken, tokenizer))
        prefix_ids.extend(close_tag_ids)
    return prefix_ids


def load_cached_pretokens_from_batch(
    batch_cached_pretokens: list[dict[str, list[float]] | None],
    batch_pretoken_slot_orders: list[tuple[str, ...]],
    device: torch.device,
) -> list[list[tuple[str, torch.Tensor]]]:
    """Read cached pretokens from the sample cache instead of recomputing them."""

    sample_pretokens: list[list[tuple[str, torch.Tensor]]] = []
    for cached_item, slot_order in zip(batch_cached_pretokens, batch_pretoken_slot_orders):
        if cached_item is None:
            raise ValueError("This cached dataset does not contain precomputed pretokens.")
        sample_slots: list[tuple[str, torch.Tensor]] = []
        for slot in slot_order:
            if slot not in cached_item:
                continue
            sample_slots.append((slot, torch.tensor(cached_item[slot], dtype=torch.float32, device=device)))
        sample_pretokens.append(sample_slots)
    return sample_pretokens


def build_model_inputs(
    batch: dict[str, Any],
    tokenizer: AutoTokenizer,
    max_length: int,
    device: torch.device,
) -> dict[str, Any]:
    """Map pretokens into token ids, then prepend them to the original instruction tokens."""

    pretokens_by_sample = load_cached_pretokens_from_batch(
        batch["cached_pretokens"],
        batch["pretoken_slot_orders"],
        device,
    )
    instruction_tokens = tokenizer(
        batch["instructions"],
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    answer_tokens = tokenizer(
        batch["answers"],
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]

    input_id_tensors = []
    label_tensors = []
    for instruction_ids, answer_ids, sample_pretokens in zip(instruction_tokens, answer_tokens, pretokens_by_sample):
        prefix_ids = build_tag_prefixed_token_ids(tokenizer, sample_pretokens)
        prompt_ids = prefix_ids + instruction_ids
        answer_ids = answer_ids[: max(0, max_length - len(prompt_ids) - 1)]
        full_ids = prompt_ids + answer_ids + [tokenizer.eos_token_id]
        full_ids = full_ids[: max_length]
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        labels = labels[: len(full_ids)]
        input_id_tensors.append(torch.tensor(full_ids, dtype=torch.long))
        label_tensors.append(torch.tensor(labels, dtype=torch.long))

    input_ids = pad_sequence(input_id_tensors, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    labels = pad_sequence(label_tensors, batch_first=True, padding_value=-100).to(device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_plain_model_inputs(
    batch: dict[str, Any],
    tokenizer: AutoTokenizer,
    max_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Use the original instruction-answer text directly, without pretokens."""

    prompt_tokens = tokenizer(
        batch["instructions"],
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    answer_tokens = tokenizer(
        batch["answers"],
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]

    input_id_tensors = []
    label_tensors = []
    for prompt_ids, answer_ids in zip(prompt_tokens, answer_tokens):
        answer_ids = answer_ids[: max(0, max_length - len(prompt_ids) - 1)]
        full_ids = prompt_ids + answer_ids + [tokenizer.eos_token_id]
        full_ids = full_ids[: max_length]
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        labels = labels[: len(full_ids)]
        input_id_tensors.append(torch.tensor(full_ids, dtype=torch.long))
        label_tensors.append(torch.tensor(labels, dtype=torch.long))

    input_ids = pad_sequence(input_id_tensors, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    labels = pad_sequence(label_tensors, batch_first=True, padding_value=-100).to(device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def append_epoch_log(log_path: Path, payload: dict[str, Any]) -> None:
    """Write one epoch summary to a jsonl log file."""

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    config: Stage2Config,
    llm: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Save Stage 2 trainable weights and optimizer state."""

    checkpoint = {
        "epoch": epoch,
        "config": asdict(config),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch}.pt")
    adapter_dir = output_dir / f"lora_epoch_{epoch}"
    llm.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)


def evaluate(
    dataloader: DataLoader | None,
    use_pretoken: bool,
    tokenizer: AutoTokenizer,
    llm: AutoModelForCausalLM,
    max_length: int,
    device: torch.device,
) -> float | None:
    """Run validation without parameter updates."""

    if dataloader is None:
        return None

    llm.eval()
    loss_sum = torch.tensor(0.0, device=device)
    batch_count = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for batch in dataloader:
            if use_pretoken:
                model_inputs = build_model_inputs(batch, tokenizer, max_length, device)
            else:
                model_inputs = build_plain_model_inputs(batch, tokenizer, max_length, device)
            outputs = llm(**model_inputs)
            loss_sum += outputs.loss.detach()
            batch_count += 1
    if is_distributed():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(batch_count, op=dist.ReduceOp.SUM)
    return (loss_sum / batch_count.clamp(min=1.0)).item()


def train() -> None:
    """Main Stage 2 loop: LoRA fine-tuning with optional cached pretokens."""

    args = parse_args()
    config = build_config(args)
    set_seed(config.seed)

    device, local_rank, world_size = setup_distributed()
    llm_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if is_main_process():
        print(f"Using device: {device}, world_size: {world_size}")
    if config.llm_cache_dir is not None and is_main_process():
        config.llm_cache_dir.mkdir(parents=True, exist_ok=True)
    ddp_barrier()

    tokenizer = AutoTokenizer.from_pretrained(
        config.llm_model_name,
        use_fast=True,
        local_files_only=config.local_files_only,
        cache_dir=str(config.llm_cache_dir) if config.llm_cache_dir is not None else None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    llm = AutoModelForCausalLM.from_pretrained(
        config.llm_model_name,
        torch_dtype=llm_dtype,
        local_files_only=config.local_files_only,
        cache_dir=str(config.llm_cache_dir) if config.llm_cache_dir is not None else None,
    ).to(device)
    llm.config.use_cache = False

    # Gradient checkpointing is optional because it saves VRAM at the cost of speed.
    if config.use_gradient_checkpointing and hasattr(llm, "gradient_checkpointing_enable"):
        llm.gradient_checkpointing_enable()
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config.lora_target_modules),
    )
    llm = get_peft_model(llm, peft_config)
    if is_main_process():
        llm.print_trainable_parameters()

    train_loader, validation_loader, train_sampler, _ = build_dataloaders(config, tokenizer, is_distributed())

    trainable_parameters = [parameter for parameter in llm.parameters() if parameter.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    if is_distributed():
        llm = DDP(llm, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    if is_main_process():
        config.output_dir.mkdir(parents=True, exist_ok=True)
        with (config.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
            json.dump(asdict(config), handle, indent=2, default=str)
        log_path = config.output_dir / config.log_file_name
        log_path.write_text("", encoding="utf-8")
    else:
        log_path = config.output_dir / config.log_file_name

    for epoch in range(1, config.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        llm.train()
        running_loss = 0.0

        iterator = train_loader
        if is_main_process():
            iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}")
        for batch in iterator:
            if config.use_pretoken:
                model_inputs = build_model_inputs(batch, tokenizer, config.max_length, device)
            else:
                model_inputs = build_plain_model_inputs(batch, tokenizer, config.max_length, device)
            outputs = llm(**model_inputs)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if is_main_process():
                iterator.set_postfix(loss=f"{loss.item():.4f}")

        train_loss_tensor = torch.tensor(running_loss, device=device)
        batch_count_tensor = torch.tensor(float(len(train_loader)), device=device)
        if is_distributed():
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(batch_count_tensor, op=dist.ReduceOp.SUM)
        train_loss = (train_loss_tensor / batch_count_tensor.clamp(min=1.0)).item()
        validation_loss = evaluate(
            validation_loader,
            config.use_pretoken,
            tokenizer,
            llm,
            config.max_length,
            device,
        )

        summary = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "epoch": epoch,
            "train_loss": train_loss,
        }
        if validation_loss is not None:
            summary["validation_loss"] = validation_loss
        if is_main_process():
            print(json.dumps(summary, indent=2))
            append_epoch_log(log_path, summary)

        if config.save_every_epoch and is_main_process():
            save_checkpoint(
                config.output_dir,
                epoch,
                config,
                llm.module if isinstance(llm, DDP) else llm,
                tokenizer,
                optimizer,
            )

    if is_main_process():
        save_checkpoint(
            config.output_dir,
            config.epochs,
            config,
            llm.module if isinstance(llm, DDP) else llm,
            tokenizer,
            optimizer,
        )
    cleanup_distributed()


if __name__ == "__main__":
    train()
