from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from pretrain import QFormerProteinAdaptor, build_adaptor_from_checkpoint


@dataclass
class Stage2Config:
    cache_path: Path
    adaptor_checkpoint: Path
    output_dir: Path
    resume_from: Path | None
    auto_resume: bool
    llm_model_name: str
    llm_cache_dir: Path | None
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    data_ratio: float
    validation_split: float
    max_length: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: tuple[str, ...]
    num_workers: int
    seed: int
    local_files_only: bool
    use_gradient_checkpointing: bool
    disable_chat_template: bool
    save_every_epoch: bool
    log_file_name: str


class PreparedAFDDataset(Dataset):
    """Dataset backed by the antibody cache produced from prepareData.py."""

    def __init__(self, samples: list[dict[str, Any]], antibody_embeddings: torch.Tensor) -> None:
        self.samples = samples
        self.antibody_embeddings = antibody_embeddings

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        return {
            "sample_id": sample["sample_id"],
            "task_name": sample["task_name"],
            "instruction": sample["instruction"],
            "answer": sample["answer"],
            "antibody_embedding": self.antibody_embeddings[index],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 LoRA fine-tuning with prepared antibody caches.")
    parser.add_argument("--cache-path", type=Path, default=Path("data/cache/afd_stage2_cache.pt"))
    parser.add_argument("--adaptor-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage2"))
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--llm-model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--llm-cache-dir", type=Path, default=Path("models"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--data-ratio", type=float, default=0.2)
    parser.add_argument("--validation-split", type=float, default=0.25)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--use-gradient-checkpointing", action="store_true")
    parser.add_argument("--disable-chat-template", action="store_true")
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--log-file-name", type=str, default="train_log.jsonl")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Stage2Config:
    if not 0.0 < args.data_ratio <= 1.0:
        raise ValueError("data_ratio must be in (0.0, 1.0].")

    return Stage2Config(
        cache_path=args.cache_path,
        adaptor_checkpoint=args.adaptor_checkpoint,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        auto_resume=args.auto_resume,
        llm_model_name=args.llm_model_name,
        llm_cache_dir=args.llm_cache_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        data_ratio=args.data_ratio,
        validation_split=args.validation_split,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=tuple(args.lora_target_modules),
        num_workers=args.num_workers,
        seed=args.seed,
        local_files_only=args.local_files_only,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        disable_chat_template=args.disable_chat_template,
        save_every_epoch=args.save_every_epoch,
        log_file_name=args.log_file_name,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


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


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def load_prepared_cache(cache_path: Path) -> tuple[list[dict[str, Any]], torch.Tensor]:
    if not cache_path.exists():
        raise FileNotFoundError(f"Prepared cache not found: {cache_path}")

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    samples = list(payload.get("samples") or [])
    antibody_embeddings = payload.get("antibody_embeddings")

    if not isinstance(samples, list) or not torch.is_tensor(antibody_embeddings):
        raise ValueError(f"Prepared cache at {cache_path} is missing required fields.")
    if len(samples) != antibody_embeddings.size(0):
        raise ValueError("Prepared cache has inconsistent sample and tensor sizes.")
    return samples, antibody_embeddings


def split_indices(num_samples: int, validation_split: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0.0 <= validation_split < 1.0:
        raise ValueError("validation_split must be in [0.0, 1.0).")

    indices = list(range(num_samples))
    random.Random(seed).shuffle(indices)
    validation_size = int(num_samples * validation_split)
    if validation_split > 0 and num_samples > 1:
        validation_size = max(1, validation_size)

    validation_indices = indices[:validation_size]
    train_indices = indices[validation_size:]
    if not train_indices:
        raise ValueError("Training split is empty after applying validation_split.")
    return train_indices, validation_indices


def apply_data_ratio(
    samples: list[dict[str, Any]],
    antibody_embeddings: torch.Tensor,
    data_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    if data_ratio >= 1.0:
        return samples, antibody_embeddings

    num_samples = len(samples)
    selected_count = max(1, int(num_samples * data_ratio))
    selected_indices = list(range(num_samples))
    random.Random(seed).shuffle(selected_indices)
    selected_indices = sorted(selected_indices[:selected_count])
    return subset_cache(samples, antibody_embeddings, selected_indices)


def subset_cache(
    samples: list[dict[str, Any]],
    antibody_embeddings: torch.Tensor,
    indices: list[int],
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    subset_samples = [samples[index] for index in indices]
    subset_embeddings = antibody_embeddings[indices]
    return subset_samples, subset_embeddings


def collate_stage2_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_ids": [item["sample_id"] for item in batch],
        "task_names": [item["task_name"] for item in batch],
        "instructions": [item["instruction"] for item in batch],
        "answers": [item["answer"] for item in batch],
        "antibody_embeddings": torch.stack([item["antibody_embedding"] for item in batch], dim=0),
    }


def filter_overlength_samples(
    samples: list[dict[str, Any]],
    antibody_embeddings: torch.Tensor,
    tokenizer: AutoTokenizer,
    prefix_length: int,
    max_length: int,
    use_template: bool,
) -> tuple[list[dict[str, Any]], torch.Tensor, int]:
    kept_indices: list[int] = []
    dropped_count = 0

    for index, sample in enumerate(samples):
        prompt_ids = build_prompt_token_ids(tokenizer, str(sample["instruction"]), use_template)
        answer_ids = list(tokenizer(str(sample["answer"]), add_special_tokens=False)["input_ids"])
        total_length = prefix_length + len(prompt_ids) + len(answer_ids) + 1
        if total_length > max_length:
            dropped_count += 1
            continue
        kept_indices.append(index)

    if not kept_indices:
        raise ValueError(
            "All Stage 2 samples were dropped because they exceed --max-length after adding the adaptor prefix. "
            "Increase --max-length or shorten the data."
        )
    return subset_cache(samples, antibody_embeddings, kept_indices)[0], antibody_embeddings[kept_indices], dropped_count


def build_dataloaders(
    samples: list[dict[str, Any]],
    antibody_embeddings: torch.Tensor,
    config: Stage2Config,
    tokenizer: AutoTokenizer,
    prefix_length: int,
    use_template: bool,
    distributed: bool,
) -> tuple[DataLoader, DataLoader | None, DistributedSampler | None, DistributedSampler | None]:
    samples, antibody_embeddings, dropped_count = filter_overlength_samples(
        samples,
        antibody_embeddings,
        tokenizer,
        prefix_length,
        config.max_length,
        use_template,
    )
    original_count = len(samples)
    samples, antibody_embeddings = apply_data_ratio(samples, antibody_embeddings, config.data_ratio, config.seed)
    train_indices, validation_indices = split_indices(len(samples), config.validation_split, config.seed)

    train_dataset = PreparedAFDDataset(*subset_cache(samples, antibody_embeddings, train_indices))
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=config.num_workers,
        collate_fn=collate_stage2_batch,
    )

    validation_loader = None
    validation_sampler = None
    if validation_indices:
        validation_dataset = PreparedAFDDataset(*subset_cache(samples, antibody_embeddings, validation_indices))
        validation_sampler = DistributedSampler(validation_dataset, shuffle=False) if distributed else None
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            sampler=validation_sampler,
            num_workers=config.num_workers,
            collate_fn=collate_stage2_batch,
        )

    if is_main_process():
        print(f"Loaded {original_count} prepared samples from {config.cache_path}")
        print(f"Dropped {dropped_count} overlength samples before Stage 2 training")
        print(f"Using {len(samples)} samples after applying data_ratio={config.data_ratio}")
        print(f"Train samples: {len(train_indices)}, validation samples: {len(validation_indices)}")
    return train_loader, validation_loader, train_sampler, validation_sampler


def use_chat_template(tokenizer: AutoTokenizer, disabled: bool) -> bool:
    return (not disabled) and hasattr(tokenizer, "apply_chat_template")


def build_prompt_token_ids(tokenizer: AutoTokenizer, instruction: str, use_template: bool) -> list[int]:
    if use_template:
        try:
            return list(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": instruction}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
        except Exception:
            pass
    return list(tokenizer(instruction, add_special_tokens=False)["input_ids"])


def build_model_inputs(
    batch: dict[str, Any],
    tokenizer: AutoTokenizer,
    llm: torch.nn.Module,
    adaptor: QFormerProteinAdaptor,
    max_length: int,
    device: torch.device,
    use_template: bool,
) -> dict[str, torch.Tensor]:
    base_llm = unwrap_model(llm)
    embedder = base_llm.get_input_embeddings()
    embedding_dtype = embedder.weight.dtype
    antibody_embeddings = batch["antibody_embeddings"].to(device=device, dtype=torch.float32)

    with torch.no_grad():
        # Restore fixed-length pre_embeddings from the cached H/L-concatenated antibody vectors.
        prefix_embeddings = adaptor(antibody_embeddings).to(dtype=embedding_dtype)

    prefix_length = prefix_embeddings.size(1)
    if max_length <= prefix_length + 1:
        raise ValueError(
            f"max_length={max_length} is too small for prefix_length={prefix_length}. Increase --max-length."
        )

    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must define either eos_token_id or pad_token_id.")

    input_embedding_tensors = []
    attention_mask_tensors = []
    label_tensors = []

    for sample_index, (instruction, answer) in enumerate(zip(batch["instructions"], batch["answers"])):
        prompt_ids = build_prompt_token_ids(tokenizer, instruction, use_template)
        answer_ids = list(tokenizer(answer, add_special_tokens=False)["input_ids"])
        total_length = prefix_length + len(prompt_ids) + len(answer_ids) + 1
        if total_length > max_length:
            sample_id = batch["sample_ids"][sample_index]
            raise ValueError(
                f"Sample {sample_id} exceeds max_length after prefix injection: "
                f"{total_length} > {max_length}. Overlength samples should be filtered before batching."
            )

        text_ids = prompt_ids + answer_ids + [eos_token_id]
        text_tensor = torch.tensor(text_ids, dtype=torch.long, device=device)
        text_embeddings = embedder(text_tensor).to(dtype=embedding_dtype)

        # Feed one antibody prefix followed by the untouched instruction/answer text embeddings.
        sample_input_embeddings = torch.cat([prefix_embeddings[sample_index], text_embeddings], dim=0)
        sample_attention_mask = torch.ones(sample_input_embeddings.size(0), dtype=torch.long, device=device)

        prefix_labels = torch.full((prefix_length,), -100, dtype=torch.long, device=device)
        prompt_labels = torch.full((len(prompt_ids),), -100, dtype=torch.long, device=device)
        answer_labels = torch.tensor(answer_ids + [eos_token_id], dtype=torch.long, device=device)
        sample_labels = torch.cat([prefix_labels, prompt_labels, answer_labels], dim=0)

        input_embedding_tensors.append(sample_input_embeddings)
        attention_mask_tensors.append(sample_attention_mask)
        label_tensors.append(sample_labels)

    inputs_embeds = pad_sequence(input_embedding_tensors, batch_first=True, padding_value=0.0)
    attention_mask = pad_sequence(attention_mask_tensors, batch_first=True, padding_value=0)
    labels = pad_sequence(label_tensors, batch_first=True, padding_value=-100)
    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def append_epoch_log(log_path: Path, payload: dict[str, Any]) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoint_paths = sorted(output_dir.glob("checkpoint_epoch_*.pt"))
    if not checkpoint_paths:
        return None
    return max(checkpoint_paths, key=lambda path: int(path.stem.split("_")[-1]))


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    config: Stage2Config,
    llm: torch.nn.Module,
    tokenizer: AutoTokenizer,
    optimizer: torch.optim.Optimizer,
) -> None:
    state_payload = {
        "epoch": epoch,
        "config": asdict(config),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state_payload, output_dir / f"checkpoint_epoch_{epoch}.pt")
    adapter_dir = output_dir / f"lora_epoch_{epoch}"
    unwrap_model(llm).save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)


def resolve_resume_checkpoint(resume_from: Path | None, auto_resume: bool, output_dir: Path) -> Path | None:
    if resume_from is not None:
        if not resume_from.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")
        return resume_from
    if auto_resume:
        return find_latest_checkpoint(output_dir)
    return None


def get_adapter_dir_for_checkpoint(checkpoint_path: Path) -> Path:
    epoch = int(checkpoint_path.stem.split("_")[-1])
    adapter_dir = checkpoint_path.parent / f"lora_epoch_{epoch}"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"LoRA adapter directory not found for checkpoint {checkpoint_path}: {adapter_dir}")
    return adapter_dir


def load_stage2_training_state(
    checkpoint_path: Path,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(checkpoint["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    resumed_epoch = int(checkpoint["epoch"])
    return resumed_epoch + 1


def evaluate(
    dataloader: DataLoader | None,
    tokenizer: AutoTokenizer,
    llm: torch.nn.Module,
    adaptor: QFormerProteinAdaptor,
    max_length: int,
    device: torch.device,
    use_template: bool,
) -> float | None:
    if dataloader is None:
        return None

    llm.eval()
    loss_sum = torch.tensor(0.0, device=device)
    batch_count = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for batch in dataloader:
            model_inputs = build_model_inputs(batch, tokenizer, llm, adaptor, max_length, device, use_template)
            outputs = llm(**model_inputs)
            loss_sum += outputs.loss.detach()
            batch_count += 1

    if is_distributed():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(batch_count, op=dist.ReduceOp.SUM)
    return (loss_sum / batch_count.clamp(min=1.0)).item()


def train() -> None:
    args = parse_args()
    config = build_config(args)
    set_seed(config.seed)

    device, local_rank, world_size = setup_distributed()
    llm_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if is_main_process():
        print(f"Using device: {device}, world_size: {world_size}")

    if config.llm_cache_dir is not None and is_main_process():
        config.llm_cache_dir.mkdir(parents=True, exist_ok=True)
    if is_distributed():
        dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(
        config.llm_model_name,
        use_fast=True,
        local_files_only=config.local_files_only,
        cache_dir=str(config.llm_cache_dir) if config.llm_cache_dir is not None else None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    use_template = use_chat_template(tokenizer, config.disable_chat_template)

    samples, antibody_embeddings = load_prepared_cache(config.cache_path)
    antibody_embedding_dim = int(antibody_embeddings.shape[-1])
    adaptor, adaptor_config = build_adaptor_from_checkpoint(config.adaptor_checkpoint, antibody_embedding_dim)
    prefix_length = int(adaptor_config["query_length"])

    llm = AutoModelForCausalLM.from_pretrained(
        config.llm_model_name,
        torch_dtype=llm_dtype,
        local_files_only=config.local_files_only,
        cache_dir=str(config.llm_cache_dir) if config.llm_cache_dir is not None else None,
    ).to(device)
    llm.config.use_cache = False

    if config.use_gradient_checkpointing and hasattr(llm, "gradient_checkpointing_enable"):
        llm.gradient_checkpointing_enable()

    resume_checkpoint = resolve_resume_checkpoint(config.resume_from, config.auto_resume, config.output_dir)
    if resume_checkpoint is None:
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(config.lora_target_modules),
        )
        llm = get_peft_model(llm, peft_config)
    else:
        adapter_dir = get_adapter_dir_for_checkpoint(resume_checkpoint)
        llm = PeftModel.from_pretrained(llm, adapter_dir, is_trainable=True)
        if is_main_process():
            print(f"Resuming Stage 2 from {resume_checkpoint}")
    if is_main_process():
        llm.print_trainable_parameters()

    train_loader, validation_loader, train_sampler, validation_sampler = build_dataloaders(
        samples,
        antibody_embeddings,
        config,
        tokenizer,
        prefix_length,
        use_template,
        is_distributed(),
    )

    adaptor = adaptor.to(device)
    adaptor.eval()
    for parameter in adaptor.parameters():
        parameter.requires_grad = False

    llm_hidden_dim = unwrap_model(llm).get_input_embeddings().weight.shape[1]
    if int(adaptor_config["output_dim"]) != int(llm_hidden_dim):
        raise ValueError(
            "Adaptor output_dim does not match LLM embedding dim: "
            f"{adaptor_config['output_dim']} != {llm_hidden_dim}"
        )

    if is_distributed():
        llm = DDP(llm, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False) if device.type == "cuda" else DDP(llm)

    trainable_parameters = [parameter for parameter in llm.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    start_epoch = 1
    if resume_checkpoint is not None:
        start_epoch = load_stage2_training_state(resume_checkpoint, optimizer, device)

    if is_main_process():
        config.output_dir.mkdir(parents=True, exist_ok=True)
        with (config.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
            json.dump({**asdict(config), "adaptor_config": adaptor_config}, handle, ensure_ascii=False, indent=2, default=str)
        log_path = config.output_dir / config.log_file_name
        if start_epoch == 1:
            log_path.write_text("", encoding="utf-8")
    else:
        log_path = config.output_dir / config.log_file_name

    if start_epoch > config.epochs:
        raise ValueError(
            f"Resume checkpoint is already at epoch {start_epoch - 1}, which exceeds --epochs={config.epochs}. "
            "Increase --epochs to continue training."
        )

    for epoch in range(start_epoch, config.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if validation_sampler is not None:
            validation_sampler.set_epoch(epoch)

        llm.train()
        running_loss = 0.0
        iterator = train_loader
        if is_main_process():
            iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}")

        for batch in iterator:
            model_inputs = build_model_inputs(batch, tokenizer, llm, adaptor, config.max_length, device, use_template)
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
        validation_loss = evaluate(validation_loader, tokenizer, llm, adaptor, config.max_length, device, use_template)

        summary = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "epoch": epoch,
            "train_loss": train_loss,
        }
        if validation_loss is not None:
            summary["validation_loss"] = validation_loss

        if is_main_process():
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            append_epoch_log(log_path, summary)
            if config.save_every_epoch:
                save_checkpoint(config.output_dir, epoch, config, llm, tokenizer, optimizer)

    if is_main_process():
        save_checkpoint(config.output_dir, config.epochs, config, llm, tokenizer, optimizer)
    cleanup_distributed()


if __name__ == "__main__":
    train()
