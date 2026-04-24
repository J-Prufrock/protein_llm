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

import esm
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


ESM_ALLOWED_RESIDUES = "ACDEFGHIKLMNPQRSTVWYXBZUO"
ESM_SEQUENCE_PATTERN = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYXBZUO]+$")


def normalize_sequence(sequence: str) -> str:
    """Remove whitespace, uppercase residues, and map gaps to X."""

    # Convert gaps to X so ESM always receives a valid token sequence.
    return "".join(sequence.split()).upper().replace("-", "X")


def is_esm_compatible_sequence(sequence: str) -> bool:
    """Check whether a sequence can be tokenized by the ESM encoder after normalization."""

    normalized_sequence = normalize_sequence(sequence)
    return bool(normalized_sequence) and bool(ESM_SEQUENCE_PATTERN.fullmatch(normalized_sequence))


def masked_mean_pooling(embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool only valid positions and ignore padded positions."""

    float_mask = mask.unsqueeze(-1).to(dtype=embeddings.dtype)
    pooled = (embeddings * float_mask).sum(dim=1)
    counts = float_mask.sum(dim=1).clamp(min=1.0)
    return pooled / counts


@dataclass
class PretrainConfig:
    data_path: Path
    output_dir: Path
    resume_from: Path | None
    auto_resume: bool
    protein_model_name: str
    text_model_name: str
    text_model_cache_dir: Path | None
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    validation_split: float
    max_text_length: int
    max_seq_length: int
    query_length: int
    adaptor_hidden_dim: int
    adaptor_num_heads: int
    dropout: float
    temperature: float
    text_embedding_dim: int | None
    num_workers: int
    seed: int
    local_files_only: bool
    save_every_epoch: bool
    log_file_name: str


@dataclass
class AFDPretrainSample:
    sample_id: str
    heavy_sequence: str
    light_sequence: str
    text: str


class AFDPretrainDataset(Dataset):
    """Dataset wrapper for AFD antibody-text alignment pretraining."""

    def __init__(self, samples: list[AFDPretrainSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, str]:
        sample = self.samples[index]
        return {
            "sample_id": sample.sample_id,
            "heavy_sequence": sample.heavy_sequence,
            "light_sequence": sample.light_sequence,
            "text": sample.text,
        }


class AFDPretrainCollator:
    """Tokenize the text target paired with one H/L antibody."""

    def __init__(self, tokenizer: AutoTokenizer, max_text_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

    def __call__(self, batch: list[dict[str, str]]) -> dict[str, Any]:
        texts = [item["text"] for item in batch]
        tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
        )
        return {
            "sample_ids": [item["sample_id"] for item in batch],
            "heavy_sequences": [item["heavy_sequence"] for item in batch],
            "light_sequences": [item["light_sequence"] for item in batch],
            "texts": texts,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }


class ProteinEncoder(nn.Module):
    """Freeze ESM and pool residue embeddings into one sequence representation."""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        pretrained_builder = getattr(esm.pretrained, model_name)
        self.model, self.alphabet = pretrained_builder()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.output_dim = int(self.model.embed_dim)

        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def forward(self, sequences: list[str]) -> torch.Tensor:
        for sequence in sequences:
            if not is_esm_compatible_sequence(sequence):
                invalid_tokens = sorted(set(normalize_sequence(sequence)) - set(ESM_ALLOWED_RESIDUES))
                raise ValueError(f"Sequence contains residues unsupported by ESM: {invalid_tokens}")

        batch = [(f"protein_{index}", sequence) for index, sequence in enumerate(sequences)]
        _, _, tokens = self.batch_converter(batch)
        tokens = tokens.to(next(self.model.parameters()).device)

        outputs = self.model(tokens, repr_layers=[self.model.num_layers], return_contacts=False)
        hidden_states = outputs["representations"][self.model.num_layers]

        residue_embeddings = []
        residue_masks = []
        for batch_index, sequence in enumerate(sequences):
            sequence_length = len(sequence)
            residue_embeddings.append(hidden_states[batch_index, 1 : sequence_length + 1])
            residue_masks.append(torch.ones(sequence_length, dtype=torch.bool, device=hidden_states.device))

        padded_embeddings = pad_sequence(residue_embeddings, batch_first=True)
        padded_masks = pad_sequence(residue_masks, batch_first=True, padding_value=False)
        return masked_mean_pooling(padded_embeddings, padded_masks)


class TextEmbeddingEncoder(nn.Module):
    """Freeze the target LLM tokenizer and embedding table for text targets."""

    def __init__(self, model_name: str, cache_dir: Path | None, local_files_only: bool) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            local_files_only=local_files_only,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            local_files_only=local_files_only,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
        embedding_weight = llm.get_input_embeddings().weight.detach().cpu()
        del llm

        self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=True)
        self.output_dim = int(embedding_weight.shape[1])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.embedding(input_ids)
        return masked_mean_pooling(token_embeddings, attention_mask.bool())


class QFormerProteinAdaptor(nn.Module):
    """Map one antibody vector into fixed-length LLM-space query embeddings."""

    def __init__(
        self,
        protein_dim: int,
        output_dim: int,
        query_length: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if output_dim % num_heads != 0:
            raise ValueError("output_dim must be divisible by num_heads.")

        self.protein_dim = protein_dim
        self.output_dim = output_dim
        self.query_length = query_length
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.protein_projection = nn.Sequential(
            nn.Linear(protein_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.query_tokens = nn.Parameter(torch.randn(query_length, output_dim) * 0.02)

        self.cross_attention_norm = nn.LayerNorm(output_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.self_attention_norm = nn.LayerNorm(output_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.feed_forward_norm = nn.LayerNorm(output_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, protein_embeddings: torch.Tensor) -> torch.Tensor:
        if protein_embeddings.ndim != 2:
            raise ValueError("protein_embeddings must have shape (batch_size, protein_dim).")

        protein_context = self.protein_projection(protein_embeddings).unsqueeze(1)
        query_tokens = self.query_tokens.unsqueeze(0).expand(protein_embeddings.size(0), -1, -1)

        cross_input = self.cross_attention_norm(query_tokens)
        cross_output, _ = self.cross_attention(cross_input, protein_context, protein_context, need_weights=False)
        query_tokens = query_tokens + cross_output

        self_input = self.self_attention_norm(query_tokens)
        self_output, _ = self.self_attention(self_input, self_input, self_input, need_weights=False)
        query_tokens = query_tokens + self_output

        feed_forward_input = self.feed_forward_norm(query_tokens)
        query_tokens = query_tokens + self.feed_forward(feed_forward_input)
        return self.output_norm(query_tokens)


class InfoNCE(nn.Module):
    """Symmetric InfoNCE over pooled pre-embedding and text embedding."""

    def __init__(self, temperature: float) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, predicted_embeddings: torch.Tensor, target_embeddings: torch.Tensor) -> torch.Tensor:
        # InfoNCE needs at least one negative sample, so a local batch of size 1 is invalid.
        if predicted_embeddings.size(0) < 2 or target_embeddings.size(0) < 2:
            raise ValueError("InfoNCE requires batch_size >= 2 on each rank.")

        predicted_embeddings = F.normalize(predicted_embeddings, dim=1)
        target_embeddings = F.normalize(target_embeddings, dim=1)

        logits = (predicted_embeddings @ target_embeddings.T) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_a + loss_b)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 pretraining from prepared caption antibody-text pairs.")
    parser.add_argument("--data-path", type=Path, default=Path("data/cache/afd_pretrain_caption.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage1"))
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--protein-model-name", type=str, default="esm2_t36_3B_UR50D")
    parser.add_argument("--text-model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--text-model-cache-dir", type=Path, default=Path("models"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--validation-split", type=float, default=0.25)
    parser.add_argument("--max-text-length", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--query-length", type=int, default=100)
    parser.add_argument("--adaptor-hidden-dim", type=int, default=8192)
    parser.add_argument("--adaptor-num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--log-file-name", type=str, default="train_log.jsonl")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PretrainConfig:
    if args.batch_size < 2:
        raise ValueError("Stage 1 InfoNCE pretraining requires --batch-size >= 2.")

    return PretrainConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        auto_resume=args.auto_resume,
        protein_model_name=args.protein_model_name,
        text_model_name=args.text_model_name,
        text_model_cache_dir=args.text_model_cache_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_split=args.validation_split,
        max_text_length=args.max_text_length,
        max_seq_length=args.max_seq_length,
        query_length=args.query_length,
        adaptor_hidden_dim=args.adaptor_hidden_dim,
        adaptor_num_heads=args.adaptor_num_heads,
        dropout=args.dropout,
        temperature=args.temperature,
        text_embedding_dim=None,
        num_workers=args.num_workers,
        seed=args.seed,
        local_files_only=args.local_files_only,
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


def load_afd_pretrain_samples(
    data_path: Path,
    max_seq_length: int,
) -> tuple[int, int, list[AFDPretrainSample]]:
    total = 0
    kept = 0
    samples: list[AFDPretrainSample] = []

    if not data_path.exists():
        raise FileNotFoundError(f"Prepared pretrain data not found: {data_path}")

    with data_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError(f"Expected {data_path} to contain a JSON list of samples.")

    for row_index, record in enumerate(payload):
        if not isinstance(record, dict):
            continue

        total += 1
        heavy_sequence = normalize_sequence(str(record.get("heavy_sequence", "")))
        light_sequence = normalize_sequence(str(record.get("light_sequence", "")))
        text = str(record.get("text", "")).strip()
        if not heavy_sequence or not light_sequence or not text:
            continue
        if not is_esm_compatible_sequence(heavy_sequence) or not is_esm_compatible_sequence(light_sequence):
            continue
        if len(heavy_sequence) > max_seq_length or len(light_sequence) > max_seq_length:
            continue

        samples.append(
            AFDPretrainSample(
                sample_id=str(record.get("sample_id", "")).strip() or f"{data_path.stem}-{row_index}",
                heavy_sequence=heavy_sequence,
                light_sequence=light_sequence,
                text=text,
            )
        )
        kept += 1

    if not samples:
        raise ValueError(f"No usable AFD pretraining samples were loaded from {data_path}.")
    return total, kept, samples


def split_samples(samples: list[AFDPretrainSample], validation_split: float, seed: int) -> tuple[list[AFDPretrainSample], list[AFDPretrainSample]]:
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


def build_dataloaders(
    config: PretrainConfig,
    tokenizer: AutoTokenizer,
    distributed: bool,
) -> tuple[DataLoader, DataLoader | None, DistributedSampler | None]:
    total, kept, samples = load_afd_pretrain_samples(config.data_path, config.max_seq_length)
    if is_main_process():
        ratio = kept / max(total, 1)
        print(f"Kept {kept}/{total} prepared caption pretraining samples (ratio={ratio:.2f})")

    train_samples, validation_samples = split_samples(samples, config.validation_split, config.seed)
    if len(train_samples) < config.batch_size:
        raise ValueError("Training set is smaller than batch_size after filtering; reduce --batch-size or relax filters.")

    collator = AFDPretrainCollator(tokenizer=tokenizer, max_text_length=config.max_text_length)
    train_dataset = AFDPretrainDataset(train_samples)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=config.num_workers,
        collate_fn=collator,
        drop_last=True,
    )

    validation_loader = None
    if validation_samples and len(validation_samples) >= config.batch_size:
        validation_dataset = AFDPretrainDataset(validation_samples)
        validation_sampler = DistributedSampler(validation_dataset, shuffle=False) if distributed else None
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            sampler=validation_sampler,
            num_workers=config.num_workers,
            collate_fn=collator,
            drop_last=True,
        )

    return train_loader, validation_loader, train_sampler


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    moved["input_ids"] = batch["input_ids"].to(device)
    moved["attention_mask"] = batch["attention_mask"].to(device)
    return moved


def run_batch(
    batch: dict[str, Any],
    protein_encoder: ProteinEncoder,
    text_encoder: TextEmbeddingEncoder,
    adaptor: nn.Module,
    criterion: InfoNCE,
) -> torch.Tensor:
    with torch.no_grad():
        heavy_embeddings = protein_encoder(batch["heavy_sequences"])
        light_embeddings = protein_encoder(batch["light_sequences"])
        antibody_embeddings = torch.cat([heavy_embeddings, light_embeddings], dim=1)
        text_embeddings = text_encoder(batch["input_ids"], batch["attention_mask"])

    pre_embeddings = adaptor(antibody_embeddings)
    pooled_pre_embeddings = pre_embeddings.mean(dim=1)
    return criterion(pooled_pre_embeddings, text_embeddings)


def evaluate(
    dataloader: DataLoader | None,
    protein_encoder: ProteinEncoder,
    text_encoder: TextEmbeddingEncoder,
    adaptor: nn.Module,
    criterion: InfoNCE,
    device: torch.device,
) -> float | None:
    if dataloader is None:
        return None

    adaptor.eval()
    loss_sum = torch.tensor(0.0, device=device)
    batch_count = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            loss_sum += run_batch(batch, protein_encoder, text_encoder, adaptor, criterion).detach()
            batch_count += 1

    if is_distributed():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(batch_count, op=dist.ReduceOp.SUM)
    return (loss_sum / batch_count.clamp(min=1.0)).item()


def append_epoch_log(log_path: Path, payload: dict[str, Any]) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoint_paths = sorted(output_dir.glob("checkpoint_epoch_*.pt"))
    if not checkpoint_paths:
        return None
    return max(checkpoint_paths, key=lambda path: int(path.stem.split("_")[-1]))


def build_adaptor_from_checkpoint(
    checkpoint_path: Path,
    protein_dim: int,
    map_location: str | torch.device = "cpu",
) -> tuple[QFormerProteinAdaptor, dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Adaptor checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    adaptor_state_dict = (
        checkpoint.get("adaptor_state_dict")
        or checkpoint.get("adaptor")
        or checkpoint.get("align_model")
        or checkpoint
    )
    if not isinstance(adaptor_state_dict, dict):
        raise ValueError("Adaptor checkpoint does not contain a usable state dict.")

    adaptor_config = dict(checkpoint.get("adaptor_config") or {})
    if not adaptor_config:
        query_tokens = adaptor_state_dict.get("query_tokens")
        protein_projection_weight = adaptor_state_dict.get("protein_projection.0.weight")
        feed_forward_weight = adaptor_state_dict.get("feed_forward.0.weight")
        if query_tokens is None or protein_projection_weight is None or feed_forward_weight is None:
            raise ValueError("Adaptor config is missing and could not be inferred from the checkpoint.")
        adaptor_config = {
            "protein_dim": int(protein_projection_weight.shape[1]),
            "output_dim": int(query_tokens.shape[1]),
            "query_length": int(query_tokens.shape[0]),
            "hidden_dim": int(feed_forward_weight.shape[0]),
            "num_heads": 8,
            "dropout": 0.0,
        }

    adaptor = QFormerProteinAdaptor(
        protein_dim=protein_dim,
        output_dim=int(adaptor_config["output_dim"]),
        query_length=int(adaptor_config["query_length"]),
        hidden_dim=int(adaptor_config["hidden_dim"]),
        num_heads=int(adaptor_config["num_heads"]),
        dropout=float(adaptor_config.get("dropout", 0.0)),
    )
    adaptor.load_state_dict(adaptor_state_dict, strict=True)
    adaptor.eval()
    return adaptor, adaptor_config


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    config: PretrainConfig,
    adaptor: QFormerProteinAdaptor,
    optimizer: torch.optim.Optimizer,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "config": asdict(config),
        "adaptor_state_dict": adaptor.state_dict(),
        "adaptor_config": {
            "protein_dim": adaptor.protein_dim,
            "output_dim": adaptor.output_dim,
            "query_length": adaptor.query_length,
            "hidden_dim": adaptor.hidden_dim,
            "num_heads": adaptor.num_heads,
            "dropout": config.dropout,
        },
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch}.pt")


def resolve_resume_checkpoint(resume_from: Path | None, auto_resume: bool, output_dir: Path) -> Path | None:
    if resume_from is not None:
        if not resume_from.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")
        return resume_from
    if auto_resume:
        return find_latest_checkpoint(output_dir)
    return None


def load_training_state(
    checkpoint_path: Path,
    adaptor: QFormerProteinAdaptor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    adaptor.load_state_dict(checkpoint["adaptor_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    resumed_epoch = int(checkpoint["epoch"])
    return resumed_epoch + 1


def train() -> None:
    args = parse_args()
    config = build_config(args)
    set_seed(config.seed)

    device, local_rank, world_size = setup_distributed()
    if is_main_process():
        print(f"Using device: {device}, world_size: {world_size}")

    text_encoder = TextEmbeddingEncoder(
        model_name=config.text_model_name,
        cache_dir=config.text_model_cache_dir,
        local_files_only=config.local_files_only,
    ).to(device)
    config.text_embedding_dim = text_encoder.output_dim

    protein_encoder = ProteinEncoder(config.protein_model_name).to(device)
    adaptor = QFormerProteinAdaptor(
        protein_dim=protein_encoder.output_dim * 2,
        output_dim=text_encoder.output_dim,
        query_length=config.query_length,
        hidden_dim=config.adaptor_hidden_dim,
        num_heads=config.adaptor_num_heads,
        dropout=config.dropout,
    ).to(device)
    criterion = InfoNCE(config.temperature)

    protein_encoder.eval()
    text_encoder.eval()
    for parameter in protein_encoder.parameters():
        parameter.requires_grad = False
    for parameter in text_encoder.parameters():
        parameter.requires_grad = False

    train_loader, validation_loader, train_sampler = build_dataloaders(config, text_encoder.tokenizer, is_distributed())
    optimizer = torch.optim.AdamW(adaptor.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    start_epoch = 1

    resume_checkpoint = resolve_resume_checkpoint(config.resume_from, config.auto_resume, config.output_dir)
    if resume_checkpoint is not None:
        start_epoch = load_training_state(resume_checkpoint, adaptor, optimizer, device)
        if is_main_process():
            print(f"Resuming Stage 1 from {resume_checkpoint} at epoch {start_epoch}")

    if is_distributed():
        adaptor = DDP(adaptor, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False) if device.type == "cuda" else DDP(adaptor)

    if is_main_process():
        config.output_dir.mkdir(parents=True, exist_ok=True)
        with (config.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
            json.dump(asdict(config), handle, ensure_ascii=False, indent=2, default=str)
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

        adaptor.train()
        running_loss = 0.0
        iterator = train_loader
        if is_main_process():
            iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}")

        for batch in iterator:
            batch = move_batch_to_device(batch, device)
            loss = run_batch(batch, protein_encoder, text_encoder, adaptor, criterion)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if is_main_process():
                iterator.set_postfix(loss=f"{loss.item():.6f}")

        train_loss_tensor = torch.tensor(running_loss, device=device)
        batch_count_tensor = torch.tensor(float(len(train_loader)), device=device)
        if is_distributed():
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(batch_count_tensor, op=dist.ReduceOp.SUM)
        train_loss = (train_loss_tensor / batch_count_tensor.clamp(min=1.0)).item()
        validation_loss = evaluate(validation_loader, protein_encoder, text_encoder, adaptor, criterion, device)

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
                save_checkpoint(
                    config.output_dir,
                    epoch,
                    config,
                    adaptor.module if isinstance(adaptor, DDP) else adaptor,
                    optimizer,
                )

    if is_main_process():
        save_checkpoint(
            config.output_dir,
            config.epochs,
            config,
            adaptor.module if isinstance(adaptor, DDP) else adaptor,
            optimizer,
        )
    cleanup_distributed()


if __name__ == "__main__":
    train()
