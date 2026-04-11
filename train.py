from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import esm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer


@dataclass
class TrainConfig:
    data_dir: Path
    output_dir: Path
    protein_model_name: str
    text_model_name: str
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    validation_split: float
    max_text_length: int
    max_seq_length:int
    hidden_dim: int
    dropout: float
    temperature: float
    num_workers: int
    seed: int
    local_files_only: bool
    log_file_name: str
    save_every_epoch: bool


@dataclass
class UniProtSample:
    sample_id: str
    sequence: str
    function: str


class UniProtDataset(Dataset):
    """Simple dataset wrapper for UniProt sequence-function pairs."""

    def __init__(self, samples: list[UniProtSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, str]:
        sample = self.samples[index]
        return {
            "id": sample.sample_id,
            "sequence": sample.sequence,
            "function": sample.function,
        }


class UniProtCollator:
    """Tokenize function text and pad it to a fixed length for batching."""

    def __init__(self, tokenizer, max_text_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

    def __call__(self, batch: list[dict[str, str]]) -> dict[str, object]:
        functions = [item["function"] for item in batch]
        tokenized = self.tokenizer(
            functions,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
        )
        return {
            "ids": [item["id"] for item in batch],
            "sequences": [item["sequence"] for item in batch],
            "functions": functions,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }


class ProteinEncoder(nn.Module):
    """Freeze ESM and pool residue embeddings into one protein embedding."""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        pretrained_builder = getattr(esm.pretrained, model_name)
        self.model, self.alphabet = pretrained_builder()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.output_dim = self.model.embed_dim

        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def forward(self, sequences: list[str]) -> torch.Tensor:
        data = [(f"protein_{index}", sequence) for index, sequence in enumerate(sequences)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(next(self.model.parameters()).device)

        outputs = self.model(tokens, repr_layers=[self.model.num_layers], return_contacts=False)
        hidden_states = outputs["representations"][self.model.num_layers]

        residue_embeddings = []
        residue_masks = []
        for index, sequence in enumerate(sequences):
            length = len(sequence)
            residue_embeddings.append(hidden_states[index, 1 : length + 1])
            residue_masks.append(torch.ones(length, dtype=torch.bool, device=hidden_states.device))

        padded_embeddings = pad_sequence(residue_embeddings, batch_first=True)
        padded_masks = pad_sequence(residue_masks, batch_first=True, padding_value=False)

        # Pool only valid residues and ignore padded positions.
        return masked_mean_pooling(padded_embeddings, padded_masks)


class TextEmbeddingEncoder:
    """Use the tokenizer output directly as the padded text representation."""

    def __init__(self, model_name: str, local_files_only: bool) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=local_files_only)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.output_dim = None

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        #do nothing
        token_values = input_ids.float()
        # token_values = input_ids.float() * attention_mask.float()
        return F.normalize(token_values, dim=1)


class ProteinTextAdaptor(nn.Module):
    """Map pooled protein embeddings into the padded token-sequence space."""

    def __init__(self, protein_dim: int, output_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, protein_embeddings: torch.Tensor) -> torch.Tensor:
        return self.layers(protein_embeddings)


class InfoNCE(nn.Module):
    """Symmetric InfoNCE over pooled protein and pooled text embeddings."""

    def __init__(self, temperature: float) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, protein_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        protein_embeddings = F.normalize(protein_embeddings, dim=1)
        text_embeddings = F.normalize(text_embeddings, dim=1)

        logits = (protein_embeddings @ text_embeddings.T) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        protein_to_text = F.cross_entropy(logits, labels)
        text_to_protein = F.cross_entropy(logits.T, labels)
        return 0.5 * (protein_to_text + text_to_protein)


def parse_args() -> argparse.Namespace:
    """Define training-time CLI options."""

    parser = argparse.ArgumentParser(description="Train a protein-to-text adaptor on UniProt TSV files.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/uniportData"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--protein-model-name", type=str, default="esm2_t36_3B_UR50D")
    parser.add_argument("--text-model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--max-text-length", type=int, default=300)
    parser.add_argument("--max-seq-length", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--log-file-name", type=str, default="train_log.jsonl")
    parser.add_argument("--save-every-epoch", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    """Pack parsed arguments into one config object."""

    return TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        protein_model_name=args.protein_model_name,
        text_model_name=args.text_model_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_split=args.validation_split,
        max_text_length=args.max_text_length,
        max_seq_length=args.max_seq_length,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        temperature=args.temperature,
        num_workers=args.num_workers,
        seed=args.seed,
        local_files_only=args.local_files_only,
        log_file_name=args.log_file_name,
        save_every_epoch=args.save_every_epoch,
    )


def set_seed(seed: int) -> None:
    """Make dataloader shuffling and training reproducible."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_sequence(sequence: str) -> str:
    """Remove whitespace and standardize amino-acid sequences."""

    return "".join(sequence.split()).upper()


def load_uniprot_samples(data_dir: Path, max_text_length: int,max_seq_length:int) -> list[UniProtSample]:
    """Load TSV files and filter by Token_Length."""
    
    total=0
    kept=0
    samples: list[UniProtSample] = []
    tsv_files = sorted(data_dir.glob("*.tsv"))
    if not tsv_files:
        raise FileNotFoundError(f"No TSV files were found in {data_dir}.")

    for tsv_path in tsv_files:
        with tsv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")

            required = {"ID", "Sequence", "Function", "Token_Length"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                missing_text = ", ".join(sorted(missing))
                raise ValueError(f"{tsv_path} is missing required columns: {missing_text}")
            for row_index, row in enumerate(reader):
                total+=1
                
                sequence = normalize_sequence(str(row.get("Sequence", "")))
                if(len(sequence)>max_seq_length):
                    continue
                function = str(row.get("Function", "")).strip()
                sample_id = str(row.get("ID", "")).strip() or f"{tsv_path.stem}-{row_index}"

                try:
                    token_length = int(row.get("Token_Length", 0))
                except ValueError:
                    continue  
                if token_length > max_text_length:
                    continue
                kept+=1
                if not sequence or not function:
                    continue
                samples.append(
                    UniProtSample(
                        sample_id=sample_id,
                        sequence=sequence,
                        function=function
                    )
                )
    if not samples:
        raise ValueError(f"No usable rows were loaded from {data_dir}.")

    return total,kept,samples


def split_samples(samples: list[UniProtSample], validation_split: float, seed: int) -> tuple[list[UniProtSample], list[UniProtSample]]:
    """Randomly split rows into train and validation subsets."""

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


def create_dataloaders(config: TrainConfig, tokenizer,max_text_length: int,max_seq_length:int) -> tuple[DataLoader, DataLoader | None]:
    """Create train/validation loaders using the UniProt TSV files."""

    total,kept,samples = load_uniprot_samples(config.data_dir,max_text_length,max_seq_length)
    print(f"Kept {kept}/{total} samples (ratio={kept/total:.2f})")
    train_samples, validation_samples = split_samples(samples, config.validation_split, config.seed)
    collator = UniProtCollator(tokenizer=tokenizer, max_text_length=config.max_text_length)

    train_loader = DataLoader(
        UniProtDataset(train_samples),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collator,
    )

    validation_loader = None
    if validation_samples:
        validation_loader = DataLoader(
            UniProtDataset(validation_samples),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collator,
        )
    return train_loader, validation_loader


def masked_mean_pooling(embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool only valid positions and ignore padded positions."""

    float_mask = mask.unsqueeze(-1).to(embeddings.dtype)
    summed = (embeddings * float_mask).sum(dim=1)
    counts = float_mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    """Move tensor fields to the selected device while keeping metadata intact."""

    moved = dict(batch)
    moved["input_ids"] = batch["input_ids"].to(device)
    moved["attention_mask"] = batch["attention_mask"].to(device)
    return moved


def run_batch(
    batch: dict[str, object],
    protein_encoder: ProteinEncoder,
    text_encoder: TextEmbeddingEncoder,
    adaptor: ProteinTextAdaptor,
    criterion: InfoNCE,
) -> torch.Tensor:
    """Encode one batch and compute the contrastive alignment loss."""

    with torch.no_grad():
        pooled_protein_embeddings = protein_encoder(batch["sequences"])
        pooled_text_embeddings = text_encoder.encode(batch["input_ids"], batch["attention_mask"])

    predicted_text_embeddings = adaptor(pooled_protein_embeddings)
    return criterion(predicted_text_embeddings, pooled_text_embeddings)


def evaluate(
    dataloader: DataLoader | None,
    protein_encoder: ProteinEncoder,
    text_encoder: TextEmbeddingEncoder,
    adaptor: ProteinTextAdaptor,
    criterion: InfoNCE,
    device: torch.device,
) -> float | None:
    """Run validation without gradient updates."""

    if dataloader is None:
        return None

    adaptor.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            loss = run_batch(batch, protein_encoder, text_encoder, adaptor, criterion)
            losses.append(loss.item())
    return sum(losses) / max(len(losses), 1)


def append_epoch_log(log_path: Path, payload: dict) -> None:
    """Append one epoch summary as a JSON line."""

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    config: TrainConfig,
    adaptor: ProteinTextAdaptor,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Save the trainable adaptor and optimizer state."""

    checkpoint = {
        "epoch": epoch,
        "config": asdict(config),
        "adaptor": adaptor.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch}.pt")


def train() -> None:
    """Main training loop for protein-text contrastive alignment."""

    args = parse_args()
    config = build_config(args)
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = config.output_dir / config.log_file_name
    with (config.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2, default=str)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("")

    text_encoder = TextEmbeddingEncoder(
        model_name=config.text_model_name,
        local_files_only=config.local_files_only,
    )
    train_loader, validation_loader = create_dataloaders(config, text_encoder.tokenizer,config.max_text_length,config.max_seq_length)

    protein_encoder = ProteinEncoder(config.protein_model_name).to(device)
    adaptor = ProteinTextAdaptor(
        protein_dim=protein_encoder.output_dim,
        output_dim=config.max_text_length,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    criterion = InfoNCE(temperature=config.temperature)

    protein_encoder.eval()
    optimizer = torch.optim.AdamW(
        adaptor.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    for epoch in range(1, config.epochs + 1):
        adaptor.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}")

        for batch in progress:
            batch = move_batch_to_device(batch, device)
            loss = run_batch(batch, protein_encoder, text_encoder, adaptor, criterion)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.6f}")

        train_loss = running_loss / max(len(train_loader), 1)
        validation_loss = evaluate(validation_loader, protein_encoder, text_encoder, adaptor, criterion, device)

        summary = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "epoch": epoch,
            "train_loss": train_loss,
        }
        if validation_loss is not None:
            summary["validation_loss"] = validation_loss
        print(json.dumps(summary, indent=2))
        append_epoch_log(log_path, summary)

        if config.save_every_epoch:
            save_checkpoint(config.output_dir, epoch, config, adaptor, optimizer)

    save_checkpoint(config.output_dir, config.epochs, config, adaptor, optimizer)


if __name__ == "__main__":
    train()
