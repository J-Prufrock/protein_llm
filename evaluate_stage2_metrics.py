from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Any

import nltk
import torch
from bert_score import score as bert_score
from nltk.translate.meteor_score import meteor_score
from peft import PeftModel
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pretrain import QFormerProteinAdaptor, build_adaptor_from_checkpoint
from train import (
    apply_data_ratio,
    build_prompt_token_ids,
    ensure_antibody_prefix_token,
    filter_overlength_samples,
    load_prepared_cache,
    split_indices,
    unwrap_model,
)

DEFAULT_BERTSCORE_MODEL = "roberta-large"
DEFAULT_BIOMED_BERTSCORE_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
DEFAULT_BIOMED_BERTSCORE_NUM_LAYERS = 9
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 runs with standard generation metrics.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs/stage2"))
    parser.add_argument("--experiments", nargs="+", default=["usePretoken", "noPretoken"])
    parser.add_argument("--experiment-epochs", nargs="*", default=None, help="Optional overrides like usePretoken=10 noPretoken=15")
    parser.add_argument("--cache-path", type=Path, default=Path("data/cache/afd_stage2_cache.pt"))
    parser.add_argument("--epoch", type=int, default=10, help="Evaluate the same epoch for all experiments unless overridden.")
    parser.add_argument("--data-ratio", type=float, default=None, help="Override data ratio for evaluation. Default: use each experiment's run_config value.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--predictions-dir", type=Path, default=Path("outputs/stage2/metrics"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--bertscore-model", type=str, default=DEFAULT_BERTSCORE_MODEL)
    parser.add_argument("--bertscore-num-layers", type=int, default=None)
    parser.add_argument("--biomed-bertscore-model", type=str, default=DEFAULT_BIOMED_BERTSCORE_MODEL)
    parser.add_argument("--biomed-bertscore-num-layers", type=int, default=DEFAULT_BIOMED_BERTSCORE_NUM_LAYERS)
    parser.add_argument("--bertscore-batch-size", type=int, default=8)
    return parser.parse_args()


def exact_match(references: list[str], hypotheses: list[str]) -> float:
    matches = 0
    for reference, hypothesis in zip(references, hypotheses):
        if reference.strip() == hypothesis.strip():
            matches += 1
    return (matches / max(len(references), 1)) * 100.0


def tokenize_for_meteor(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def ensure_meteor_resources() -> None:
    for resource_name, resource_path in (("wordnet", "corpora/wordnet"), ("omw-1.4", "corpora/omw-1.4")):
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)


def parse_experiment_epoch_overrides(raw_items: list[str] | None) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for item in raw_items or []:
        if "=" not in item:
            raise ValueError(f"Invalid --experiment-epochs item: {item}. Expected format experiment=epoch.")
        experiment, raw_epoch = item.split("=", 1)
        experiment = experiment.strip()
        if not experiment:
            raise ValueError(f"Invalid --experiment-epochs item: {item}. Experiment name is empty.")
        overrides[experiment] = int(raw_epoch)
    return overrides


def resolve_data_ratio(run_config: dict[str, Any], override: float | None) -> float:
    data_ratio = float(run_config.get("data_ratio", 1.0) if override is None else override)
    if not 0.0 < data_ratio <= 1.0:
        raise ValueError(f"data_ratio must be in (0.0, 1.0], got {data_ratio}.")
    return data_ratio


def find_latest_adapter_dir(experiment_dir: Path) -> Path:
    candidates = sorted(path for path in experiment_dir.glob("lora_epoch_*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(
            f"No LoRA adapter directory found under {experiment_dir}. Expected something like lora_epoch_10/."
        )
    return max(candidates, key=lambda path: int(path.name.split("_")[-1]))


def find_adapter_dir_for_epoch(experiment_dir: Path, epoch: int) -> Path:
    adapter_dir = experiment_dir / f"lora_epoch_{epoch}"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Missing adapter directory for epoch {epoch}: {adapter_dir}")
    return adapter_dir


def resolve_adapter_dir(experiment_dir: Path, experiment: str, args: argparse.Namespace) -> Path:
    experiment_epoch_overrides = parse_experiment_epoch_overrides(args.experiment_epochs)
    if experiment in experiment_epoch_overrides:
        return find_adapter_dir_for_epoch(experiment_dir, experiment_epoch_overrides[experiment])
    if args.epoch is not None:
        return find_adapter_dir_for_epoch(experiment_dir, args.epoch)
    return find_latest_adapter_dir(experiment_dir)


def find_matching_checkpoint(experiment_dir: Path, epoch: int) -> Path | None:
    checkpoint_path = experiment_dir / f"checkpoint_epoch_{epoch}.pt"
    if checkpoint_path.exists():
        return checkpoint_path
    return None


def load_run_config(experiment_dir: Path) -> dict[str, Any]:
    config_path = experiment_dir / "run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run_config.json: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def infer_bertscore_num_layers(
    model_name: str,
    explicit_num_layers: int | None,
    cache_dir: str | None,
    local_files_only: bool,
) -> int | None:
    if explicit_num_layers is not None:
        return explicit_num_layers

    if model_name == DEFAULT_BIOMED_BERTSCORE_MODEL:
        return DEFAULT_BIOMED_BERTSCORE_NUM_LAYERS

    try:
        config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=local_files_only)
    except Exception:
        return None

    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    if not isinstance(num_hidden_layers, int) or num_hidden_layers <= 0:
        return None
    return max(1, num_hidden_layers - 3)


def get_model_max_length(
    model_name: str,
    cache_dir: str | None,
    local_files_only: bool,
) -> int | None:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=local_files_only)
        config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=local_files_only)
    except Exception:
        return None

    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if not isinstance(tokenizer_limit, int) or tokenizer_limit <= 0 or tokenizer_limit > 1_000_000:
        tokenizer_limit = None

    config_limit = getattr(config, "max_position_embeddings", None)
    if not isinstance(config_limit, int) or config_limit <= 0:
        config_limit = None

    limits = [limit for limit in (tokenizer_limit, config_limit) if limit is not None]
    if not limits:
        return None
    return min(limits)


def truncate_texts_for_bertscore(
    texts: list[str],
    model_name: str,
    cache_dir: str | None,
    local_files_only: bool,
) -> tuple[list[str], int, int | None]:
    max_length = get_model_max_length(model_name, cache_dir, local_files_only)
    if max_length is None:
        return texts, 0, None

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=local_files_only)
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    text_max_length = max(1, max_length - special_tokens)

    truncated_texts: list[str] = []
    truncated_count = 0
    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        if len(token_ids) > text_max_length:
            token_ids = token_ids[:text_max_length]
            truncated_count += 1
        truncated_texts.append(tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True))
    return truncated_texts, truncated_count, max_length


def build_prompt_embeddings(
    instruction: str,
    tokenizer: AutoTokenizer,
    llm: torch.nn.Module,
    adaptor: QFormerProteinAdaptor,
    antibody_embedding: torch.Tensor,
    antibody_prefix_token_id: int,
    use_template: bool,
    use_pretoken: bool,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    prompt_ids = build_prompt_token_ids(tokenizer, instruction, use_template, use_pretoken)
    embedder = unwrap_model(llm).get_input_embeddings()
    embedding_dtype = embedder.weight.dtype
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    prompt_token_embeddings = embedder(prompt_tensor).to(dtype=embedding_dtype)

    if not use_pretoken:
        return prompt_token_embeddings.unsqueeze(0), prompt_token_embeddings.size(0)

    if prompt_ids.count(antibody_prefix_token_id) != 1:
        raise ValueError("Prompt must contain exactly one antibody prefix placeholder.")

    # Replace the single placeholder token with the adaptor-generated antibody prefix embeddings.
    prefix_embeddings = adaptor(antibody_embedding.unsqueeze(0).to(device=device, dtype=torch.float32)).to(dtype=embedding_dtype)
    placeholder_position = prompt_ids.index(antibody_prefix_token_id)
    prompt_embeddings = torch.cat(
        [
            prompt_token_embeddings[:placeholder_position],
            prefix_embeddings[0],
            prompt_token_embeddings[placeholder_position + 1 :],
        ],
        dim=0,
    )
    return prompt_embeddings.unsqueeze(0), prompt_embeddings.size(0)


def left_pad_tensors(
    tensors: list[torch.Tensor],
    padding_value: float | int,
) -> torch.Tensor:
    max_length = max(tensor.size(0) for tensor in tensors)
    padded_tensors: list[torch.Tensor] = []
    for tensor in tensors:
        pad_length = max_length - tensor.size(0)
        if tensor.dim() == 1:
            padded = torch.full((max_length,), padding_value, dtype=tensor.dtype, device=tensor.device)
            padded[pad_length:] = tensor
        else:
            padded = torch.full(
                (max_length, tensor.size(1)),
                padding_value,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            padded[pad_length:, :] = tensor
        padded_tensors.append(padded)
    return torch.stack(padded_tensors, dim=0)


def build_batched_prompt_embeddings(
    instructions: list[str],
    antibody_embeddings: torch.Tensor,
    tokenizer: AutoTokenizer,
    llm: torch.nn.Module,
    adaptor: QFormerProteinAdaptor,
    antibody_prefix_token_id: int,
    use_template: bool,
    use_pretoken: bool,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    del max_new_tokens
    embedder = unwrap_model(llm).get_input_embeddings()
    embedding_dtype = embedder.weight.dtype
    prompt_embedding_tensors: list[torch.Tensor] = []
    attention_mask_tensors: list[torch.Tensor] = []

    batched_prefix_embeddings = None
    if use_pretoken:
        batched_prefix_embeddings = adaptor(antibody_embeddings.to(device=device, dtype=torch.float32)).to(dtype=embedding_dtype)

    for sample_index, instruction in enumerate(instructions):
        prompt_ids = build_prompt_token_ids(tokenizer, instruction, use_template, use_pretoken)
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        prompt_token_embeddings = embedder(prompt_tensor).to(dtype=embedding_dtype)

        if use_pretoken:
            if prompt_ids.count(antibody_prefix_token_id) != 1:
                raise ValueError("Prompt must contain exactly one antibody prefix placeholder.")
            placeholder_position = prompt_ids.index(antibody_prefix_token_id)
            sample_prompt_embeddings = torch.cat(
                [
                    prompt_token_embeddings[:placeholder_position],
                    batched_prefix_embeddings[sample_index],
                    prompt_token_embeddings[placeholder_position + 1 :],
                ],
                dim=0,
            )
        else:
            sample_prompt_embeddings = prompt_token_embeddings

        prompt_embedding_tensors.append(sample_prompt_embeddings)
        attention_mask_tensors.append(torch.ones(sample_prompt_embeddings.size(0), dtype=torch.long, device=device))

    # Left padding keeps every sample's effective prompt aligned to the batch tail,
    # which is safer for decoder-only generation than right-padded prompt embeddings.
    inputs_embeds = left_pad_tensors(prompt_embedding_tensors, padding_value=0.0)
    attention_mask = left_pad_tensors(attention_mask_tensors, padding_value=0)
    padded_prompt_length = int(inputs_embeds.size(1))
    return inputs_embeds, attention_mask, padded_prompt_length


def generate_predictions_batch(
    instructions: list[str],
    antibody_embeddings: torch.Tensor,
    tokenizer: AutoTokenizer,
    llm: torch.nn.Module,
    adaptor: QFormerProteinAdaptor,
    antibody_prefix_token_id: int,
    use_template: bool,
    use_pretoken: bool,
    max_new_tokens: int,
    device: torch.device,
) -> list[str]:
    inputs_embeds, attention_mask, padded_prompt_length = build_batched_prompt_embeddings(
        instructions,
        antibody_embeddings,
        tokenizer,
        llm,
        adaptor,
        antibody_prefix_token_id,
        use_template,
        use_pretoken,
        max_new_tokens,
        device,
    )

    with torch.no_grad():
        generated = llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    predictions: list[str] = []
    for generated_ids_tensor in generated:
        generated_ids = generated_ids_tensor.tolist()
        answer_ids = generated_ids[padded_prompt_length:] if len(generated_ids) > padded_prompt_length else generated_ids
        predictions.append(tokenizer.decode(answer_ids, skip_special_tokens=True).strip())
    return predictions


def load_test_subset(
    run_config: dict[str, Any],
    tokenizer: AutoTokenizer,
    antibody_prefix_token_id: int,
    prefix_length: int,
    use_template: bool,
    cache_path: Path,
    use_pretoken: bool,
    data_ratio: float,
) -> tuple[list[dict[str, Any]], torch.Tensor, int, int]:
    samples, antibody_embeddings = load_prepared_cache(cache_path)
    samples, antibody_embeddings, _ = filter_overlength_samples(
        samples,
        antibody_embeddings,
        tokenizer,
        antibody_prefix_token_id,
        prefix_length,
        int(run_config["max_length"]),
        use_template,
        use_pretoken,
    )
    filtered_count = len(samples)
    samples, antibody_embeddings = apply_data_ratio(samples, antibody_embeddings, data_ratio, int(run_config["seed"]))
    train_indices, validation_indices, test_indices = split_indices(
        len(samples),
        float(run_config["validation_split"]),
        float(run_config["test_split"]),
        int(run_config["seed"]),
    )
    del train_indices, validation_indices
    return [samples[index] for index in test_indices], antibody_embeddings[test_indices], filtered_count, len(samples)


def compute_rouge_scores(references: list[str], predictions: list[str]) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge1_scores: list[float] = []
    rouge2_scores: list[float] = []
    rouge_l_scores: list[float] = []

    for reference, prediction in zip(references, predictions):
        score = scorer.score(reference, prediction)
        rouge1_scores.append(score["rouge1"].fmeasure)
        rouge2_scores.append(score["rouge2"].fmeasure)
        rouge_l_scores.append(score["rougeL"].fmeasure)

    return {
        "rouge_1": (sum(rouge1_scores) / max(len(rouge1_scores), 1)) * 100.0,
        "rouge_2": (sum(rouge2_scores) / max(len(rouge2_scores), 1)) * 100.0,
        "rouge_l": (sum(rouge_l_scores) / max(len(rouge_l_scores), 1)) * 100.0,
    }


def compute_meteor_score(references: list[str], predictions: list[str]) -> float:
    ensure_meteor_resources()
    meteor_scores: list[float] = []
    for reference, prediction in zip(references, predictions):
        reference_tokens = tokenize_for_meteor(reference)
        prediction_tokens = tokenize_for_meteor(prediction)
        meteor_scores.append(meteor_score([reference_tokens], prediction_tokens))
    return (sum(meteor_scores) / max(len(meteor_scores), 1)) * 100.0


def compute_bert_f1(
    references: list[str],
    predictions: list[str],
    model_name: str,
    num_layers: int | None,
    batch_size: int,
    device: torch.device,
    cache_dir: str | None,
    local_files_only: bool,
) -> float:
    truncated_references, truncated_ref_count, max_length = truncate_texts_for_bertscore(
        references,
        model_name,
        cache_dir,
        local_files_only,
    )
    truncated_predictions, truncated_pred_count, _ = truncate_texts_for_bertscore(
        predictions,
        model_name,
        cache_dir,
        local_files_only,
    )
    if truncated_ref_count or truncated_pred_count:
        print(
            f"Truncated texts for {model_name}: references={truncated_ref_count}, "
            f"predictions={truncated_pred_count}, max_length={max_length}"
        )

    score_kwargs: dict[str, Any] = {
        "model_type": model_name,
        "lang": "en",
        "batch_size": batch_size,
        "device": str(device),
        "verbose": False,
        "rescale_with_baseline": False,
    }
    if num_layers is not None:
        score_kwargs["num_layers"] = num_layers

    _, _, f1 = bert_score(truncated_predictions, truncated_references, **score_kwargs)
    result = float(f1.mean().item() * 100.0)
    del f1
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def compute_standard_metrics(
    references: list[str],
    predictions: list[str],
    args: argparse.Namespace,
    device: torch.device,
    cache_dir: str | None,
    local_files_only: bool,
) -> dict[str, float]:
    bleu2 = BLEU(max_ngram_order=2, effective_order=True)
    bleu4 = BLEU(max_ngram_order=4, effective_order=True)
    print("Computing BLEU and ROUGE...")
    rouge_scores = compute_rouge_scores(references, predictions)
    print("Computing METEOR...")
    meteor_score_value = compute_meteor_score(references, predictions)
    bertscore_num_layers = infer_bertscore_num_layers(
        args.bertscore_model,
        args.bertscore_num_layers,
        cache_dir,
        local_files_only,
    )
    biomed_bertscore_num_layers = infer_bertscore_num_layers(
        args.biomed_bertscore_model,
        args.biomed_bertscore_num_layers,
        cache_dir,
        local_files_only,
    )
    print(f"Computing BERTScore with model={args.bertscore_model}, num_layers={bertscore_num_layers}...")
    bertscore_value = compute_bert_f1(
        references,
        predictions,
        args.bertscore_model,
        bertscore_num_layers,
        args.bertscore_batch_size,
        device,
        cache_dir,
        local_files_only,
    )
    print(
        f"Computing BiomedBERTScore with model={args.biomed_bertscore_model}, "
        f"num_layers={biomed_bertscore_num_layers}..."
    )
    biomed_bertscore_value = compute_bert_f1(
        references,
        predictions,
        args.biomed_bertscore_model,
        biomed_bertscore_num_layers,
        args.bertscore_batch_size,
        device,
        cache_dir,
        local_files_only,
    )

    return {
        "bleu_2": float(bleu2.corpus_score(predictions, [references]).score),
        "bleu_4": float(bleu4.corpus_score(predictions, [references]).score),
        "rouge_1": rouge_scores["rouge_1"],
        "rouge_2": rouge_scores["rouge_2"],
        "rouge_l": rouge_scores["rouge_l"],
        "meteor": meteor_score_value,
        "bertscore": bertscore_value,
        "biomed_bertscore": biomed_bertscore_value,
    }


def evaluate_experiment(experiment_dir: Path, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_config = load_run_config(experiment_dir)
    use_pretoken = bool(run_config.get("use_pretoken", False))
    use_template = not bool(run_config.get("disable_chat_template", False))
    data_ratio = resolve_data_ratio(run_config, args.data_ratio)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        run_config["llm_model_name"],
        use_fast=True,
        local_files_only=args.local_files_only or bool(run_config.get("local_files_only", False)),
        cache_dir=run_config.get("llm_cache_dir"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    antibody_prefix_token_id = ensure_antibody_prefix_token(tokenizer) if use_pretoken else -1

    adapter_dir = resolve_adapter_dir(experiment_dir, experiment_dir.name, args)
    llm = AutoModelForCausalLM.from_pretrained(
        run_config["llm_model_name"],
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        local_files_only=args.local_files_only or bool(run_config.get("local_files_only", False)),
        cache_dir=run_config.get("llm_cache_dir"),
    ).to(device)
    if use_pretoken:
        llm.resize_token_embeddings(len(tokenizer))
    llm = PeftModel.from_pretrained(llm, adapter_dir, is_trainable=False).to(device)
    llm.eval()

    antibody_embedding_dim = 5120
    if args.cache_path.exists():
        _, cached_antibody_embeddings = load_prepared_cache(args.cache_path)
        antibody_embedding_dim = int(cached_antibody_embeddings.shape[-1])

    adaptor, adaptor_config = build_adaptor_from_checkpoint(Path(run_config["adaptor_checkpoint"]), antibody_embedding_dim)
    prefix_length = int(adaptor_config["query_length"])
    adapter_epoch = int(adapter_dir.name.split("_")[-1])
    checkpoint_path = find_matching_checkpoint(experiment_dir, adapter_epoch)
    if checkpoint_path is not None:
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        adaptor_state = checkpoint_payload.get("adaptor_state")
        if adaptor_state is not None:
            adaptor.load_state_dict(adaptor_state, strict=True)
    adaptor = adaptor.to(device)
    adaptor.eval()

    test_samples, test_antibody_embeddings, filtered_count, selected_count = load_test_subset(
        run_config,
        tokenizer,
        antibody_prefix_token_id,
        prefix_length,
        use_template,
        args.cache_path,
        use_pretoken,
        data_ratio,
    )
    if not test_samples:
        raise ValueError(f"No test samples available for evaluation in {experiment_dir}.")

    print(
        f"[{experiment_dir.name}] device={device} epoch={adapter_epoch} "
        f"use_pretoken={use_pretoken} use_template={use_template}"
    )
    print(
        f"[{experiment_dir.name}] cache={args.cache_path} filtered_samples={filtered_count} "
        f"selected_samples={selected_count} test_samples={len(test_samples)} data_ratio={data_ratio}"
    )
    print(
        f"[{experiment_dir.name}] max_length={run_config['max_length']} max_new_tokens={args.max_new_tokens} "
        f"eval_batch_size={args.eval_batch_size} adapter_dir={adapter_dir}"
    )

    prediction_rows: list[dict[str, Any]] = []
    references_text: list[str] = []
    predictions_text: list[str] = []

    prediction_iterator = tqdm(total=len(test_samples), desc=f"{experiment_dir.name} epoch {adapter_epoch}", unit="sample")
    for batch_start in range(0, len(test_samples), args.eval_batch_size):
        batch_end = min(batch_start + args.eval_batch_size, len(test_samples))
        batch_samples = test_samples[batch_start:batch_end]
        batch_antibody_embeddings = test_antibody_embeddings[batch_start:batch_end].to(device=device)
        batch_predictions = generate_predictions_batch(
            instructions=[str(sample["instruction"]) for sample in batch_samples],
            antibody_embeddings=batch_antibody_embeddings,
            tokenizer=tokenizer,
            llm=llm,
            adaptor=adaptor,
            antibody_prefix_token_id=antibody_prefix_token_id,
            use_template=use_template,
            use_pretoken=use_pretoken,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )

        for sample, prediction in zip(batch_samples, batch_predictions):
            reference = str(sample["answer"]).strip()
            prediction_rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "task_name": sample.get("task_name", ""),
                    "reference": reference,
                    "prediction": prediction,
                }
            )
            references_text.append(reference)
            predictions_text.append(prediction)

        prediction_iterator.update(len(batch_samples))

    prediction_iterator.close()

    print(
        f"[{experiment_dir.name}] computing metrics for {len(test_samples)} test samples "
        f"(filtered={filtered_count}, selected={selected_count}, data_ratio={data_ratio})"
    )
    del llm
    del adaptor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    metric_scores = compute_standard_metrics(
        references_text,
        predictions_text,
        args,
        device,
        run_config.get("llm_cache_dir"),
        args.local_files_only or bool(run_config.get("local_files_only", False)),
    )

    summary = {
        "experiment": experiment_dir.name,
        "epoch": adapter_epoch,
        "num_test_samples": len(test_samples),
        "data_ratio": data_ratio,
        "filtered_samples": filtered_count,
        "selected_samples": selected_count,
        "bleu_2": metric_scores["bleu_2"],
        "bleu_4": metric_scores["bleu_4"],
        "rouge_1": metric_scores["rouge_1"],
        "rouge_2": metric_scores["rouge_2"],
        "rouge_l": metric_scores["rouge_l"],
        "meteor": metric_scores["meteor"],
        "bertscore": metric_scores["bertscore"],
        "biomed_bertscore": metric_scores["biomed_bertscore"],
        "exact_match": exact_match(references_text, predictions_text),
        "avg_pred_chars": sum(len(text) for text in predictions_text) / max(len(predictions_text), 1),
        "avg_ref_chars": sum(len(text) for text in references_text) / max(len(references_text), 1),
        "adapter_dir": str(adapter_dir),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "use_pretoken": use_pretoken,
        "use_template": use_template,
        "bertscore_model": args.bertscore_model,
        "bertscore_num_layers": infer_bertscore_num_layers(
            args.bertscore_model,
            args.bertscore_num_layers,
            run_config.get("llm_cache_dir"),
            args.local_files_only or bool(run_config.get("local_files_only", False)),
        ),
        "biomed_bertscore_model": args.biomed_bertscore_model,
        "biomed_bertscore_num_layers": infer_bertscore_num_layers(
            args.biomed_bertscore_model,
            args.biomed_bertscore_num_layers,
            run_config.get("llm_cache_dir"),
            args.local_files_only or bool(run_config.get("local_files_only", False)),
        ),
    }
    return summary, prediction_rows


def save_outputs(
    predictions_dir: Path,
    summary_rows: list[dict[str, Any]],
    prediction_payloads: dict[str, list[dict[str, Any]]],
) -> None:
    predictions_dir.mkdir(parents=True, exist_ok=True)
    summary_path = predictions_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    for experiment, rows in prediction_payloads.items():
        output_path = predictions_dir / f"{experiment}_predictions.json"
        output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown_lines = [
        "| Experiment | Epoch | Pretoken | Template | Test Samples | BLEU-2 | BLEU-4 | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR | BERTScore | BiomedBERTScore | EM |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summary_rows:
        markdown_lines.append(
            "| {experiment} | {epoch} | {use_pretoken} | {use_template} | {num_test_samples} | "
            "{bleu_2:.2f} | {bleu_4:.2f} | {rouge_1:.2f} | {rouge_2:.2f} | {rouge_l:.2f} | "
            "{meteor:.2f} | {bertscore:.2f} | {biomed_bertscore:.2f} | {exact_match:.2f} |".format(**summary)
        )
    (predictions_dir / "metrics_table.md").write_text("\n".join(markdown_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary_rows: list[dict[str, Any]] = []
    prediction_payloads: dict[str, list[dict[str, Any]]] = {}

    experiment_iterator = tqdm(args.experiments, desc="Experiments", unit="exp")
    for experiment in experiment_iterator:
        experiment_dir = args.outputs_dir / experiment
        summary, prediction_rows = evaluate_experiment(experiment_dir, args)
        summary_rows.append(summary)
        prediction_payloads[experiment] = prediction_rows

    save_outputs(args.predictions_dir, summary_rows, prediction_payloads)

    print("| Experiment | Epoch | Pretoken | Template | Test Samples | BLEU-2 | BLEU-4 | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR | BERTScore | BiomedBERTScore | EM |")
    print("| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for summary in summary_rows:
        print(
            "| {experiment} | {epoch} | {use_pretoken} | {use_template} | {num_test_samples} | "
            "{bleu_2:.2f} | {bleu_4:.2f} | {rouge_1:.2f} | {rouge_2:.2f} | {rouge_l:.2f} | "
            "{meteor:.2f} | {bertscore:.2f} | {biomed_bertscore:.2f} | {exact_match:.2f} |".format(**summary)
        )


if __name__ == "__main__":
    main()
