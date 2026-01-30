import math
import os
import random
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import optuna
import torch
import torch.nn.functional as F
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
import wandb

try:
    import preprocess
    import model as model_lib
except ImportError:  # pragma: no cover
    from src import preprocess
    from src import model as model_lib


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "y", "t"}:
            return True
        if token in {"false", "0", "no", "n", "f", "none"}:
            return False
        return default
    return default


def parse_exclude_keypoints(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"false", "0", "no", "none", "disable", "disabled"}:
            return False
        return True
    return True


def validate_cfg(cfg: DictConfig) -> None:
    assert cfg.training.batch_size > 0, "training.batch_size must be > 0"
    assert cfg.training.grad_accum_steps > 0, "training.grad_accum_steps must be > 0"
    assert cfg.training.epochs > 0, "training.epochs must be > 0"
    assert cfg.training.learning_rate > 0, "training.learning_rate must be > 0"
    assert cfg.dataset.preprocessing.max_seq_len > 0, "dataset.preprocessing.max_seq_len must be > 0"
    assert cfg.dataset.splits.train > 0, "dataset.splits.train must be > 0"
    assert cfg.dataset.splits.test > 0, "dataset.splits.test must be > 0"
    if cfg.training.optimizer and str(cfg.training.optimizer).lower() not in {"adamw", "adam"}:
        raise ValueError(f"Unsupported optimizer: {cfg.training.optimizer}")
    if cfg.training.loss is None or cfg.training.loss.get("name") is None:
        raise ValueError("training.loss.name must be specified")


def load_run_config(cfg: DictConfig) -> DictConfig:
    if cfg.run is None:
        raise ValueError("run must be provided (e.g., run=proposed-qwen3-1.7b-gsm8k)")
    run_id = str(cfg.run)
    config_dir = Path(get_original_cwd()) / "config"
    run_path = config_dir / "runs" / f"{run_id}.yaml"
    if not run_path.exists():
        raise FileNotFoundError(f"Run config not found: {run_path}")
    run_cfg = OmegaConf.load(run_path)
    merged = OmegaConf.merge(cfg, run_cfg)
    merged.run_name = run_id
    merged.run = OmegaConf.create({"run_id": run_cfg.get("run_id", run_id)})
    merged.run_id = run_cfg.get("run_id", run_id)
    return merged


def apply_mode_overrides(cfg: DictConfig) -> DictConfig:
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.epochs = 1
        cfg.training.eval_every_steps = 1
        cfg.training.grad_accum_steps = 1
        cfg.training.batch_size = min(int(cfg.training.batch_size), 2)
        cfg.training.max_steps = 2
        cfg.dataset.splits.train = min(int(cfg.dataset.splits.train), 32)
        cfg.dataset.splits.dev = min(int(cfg.dataset.splits.dev), 16)
        cfg.dataset.splits.test = min(int(cfg.dataset.splits.test), 16)
        teacher_max = OmegaConf.select(cfg, "dataset.preprocessing.teacher_cot.max_new_tokens_rationale")
        if teacher_max is not None:
            cfg.dataset.preprocessing.teacher_cot.max_new_tokens_rationale = min(int(teacher_max), 64)
        cfg.evaluation.max_eval_batches = 1
        cfg.evaluation.rationale_eval_samples = min(int(cfg.evaluation.rationale_eval_samples), 5)
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")
    return cfg


def resolve_unk_id(tokenizer) -> int:
    if tokenizer.unk_token_id is not None:
        return tokenizer.unk_token_id
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    raise ValueError("Tokenizer has no unk/eos/pad token id.")


def prepare_datasets(cfg: DictConfig, tokenizer):
    train_dataset, dev_dataset, test_dataset = preprocess.load_datasets(cfg, tokenizer)
    collate = partial(preprocess.collate_fn, pad_token_id=tokenizer.pad_token_id)
    return train_dataset, dev_dataset, test_dataset, collate


def evaluate_prompts_list(
    model,
    tokenizer,
    prompts: List[str],
    answers: List[str],
    device: torch.device,
    max_new_tokens: int,
    batch_size: int,
    max_seq_len: int,
    max_examples: int = None,
) -> List[bool]:
    if max_examples is not None:
        prompts = prompts[:max_examples]
        answers = answers[:max_examples]
    results: List[bool] = []
    was_training = model.training
    model.eval()
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        batch_answers = answers[start : start + batch_size]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            outputs = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for text, gold in zip(texts, batch_answers):
            pred = preprocess.extract_answer(text)
            results.append(preprocess.is_correct(pred, gold))
    if was_training:
        model.train()
    return results


def evaluate_accuracy(
    model,
    tokenizer,
    dataset,
    device: torch.device,
    max_new_tokens: int,
    batch_size: int,
    max_seq_len: int,
    max_examples: int = None,
) -> Tuple[float, List[bool]]:
    prompts = [ex["prompt_text"] for ex in dataset.examples]
    answers = [ex["answer"] for ex in dataset.examples]
    results = evaluate_prompts_list(
        model,
        tokenizer,
        prompts,
        answers,
        device,
        max_new_tokens,
        batch_size,
        max_seq_len=max_seq_len,
        max_examples=max_examples,
    )
    if not results:
        return 0.0, []
    return float(np.mean(results)), results


def _compute_confusion(clean: List[bool], ood: List[bool]) -> Dict[str, int]:
    cc = sum(c and o for c, o in zip(clean, ood))
    ci = sum(c and not o for c, o in zip(clean, ood))
    ic = sum((not c) and o for c, o in zip(clean, ood))
    ii = sum((not c) and (not o) for c, o in zip(clean, ood))
    return {"cc": cc, "ci": ci, "ic": ic, "ii": ii}


def evaluate_ood_metrics(
    model,
    tokenizer,
    dataset,
    device: torch.device,
    max_new_tokens: int,
    batch_size: int,
    max_seq_len: int,
    seed: int,
    max_examples: int = None,
) -> Dict[str, object]:
    questions = [ex["question"] for ex in dataset.examples]
    answers = [ex["answer"] for ex in dataset.examples]
    rng = random.Random(seed)

    clean_prompts = [preprocess.build_prompt(q) for q in questions]
    clean_results = evaluate_prompts_list(
        model,
        tokenizer,
        clean_prompts,
        answers,
        device,
        max_new_tokens,
        batch_size,
        max_seq_len=max_seq_len,
        max_examples=max_examples,
    )
    clean_acc = float(np.mean(clean_results)) if clean_results else 0.0

    strengths = [1, 2, 3]
    fluff_accs = []
    irr_accs = []
    fluff_results_base = []
    irr_results_base = []
    for strength in strengths:
        fluff_questions = [preprocess.apply_fluff(q, rng, strength=strength) for q in questions]
        fluff_prompts = [preprocess.build_prompt(q) for q in fluff_questions]
        fluff_results = evaluate_prompts_list(
            model,
            tokenizer,
            fluff_prompts,
            answers,
            device,
            max_new_tokens,
            batch_size,
            max_seq_len=max_seq_len,
            max_examples=max_examples,
        )
        fluff_accs.append(float(np.mean(fluff_results)) if fluff_results else 0.0)
        if strength == 1:
            fluff_results_base = fluff_results

        irr_questions = [preprocess.apply_irrelevant_numbers(q, rng, strength=strength) for q in questions]
        irr_prompts = [preprocess.build_prompt(q) for q in irr_questions]
        irr_results = evaluate_prompts_list(
            model,
            tokenizer,
            irr_prompts,
            answers,
            device,
            max_new_tokens,
            batch_size,
            max_seq_len=max_seq_len,
            max_examples=max_examples,
        )
        irr_accs.append(float(np.mean(irr_results)) if irr_results else 0.0)
        if strength == 1:
            irr_results_base = irr_results

    fluff_acc = fluff_accs[0] if fluff_accs else 0.0
    irr_acc = irr_accs[0] if irr_accs else 0.0

    drops = {
        "prompt_fluff": clean_acc - fluff_acc,
        "prompt_irrelevant_numbers": clean_acc - irr_acc,
    }
    mean_drop = float(np.mean(list(drops.values()))) if drops else 0.0

    clean_correct_indices = [i for i, c in enumerate(clean_results) if c]
    if clean_correct_indices:
        irr_errors = [not irr_results_base[i] for i in clean_correct_indices if i < len(irr_results_base)]
        numeric_distraction_error_rate = float(np.mean(irr_errors)) if irr_errors else 0.0
    else:
        numeric_distraction_error_rate = 0.0

    confusion_fluff = _compute_confusion(clean_results, fluff_results_base) if fluff_results_base else {}
    confusion_irr = _compute_confusion(clean_results, irr_results_base) if irr_results_base else {}

    return {
        "clean_acc": clean_acc,
        "fluff_acc": fluff_acc,
        "irrelevant_numbers_acc": irr_acc,
        "ood_accuracy_drop": mean_drop,
        "numeric_distraction_error_rate": numeric_distraction_error_rate,
        "confusion_clean_vs_fluff": confusion_fluff,
        "confusion_clean_vs_irrelevant_numbers": confusion_irr,
        "ood_fluff_curve": {"strengths": strengths, "acc": fluff_accs},
        "ood_irrelevant_curve": {"strengths": strengths, "acc": irr_accs},
        "clean_correct_list": clean_results,
        "fluff_correct_list": fluff_results_base,
        "irrelevant_correct_list": irr_results_base,
    }


def compute_answer_invariance_kl(
    model,
    tokenizer,
    dataset,
    collate_fn,
    device: torch.device,
    cfg: DictConfig,
    max_batches: int = None,
) -> Tuple[float, List[float]]:
    model.eval()
    loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate_fn)
    unk_id = resolve_unk_id(tokenizer)
    loss_cfg = cfg.training.loss
    use_keypoints = False
    if loss_cfg.name == "mci_cot":
        topk_influence = int(loss_cfg.mci_cot.topk_influence)
        low_quantile = float(loss_cfg.mci_cot.low_quantile)
        exclude_keypoints = parse_exclude_keypoints(loss_cfg.mci_cot.get("exclude_from_low_set", True))
        use_keypoints = True
    elif loss_cfg.name == "ci_uf_kcot":
        topk_influence = int(loss_cfg.ci_uf_kcot.topk_influence)
        low_quantile = 0.5
        exclude_keypoints = True
        use_keypoints = parse_bool(loss_cfg.ci_uf_kcot.use_keypoints, True)
    else:
        topk_influence = 4
        low_quantile = 0.5
        exclude_keypoints = True
        use_keypoints = False
    eps = 1e-8

    kl_values: List[float] = []
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        rationale_mask = batch["rationale_mask"].to(device)
        answer_mask = batch["answer_mask"].to(device)

        x = input_ids[:, :-1]
        y = input_ids[:, 1:]
        attn = attention_mask[:, :-1]
        attn_bool = attn.bool()
        rat_m = rationale_mask[:, 1:] & attn_bool
        ans_m = answer_mask[:, 1:] & attn_bool

        with torch.no_grad():
            logits = model(input_ids=x, attention_mask=attn).logits
            logp = F.log_softmax(logits, dim=-1)
            pt = logp.gather(-1, y.unsqueeze(-1)).squeeze(-1).exp()
            u = (1.0 - pt).clamp(0.0, 1.0)
            ce = model_lib.token_cross_entropy(logits, y) * attn_bool.float()
            orig_ans_nll = (ce * ans_m.float()).sum(dim=1)

        influence = torch.zeros_like(u)
        u_rat = u.masked_fill(~rat_m, -1e9)
        k = max(1, min(topk_influence, u_rat.size(1)))
        idx = torch.topk(u_rat, k=k, dim=1).indices
        for j in range(idx.size(1)):
            t_idx = idx[:, j]
            valid = rat_m[torch.arange(x.size(0), device=device), t_idx]
            if not torch.any(valid):
                continue
            x_corr = x.clone()
            x_corr[valid, t_idx[valid]] = unk_id
            with torch.no_grad():
                logits_corr = model(input_ids=x_corr, attention_mask=attn).logits
                ce_corr = model_lib.token_cross_entropy(logits_corr, y) * attn_bool.float()
                corr_ans_nll = (ce_corr * ans_m.float()).sum(dim=1)
                delta = (corr_ans_nll - orig_ans_nll).clamp(min=0.0)
            influence[valid, t_idx[valid]] = delta[valid]

        with torch.no_grad():
            infl_mean = (influence * rat_m.float()).sum(dim=1) / (rat_m.float().sum(dim=1) + eps)
            infl_norm = (influence / (infl_mean.unsqueeze(1) + eps)).clamp(0.0, 5.0)
            kp_w = model_lib.compute_keypoint_weights(y, tokenizer) if use_keypoints else torch.ones_like(u)
            r = kp_w * u * infl_norm
            r = r.masked_fill(~rat_m, 0.0)
            mean_r = (r.sum(dim=1) / (rat_m.float().sum(dim=1) + eps)).unsqueeze(1)
            alpha = (r / (mean_r + eps)).masked_fill(~rat_m, 0.0)

        if exclude_keypoints:
            keypoint_mask = model_lib.compute_keypoint_mask(y, tokenizer)
            kp_mask = keypoint_mask & rat_m
        else:
            kp_mask = torch.zeros_like(rat_m)

        eligible = rat_m & (~kp_mask)
        low_mask = torch.zeros_like(rat_m)
        with torch.no_grad():
            for b in range(x.size(0)):
                eligible_indices = torch.where(eligible[b])[0]
                if eligible_indices.numel() == 0:
                    continue
                m = max(1, int(eligible_indices.numel() * low_quantile))
                vals = alpha[b].masked_fill(~eligible[b], 1e9)
                _, low_idx = torch.topk(-vals, k=m)
                low_mask[b, low_idx] = True

        x_junk = x.clone()
        x_junk[low_mask] = unk_id
        with torch.no_grad():
            logits_junk = model(input_ids=x_junk, attention_mask=attn).logits
            logp_junk = F.log_softmax(logits_junk, dim=-1)
            logp_orig = F.log_softmax(logits, dim=-1)
            kl_tok = model_lib.token_kl(logp_orig, logp_junk)
            kl_per_example = (kl_tok * ans_m.float()).sum(dim=1) / (ans_m.float().sum(dim=1) + eps)
        kl_values.extend(kl_per_example.cpu().tolist())

    mean_kl = float(np.mean(kl_values)) if kl_values else 0.0
    return mean_kl, kl_values


def compute_rationale_stats(
    model,
    tokenizer,
    dataset,
    device: torch.device,
    max_new_tokens: int,
    batch_size: int,
    max_seq_len: int,
    sample_size: int,
) -> Tuple[float, float, List[int], float, List[float]]:
    model.eval()
    prompts = [ex["prompt_text"] for ex in dataset.examples]
    if sample_size is not None:
        prompts = prompts[:sample_size]
    lengths: List[int] = []
    fluff_rates: List[float] = []
    fluff_tokens = preprocess.get_fluff_tokens()
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            outputs = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for text in texts:
            rationale_text = preprocess.extract_rationale_segment(text)
            if rationale_text:
                length = len(tokenizer.encode(rationale_text, add_special_tokens=False))
            else:
                length = 0
            lengths.append(length)
            fluff_rates.append(preprocess.compute_fluff_rate(rationale_text, fluff_tokens))
    if not lengths:
        return 0.0, 0.0, [], 0.0, []
    mean_len = float(np.mean(lengths))
    median_len = float(np.median(lengths))
    mean_fluff_rate = float(np.mean(fluff_rates)) if fluff_rates else 0.0
    return mean_len, median_len, lengths, mean_fluff_rate, fluff_rates


def assert_valid_gradients(optimizer: torch.optim.Optimizer) -> None:
    has_nonzero = False
    for group in optimizer.param_groups:
        for param in group["params"]:
            if not param.requires_grad:
                continue
            assert param.grad is not None, "Missing gradient for trainable parameter"
            if torch.any(param.grad.detach() != 0):
                has_nonzero = True
    assert has_nonzero, "All gradients are zero; optimizer step would be invalid."


def compute_loss(cfg: DictConfig, model, tokenizer, batch: Dict[str, torch.Tensor]):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    rationale_mask = batch["rationale_mask"]
    answer_mask = batch["answer_mask"]
    unk_id = resolve_unk_id(tokenizer)

    loss_name = cfg.training.loss.name
    if loss_name == "mci_cot":
        params = cfg.training.loss.mci_cot
        exclude_keypoints = parse_exclude_keypoints(params.get("exclude_from_low_set", True))
        loss, metrics = model_lib.mci_cot_loss(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            rationale_mask=rationale_mask,
            answer_mask=answer_mask,
            tokenizer=tokenizer,
            unk_id=unk_id,
            topk_influence=int(params.topk_influence),
            low_quantile=float(params.low_quantile),
            beta_inv=float(params.beta_inv),
            gamma_budget=float(params.gamma_budget),
            tau_budget=float(params.tau_budget),
            exclude_keypoints=exclude_keypoints,
        )
    elif loss_name == "ci_uf_kcot":
        params = cfg.training.loss.ci_uf_kcot
        use_keypoints = parse_bool(params.use_keypoints, True)
        loss, metrics = model_lib.ci_uf_kcot_loss(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            rationale_mask=rationale_mask,
            answer_mask=answer_mask,
            tokenizer=tokenizer,
            unk_id=unk_id,
            topk_influence=int(params.topk_influence),
            use_keypoints=bool(use_keypoints),
        )
    else:
        loss, metrics = model_lib.uniform_cot_loss(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            rationale_mask=rationale_mask,
            answer_mask=answer_mask,
        )
    return loss, metrics


def train_model(
    cfg: DictConfig,
    tokenizer,
    datasets,
    collate_fn,
    device: torch.device,
    log_wandb: bool,
    eval_full: bool,
) -> Tuple[float, Dict[str, object]]:
    train_dataset, dev_dataset, test_dataset = datasets
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = model_lib.build_model(cfg, tokenizer, device)
    assert tokenizer.pad_token_id is not None, "Tokenizer pad_token_id must be set"
    assert tokenizer.unk_token_id is not None, "Tokenizer unk_token_id must be set"
    assert model.get_output_embeddings().weight.shape[0] == len(tokenizer), "Model vocab size mismatch"
    assert model.config.pad_token_id == tokenizer.pad_token_id, "Model pad_token_id mismatch"

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found in the model.")

    optimizer_name = str(cfg.training.optimizer).lower() if cfg.training.optimizer else "adamw"
    optimizer_cls = torch.optim.AdamW if optimizer_name in {"adamw", "adam"} else torch.optim.AdamW
    optimizer = optimizer_cls(
        trainable_params,
        lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
    )

    steps_per_epoch = max(1, math.ceil(len(train_loader) / cfg.training.grad_accum_steps))
    total_updates = steps_per_epoch * cfg.training.epochs
    max_steps = getattr(cfg.training, "max_steps", None)
    if max_steps is not None:
        total_updates = min(total_updates, math.ceil(max_steps / cfg.training.grad_accum_steps))
    total_updates = max(total_updates, 1)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.training.warmup_steps),
        num_training_steps=total_updates,
    )

    use_amp = cfg.model.precision in {"bf16", "fp16"} and device.type == "cuda"
    amp_dtype = torch.bfloat16 if cfg.model.precision == "bf16" else torch.float16
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and cfg.model.precision == "fp16")

    batch_step = 0
    update_step = 0
    best_val_acc = -1.0
    model.train()

    for epoch in range(cfg.training.epochs):
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.training.epochs}")
        optimizer.zero_grad(set_to_none=True)
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            rationale_mask = batch["rationale_mask"].to(device)
            answer_mask = batch["answer_mask"].to(device)

            if epoch == 0 and batch_step == 0:
                assert input_ids.shape == rationale_mask.shape == answer_mask.shape == attention_mask.shape, (
                    "Mask shapes must match input_ids"
                )
                assert input_ids.dim() == 2, "input_ids must be 2D"

            batch = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "rationale_mask": rationale_mask,
                "answer_mask": answer_mask,
            }

            with torch.autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
                loss, metrics = compute_loss(cfg, model, tokenizer, batch)
                loss_scaled = loss / cfg.training.grad_accum_steps

            aux_grad_norm = None
            if log_wandb:
                aux_grads = torch.autograd.grad(
                    loss_scaled,
                    trainable_params[:1],
                    retain_graph=True,
                    create_graph=False,
                )
                aux_grad_norm = aux_grads[0].detach().norm().item()

            if scaler.is_enabled():
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            if log_wandb:
                log_payload = {
                    "train_loss": float(metrics["loss"]),
                    "train_ans_loss": float(metrics.get("ans_loss", 0.0)),
                    "train_rat_loss": float(metrics.get("rat_loss", 0.0)),
                    "train_inv_loss": float(metrics.get("inv_loss", 0.0)),
                    "train_budget_loss": float(metrics.get("budget_loss", 0.0)),
                    "train_token_acc": float(metrics.get("token_acc", 0.0)),
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch + 1,
                }
                if aux_grad_norm is not None:
                    log_payload["aux_grad_norm"] = aux_grad_norm
                wandb.log(log_payload, step=batch_step)

            if (batch_step + 1) % cfg.training.grad_accum_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, float(cfg.training.gradient_clip))
                assert_valid_gradients(optimizer)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1
                if log_wandb:
                    wandb.log({"grad_norm": float(grad_norm), "update_step": update_step}, step=batch_step)

            if cfg.training.eval_every_steps and (batch_step + 1) % cfg.training.eval_every_steps == 0:
                max_eval = cfg.evaluation.max_eval_batches
                max_examples = None
                if max_eval is not None:
                    max_examples = max_eval * max(1, cfg.training.batch_size)
                val_acc, _ = evaluate_accuracy(
                    model,
                    tokenizer,
                    dev_dataset,
                    device,
                    max_new_tokens=min(128, int(cfg.dataset.preprocessing.teacher_cot.max_new_tokens_rationale) + 32),
                    batch_size=max(1, min(cfg.training.batch_size, 4)),
                    max_seq_len=int(cfg.dataset.preprocessing.max_seq_len),
                    max_examples=max_examples,
                )
                best_val_acc = max(best_val_acc, val_acc)
                if log_wandb:
                    wandb.log({"val_acc": val_acc, "best_val_acc": best_val_acc}, step=batch_step)

            batch_step += 1
            progress.set_postfix(loss=float(metrics["loss"]))

            if max_steps is not None and batch_step >= max_steps:
                break
        if max_steps is not None and batch_step >= max_steps:
            break

    if not eval_full:
        if best_val_acc < 0:
            best_val_acc, _ = evaluate_accuracy(
                model,
                tokenizer,
                dev_dataset,
                device,
                max_new_tokens=min(128, int(cfg.dataset.preprocessing.teacher_cot.max_new_tokens_rationale) + 32),
                batch_size=max(1, min(cfg.training.batch_size, 4)),
                max_seq_len=int(cfg.dataset.preprocessing.max_seq_len),
                max_examples=None,
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return best_val_acc, {}

    max_new_tokens = min(128, int(cfg.dataset.preprocessing.teacher_cot.max_new_tokens_rationale) + 32)
    max_seq_len = int(cfg.dataset.preprocessing.max_seq_len)
    test_acc, test_correct_list = evaluate_accuracy(
        model,
        tokenizer,
        test_dataset,
        device,
        max_new_tokens=max_new_tokens,
        batch_size=max(1, min(cfg.training.batch_size, 4)),
        max_seq_len=max_seq_len,
    )

    ood_metrics = evaluate_ood_metrics(
        model,
        tokenizer,
        test_dataset,
        device,
        max_new_tokens=max_new_tokens,
        batch_size=max(1, min(cfg.training.batch_size, 4)),
        max_seq_len=max_seq_len,
        seed=int(cfg.training.seed),
        max_examples=cfg.evaluation.max_eval_batches * max(1, cfg.training.batch_size)
        if cfg.evaluation.max_eval_batches
        else None,
    )

    inv_kl, inv_list = compute_answer_invariance_kl(
        model,
        tokenizer,
        test_dataset,
        collate_fn,
        device,
        cfg,
        max_batches=cfg.evaluation.max_eval_batches,
    )

    rationale_mean, rationale_median, rationale_list, fluff_rate, fluff_rates = compute_rationale_stats(
        model,
        tokenizer,
        test_dataset,
        device,
        max_new_tokens=max_new_tokens,
        batch_size=max(1, min(cfg.training.batch_size, 4)),
        max_seq_len=max_seq_len,
        sample_size=cfg.evaluation.rationale_eval_samples,
    )

    summary_metrics: Dict[str, object] = {
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "accuracy": float(test_acc),
        "ood_accuracy_drop": float(ood_metrics["ood_accuracy_drop"]),
        "answer_invariance_kl": float(inv_kl),
        "numeric_distraction_error_rate": float(ood_metrics["numeric_distraction_error_rate"]),
        "rationale_length_mean": float(rationale_mean),
        "rationale_length_median": float(rationale_median),
        "fluff_imitation_rate": float(fluff_rate),
        "clean_acc": float(ood_metrics["clean_acc"]),
        "fluff_acc": float(ood_metrics["fluff_acc"]),
        "irrelevant_numbers_acc": float(ood_metrics["irrelevant_numbers_acc"]),
        "confusion_clean_vs_fluff": ood_metrics["confusion_clean_vs_fluff"],
        "confusion_clean_vs_irrelevant_numbers": ood_metrics["confusion_clean_vs_irrelevant_numbers"],
        "ood_fluff_curve": ood_metrics["ood_fluff_curve"],
        "ood_irrelevant_curve": ood_metrics["ood_irrelevant_curve"],
        "answer_invariance_kl_list": [float(x) for x in inv_list],
        "rationale_length_list": [int(x) for x in rationale_list],
        "fluff_imitation_rates": [float(x) for x in fluff_rates],
        "test_correct_list": [int(x) for x in test_correct_list],
        "clean_correct_list": [int(x) for x in ood_metrics["clean_correct_list"]],
        "fluff_correct_list": [int(x) for x in ood_metrics["fluff_correct_list"]],
        "irrelevant_correct_list": [int(x) for x in ood_metrics["irrelevant_correct_list"]],
    }

    if log_wandb:
        wandb.log(
            {
                "test_acc": float(test_acc),
                "accuracy": float(test_acc),
                "ood_accuracy_drop": float(ood_metrics["ood_accuracy_drop"]),
                "answer_invariance_kl": float(inv_kl),
                "numeric_distraction_error_rate": float(ood_metrics["numeric_distraction_error_rate"]),
                "rationale_length_mean": float(rationale_mean),
                "fluff_imitation_rate": float(fluff_rate),
            },
            step=batch_step,
        )
        for key, value in summary_metrics.items():
            wandb.summary[key] = value

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_val_acc, summary_metrics


def apply_trial_params(cfg: DictConfig, trial: optuna.Trial) -> DictConfig:
    cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    for space in cfg_copy.optuna.search_spaces:
        name = space.param_name
        dist = space.distribution_type
        if dist == "loguniform":
            value = trial.suggest_float(name, float(space.low), float(space.high), log=True)
        elif dist == "categorical":
            value = trial.suggest_categorical(name, list(space.choices))
        else:
            raise ValueError(f"Unsupported distribution: {dist}")

        if name == "learning_rate":
            cfg_copy.training.learning_rate = float(value)
        elif name == "lora_r":
            cfg_copy.model.lora.r = int(value)
        elif name == "topk_influence":
            if cfg_copy.training.loss.name == "mci_cot":
                cfg_copy.training.loss.mci_cot.topk_influence = int(value)
            else:
                cfg_copy.training.loss.ci_uf_kcot.topk_influence = int(value)
        elif name == "low_quantile":
            cfg_copy.training.loss.mci_cot.low_quantile = float(value)
        elif name == "beta_inv":
            cfg_copy.training.loss.mci_cot.beta_inv = float(value)
        elif name == "gamma_budget":
            cfg_copy.training.loss.mci_cot.gamma_budget = float(value)
        elif name == "use_keypoints":
            cfg_copy.training.loss.ci_uf_kcot.use_keypoints = bool(int(value))
        else:
            OmegaConf.update(cfg_copy, name, value)
    return cfg_copy


def apply_best_params(cfg: DictConfig, best_params: Dict[str, float]) -> DictConfig:
    for name, value in best_params.items():
        if name == "learning_rate":
            cfg.training.learning_rate = float(value)
        elif name == "lora_r":
            cfg.model.lora.r = int(value)
        elif name == "topk_influence":
            if cfg.training.loss.name == "mci_cot":
                cfg.training.loss.mci_cot.topk_influence = int(value)
            else:
                cfg.training.loss.ci_uf_kcot.topk_influence = int(value)
        elif name == "low_quantile":
            cfg.training.loss.mci_cot.low_quantile = float(value)
        elif name == "beta_inv":
            cfg.training.loss.mci_cot.beta_inv = float(value)
        elif name == "gamma_budget":
            cfg.training.loss.mci_cot.gamma_budget = float(value)
        elif name == "use_keypoints":
            cfg.training.loss.ci_uf_kcot.use_keypoints = bool(int(value))
        else:
            OmegaConf.update(cfg, name, value)
    return cfg


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    os.environ.setdefault("HF_HOME", ".cache/")
    os.environ.setdefault("HF_DATASETS_CACHE", ".cache/")
    os.environ.setdefault("TRANSFORMERS_CACHE", ".cache/")

    cfg = load_run_config(cfg)
    cfg = apply_mode_overrides(cfg)
    validate_cfg(cfg)
    set_seed(int(cfg.training.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = model_lib.load_tokenizer(cfg)
    assert tokenizer.pad_token_id is not None, "Tokenizer pad_token_id must be defined"
    assert tokenizer.unk_token_id is not None, "Tokenizer unk_token_id must be defined"

    train_dataset, dev_dataset, test_dataset, collate_fn = prepare_datasets(cfg, tokenizer)

    if cfg.optuna.n_trials and int(cfg.optuna.n_trials) > 0:
        study = optuna.create_study(direction="maximize")

        def objective(trial: optuna.Trial) -> float:
            cfg_trial = apply_trial_params(cfg, trial)
            val_acc, _ = train_model(
                cfg_trial,
                tokenizer,
                (train_dataset, dev_dataset, test_dataset),
                collate_fn,
                device,
                log_wandb=False,
                eval_full=False,
            )
            return val_acc

        study.optimize(objective, n_trials=int(cfg.optuna.n_trials))
        cfg = apply_best_params(cfg, study.best_params)

    if cfg.wandb.mode != "disabled":
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
            mode=cfg.wandb.mode,
        )
        print(f"WandB URL: {wandb.run.get_url()}")

    train_model(
        cfg,
        tokenizer,
        (train_dataset, dev_dataset, test_dataset),
        collate_fn,
        device,
        log_wandb=cfg.wandb.mode != "disabled",
        eval_full=cfg.mode == "full",
    )

    if cfg.wandb.mode != "disabled":
        wandb.finish()


if __name__ == "__main__":
    main()
