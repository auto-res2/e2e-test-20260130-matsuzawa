import re
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_model_name(model_name: str) -> str:
    mapping = {
        "qwen3-1.7b": "Qwen/Qwen3-1.7B",
        "qwen3-1.7b (1.7b parameters)": "Qwen/Qwen3-1.7B",
        "qwen3": "Qwen/Qwen3-1.7B",
        "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
    }
    lower = str(model_name).lower()
    for key, value in mapping.items():
        if key in lower:
            return value
    return model_name


def load_tokenizer(cfg) -> AutoTokenizer:
    if "synthetic" in str(cfg.dataset.name).lower() or str(cfg.model.name).lower().startswith("synthetic"):
        tokenizer = build_synthetic_tokenizer()
        tokenizer.padding_side = "right"
        if hasattr(cfg.dataset, "preprocessing") and hasattr(cfg.dataset.preprocessing, "max_seq_len"):
            tokenizer.model_max_length = int(cfg.dataset.preprocessing.max_seq_len)
        assert tokenizer.pad_token_id is not None, "Synthetic tokenizer must have pad_token_id"
        assert tokenizer.unk_token_id is not None, "Synthetic tokenizer must have unk_token_id"
        return tokenizer

    model_name = resolve_model_name(cfg.model.name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=".cache/", use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    if tokenizer.unk_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.unk_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"unk_token": "<unk>"})
    tokenizer.padding_side = "right"
    if hasattr(cfg.dataset, "preprocessing") and hasattr(cfg.dataset.preprocessing, "max_seq_len"):
        tokenizer.model_max_length = int(cfg.dataset.preprocessing.max_seq_len)
    assert tokenizer.pad_token_id is not None, "Tokenizer pad_token_id must be set"
    assert tokenizer.unk_token_id is not None, "Tokenizer unk_token_id must be set"
    return tokenizer


def _filter_lora_target_modules(model, target_modules: List[str]) -> List[str]:
    available = []
    for module_name in target_modules:
        for name, _ in model.named_modules():
            if name.endswith(module_name):
                available.append(module_name)
                break
    available = sorted(set(available))
    if not available:
        raise ValueError(f"No LoRA target modules found in model for {target_modules}")
    return available


@dataclass
class SmallTransformerConfig:
    vocab_size: int
    n_layers: int = 6
    n_heads: int = 8
    d_model: int = 256
    d_ff: int = 1024
    max_seq_len: int = 1024
    dropout: float = 0.1


class SmallTransformerLM(nn.Module):
    def __init__(self, cfg: SmallTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(cfg.max_seq_len, cfg.max_seq_len), diagonal=1).bool(),
            persistent=False,
        )

        self.config = type("cfg", (), {})()
        self.config.use_cache = False
        self.config.pad_token_id = None
        self.config.vocab_size = cfg.vocab_size
        self.config.max_position_embeddings = cfg.max_seq_len
        self.generation_config = type("gen", (), {})()
        self.generation_config.pad_token_id = None
        self.generation_config.eos_token_id = None

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        bsz, seq_len = input_ids.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.cfg.max_seq_len}")
        pos = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)

        causal_mask = self.causal_mask[:seq_len, :seq_len]
        src_key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                raise ValueError("attention_mask must be 2D [batch, seq]")
            src_key_padding_mask = attention_mask == 0

        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return type("out", (), {"logits": logits})

    def get_input_embeddings(self):
        return self.token_emb

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_emb: nn.Module):
        self.lm_head = new_emb

    def resize_token_embeddings(self, new_size: int):
        if new_size == self.cfg.vocab_size:
            return
        old_emb = self.token_emb.weight.data
        old_head = self.lm_head.weight.data
        self.token_emb = nn.Embedding(new_size, self.cfg.d_model)
        self.lm_head = nn.Linear(self.cfg.d_model, new_size, bias=False)
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        num_copy = min(old_emb.size(0), new_size)
        self.token_emb.weight.data[:num_copy] = old_emb[:num_copy]
        self.lm_head.weight.data[:num_copy] = old_head[:num_copy]
        self.cfg.vocab_size = new_size
        self.config.vocab_size = new_size

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 20,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        self.eval()
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

        generated = input_ids
        attn = attention_mask
        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)

        for _ in range(max_new_tokens):
            logits = self.forward(generated, attention_mask=attn).logits
            next_logits = logits[:, -1, :]
            if do_sample:
                logits_scaled = next_logits / max(temperature, 1e-6)
                probs = F.softmax(logits_scaled, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cumulative = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumulative > top_p
                    mask[..., 0] = False
                    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    next_tokens = torch.multinomial(sorted_probs, num_samples=1)
                    next_tokens = sorted_idx.gather(-1, next_tokens)
                else:
                    next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(next_logits, dim=-1, keepdim=True)

            if eos_token_id is not None:
                next_tokens = next_tokens.masked_fill(finished.unsqueeze(1), eos_token_id)

            generated = torch.cat([generated, next_tokens], dim=1)
            attn = torch.cat([attn, torch.ones_like(next_tokens, dtype=attn.dtype)], dim=1)

            if eos_token_id is not None:
                finished = finished | (next_tokens.squeeze(1) == eos_token_id)
                if torch.all(finished):
                    break
            if generated.size(1) >= self.cfg.max_seq_len:
                break

        return generated


class SimpleTokenizer:
    def __init__(self, vocab: List[str]):
        self.vocab = {tok: idx for idx, tok in enumerate(vocab)}
        self.inv_vocab = {idx: tok for tok, idx in self.vocab.items()}
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.eos_token = "</s>"
        self.bos_token = "<s>"
        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unk_token]
        self.eos_token_id = self.vocab[self.eos_token]
        self.bos_token_id = self.vocab[self.bos_token]
        self.padding_side = "right"
        self.model_max_length = None

    def __len__(self):
        return len(self.vocab)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        tokens = re.findall(r"\w+|[^\w\s]", text)
        ids = [self.vocab.get(tok.lower(), self.unk_token_id) for tok in tokens]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for idx in ids:
            tok = self.inv_vocab.get(int(idx), self.unk_token)
            if skip_special_tokens and tok in {self.pad_token, self.unk_token, self.eos_token, self.bos_token}:
                continue
            tokens.append(tok)
        return " ".join(tokens)

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self.inv_vocab.get(int(ids), self.unk_token)
        return [self.inv_vocab.get(int(idx), self.unk_token) for idx in ids]

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.vocab.get(tokens, self.unk_token_id)
        return [self.vocab.get(tok, self.unk_token_id) for tok in tokens]

    def batch_decode(self, sequences, skip_special_tokens: bool = True) -> List[str]:
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()
        return [self.decode(seq, skip_special_tokens=skip_special_tokens) for seq in sequences]

    def __call__(
        self,
        texts: List[str],
        return_tensors: str = None,
        padding: bool = True,
        truncation: bool = True,
        max_length: int = None,
    ) -> dict:
        encoded = [self.encode(t, add_special_tokens=False) for t in texts]
        max_len = max(len(ids) for ids in encoded) if encoded else 0
        if max_length is not None:
            max_len = min(max_len, max_length)
        input_ids = []
        attention_mask = []
        for ids in encoded:
            ids = ids[:max_len]
            padding_len = max_len - len(ids)
            if self.padding_side == "right":
                input_ids.append(ids + [self.pad_token_id] * padding_len)
                attention_mask.append([1] * len(ids) + [0] * padding_len)
            else:
                input_ids.append([self.pad_token_id] * padding_len + ids)
                attention_mask.append([0] * padding_len + [1] * len(ids))
        result = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
        return result


def build_synthetic_tokenizer() -> SimpleTokenizer:
    base_vocab = ["<pad>", "<unk>", "</s>", "<s>"]
    vocab = base_vocab + [
        "question",
        "rationale",
        "answer",
        ":",
        "+",
        "-",
        "=",
        ",",
        ".",
    ]
    vocab += [str(i) for i in range(0, 101)]
    vocab += [
        "john",
        "has",
        "apples",
        "and",
        "buys",
        "more",
        "how",
        "many",
        "does",
        "he",
        "have",
        "starts",
        "with",
        "so",
        "let",
        "s",
        "think",
        "step",
        "by",
        "the",
        "is",
        "just",
        "a",
        "friendly",
        "note",
        "appreciate",
        "your",
        "time",
        "thank",
        "you",
        "for",
        "help",
        "unrelated",
        "fact",
        "there",
        "are",
        "marbles",
        "coins",
        "on",
        "table",
        "box",
        "contains",
        "pens",
        "markers",
        "shelf",
        "books",
        "magazines",
    ]
    return SimpleTokenizer(list(dict.fromkeys(vocab)))


def build_model(cfg, tokenizer, device: torch.device):
    model_name = resolve_model_name(cfg.model.name)
    if str(cfg.model.name).lower().startswith("synthetic"):
        synth_cfg = SmallTransformerConfig(
            vocab_size=len(tokenizer),
            n_layers=int(getattr(cfg.model, "n_layers", 6)),
            n_heads=int(getattr(cfg.model, "n_heads", 8)),
            d_model=int(getattr(cfg.model, "d_model", 256)),
            d_ff=int(getattr(cfg.model, "d_ff", 1024)),
            max_seq_len=int(cfg.dataset.preprocessing.max_seq_len),
            dropout=float(getattr(cfg.model, "dropout", 0.1)),
        )
        model = SmallTransformerLM(synth_cfg).to(device)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        return model

    precision = str(cfg.model.precision).lower()
    if device.type == "cpu":
        dtype = torch.float32
    elif precision == "bf16":
        dtype = torch.bfloat16
    elif precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=".cache/",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    finetune = str(getattr(cfg.model, "finetune", "none")).lower()
    if finetune == "lora":
        if not hasattr(cfg.model, "lora"):
            raise ValueError("LoRA config missing under model.lora")
        target_modules = _filter_lora_target_modules(model, list(cfg.model.lora.target_modules))
        lora_cfg = LoraConfig(
            r=int(cfg.model.lora.r),
            lora_alpha=int(cfg.model.lora.alpha),
            lora_dropout=float(cfg.model.lora.dropout),
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)

    model.to(device)
    return model


def token_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bsz, seq_len, vocab = logits.shape
    loss = F.cross_entropy(logits.reshape(bsz * seq_len, vocab), targets.reshape(bsz * seq_len), reduction="none")
    return loss.view(bsz, seq_len)


def token_kl(p_log: torch.Tensor, q_log: torch.Tensor) -> torch.Tensor:
    p = p_log.exp()
    return (p * (p_log - q_log)).sum(dim=-1)


def compute_keypoint_mask(token_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    tokens = tokenizer.convert_ids_to_tokens(token_ids.reshape(-1).tolist()) if hasattr(tokenizer, "convert_ids_to_tokens") else []
    if not tokens:
        return torch.zeros_like(token_ids, dtype=torch.bool)
    mask = []
    for tok in tokens:
        tok_clean = tok.replace("▁", "").replace("Ġ", "")
        if re.search(r"\d", tok_clean):
            mask.append(True)
        elif tok_clean in {"+", "-", "*", "/", "×", "÷", "="}:
            mask.append(True)
        else:
            mask.append(False)
    return torch.tensor(mask, device=token_ids.device).reshape(token_ids.shape)


def compute_keypoint_weights(token_ids: torch.Tensor, tokenizer, weight: float = 1.5) -> torch.Tensor:
    mask = compute_keypoint_mask(token_ids, tokenizer).float()
    return 1.0 + mask * (weight - 1.0)


def _no_grad_forward(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    if was_training:
        model.train()
    return logits


def uniform_cot_loss(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    rationale_mask: torch.Tensor,
    answer_mask: torch.Tensor,
    eps: float = 1e-8,
) -> tuple:
    x = input_ids[:, :-1]
    y = input_ids[:, 1:]
    attn = attention_mask[:, :-1]
    attn_bool = attn.bool()
    rat_m = rationale_mask[:, 1:] & attn_bool
    ans_m = answer_mask[:, 1:] & attn_bool
    logits = model(input_ids=x, attention_mask=attn).logits
    ce = token_cross_entropy(logits, y) * attn_bool.float()
    loss = ce.sum() / (attn_bool.float().sum() + eps)
    ans_loss = (ce * ans_m.float()).sum() / (ans_m.float().sum() + eps)
    rat_loss = (ce * rat_m.float()).sum() / (rat_m.float().sum() + eps)
    pred = logits.argmax(dim=-1)
    token_acc = ((pred == y) & ans_m).float().sum() / (ans_m.float().sum() + eps)
    return loss, {
        "loss": loss.item(),
        "token_acc": token_acc.item(),
        "ans_loss": ans_loss.item(),
        "rat_loss": rat_loss.item(),
        "inv_loss": 0.0,
        "budget_loss": 0.0,
    }


def ci_uf_kcot_loss(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    rationale_mask: torch.Tensor,
    answer_mask: torch.Tensor,
    tokenizer,
    unk_id: int,
    topk_influence: int = 8,
    use_keypoints: bool = True,
    eps: float = 1e-8,
) -> tuple:
    x = input_ids[:, :-1]
    y = input_ids[:, 1:]
    attn = attention_mask[:, :-1]
    attn_bool = attn.bool()
    rat_m = rationale_mask[:, 1:] & attn_bool
    ans_m = answer_mask[:, 1:] & attn_bool

    logits = model(input_ids=x, attention_mask=attn).logits
    ce = token_cross_entropy(logits, y) * attn_bool.float()

    ans_loss = (ce * ans_m.float()).sum() / (ans_m.float().sum() + eps)

    with torch.no_grad():
        logp = F.log_softmax(logits, dim=-1)
        pt = logp.gather(-1, y.unsqueeze(-1)).squeeze(-1).exp()
        u = (1.0 - pt).clamp(0.0, 1.0)
        orig_ans_nll = (ce * ans_m.float()).sum(dim=1)

    influence = torch.zeros_like(u)
    u_rat = u.masked_fill(~rat_m, -1e9)
    k = max(1, min(topk_influence, u_rat.size(1)))
    idx = torch.topk(u_rat, k=k, dim=1).indices
    for j in range(idx.size(1)):
        t_idx = idx[:, j]
        valid = rat_m[torch.arange(x.size(0), device=x.device), t_idx]
        if not torch.any(valid):
            continue
        x_corr = x.clone()
        x_corr[valid, t_idx[valid]] = unk_id
        logits_corr = _no_grad_forward(model, x_corr, attn)
        ce_corr = token_cross_entropy(logits_corr, y) * attn_bool.float()
        corr_ans_nll = (ce_corr * ans_m.float()).sum(dim=1)
        delta = (corr_ans_nll - orig_ans_nll).clamp(min=0.0)
        influence[valid, t_idx[valid]] = delta[valid]

    with torch.no_grad():
        infl_mean = (influence * rat_m.float()).sum(dim=1) / (rat_m.float().sum(dim=1) + eps)
        infl_norm = (influence / (infl_mean.unsqueeze(1) + eps)).clamp(0.0, 5.0)
        kp_w = compute_keypoint_weights(y, tokenizer) if use_keypoints else torch.ones_like(u)
        r = kp_w * u * infl_norm
        r_rat = r.masked_fill(~rat_m, 0.0)
        mean_r = (r_rat.sum(dim=1) / (rat_m.float().sum(dim=1) + eps)).unsqueeze(1)
        alpha = (r / (mean_r + eps)).masked_fill(~rat_m, 0.0)

    rat_loss = (alpha * ce * rat_m.float()).sum() / (alpha * rat_m.float()).sum().clamp(min=eps)
    loss = ans_loss + rat_loss

    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        token_acc = ((pred == y) & ans_m).float().sum() / (ans_m.float().sum() + eps)

    metrics = {
        "loss": loss.item(),
        "ans_loss": ans_loss.item(),
        "rat_loss": rat_loss.item(),
        "inv_loss": 0.0,
        "budget_loss": 0.0,
        "token_acc": token_acc.item(),
    }
    return loss, metrics


def mci_cot_loss(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    rationale_mask: torch.Tensor,
    answer_mask: torch.Tensor,
    tokenizer,
    unk_id: int,
    topk_influence: int = 8,
    low_quantile: float = 0.5,
    beta_inv: float = 0.2,
    gamma_budget: float = 0.01,
    tau_budget: float = 1.0,
    exclude_keypoints: bool = True,
    eps: float = 1e-8,
) -> tuple:
    x = input_ids[:, :-1]
    y = input_ids[:, 1:]
    attn = attention_mask[:, :-1]
    attn_bool = attn.bool()
    rat_m = rationale_mask[:, 1:] & attn_bool
    ans_m = answer_mask[:, 1:] & attn_bool

    logits = model(input_ids=x, attention_mask=attn).logits
    ce = token_cross_entropy(logits, y) * attn_bool.float()

    ans_loss = (ce * ans_m.float()).sum() / (ans_m.float().sum() + eps)

    with torch.no_grad():
        logp = F.log_softmax(logits, dim=-1)
        pt = logp.gather(-1, y.unsqueeze(-1)).squeeze(-1).exp()
        u = (1.0 - pt).clamp(0.0, 1.0)
        orig_ans_nll = (ce * ans_m.float()).sum(dim=1)

    influence = torch.zeros_like(u)
    u_rat = u.masked_fill(~rat_m, -1e9)
    k = max(1, min(topk_influence, u_rat.size(1)))
    idx = torch.topk(u_rat, k=k, dim=1).indices
    for j in range(idx.size(1)):
        t_idx = idx[:, j]
        valid = rat_m[torch.arange(x.size(0), device=x.device), t_idx]
        if not torch.any(valid):
            continue
        x_corr = x.clone()
        x_corr[valid, t_idx[valid]] = unk_id
        logits_corr = _no_grad_forward(model, x_corr, attn)
        ce_corr = token_cross_entropy(logits_corr, y) * attn_bool.float()
        corr_ans_nll = (ce_corr * ans_m.float()).sum(dim=1)
        delta = (corr_ans_nll - orig_ans_nll).clamp(min=0.0)
        influence[valid, t_idx[valid]] = delta[valid]

    with torch.no_grad():
        infl_mean = (influence * rat_m.float()).sum(dim=1) / (rat_m.float().sum(dim=1) + eps)
        infl_norm = (influence / (infl_mean.unsqueeze(1) + eps)).clamp(0.0, 5.0)
        kp_w = compute_keypoint_weights(y, tokenizer)
        r = kp_w * u * infl_norm
        r_rat = r.masked_fill(~rat_m, 0.0)
        mean_r = (r_rat.sum(dim=1) / (rat_m.float().sum(dim=1) + eps)).unsqueeze(1)
        alpha = (r / (mean_r + eps)).masked_fill(~rat_m, 0.0)

    rat_loss = (alpha * ce * rat_m.float()).sum() / (alpha * rat_m.float()).sum().clamp(min=eps)

    with torch.no_grad():
        alpha_mean = (alpha * rat_m.float()).sum(dim=1) / (rat_m.float().sum(dim=1) + eps)
    budget_loss = ((alpha_mean - tau_budget) ** 2).mean()

    keypoint_mask = compute_keypoint_mask(y, tokenizer) if exclude_keypoints else torch.zeros_like(rat_m)
    eligible = rat_m & (~keypoint_mask) if exclude_keypoints else rat_m
    low_mask = torch.zeros_like(rat_m)
    with torch.no_grad():
        for b in range(x.size(0)):
            eligible_idx = torch.where(eligible[b])[0]
            if eligible_idx.numel() == 0:
                continue
            m = max(1, int(eligible_idx.numel() * low_quantile))
            vals = alpha[b].masked_fill(~eligible[b], 1e9)
            _, low_idx = torch.topk(-vals, k=m)
            low_mask[b, low_idx] = True

    x_junk = x.clone()
    x_junk[low_mask] = unk_id
    logits_junk = model(input_ids=x_junk, attention_mask=attn).logits
    logp_junk = F.log_softmax(logits_junk, dim=-1)

    with torch.no_grad():
        logp_orig_det = F.log_softmax(logits, dim=-1)

    kl_tok = token_kl(logp_orig_det, logp_junk)
    inv_loss = (kl_tok * ans_m.float()).sum() / (ans_m.float().sum() + eps)

    loss = ans_loss + rat_loss + beta_inv * inv_loss + gamma_budget * budget_loss

    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        token_acc = ((pred == y) & ans_m).float().sum() / (ans_m.float().sum() + eps)

    metrics = {
        "loss": loss.item(),
        "ans_loss": ans_loss.item(),
        "rat_loss": rat_loss.item(),
        "inv_loss": inv_loss.item(),
        "budget_loss": budget_loss.item(),
        "token_acc": token_acc.item(),
    }
    return loss, metrics
