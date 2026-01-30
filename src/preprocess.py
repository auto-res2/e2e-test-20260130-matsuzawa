import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM

try:
    import model as model_lib
except ImportError:  # pragma: no cover
    from src import model as model_lib

FLUFF_PHRASES = [
    "Thank you for your help.",
    "I hope you are having a great day.",
    "This is just a friendly note.",
    "Appreciate your time!",
    "Wishing you the best.",
]

OOD_FLUFF_PHRASES = [
    "Just a quick aside: hope all is well.",
    "Kindly note this is unrelated.",
    "Sending positive vibes your way.",
    "Thanks a bunch for reading.",
]

IRRELEVANT_NUMBER_TEMPLATES = [
    "By the way, there are {a} marbles and {b} coins on the table.",
    "Unrelated fact: {a} bikes and {b} scooters are parked nearby.",
    "Note: a box contains {a} pens and {b} markers, unrelated to the question.",
    "Side info: a shelf has {a} books and {b} magazines.",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_gsm8k_answer(answer: str) -> Tuple[str, str]:
    if "####" in answer:
        rationale, final = answer.rsplit("####", 1)
        return rationale.strip(), final.strip()
    return "", answer.strip()


def build_prompt(question: str) -> str:
    return f"Question: {question.strip()}\nRationale:"


def build_full_text(question: str, rationale: str, answer: str) -> Tuple[str, str, str, str]:
    prompt = build_prompt(question)
    rationale_text = f" {rationale.strip()}\nAnswer:"
    answer_text = f" {answer.strip()}"
    return prompt + rationale_text + answer_text, prompt, rationale_text, answer_text


def normalize_answer(ans: str) -> str:
    ans = ans.strip().replace(",", "").replace("$", "").replace("€", "")
    ans = ans.rstrip(".")
    if ans == "":
        return ""
    try:
        if re.search(r"\.\d", ans):
            val = float(ans)
            if val.is_integer():
                return str(int(val))
            return f"{val:.10f}".rstrip("0").rstrip(".")
        return str(int(float(ans)))
    except Exception:
        return ans


def extract_last_number(text: str) -> str:
    nums = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    return nums[-1] if nums else ""


def extract_answer(text: str) -> str:
    if "Answer:" in text:
        candidate = text.split("Answer:")[-1]
    elif "####" in text:
        candidate = text.split("####")[-1]
    else:
        candidate = text
    return extract_last_number(candidate)


def is_correct(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def extract_rationale_segment(text: str) -> str:
    if "Rationale:" in text and "Answer:" in text:
        start_idx = text.rfind("Rationale:") + len("Rationale:")
        end_idx = text.rfind("Answer:")
        return text[start_idx:end_idx].strip()
    if "Let's think" in text and "Answer:" in text:
        start_idx = text.rfind("Let's think")
        end_idx = text.rfind("Answer:")
        return text[start_idx:end_idx].strip()
    return ""


def get_fluff_tokens() -> List[str]:
    tokens = set()
    for phrase in FLUFF_PHRASES + OOD_FLUFF_PHRASES:
        for tok in re.findall(r"\b\w+\b", phrase.lower()):
            tokens.add(tok)
    return list(tokens)


def compute_fluff_rate(text: str, fluff_tokens: List[str]) -> float:
    if not text:
        return 0.0
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0
    fluff_set = set(fluff_tokens)
    count = sum(1 for w in words if w in fluff_set)
    return count / len(words)


def apply_fluff(question: str, rng: random.Random, strength: int = 1, ood: bool = False) -> str:
    lexicon = OOD_FLUFF_PHRASES if ood else FLUFF_PHRASES
    k = min(len(lexicon), max(1, strength))
    fluff = " ".join(rng.sample(lexicon, k=k))
    return f"{question.strip()} {fluff}"


def apply_irrelevant_numbers(question: str, rng: random.Random, strength: int = 1) -> str:
    clauses = []
    for _ in range(max(1, strength)):
        a, b = rng.randint(2, 50), rng.randint(2, 50)
        template = rng.choice(IRRELEVANT_NUMBER_TEMPLATES)
        clauses.append(template.format(a=a, b=b))
    return f"{question.strip()} {' '.join(clauses)}"


def build_sequence(tokenizer, question: str, rationale: str, answer: str, max_seq_len: int) -> Dict:
    _, prompt, rationale_text, answer_text = build_full_text(question, rationale, answer)

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    rat_ids = tokenizer.encode(rationale_text, add_special_tokens=False)
    ans_ids = tokenizer.encode(answer_text, add_special_tokens=False)

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    prefix_ids = []
    prefix_rat_mask = []
    if bos_id is not None:
        prefix_ids.append(bos_id)
        prefix_rat_mask.append(False)
    prefix_ids.extend(prompt_ids)
    prefix_rat_mask.extend([False] * len(prompt_ids))
    prefix_ids.extend(rat_ids)
    prefix_rat_mask.extend([True] * len(rat_ids))

    ans_ids_only = ans_ids
    ans_mask = [True] * len(ans_ids_only)
    if eos_id is not None:
        ans_ids = ans_ids_only + [eos_id]
        ans_mask = ans_mask + [False]
    else:
        ans_ids = ans_ids_only

    max_prefix = max_seq_len - len(ans_ids)
    if max_prefix < 0:
        ans_ids = ans_ids[-max_seq_len:]
        ans_mask = ans_mask[-max_seq_len:]
        prefix_ids = []
        prefix_rat_mask = []
    else:
        if len(prefix_ids) > max_prefix:
            drop = len(prefix_ids) - max_prefix
            prefix_ids = prefix_ids[drop:]
            prefix_rat_mask = prefix_rat_mask[drop:]

    input_ids = prefix_ids + ans_ids
    rationale_mask = prefix_rat_mask + [False] * len(ans_ids)
    answer_mask = [False] * len(prefix_ids) + ans_mask

    return {
        "input_ids": input_ids,
        "rationale_mask": rationale_mask,
        "answer_mask": answer_mask,
        "prompt_text": prompt,
        "question": question,
        "answer": answer,
    }


def _save_cache(path: Path, examples: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def _load_cache(path: Path) -> List[Dict]:
    examples = []
    with path.open() as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def _sanitize_model_tag(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()


def _teacher_cache_path(cfg, split: str, count: int) -> Path:
    tc = cfg.dataset.preprocessing.teacher_cot
    model_tag = _sanitize_model_tag(cfg.model.name)
    cache_dir = Path(".cache") / "teacher_cot"
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = (
        f"gsm8k_{split}_{count}_{model_tag}_mt{tc.max_new_tokens_rationale}"
        f"_t{tc.temperature}_p{tc.top_p}_s{tc.seed}.jsonl"
    )
    return cache_dir / name


def _processed_cache_path(cfg, split: str, count: int) -> Path:
    tc = cfg.dataset.preprocessing.teacher_cot
    model_tag = _sanitize_model_tag(cfg.model.name)
    cache_dir = Path(".cache") / "processed"
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = f"gsm8k_{split}_{count}_{model_tag}_seq{cfg.dataset.preprocessing.max_seq_len}_mt{tc.max_new_tokens_rationale}.jsonl"
    return cache_dir / name


def _extract_generated_rationale(text: str, prompt: str, tokenizer, max_tokens: int) -> str:
    if prompt in text:
        generated = text.split(prompt, 1)[-1]
    else:
        generated = text
    for marker in ["Answer:", "####"]:
        if marker in generated:
            generated = generated.split(marker)[0]
    generated = generated.strip()
    if not generated:
        return ""
    token_ids = tokenizer.encode(generated, add_special_tokens=False)
    token_ids = token_ids[:max_tokens]
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def generate_teacher_rationales(
    cfg, tokenizer, questions: List[str], split: str, fallback_rationales: List[str] = None
) -> List[str]:
    if getattr(cfg, "mode", "full") == "trial":
        return fallback_rationales or ["" for _ in questions]

    cache_path = _teacher_cache_path(cfg, split, len(questions))
    if cache_path.exists():
        cached = _load_cache(cache_path)
        if len(cached) >= len(questions):
            return [item["rationale"] for item in cached[: len(questions)]]

    model_name = model_lib.resolve_model_name(cfg.model.name)
    precision = str(cfg.model.precision).lower()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16 if precision == "fp16" else torch.float32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=".cache/",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    teacher_cfg = cfg.dataset.preprocessing.teacher_cot
    max_tokens = int(teacher_cfg.max_new_tokens_rationale)
    batch_size = 4
    rationales: List[str] = []

    for start in range(0, len(questions), batch_size):
        batch_questions = questions[start : start + batch_size]
        prompts = [build_prompt(q) for q in batch_questions]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(cfg.dataset.preprocessing.max_seq_len),
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            outputs = model.generate(
                **enc,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=float(teacher_cfg.temperature),
                top_p=float(teacher_cfg.top_p),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for text, prompt in zip(decoded, prompts):
            rationale = _extract_generated_rationale(text, prompt, tokenizer, max_tokens)
            if not rationale:
                rationale = "Let's think step by step."
            rationales.append(rationale)

    _save_cache(cache_path, [{"rationale": r} for r in rationales])

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rationales


def build_examples_from_dataset(ds, tokenizer, max_seq_len: int, rationales: List[str]) -> List[Dict]:
    examples = []
    for item, rationale in zip(ds, rationales):
        parsed_rationale, answer = parse_gsm8k_answer(item["answer"])
        rationale = rationale if rationale else (parsed_rationale or "Let's think step by step.")
        ex = build_sequence(tokenizer, item["question"], rationale, answer, max_seq_len)
        examples.append(ex)
    return examples


def load_gsm8k_datasets(cfg, tokenizer) -> Tuple[Dataset, Dataset, Dataset]:
    max_seq_len = int(cfg.dataset.preprocessing.max_seq_len)
    train_n = int(cfg.dataset.splits.train)
    dev_n = int(cfg.dataset.splits.dev)
    test_n = int(cfg.dataset.splits.test)
    seed = int(cfg.training.seed)

    train_cache = _processed_cache_path(cfg, "train", train_n)
    dev_cache = _processed_cache_path(cfg, "dev", dev_n)
    test_cache = _processed_cache_path(cfg, "test", test_n)

    if train_cache.exists() and dev_cache.exists() and test_cache.exists():
        train_examples = _load_cache(train_cache)
        dev_examples = _load_cache(dev_cache)
        test_examples = _load_cache(test_cache)
    else:
        ds_train = load_dataset("gsm8k", "main", split="train", cache_dir=".cache/")
        ds_test = load_dataset("gsm8k", "main", split="test", cache_dir=".cache/")
        ds_train = ds_train.shuffle(seed=seed)
        ds_test = ds_test.shuffle(seed=seed)

        train_slice = ds_train.select(range(min(train_n, len(ds_train))))
        dev_slice = ds_train.select(
            range(min(train_n, len(ds_train)), min(train_n + dev_n, len(ds_train)))
        )
        test_slice = ds_test.select(range(min(test_n, len(ds_test))))

        train_questions = [x["question"] for x in train_slice]
        dev_questions = [x["question"] for x in dev_slice]
        test_questions = [x["question"] for x in test_slice]

        train_fallback = [parse_gsm8k_answer(x["answer"])[0] for x in train_slice]
        dev_fallback = [parse_gsm8k_answer(x["answer"])[0] for x in dev_slice]
        test_fallback = [parse_gsm8k_answer(x["answer"])[0] for x in test_slice]

        train_rationales = generate_teacher_rationales(cfg, tokenizer, train_questions, "train", train_fallback)
        dev_rationales = generate_teacher_rationales(cfg, tokenizer, dev_questions, "dev", dev_fallback)
        test_rationales = generate_teacher_rationales(cfg, tokenizer, test_questions, "test", test_fallback)

        train_examples = build_examples_from_dataset(train_slice, tokenizer, max_seq_len, train_rationales)
        dev_examples = build_examples_from_dataset(dev_slice, tokenizer, max_seq_len, dev_rationales)
        test_examples = build_examples_from_dataset(test_slice, tokenizer, max_seq_len, test_rationales)

        _save_cache(train_cache, train_examples)
        _save_cache(dev_cache, dev_examples)
        _save_cache(test_cache, test_examples)

    return CoTDataset(train_examples), CoTDataset(dev_examples), CoTDataset(test_examples)


def generate_synthetic_examples(
    n: int, tokenizer, max_seq_len: int, seed: int, fluff_prob: float, irr_prob: float
) -> List[Dict]:
    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        a, b = rng.randint(1, 99), rng.randint(1, 99)
        question = f"John has {a} apples and buys {b} more. How many apples does he have?"
        rationale = f"He starts with {a} apples and buys {b} more, so {a} + {b} = {a + b}."
        answer = str(a + b)
        if rng.random() < fluff_prob:
            rationale += " " + rng.choice(FLUFF_PHRASES)
        if rng.random() < irr_prob:
            rationale += " " + rng.choice(IRRELEVANT_NUMBER_TEMPLATES).format(
                a=rng.randint(1, 20), b=rng.randint(1, 20)
            )
        examples.append(build_sequence(tokenizer, question, rationale, answer, max_seq_len))
    return examples


def load_synthetic_datasets(cfg, tokenizer) -> Tuple[Dataset, Dataset, Dataset]:
    max_seq_len = int(cfg.dataset.preprocessing.max_seq_len)
    train_n = int(cfg.dataset.splits.train)
    dev_n = int(cfg.dataset.splits.dev)
    test_n = int(cfg.dataset.splits.test)
    seed = int(cfg.training.seed)

    train_examples = generate_synthetic_examples(train_n, tokenizer, max_seq_len, seed, fluff_prob=0.4, irr_prob=0.4)
    dev_examples = generate_synthetic_examples(dev_n, tokenizer, max_seq_len, seed + 1, fluff_prob=0.4, irr_prob=0.4)
    test_examples = generate_synthetic_examples(
        test_n, tokenizer, max_seq_len, seed + 2, fluff_prob=0.7, irr_prob=0.7
    )
    return CoTDataset(train_examples), CoTDataset(dev_examples), CoTDataset(test_examples)


def load_datasets(cfg, tokenizer) -> Tuple[Dataset, Dataset, Dataset]:
    name = cfg.dataset.name.lower()
    if name == "gsm8k":
        return load_gsm8k_datasets(cfg, tokenizer)
    if "synthetic" in name:
        return load_synthetic_datasets(cfg, tokenizer)
    raise ValueError(f"Unsupported dataset: {cfg.dataset.name}")


class CoTDataset(Dataset):
    def __init__(self, examples: List[Dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]


def collate_fn(batch: List[Dict], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(len(ex["input_ids"]) for ex in batch)
    input_ids = []
    attention_mask = []
    rationale_mask = []
    answer_mask = []
    prompt_text = []
    answers = []
    questions = []

    for ex in batch:
        seq_len = len(ex["input_ids"])
        padding = max_len - seq_len
        input_ids.append(ex["input_ids"] + [pad_token_id] * padding)
        attention_mask.append([1] * seq_len + [0] * padding)
        rationale_mask.append(ex["rationale_mask"] + [False] * padding)
        answer_mask.append(ex["answer_mask"] + [False] * padding)
        prompt_text.append(ex["prompt_text"])
        answers.append(ex["answer"])
        questions.append(ex["question"])

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "rationale_mask": torch.tensor(rationale_mask, dtype=torch.bool),
        "answer_mask": torch.tensor(answer_mask, dtype=torch.bool),
        "prompt_text": prompt_text,
        "answer": answers,
        "question": questions,
    }
