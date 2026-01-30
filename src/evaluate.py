import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from omegaconf import OmegaConf
from scipy import stats

PRIMARY_METRIC = "accuracy"

EXPECTED_RESULTS_TEXT = """
**Synthetic Arithmetic-with-Fluff+Noise (same model/steps):**
- A Uniform CE: accuracy 88–91% (OOD irrelevant-number: 72–78%)
- C UF-KCoT: accuracy 93–95% (OOD: 82–86%)
- D CI-UF-KCoT: accuracy 95–97% (OOD: 86–90%)
- **E MCI-CoT (ours): accuracy 96.5–98% (OOD: 90–94%)**

Secondary expectations (E vs D):
- Fluff-token imitation rate: −25% to −45% relative
- Numeric distraction susceptibility: −20% to −35% relative error rate
- Answer invariance KL under synthetic junking: −30% to −50%
- Rationale length when generating CoT: −10% to −25%

**GSM8K (500 train / 200 dev, LoRA if available):**
- Expect +1.0 to +2.0 absolute accuracy over CI-UF-KCoT, with larger gains on items whose teacher CoTs contain redundant prose or extra numerals.

(Higher accuracy is better; lower OOD drop / KL / fluff imitation is better.)
""".strip()

sns.set_theme(style="whitegrid")


def normalize_key(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def derive_metric_preferences(text: str) -> Dict[str, str]:
    preferences: Dict[str, str] = {}
    for clause in re.split(r"[;\n]", text):
        clause = clause.strip()
        if not clause:
            continue
        match = re.search(r"(higher|lower)\s+(.+?)\s+is better", clause, flags=re.IGNORECASE)
        if match:
            direction = "max" if match.group(1).lower() == "higher" else "min"
            metrics_part = match.group(2)
            for part in re.split(r"/|,|and", metrics_part):
                key = part.strip().lower()
                if key:
                    preferences[key] = direction
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        metric_name, remainder = line.split(":", 1)
        if "−" in remainder or ("-" in remainder and "relative" in remainder):
            preferences[metric_name.strip().lower()] = "min"
    return preferences


def metric_direction(metric_name: str) -> str:
    name = metric_name.lower()
    preferences = derive_metric_preferences(EXPECTED_RESULTS_TEXT)
    name_norm = normalize_key(name)
    for key, direction in preferences.items():
        key_norm = normalize_key(key)
        if key_norm and (key_norm in name_norm or name_norm in key_norm):
            return direction
    if "accuracy" in name_norm and "drop" not in name_norm:
        return "max"
    if any(token in name_norm for token in ["loss", "error", "perplexity", "kl", "drop", "length", "rate"]):
        return "min"
    return "max"


def to_json_serializable(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_serializable(v) for v in obj]
    return obj


def load_wandb_config() -> Dict[str, str]:
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config file: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    return {"entity": cfg.wandb.entity, "project": cfg.wandb.project}


def parse_kv_args(argv: List[str]) -> Dict[str, str]:
    args: Dict[str, str] = {}
    for arg in argv[1:]:
        if "=" not in arg:
            raise ValueError(f"Invalid argument format: {arg}. Expected key=value.")
        key, value = arg.split("=", 1)
        args[key] = value
    if "results_dir" not in args or "run_ids" not in args:
        raise ValueError("Expected arguments: results_dir=... run_ids='[...]'")
    return args


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, default=to_json_serializable)


def plot_learning_curve(history: pd.DataFrame, run_id: str, out_dir: Path) -> Optional[Path]:
    if history.empty:
        return None
    fig, ax1 = plt.subplots(figsize=(8, 5))
    steps = history["_step"] if "_step" in history.columns else range(len(history))

    has_any = False
    if "train_loss" in history.columns:
        ax1.plot(steps, history["train_loss"], label="train_loss", color="tab:blue")
        ax1.set_ylabel("Loss")
        has_any = True
    ax1.set_xlabel("Step")

    ax2 = ax1.twinx()
    if "val_acc" in history.columns:
        ax2.plot(steps, history["val_acc"], label="val_acc", color="tab:green")
        ax2.set_ylabel("Accuracy")
        has_any = True

    if has_any:
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    else:
        ax1.text(0.5, 0.5, "No learning curve data", ha="center", va="center")

    plt.title(f"Learning Curve: {run_id}")
    plt.tight_layout()

    out_path = out_dir / f"{run_id}_learning_curve.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_ood_accuracy(summary: Dict[str, float], run_id: str, out_dir: Path) -> Path:
    labels = ["clean", "fluff", "irrelevant_numbers"]
    values = [
        summary.get("clean_acc", 0.0),
        summary.get("fluff_acc", 0.0),
        summary.get("irrelevant_numbers_acc", 0.0),
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_ylabel("Accuracy")
    ax.set_title(f"OOD Accuracy: {run_id}")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    out_path = out_dir / f"{run_id}_ood_accuracy_bar_chart.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_numeric_distraction_rate(summary: Dict[str, float], run_id: str, out_dir: Path) -> Path:
    rate = summary.get("numeric_distraction_error_rate", 0.0)
    fig, ax = plt.subplots(figsize=(4, 4))
    bars = ax.bar(["error_rate"], [rate], color="#C44E52")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Error Rate")
    ax.set_title(f"Numeric Distraction Error: {run_id}")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{rate:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    out_path = out_dir / f"{run_id}_numeric_distraction_error_rate_bar.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_confusion_matrix(confusion: Dict[str, int], run_id: str, out_dir: Path, suffix: str) -> Path:
    matrix = np.array(
        [
            [confusion.get("cc", 0), confusion.get("ci", 0)],
            [confusion.get("ic", 0), confusion.get("ii", 0)],
        ]
    )
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("OOD Correctness")
    ax.set_ylabel("Clean Correctness")
    ax.set_xticklabels(["Correct", "Incorrect"])
    ax.set_yticklabels(["Correct", "Incorrect"], rotation=0)
    ax.set_title(f"Confusion Matrix ({suffix}): {run_id}")
    plt.tight_layout()
    out_path = out_dir / f"{run_id}_confusion_matrix_{suffix}.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_invariance_kl(summary: Dict, history: pd.DataFrame, run_id: str, out_dir: Path) -> Path:
    kl_list = summary.get("answer_invariance_kl_list")
    if not kl_list and "answer_invariance_kl" in history.columns:
        kl_list = history["answer_invariance_kl"].dropna().tolist()
    kl_list = kl_list or [summary.get("answer_invariance_kl", 0.0)]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x=kl_list, ax=ax, color="#4C72B0")
    ax.set_xlabel("Answer Invariance KL")
    ax.set_title(f"Answer Invariance KL: {run_id}")
    mean_val = float(np.mean(kl_list)) if kl_list else 0.0
    ax.annotate(f"mean={mean_val:.3f}", xy=(0.02, 0.9), xycoords="axes fraction")
    plt.tight_layout()
    out_path = out_dir / f"{run_id}_answer_invariance_kl_boxplot.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_rationale_length(summary: Dict, run_id: str, out_dir: Path) -> Path:
    lengths = summary.get("rationale_length_list", [])
    if not lengths:
        lengths = [summary.get("rationale_length_mean", 0.0)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(lengths, bins=20, color="#55A868", alpha=0.75)
    ax.set_xlabel("Rationale Length (tokens)")
    ax.set_ylabel("Count")
    ax.set_title(f"Rationale Length Distribution: {run_id}")
    plt.tight_layout()
    out_path = out_dir / f"{run_id}_rationale_length_histogram.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_fluff_imitation(summary: Dict, run_id: str, out_dir: Path) -> Path:
    rates = summary.get("fluff_imitation_rates", [])
    if not rates:
        rates = [summary.get("fluff_imitation_rate", 0.0)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(rates, bins=20, color="#C44E52", alpha=0.75)
    ax.set_xlabel("Fluff Imitation Rate")
    ax.set_ylabel("Count")
    ax.set_title(f"Fluff Imitation: {run_id}")
    plt.tight_layout()
    out_path = out_dir / f"{run_id}_fluff_imitation_histogram.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_ood_curve(summary: Dict, run_id: str, out_dir: Path, key: str, suffix: str) -> Optional[Path]:
    curve = summary.get(key, {})
    strengths = curve.get("strengths", [])
    accs = curve.get("acc", [])
    if not strengths or not accs:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(strengths, accs, marker="o", color="#4C72B0")
    ax.set_xlabel("Perturbation Strength")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"OOD Curve ({suffix}): {run_id}")
    for s, a in zip(strengths, accs):
        ax.annotate(f"{a:.3f}", (s, a), textcoords="offset points", xytext=(0, 6), ha="center")
    plt.tight_layout()
    out_path = out_dir / f"{run_id}_ood_curve_{suffix}.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_comparison_bar(metrics: Dict[str, Dict[str, float]], metric_name: str, out_dir: Path) -> Optional[Path]:
    values = metrics.get(metric_name, {})
    if not values:
        return None
    run_ids = list(values.keys())
    vals = [values[r] for r in run_ids]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(run_ids)), vals, color="#4C72B0")
    ax.set_ylabel(metric_name)
    ax.set_xticks(range(len(run_ids)))
    ax.set_xticklabels(run_ids, rotation=45, ha="right")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    out_path = out_dir / f"comparison_{metric_name}_bar_chart.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_metrics_table(metrics: Dict[str, Dict[str, float]], out_dir: Path) -> Path:
    df = pd.DataFrame(metrics).T
    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(df))))
    ax.axis("off")
    table = ax.table(
        cellText=np.round(df.values, 4),
        rowLabels=df.index,
        colLabels=df.columns,
        loc="center",
    )
    table.scale(1, 1.5)
    plt.tight_layout()
    out_path = out_dir / "comparison_metrics_table.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_comparison_boxplot(metric_lists: Dict[str, List[float]], metric_name: str, out_dir: Path) -> Optional[Path]:
    if not metric_lists:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    data = [metric_lists[k] for k in metric_lists]
    ax.boxplot(data, labels=list(metric_lists.keys()), showmeans=True)
    ax.set_ylabel(metric_name)
    ax.set_xticklabels(list(metric_lists.keys()), rotation=45, ha="right")
    ax.set_title(f"{metric_name} Distribution")
    plt.tight_layout()
    out_path = out_dir / f"comparison_{metric_name}_boxplot.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def compute_mcnemar_significance(correct_lists: Dict[str, List[int]]) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    run_ids = list(correct_lists.keys())
    for i, run_a in enumerate(run_ids):
        for run_b in run_ids[i + 1 :]:
            a_vals = correct_lists[run_a]
            b_vals = correct_lists[run_b]
            n = min(len(a_vals), len(b_vals))
            if n == 0:
                continue
            b_count = sum((a_vals[idx] == 1) and (b_vals[idx] == 0) for idx in range(n))
            c_count = sum((a_vals[idx] == 0) and (b_vals[idx] == 1) for idx in range(n))
            total = b_count + c_count
            if total == 0:
                p_val = 1.0
            else:
                p_val = stats.binomtest(min(b_count, c_count), n=total, p=0.5).pvalue
            results.setdefault(run_a, {})[run_b] = float(p_val)
            results.setdefault(run_b, {})[run_a] = float(p_val)
    return results


def compute_ttest_significance(metric_lists: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    run_ids = list(metric_lists.keys())
    for i, run_a in enumerate(run_ids):
        for run_b in run_ids[i + 1 :]:
            a_vals = np.array(metric_lists[run_a], dtype=float)
            b_vals = np.array(metric_lists[run_b], dtype=float)
            if len(a_vals) < 2 or len(b_vals) < 2:
                continue
            t_stat, p_val = stats.ttest_ind(a_vals, b_vals, equal_var=False)
            results.setdefault(run_a, {})[run_b] = float(p_val)
            results.setdefault(run_b, {})[run_a] = float(p_val)
    return results


def plot_significance_heatmap(significance: Dict[str, Dict[str, float]], out_dir: Path, name: str) -> Optional[Path]:
    if not significance:
        return None
    run_ids = sorted(significance.keys())
    matrix = np.ones((len(run_ids), len(run_ids)))
    for i, run_a in enumerate(run_ids):
        for j, run_b in enumerate(run_ids):
            if run_a == run_b:
                matrix[i, j] = 0.0
            else:
                matrix[i, j] = significance.get(run_a, {}).get(run_b, 1.0)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, cmap="viridis", xticklabels=run_ids, yticklabels=run_ids, ax=ax)
    ax.set_title("Significance (p-values)")
    plt.tight_layout()
    out_path = out_dir / name
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_kv_args(sys.argv)
    run_ids = json.loads(args["run_ids"])
    if not isinstance(run_ids, list):
        raise ValueError("run_ids must be a JSON list")
    cfg = load_wandb_config()
    entity = cfg["entity"]
    project = cfg["project"]

    results_dir = Path(args["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()

    histories: Dict[str, pd.DataFrame] = {}
    summaries: Dict[str, Dict] = {}
    configs: Dict[str, Dict] = {}
    generated_paths: List[str] = []
    correct_lists: Dict[str, List[int]] = {}
    kl_lists: Dict[str, List[float]] = {}

    for run_id in run_ids:
        run = api.run(f"{entity}/{project}/{run_id}")
        history = run.history()
        summary = run.summary._json_dict
        config = dict(run.config)
        if str(config.get("mode", "full")) == "trial":
            raise ValueError(f"Run {run_id} was executed in trial mode; evaluation requires full runs.")

        run_dir = results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = run_dir / "metrics.json"
        save_json(
            metrics_path,
            {
                "history": history.to_dict(orient="list"),
                "summary": summary,
                "config": config,
            },
        )
        generated_paths.append(str(metrics_path))

        curve_path = plot_learning_curve(history, run_id, run_dir)
        if curve_path:
            generated_paths.append(str(curve_path))
        generated_paths.append(str(plot_ood_accuracy(summary, run_id, run_dir)))
        generated_paths.append(str(plot_numeric_distraction_rate(summary, run_id, run_dir)))
        generated_paths.append(str(plot_invariance_kl(summary, history, run_id, run_dir)))
        generated_paths.append(str(plot_rationale_length(summary, run_id, run_dir)))
        generated_paths.append(str(plot_fluff_imitation(summary, run_id, run_dir)))

        if summary.get("confusion_clean_vs_fluff"):
            generated_paths.append(
                str(plot_confusion_matrix(summary["confusion_clean_vs_fluff"], run_id, run_dir, "fluff"))
            )
        if summary.get("confusion_clean_vs_irrelevant_numbers"):
            generated_paths.append(
                str(
                    plot_confusion_matrix(
                        summary["confusion_clean_vs_irrelevant_numbers"],
                        run_id,
                        run_dir,
                        "irrelevant_numbers",
                    )
                )
            )
        if summary.get("ood_fluff_curve"):
            fluff_curve = plot_ood_curve(summary, run_id, run_dir, "ood_fluff_curve", "fluff")
            if fluff_curve:
                generated_paths.append(str(fluff_curve))
        if summary.get("ood_irrelevant_curve"):
            irr_curve = plot_ood_curve(summary, run_id, run_dir, "ood_irrelevant_curve", "irrelevant_numbers")
            if irr_curve:
                generated_paths.append(str(irr_curve))

        histories[run_id] = history
        summaries[run_id] = summary
        configs[run_id] = config

        if "test_correct_list" in summary:
            correct_lists[run_id] = [int(x) for x in summary["test_correct_list"]]
        if "answer_invariance_kl_list" in summary:
            kl_lists[run_id] = [float(x) for x in summary["answer_invariance_kl_list"]]

    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    metrics: Dict[str, Dict[str, float]] = {}
    for run_id, summary in summaries.items():
        for key, value in summary.items():
            if key.startswith("_"):
                continue
            if isinstance(value, (int, float)):
                metrics.setdefault(key, {})[run_id] = float(value)

    primary_metric = PRIMARY_METRIC
    if "accuracy" not in metrics:
        if "test_acc" in metrics:
            metrics["accuracy"] = metrics["test_acc"]
        elif "val_acc" in metrics:
            metrics["accuracy"] = metrics["val_acc"]
        else:
            metrics["accuracy"] = {rid: 0.0 for rid in run_ids}

    proposed_runs = {rid: metrics[primary_metric].get(rid, 0.0) for rid in run_ids if "proposed" in rid}
    baseline_runs = {
        rid: metrics[primary_metric].get(rid, 0.0)
        for rid in run_ids
        if "comparative" in rid or "baseline" in rid
    }

    direction = metric_direction(primary_metric)

    best_proposed_id, best_proposed_val = ("", float("inf") if direction == "min" else float("-inf"))
    if proposed_runs:
        best_proposed_id = min(proposed_runs, key=proposed_runs.get) if direction == "min" else max(
            proposed_runs, key=proposed_runs.get
        )
        best_proposed_val = proposed_runs[best_proposed_id]

    best_baseline_id, best_baseline_val = ("", float("inf") if direction == "min" else float("-inf"))
    if baseline_runs:
        best_baseline_id = min(baseline_runs, key=baseline_runs.get) if direction == "min" else max(
            baseline_runs, key=baseline_runs.get
        )
        best_baseline_val = baseline_runs[best_baseline_id]

    if best_baseline_val in (0.0, float("-inf"), float("inf")):
        gap = 0.0
    else:
        gap = (best_proposed_val - best_baseline_val) / abs(best_baseline_val) * 100.0
        if direction == "min":
            gap = -gap

    aggregated = {
        "primary_metric": primary_metric,
        "metrics": metrics,
        "best_proposed": {"run_id": best_proposed_id, "value": best_proposed_val},
        "best_baseline": {"run_id": best_baseline_id, "value": best_baseline_val},
        "gap": gap,
    }

    aggregated_path = comparison_dir / "aggregated_metrics.json"
    save_json(aggregated_path, aggregated)
    generated_paths.append(str(aggregated_path))

    for metric_name in [
        "accuracy",
        "ood_accuracy_drop",
        "answer_invariance_kl",
        "numeric_distraction_error_rate",
        "fluff_imitation_rate",
        "rationale_length_mean",
    ]:
        bar_path = plot_comparison_bar(metrics, metric_name, comparison_dir)
        if bar_path:
            generated_paths.append(str(bar_path))

    metrics_table_path = plot_metrics_table(metrics, comparison_dir)
    generated_paths.append(str(metrics_table_path))

    kl_boxplot = plot_comparison_boxplot(kl_lists, "answer_invariance_kl", comparison_dir)
    if kl_boxplot:
        generated_paths.append(str(kl_boxplot))

    significance = compute_mcnemar_significance(correct_lists)
    sig_path = comparison_dir / "comparison_significance_accuracy.json"
    save_json(sig_path, significance)
    generated_paths.append(str(sig_path))

    sig_heatmap = plot_significance_heatmap(significance, comparison_dir, "comparison_significance_heatmap_accuracy.pdf")
    if sig_heatmap:
        generated_paths.append(str(sig_heatmap))

    kl_significance = compute_ttest_significance(kl_lists)
    kl_sig_path = comparison_dir / "comparison_significance_invariance_kl.json"
    save_json(kl_sig_path, kl_significance)
    generated_paths.append(str(kl_sig_path))

    kl_heatmap = plot_significance_heatmap(
        kl_significance, comparison_dir, "comparison_significance_heatmap_invariance_kl.pdf"
    )
    if kl_heatmap:
        generated_paths.append(str(kl_heatmap))

    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
