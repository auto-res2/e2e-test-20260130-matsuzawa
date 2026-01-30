import subprocess
import sys
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if cfg.run is None:
        raise ValueError("run must be provided, e.g., run=proposed-qwen3-1.7b-gsm8k")
    if cfg.mode not in {"trial", "full"}:
        raise ValueError("mode must be 'trial' or 'full'")
    run_path = Path(get_original_cwd()) / "config" / "runs" / f"{cfg.run}.yaml"
    if not run_path.exists():
        raise FileNotFoundError(f"Run config not found: {run_path}")

    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        f"run={cfg.run}",
        f"results_dir={results_dir}",
        f"mode={cfg.mode}",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
