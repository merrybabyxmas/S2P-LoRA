"""Shared configuration and artifact helpers for the training entry points."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def parse_args_with_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Load YAML defaults from ``--config`` while preserving CLI overrides."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    known, _ = pre_parser.parse_known_args()
    if known.config:
        with open(known.config, encoding="utf-8") as handle:
            defaults = yaml.safe_load(handle) or {}
        valid = {action.dest for action in parser._actions}
        unknown = sorted(set(defaults) - valid)
        if unknown:
            parser.error(f"Unknown keys in {known.config}: {unknown}")
        parser.set_defaults(**defaults)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "--", value.strip()).strip("-")


def make_run_dir(
    root: str | Path,
    *,
    family: str,
    task: str,
    model: str,
    rank: int,
    switch_ratio: float,
    seed: int,
    overwrite: bool = False,
) -> Path:
    run_dir = (
        Path(root)
        / slug(family)
        / slug(task)
        / slug(model)
        / f"s2p-r{rank}-sw{switch_ratio:g}"
        / f"seed-{seed}"
    )
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Run directory already contains files: {run_dir}. Use --overwrite-output to reuse it."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def runtime_versions() -> dict[str, str]:
    import datasets
    import transformers

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "cuda": torch.version.cuda or "none",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
