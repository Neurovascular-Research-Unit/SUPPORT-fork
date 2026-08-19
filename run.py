#!/usr/bin/env python3
"""Train one generalized SUPPORT model from every TIFF in one directory."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_PARAMETERS = REPO_DIR / "notebooks" / "parameters.json"
TIFF_SUFFIXES = {".tif", ".tiff"}


def slurm_cpu_count(default: int = 2) -> int:
    """Use the CPUs allocated by Slurm when available."""
    value = os.environ.get("SLURM_CPUS_PER_TASK")
    if value is None:
        return default
    try:
        count = int(value)
    except ValueError:
        return default
    return count if count >= 0 else default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a single generalized SUPPORT model using all top-level TIFF "
            "stacks in one directory."
        )
    )
    parser.add_argument(
        "--in-path",
        type=Path,
        required=True,
        metavar="DIRECTORY",
        help="directory containing the training .tif/.tiff stacks",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_DIR / "outputs",
        help="output root (the experiment name is added beneath it)",
    )
    parser.add_argument(
        "--exp-name",
        help="experiment/model name (default: input directory name)",
    )
    parser.add_argument(
        "--dimensionality",
        choices=("2+1D", "3D", "3+1D"),
        default="2+1D",
        help="training mode used to select parameters (default: %(default)s)",
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=DEFAULT_PARAMETERS,
        help="JSON parameter file (default: %(default)s)",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        help="final total number of epochs (overrides the parameter file)",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=0,
        help=(
            "epoch at which to start; N loads model/optimizer/scaler from "
            "epoch N-1, matching the original SUPPORT trainer (default: 0)"
        ),
    )
    parser.add_argument(
        "--n-cpu",
        type=int,
        default=None,
        help="DataLoader workers (default: SLURM_CPUS_PER_TASK, otherwise 2)",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        help="batches prefetched by each worker (overrides the parameter file)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="training batch size (overrides the parameter file)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=3,
        metavar=("T_OR_Z", "Y", "X"),
        help="training patch size (overrides the parameter file)",
    )
    parser.add_argument(
        "--patch-interval",
        type=int,
        nargs=3,
        metavar=("T_OR_Z", "Y", "X"),
        help="training patch interval (overrides the parameter file)",
    )
    memory_group = parser.add_mutually_exclusive_group()
    memory_group.add_argument(
        "--load-to-memory",
        dest="load_to_memory",
        action="store_true",
        help="load all TIFF stacks into RAM",
    )
    memory_group.add_argument(
        "--no-load-to-memory",
        dest="load_to_memory",
        action="store_false",
        help="use TIFF memory mapping where possible",
    )
    parser.set_defaults(load_to_memory=None)
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument(
        "--use-amp",
        dest="use_amp",
        action="store_true",
        help="enable automatic mixed precision",
    )
    amp_group.add_argument(
        "--no-use-amp",
        dest="use_amp",
        action="store_false",
        help="disable automatic mixed precision",
    )
    parser.set_defaults(use_amp=None)
    parser.add_argument(
        "--use-cpu",
        action="store_true",
        help="train on CPU intentionally (otherwise a CUDA GPU is required)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        help="training random seed (overrides the parameter file)",
    )
    return parser


def find_tiffs(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES
    )


def load_training_parameters(path: Path, dimensionality: str) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            parameters = json.load(file)
    except FileNotFoundError as exc:
        raise ValueError(f"Parameter file does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read parameter file {path}: {exc}") from exc

    try:
        training = parameters[dimensionality]["train"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Parameter file {path} has no '{dimensionality}.train' object"
        ) from exc
    if not isinstance(training, dict):
        raise ValueError(f"'{dimensionality}.train' in {path} must be an object")
    return dict(training)


def build_training_parameters(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> Dict[str, Any]:
    in_path = args.in_path.expanduser().resolve()
    if not in_path.is_dir():
        parser.error(f"--in-path must be an existing directory: {in_path}")
    tiffs = find_tiffs(in_path)
    if not tiffs:
        parser.error(f"no top-level .tif or .tiff files found in {in_path}")

    parameters_path = args.parameters.expanduser().resolve()
    try:
        training = load_training_parameters(parameters_path, args.dimensionality)
    except ValueError as exc:
        parser.error(str(exc))

    overrides = {
        "n_epochs": args.n_epochs,
        "n_cpu": args.n_cpu,
        "prefetch_factor": args.prefetch_factor,
        "batch_size": args.batch_size,
        "patch_size": args.patch_size,
        "patch_interval": args.patch_interval,
        "load_to_memory": args.load_to_memory,
        "use_amp": args.use_amp,
        "rnd_seed": args.random_seed,
    }
    training.update({key: value for key, value in overrides.items() if value is not None})
    training.setdefault("n_cpu", slurm_cpu_count())
    training.update(
        {
            "in_path": str(in_path),
            "out_dir": str(args.out_dir.expanduser().resolve()),
            "exp_name": args.exp_name or in_path.name,
            "dimensionality": args.dimensionality,
            "rm_dir": False,
            "epoch": args.epoch,
            "checkpoints": True,
            "resume": False,
            "rm_intermediaries": False,
            "use_cpu": args.use_cpu,
        }
    )

    n_epochs = training.get("n_epochs")
    if not isinstance(n_epochs, int) or n_epochs < 1:
        parser.error("n_epochs must be a positive integer")
    if args.epoch < 0 or args.epoch > n_epochs:
        parser.error("--epoch must be between 0 and n_epochs, inclusive")
    if training["n_cpu"] < 0:
        parser.error("--n-cpu must be non-negative")
    if training.get("prefetch_factor", 1) < 1:
        parser.error("--prefetch-factor must be positive")
    if training.get("batch_size", 0) < 1:
        parser.error("--batch-size must be positive")

    print(f"Found {len(tiffs)} training TIFF stack(s) in {in_path}")
    return training


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    training = build_training_parameters(args, parser)

    # These must be set before importing torch through TrainingClass.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("MPLBACKEND", "Agg")

    import torch

    if not args.use_cpu and not torch.cuda.is_available():
        parser.error("no CUDA GPU is visible; request a GPU or pass --use-cpu")

    from utils.train_collection import TrainingClass

    trainer = TrainingClass(training)
    try:
        trainer.run()
    finally:
        trainer.release_resources()

    print(f"Training complete. Checkpoints: {trainer.pth_dir}")
    print(f"Final model: {trainer.model_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
