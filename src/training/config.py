"""Lightweight command-line configuration for the existing Qwen training pipeline.

Relative paths are interpreted from the current working directory. Run the
examples from the repository root, or supply absolute paths.
"""
import argparse
from pathlib import Path


DEFAULT_ACTION_BASE = Path("models/qwen-action-base")


def positive_int(value):
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_training_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a Qwen action-token LoRA adapter (CUDA required).")
    parser.add_argument("--base-model-dir", type=Path, default=DEFAULT_ACTION_BASE)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/samples/project/ftg_messages_plain"),
                        help="Directory containing train.jsonl and valid.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen-v3"))
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--epochs", type=positive_int, default=3)
    parser.add_argument("--max-steps", type=positive_int, default=None,
                        help="Optional optimizer-step cap for smoke tests; omitted means all epochs")
    return parser.parse_args(argv)


def parse_builder_args(argv=None):
    parser = argparse.ArgumentParser(description="Add FightingICE action tokens to the Qwen2.5 base model.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B",
                        help="Hugging Face model ID or local source model directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ACTION_BASE)
    return parser.parse_args(argv)


def validate_training_paths(args):
    if not args.base_model_dir.is_dir() or not (args.base_model_dir / "config.json").is_file():
        raise ValueError(f"--base-model-dir must contain a local model and config.json: {args.base_model_dir}")
    for split in ("train.jsonl", "valid.jsonl"):
        if not (args.data_dir / split).is_file():
            raise ValueError(f"--data-dir is missing {split}: {args.data_dir}")
    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise ValueError(f"--output-dir must be a directory: {args.output_dir}")
