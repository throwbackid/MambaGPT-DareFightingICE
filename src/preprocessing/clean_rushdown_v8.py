#!/usr/bin/env python3

"""
Prepare the rushdown v8 training dataset.

Steps:
1. Read dagger_analysis/cleaned.jsonl.
2. Remove rows where self.remaining_frames or
   opponent.remaining_frames is negative.
3. Read rushdownv3data/train.jsonl.
4. Stratified-sample 60,000 original v3 rows according to assistant action.
5. Merge the sampled v3 rows with the cleaned DAgger rows.
6. Shuffle everything.
7. Write rushdownv8data/train.jsonl.

Run from the Python directory:

    python prepare_rushdownv8_data.py

Optional:

    python prepare_rushdownv8_data.py --sample-size 60000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DAGGER_PATH = Path("dagger_analysis/cleaned.jsonl")
DEFAULT_V3_PATH = Path("rushdownv3data/train.jsonl")
DEFAULT_OUTPUT_PATH = Path("rushdownv8data/train.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a merged v3 + DAgger training dataset."
    )

    parser.add_argument(
        "--dagger-file",
        type=Path,
        default=DEFAULT_DAGGER_PATH,
        help=f"DAgger input file. Default: {DEFAULT_DAGGER_PATH}",
    )

    parser.add_argument(
        "--v3-file",
        type=Path,
        default=DEFAULT_V3_PATH,
        help=f"Original v3 input file. Default: {DEFAULT_V3_PATH}",
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output file. Default: {DEFAULT_OUTPUT_PATH}",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=60_000,
        help="Number of original v3 rows to sample. Default: 60000",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )

    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exception:
                raise ValueError(
                    f"Malformed JSON in {path}, line {line_number}: "
                    f"{exception.msg}"
                ) from exception

            if not isinstance(row, dict):
                raise ValueError(
                    f"Row in {path}, line {line_number} is not a JSON object."
                )

            rows.append(row)

    return rows


def get_assistant_action(row: dict[str, Any]) -> str:
    messages = row.get("messages")

    if not isinstance(messages, list):
        raise ValueError("Row does not contain a valid messages list.")

    for message in reversed(messages):
        if (
            isinstance(message, dict)
            and message.get("role") == "assistant"
            and isinstance(message.get("content"), str)
        ):
            action = message["content"].strip()

            if action:
                return action

    raise ValueError("Could not find a valid assistant action in row.")


def parse_state(row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("messages")

    if not isinstance(messages, list):
        raise ValueError("Row does not contain a valid messages list.")

    for message in messages:
        if (
            isinstance(message, dict)
            and message.get("role") == "user"
            and isinstance(message.get("content"), str)
        ):
            try:
                state = json.loads(message["content"])
            except json.JSONDecodeError as exception:
                raise ValueError(
                    f"User content contains malformed state JSON: {exception.msg}"
                ) from exception

            if not isinstance(state, dict):
                raise ValueError("Decoded user state is not a JSON object.")

            return state

    raise ValueError("Could not find user state in row.")


def has_negative_remaining_frames(row: dict[str, Any]) -> bool:
    state = parse_state(row)

    for character_name in ("self", "opponent"):
        character = state.get(character_name)

        if not isinstance(character, dict):
            raise ValueError(f"State is missing {character_name} data.")

        remaining_frames = character.get("remaining_frames")

        if not isinstance(remaining_frames, (int, float)):
            raise ValueError(
                f"{character_name}.remaining_frames is missing or non-numeric."
            )

        if remaining_frames < 0:
            return True

    return False


def clean_dagger_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept_rows: list[dict[str, Any]] = []
    removed_rows: list[dict[str, Any]] = []

    for row in rows:
        if has_negative_remaining_frames(row):
            removed_rows.append(row)
        else:
            kept_rows.append(row)

    return kept_rows, removed_rows


def calculate_stratified_quotas(
    group_sizes: dict[str, int],
    sample_size: int,
) -> dict[str, int]:
    """
    Calculate per-action sample counts while preserving the original
    distribution as closely as possible.

    Uses the largest-remainder method so the quotas sum exactly to sample_size.
    """

    total_rows = sum(group_sizes.values())

    if sample_size > total_rows:
        raise ValueError(
            f"Requested {sample_size:,} rows, but only {total_rows:,} exist."
        )

    exact_quotas: dict[str, float] = {
        action: size * sample_size / total_rows
        for action, size in group_sizes.items()
    }

    quotas: dict[str, int] = {
        action: math.floor(exact)
        for action, exact in exact_quotas.items()
    }

    assigned = sum(quotas.values())
    remaining = sample_size - assigned

    remainder_order = sorted(
        exact_quotas,
        key=lambda action: exact_quotas[action] - quotas[action],
        reverse=True,
    )

    for action in remainder_order[:remaining]:
        quotas[action] += 1

    if sum(quotas.values()) != sample_size:
        raise RuntimeError("Stratified quotas do not sum to requested sample size.")

    for action, quota in quotas.items():
        if quota > group_sizes[action]:
            raise RuntimeError(
                f"Quota for {action} exceeds available rows: "
                f"{quota} > {group_sizes[action]}"
            )

    return quotas


def stratified_sample(
    rows: list[dict[str, Any]],
    sample_size: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row_number, row in enumerate(rows, start=1):
        try:
            action = get_assistant_action(row)
        except ValueError as exception:
            raise ValueError(
                f"Invalid v3 row at loaded row {row_number}: {exception}"
            ) from exception

        grouped_rows[action].append(row)

    group_sizes = {
        action: len(action_rows)
        for action, action_rows in grouped_rows.items()
    }

    quotas = calculate_stratified_quotas(
        group_sizes=group_sizes,
        sample_size=sample_size,
    )

    sampled_rows: list[dict[str, Any]] = []

    for action, action_rows in grouped_rows.items():
        quota = quotas[action]

        if quota == len(action_rows):
            selected = list(action_rows)
        else:
            selected = rng.sample(action_rows, quota)

        sampled_rows.extend(selected)

    rng.shuffle(sampled_rows)

    return sampled_rows, quotas


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def print_distribution(
    title: str,
    rows: list[dict[str, Any]],
    limit: int = 40,
) -> None:
    counts = Counter(get_assistant_action(row) for row in rows)
    total = len(rows)

    print()
    print(title)
    print("-" * 62)

    for action, count in counts.most_common(limit):
        percentage = count / total * 100 if total else 0
        print(f"{action:<25} {count:>8,}  {percentage:>7.2f}%")

    print(f"{'TOTAL':<25} {total:>8,}")


def main() -> None:
    args = parse_args()

    if args.sample_size <= 0:
        raise ValueError("--sample-size must be greater than zero.")

    rng = random.Random(args.seed)

    print(f"Reading DAgger data: {args.dagger_file}")
    dagger_rows = read_jsonl(args.dagger_file)

    cleaned_dagger_rows, removed_negative_rows = clean_dagger_rows(
        dagger_rows
    )

    print(f"DAgger rows loaded:             {len(dagger_rows):,}")
    print(f"Negative-frame rows removed:   {len(removed_negative_rows):,}")
    print(f"Clean DAgger rows:             {len(cleaned_dagger_rows):,}")

    print()
    print(f"Reading original v3 data: {args.v3_file}")
    v3_rows = read_jsonl(args.v3_file)

    print(f"Original v3 rows loaded:       {len(v3_rows):,}")

    sampled_v3_rows, quotas = stratified_sample(
        rows=v3_rows,
        sample_size=args.sample_size,
        rng=rng,
    )

    print(f"Stratified v3 rows sampled:    {len(sampled_v3_rows):,}")
    print(f"Actions represented:           {len(quotas):,}")

    merged_rows = sampled_v3_rows + cleaned_dagger_rows
    rng.shuffle(merged_rows)

    write_jsonl(args.output_file, merged_rows)

    print_distribution(
        "Original v3 action distribution",
        v3_rows,
    )

    print_distribution(
        "Sampled v3 action distribution",
        sampled_v3_rows,
    )

    print_distribution(
        "Clean DAgger action distribution",
        cleaned_dagger_rows,
    )

    print_distribution(
        "Final merged action distribution",
        merged_rows,
    )

    print()
    print("=" * 62)
    print("Dataset preparation complete")
    print(f"Sampled original v3 rows:      {len(sampled_v3_rows):,}")
    print(f"Clean DAgger rows:             {len(cleaned_dagger_rows):,}")
    print(f"Final merged rows:             {len(merged_rows):,}")
    print(f"Random seed:                   {args.seed}")
    print(f"Output:                        {args.output_file.resolve()}")


if __name__ == "__main__":
    main()