#!/usr/bin/env python3
"""
Analyze a disagreement-only DAgger JSONL dataset.

Expected row format:
{
  "messages": [
    {"role": "system", "content": "Predict the next FightingICE action."},
    {"role": "user", "content": "{\"opponent\": {...}, \"self\": {...}}"},
    {"role": "assistant", "content": "DASH"}
  ],
  "metadata": {
    "learner_action": "...",
    "expert_action": "DASH",
    "source": "dagger_disagreement",
    "blackmamba_raw_action": "DASH",
    "blackmamba_action_index": 29,
    "frame_num": 1566
  }
}

Usage:
    python analyze_dagger.py dagger_data.jsonl

Optional legal-action validation:
    python analyze_dagger.py dagger_data.jsonl \
        --legal-actions-file legal_actions.txt

legal_actions.txt should contain one action per line.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXPECTED_SYSTEM_PROMPT = "Predict the next FightingICE action."

REQUIRED_CHARACTERS = ("self", "opponent")
REQUIRED_CHARACTER_FIELDS = (
    "remaining_frames",
    "hp",
    "action",
    "state",
    "position",
    "speed",
    "energy",
    "projectiles",
)
REQUIRED_POSITION_FIELDS = ("top", "left", "bottom", "right")
REQUIRED_SPEED_FIELDS = ("x", "y")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and analyze a FightingICE DAgger JSONL dataset."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Input DAgger JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dagger_analysis"),
        help="Directory for analysis outputs. Default: dagger_analysis",
    )
    parser.add_argument(
        "--legal-actions-file",
        type=Path,
        default=None,
        help="Optional text file containing one legal action per line.",
    )
    parser.add_argument(
        "--system-prompt",
        default=EXPECTED_SYSTEM_PROMPT,
        help="Expected system prompt.",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep exact duplicate state-label rows in cleaned.jsonl.",
    )
    return parser.parse_args()


def load_legal_actions(path: Path | None) -> set[str] | None:
    if path is None:
        return None

    if not path.exists():
        raise FileNotFoundError(f"Legal-actions file not found: {path}")

    actions: set[str] = set()

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            action = line.strip()

            if not action or action.startswith("#"):
                continue

            actions.add(action)

    if not actions:
        raise ValueError(f"No legal actions found in {path}")

    return actions


def is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def validate_character(
    character: Any,
    character_name: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    if not isinstance(character, dict):
        errors.append(f"{character_name} must be an object")
        return

    for field in REQUIRED_CHARACTER_FIELDS:
        if field not in character:
            errors.append(f"{character_name}.{field} is missing")

    numeric_fields = ("remaining_frames", "hp", "energy")

    for field in numeric_fields:
        if field in character and not is_number(character[field]):
            errors.append(f"{character_name}.{field} must be numeric")

    if "remaining_frames" in character:
        value = character["remaining_frames"]
        if is_number(value) and value < 0:
            warnings.append(f"{character_name}.remaining_frames is negative")

    if "hp" in character:
        value = character["hp"]
        if is_number(value) and value < 0:
            warnings.append(f"{character_name}.hp is negative")

    if "energy" in character:
        value = character["energy"]
        if is_number(value) and value < 0:
            warnings.append(f"{character_name}.energy is negative")

    for field in ("action", "state"):
        if field in character:
            value = character[field]
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{character_name}.{field} must be a non-empty string")

    position = character.get("position")

    if not isinstance(position, dict):
        errors.append(f"{character_name}.position must be an object")
    else:
        for field in REQUIRED_POSITION_FIELDS:
            if field not in position:
                errors.append(f"{character_name}.position.{field} is missing")
            elif not is_number(position[field]):
                errors.append(
                    f"{character_name}.position.{field} must be numeric"
                )

        if all(
            field in position and is_number(position[field])
            for field in REQUIRED_POSITION_FIELDS
        ):
            if position["left"] > position["right"]:
                warnings.append(
                    f"{character_name}.position.left is greater than right"
                )

            if position["top"] > position["bottom"]:
                warnings.append(
                    f"{character_name}.position.top is greater than bottom"
                )

    speed = character.get("speed")

    if not isinstance(speed, dict):
        errors.append(f"{character_name}.speed must be an object")
    else:
        for field in REQUIRED_SPEED_FIELDS:
            if field not in speed:
                errors.append(f"{character_name}.speed.{field} is missing")
            elif not is_number(speed[field]):
                errors.append(f"{character_name}.speed.{field} must be numeric")

    projectiles = character.get("projectiles")

    if projectiles is not None and not isinstance(projectiles, list):
        errors.append(f"{character_name}.projectiles must be a list")


def validate_state(
    state: Any,
    errors: list[str],
    warnings: list[str],
) -> None:
    if not isinstance(state, dict):
        errors.append("user content must decode to a JSON object")
        return

    for character_name in REQUIRED_CHARACTERS:
        if character_name not in state:
            errors.append(f"state.{character_name} is missing")
            continue

        validate_character(
            state[character_name],
            character_name,
            errors,
            warnings,
        )


def extract_messages(
    row: dict[str, Any],
    expected_system_prompt: str,
    errors: list[str],
    warnings: list[str],
) -> tuple[dict[str, Any] | None, str | None]:
    messages = row.get("messages")

    if not isinstance(messages, list):
        errors.append("messages must be a list")
        return None, None

    if len(messages) != 3:
        errors.append(f"messages must contain exactly 3 entries, found {len(messages)}")
        return None, None

    expected_roles = ("system", "user", "assistant")

    for index, expected_role in enumerate(expected_roles):
        message = messages[index]

        if not isinstance(message, dict):
            errors.append(f"messages[{index}] must be an object")
            continue

        role = message.get("role")
        content = message.get("content")

        if role != expected_role:
            errors.append(
                f"messages[{index}].role should be {expected_role!r}, found {role!r}"
            )

        if not isinstance(content, str):
            errors.append(f"messages[{index}].content must be a string")

    if errors:
        return None, None

    system_content = messages[0]["content"]
    user_content = messages[1]["content"]
    assistant_content = messages[2]["content"].strip()

    if system_content != expected_system_prompt:
        warnings.append("system prompt differs from expected prompt")

    if not assistant_content:
        errors.append("assistant action is empty")

    try:
        state = json.loads(user_content)
    except json.JSONDecodeError as exception:
        errors.append(
            "user content contains invalid JSON: "
            f"{exception.msg} at position {exception.pos}"
        )
        return None, assistant_content

    validate_state(state, errors, warnings)

    return state, assistant_content


def validate_metadata(
    row: dict[str, Any],
    assistant_action: str | None,
    legal_actions: set[str] | None,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    metadata = row.get("metadata")

    if metadata is None:
        warnings.append("metadata is missing")
        return {}

    if not isinstance(metadata, dict):
        errors.append("metadata must be an object")
        return {}

    learner_action = metadata.get("learner_action")
    expert_action = metadata.get("expert_action")
    raw_action = metadata.get("blackmamba_raw_action")
    source = metadata.get("source")
    frame_num = metadata.get("frame_num")
    action_index = metadata.get("blackmamba_action_index")

    if learner_action is not None and not isinstance(learner_action, str):
        errors.append("metadata.learner_action must be a string")

    if expert_action is not None and not isinstance(expert_action, str):
        errors.append("metadata.expert_action must be a string")

    if raw_action is not None and not isinstance(raw_action, str):
        errors.append("metadata.blackmamba_raw_action must be a string")

    if source is not None and not isinstance(source, str):
        errors.append("metadata.source must be a string")

    if frame_num is not None and not isinstance(frame_num, int):
        warnings.append("metadata.frame_num is not an integer")

    if action_index is not None and not isinstance(action_index, int):
        warnings.append("metadata.blackmamba_action_index is not an integer")

    if (
        assistant_action is not None
        and isinstance(expert_action, str)
        and assistant_action != expert_action
    ):
        errors.append(
            "assistant action does not match metadata.expert_action: "
            f"{assistant_action!r} != {expert_action!r}"
        )

    if (
        assistant_action is not None
        and isinstance(raw_action, str)
        and assistant_action != raw_action
    ):
        warnings.append(
            "assistant action differs from metadata.blackmamba_raw_action"
        )

    if (
        assistant_action is not None
        and isinstance(learner_action, str)
        and assistant_action == learner_action
    ):
        warnings.append(
            "learner action equals expert action in a disagreement-only dataset"
        )

    if legal_actions is not None:
        actions_to_check = {
            "assistant action": assistant_action,
            "metadata.expert_action": expert_action,
            "metadata.learner_action": learner_action,
        }

        for name, action in actions_to_check.items():
            if isinstance(action, str) and action not in legal_actions:
                errors.append(f"{name} is not in legal action list: {action}")

    return metadata


def canonical_state_label_hash(
    state: dict[str, Any],
    expert_action: str,
) -> str:
    canonical = json.dumps(
        {
            "state": state,
            "expert_action": expert_action,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_counter_csv(
    path: Path,
    counter: Counter[Any],
    column_names: tuple[str, ...],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([*column_names, "count"])

        for key, count in counter.most_common():
            if isinstance(key, tuple):
                writer.writerow([*key, count])
            else:
                writer.writerow([key, count])


def main() -> None:
    args = parse_args()

    if not args.input_file.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    legal_actions = load_legal_actions(args.legal_actions_file)

    valid_path = args.output_dir / "valid_rows.jsonl"
    cleaned_path = args.output_dir / "cleaned.jsonl"
    malformed_path = args.output_dir / "malformed_rows.jsonl"
    warnings_path = args.output_dir / "warnings.jsonl"
    duplicates_path = args.output_dir / "exact_duplicates.jsonl"

    expert_action_counts: Counter[str] = Counter()
    learner_action_counts: Counter[str] = Counter()
    disagreement_pair_counts: Counter[tuple[str, str]] = Counter()
    self_current_action_counts: Counter[str] = Counter()
    opponent_current_action_counts: Counter[str] = Counter()
    self_state_counts: Counter[str] = Counter()
    opponent_state_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()

    unique_hashes: set[str] = set()
    duplicate_hash_counts: Counter[str] = Counter()

    total_lines = 0
    blank_lines = 0
    valid_rows = 0
    malformed_rows = 0
    rows_with_warnings = 0
    duplicate_rows = 0
    disagreement_rows = 0
    accidental_agreements = 0
    metadata_missing_rows = 0

    frame_numbers: list[int] = []
    expert_energy_values: defaultdict[str, list[float]] = defaultdict(list)

    with (
        args.input_file.open("r", encoding="utf-8") as input_file,
        valid_path.open("w", encoding="utf-8") as valid_file,
        cleaned_path.open("w", encoding="utf-8") as cleaned_file,
        malformed_path.open("w", encoding="utf-8") as malformed_file,
        warnings_path.open("w", encoding="utf-8") as warning_file,
        duplicates_path.open("w", encoding="utf-8") as duplicate_file,
    ):
        for line_number, raw_line in enumerate(input_file, start=1):
            total_lines += 1
            stripped = raw_line.strip()

            if not stripped:
                blank_lines += 1
                malformed_rows += 1

                record = {
                    "line_number": line_number,
                    "errors": ["blank line"],
                    "raw_line": raw_line.rstrip("\n"),
                }
                malformed_file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                error_counts["blank line"] += 1
                continue

            errors: list[str] = []
            warnings: list[str] = []

            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exception:
                error = (
                    f"invalid outer JSON: {exception.msg} "
                    f"at position {exception.pos}"
                )
                errors.append(error)

                malformed_file.write(
                    json.dumps(
                        {
                            "line_number": line_number,
                            "errors": errors,
                            "raw_line": stripped,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                malformed_rows += 1
                error_counts[error] += 1
                continue

            if not isinstance(row, dict):
                errors.append("outer row must be a JSON object")
                state = None
                assistant_action = None
                metadata = {}
            else:
                state, assistant_action = extract_messages(
                    row=row,
                    expected_system_prompt=args.system_prompt,
                    errors=errors,
                    warnings=warnings,
                )

                metadata = validate_metadata(
                    row=row,
                    assistant_action=assistant_action,
                    legal_actions=legal_actions,
                    errors=errors,
                    warnings=warnings,
                )

            if errors:
                malformed_rows += 1

                for error in errors:
                    error_counts[error] += 1

                malformed_file.write(
                    json.dumps(
                        {
                            "line_number": line_number,
                            "errors": errors,
                            "warnings": warnings,
                            "row": row,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            assert isinstance(row, dict)
            assert isinstance(state, dict)
            assert isinstance(assistant_action, str)

            valid_rows += 1
            expert_action_counts[assistant_action] += 1

            learner_action = metadata.get("learner_action")
            source = metadata.get("source")
            frame_num = metadata.get("frame_num")

            if not metadata:
                metadata_missing_rows += 1

            if isinstance(learner_action, str):
                learner_action_counts[learner_action] += 1

                if learner_action == assistant_action:
                    accidental_agreements += 1
                else:
                    disagreement_rows += 1
                    disagreement_pair_counts[
                        (learner_action, assistant_action)
                    ] += 1

            if isinstance(source, str):
                source_counts[source] += 1

            if isinstance(frame_num, int):
                frame_numbers.append(frame_num)

            self_data = state["self"]
            opponent_data = state["opponent"]

            self_current_action_counts[self_data["action"]] += 1
            opponent_current_action_counts[opponent_data["action"]] += 1
            self_state_counts[self_data["state"]] += 1
            opponent_state_counts[opponent_data["state"]] += 1

            self_energy = self_data.get("energy")
            if is_number(self_energy):
                expert_energy_values[assistant_action].append(float(self_energy))

            row_hash = canonical_state_label_hash(state, assistant_action)
            is_duplicate = row_hash in unique_hashes

            if is_duplicate:
                duplicate_rows += 1
                duplicate_hash_counts[row_hash] += 1

                duplicate_file.write(
                    json.dumps(
                        {
                            "line_number": line_number,
                            "duplicate_hash": row_hash,
                            "row": row,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            else:
                unique_hashes.add(row_hash)
                duplicate_hash_counts[row_hash] = 1

            valid_file.write(json.dumps(row, ensure_ascii=False) + "\n")

            if args.keep_duplicates or not is_duplicate:
                # Training-ready row: keep only the messages.
                cleaned_row = {"messages": row["messages"]}
                cleaned_file.write(
                    json.dumps(cleaned_row, ensure_ascii=False) + "\n"
                )

            if warnings:
                rows_with_warnings += 1

                for warning in warnings:
                    warning_counts[warning] += 1

                warning_file.write(
                    json.dumps(
                        {
                            "line_number": line_number,
                            "warnings": warnings,
                            "row": row,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    write_counter_csv(
        args.output_dir / "expert_action_counts.csv",
        expert_action_counts,
        ("expert_action",),
    )
    write_counter_csv(
        args.output_dir / "learner_action_counts.csv",
        learner_action_counts,
        ("learner_action",),
    )
    write_counter_csv(
        args.output_dir / "disagreement_pairs.csv",
        disagreement_pair_counts,
        ("learner_action", "expert_action"),
    )
    write_counter_csv(
        args.output_dir / "self_current_action_counts.csv",
        self_current_action_counts,
        ("self_current_action",),
    )
    write_counter_csv(
        args.output_dir / "opponent_current_action_counts.csv",
        opponent_current_action_counts,
        ("opponent_current_action",),
    )
    write_counter_csv(
        args.output_dir / "self_state_counts.csv",
        self_state_counts,
        ("self_state",),
    )
    write_counter_csv(
        args.output_dir / "opponent_state_counts.csv",
        opponent_state_counts,
        ("opponent_state",),
    )
    write_counter_csv(
        args.output_dir / "source_counts.csv",
        source_counts,
        ("source",),
    )
    write_counter_csv(
        args.output_dir / "warning_counts.csv",
        warning_counts,
        ("warning",),
    )
    write_counter_csv(
        args.output_dir / "error_counts.csv",
        error_counts,
        ("error",),
    )

    with (
        args.output_dir / "expert_action_energy_stats.csv"
    ).open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "expert_action",
                "count",
                "min_self_energy",
                "mean_self_energy",
                "max_self_energy",
            ]
        )

        for action, values in sorted(
            expert_energy_values.items(),
            key=lambda item: (-len(item[1]), item[0]),
        ):
            writer.writerow(
                [
                    action,
                    len(values),
                    min(values),
                    sum(values) / len(values),
                    max(values),
                ]
            )

    unique_cleaned_rows = (
        valid_rows if args.keep_duplicates else valid_rows - duplicate_rows
    )

    summary = {
        "input_file": str(args.input_file),
        "total_lines": total_lines,
        "blank_lines": blank_lines,
        "valid_rows": valid_rows,
        "malformed_rows": malformed_rows,
        "rows_with_warnings": rows_with_warnings,
        "exact_duplicate_rows": duplicate_rows,
        "cleaned_output_rows": unique_cleaned_rows,
        "disagreement_rows_with_metadata": disagreement_rows,
        "accidental_agreements": accidental_agreements,
        "metadata_missing_rows": metadata_missing_rows,
        "unique_expert_actions": len(expert_action_counts),
        "unique_learner_actions": len(learner_action_counts),
        "legal_action_validation_enabled": legal_actions is not None,
        "legal_action_count": len(legal_actions) if legal_actions else None,
        "minimum_frame_num": min(frame_numbers) if frame_numbers else None,
        "maximum_frame_num": max(frame_numbers) if frame_numbers else None,
        "top_20_expert_actions": expert_action_counts.most_common(20),
        "top_20_learner_actions": learner_action_counts.most_common(20),
        "top_20_disagreement_pairs": [
            {
                "learner_action": pair[0],
                "expert_action": pair[1],
                "count": count,
            }
            for pair, count in disagreement_pair_counts.most_common(20)
        ],
        "top_errors": error_counts.most_common(20),
        "top_warnings": warning_counts.most_common(20),
    }

    summary_path = args.output_dir / "summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print()
    print("DAgger dataset analysis complete")
    print("=" * 60)
    print(f"Input lines:               {total_lines:,}")
    print(f"Valid rows:                {valid_rows:,}")
    print(f"Malformed rows:            {malformed_rows:,}")
    print(f"Rows with warnings:        {rows_with_warnings:,}")
    print(f"Exact duplicate rows:      {duplicate_rows:,}")
    print(f"Cleaned output rows:       {unique_cleaned_rows:,}")
    print(f"Accidental agreements:     {accidental_agreements:,}")
    print(f"Unique expert actions:     {len(expert_action_counts):,}")
    print(f"Unique learner actions:    {len(learner_action_counts):,}")
    print()

    print("Most common expert actions:")
    for action, count in expert_action_counts.most_common(15):
        percentage = 100 * count / valid_rows if valid_rows else 0
        print(f"  {action:<25} {count:>8,}  ({percentage:6.2f}%)")

    print()
    print("Most common learner -> expert disagreements:")
    for (learner, expert), count in disagreement_pair_counts.most_common(15):
        print(f"  {learner:<24} -> {expert:<24} {count:>8,}")

    print()
    print(f"All outputs written to: {args.output_dir.resolve()}")
    print(f"Summary:                {summary_path}")
    print(f"Training-ready file:    {cleaned_path}")
    print(f"Malformed rows:         {malformed_path}")


if __name__ == "__main__":
    main()