import json
import random
from collections import defaultdict, Counter
from pathlib import Path

INPUT_PATH = Path("rebalanced_rushdown_v6.jsonl")
TRAIN_PATH = Path("rushdownv6data/train.jsonl")
VALID_PATH = Path("rushdownv6data/valid.jsonl")

VALID_RATIO = 0.10
SEED = 42

ALL_ACTIONS = [
    "FORWARD_WALK", "DASH", "BACK_STEP", "CROUCH", "JUMP", "FOR_JUMP",
    "BACK_JUMP", "STAND_GUARD", "CROUCH_GUARD", "AIR_GUARD", "STAND_A", "STAND_B",
    "THROW_A", "THROW_B", "CROUCH_A", "CROUCH_B", "STAND_FA", "STAND_FB",
    "CROUCH_FA", "CROUCH_FB", "STAND_F_D_DFA", "STAND_F_D_DFB",
    "STAND_D_DB_BA", "STAND_D_DB_BB", "STAND_D_DF_FA", "STAND_D_DF_FB",
    "STAND_D_DF_FC", "AIR_A", "AIR_B", "AIR_DA", "AIR_DB", "AIR_FA",
    "AIR_FB", "AIR_UA", "AIR_UB", "AIR_F_D_DFA", "AIR_F_D_DFB",
    "AIR_D_DB_BA", "AIR_D_DB_BB", "AIR_D_DF_FA", "AIR_D_DF_FB",
]
VALID_ACTIONS = set(ALL_ACTIONS)


def extract_action(row: dict) -> str:
    if "messages" in row:
        msgs = row["messages"]
        if len(msgs) < 3:
            raise ValueError(f"Row has fewer than 3 messages: {row}")
        return msgs[2]["content"].strip()

    if "output" in row:
        return str(row["output"]).strip()

    raise ValueError(f"Could not find action field in row: {row}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_distribution(title: str, rows: list[dict]) -> None:
    counter = Counter()
    for row in rows:
        action = extract_action(row)
        counter[action] += 1

    total = len(rows)
    print(f"\n=== {title} ===")
    print(f"Total rows: {total}")
    print(f"Unique actions: {len(counter)}")

    for i, (action, count) in enumerate(counter.most_common(), start=1):
        pct = (count / total * 100) if total else 0.0
        print(f"{i:2d}) {action:20s} {count:6d}  {pct:6.2f}%")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH.resolve()}")

    random.seed(SEED)

    grouped = defaultdict(list)
    invalid_rows = 0
    total_rows = 0

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
                action = extract_action(row)
                total_rows += 1

                if action not in VALID_ACTIONS:
                    invalid_rows += 1
                    continue

                grouped[action].append(row)

            except Exception as e:
                print(f"[WARN] Skipping line {line_num}: {e}")

    print(f"Loaded rows: {total_rows}")
    print(f"Invalid rows skipped: {invalid_rows}")

    train_rows = []
    valid_rows = []

    for action in sorted(grouped.keys()):
        rows = grouped[action]
        random.shuffle(rows)

        n_total = len(rows)
        n_valid = max(1, round(n_total * VALID_RATIO)) if n_total >= 2 else 0
        n_valid = min(n_valid, n_total - 1) if n_total >= 2 else 0

        valid_part = rows[:n_valid]
        train_part = rows[n_valid:]

        valid_rows.extend(valid_part)
        train_rows.extend(train_part)

        print(
            f"{action:20s} total={n_total:6d}  "
            f"train={len(train_part):6d}  valid={len(valid_part):6d}"
        )

    random.shuffle(train_rows)
    random.shuffle(valid_rows)

    write_jsonl(TRAIN_PATH, train_rows)
    write_jsonl(VALID_PATH, valid_rows)

    print(f"\nSaved train to: {TRAIN_PATH}")
    print(f"Saved valid to: {VALID_PATH}")

    print_distribution("TRAIN DISTRIBUTION", train_rows)
    print_distribution("VALID DISTRIBUTION", valid_rows)


if __name__ == "__main__":
    main()