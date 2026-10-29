import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

INPUT_PATH = Path("zipf_rebalanced_v2.jsonl")
OUTPUT_PATH = Path("head_compressed.jsonl")

SEED = 42

# Actions at or below this count stay unchanged
KEEP_BELOW = 2500

# The largest class will be compressed to this count
TOP_CAP = 10000

# Compression strength for the head.
# Larger = harder compression of the biggest classes.
LOG_K = 1.0

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


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] Skipping line {line_num}: {e}")
    return rows


def compute_target_counts(counter: Counter, keep_below: int, top_cap: int, log_k: float):
    top_count = max(counter.values())
    targets = {}

    for action, old_count in counter.items():
        if old_count <= keep_below:
            targets[action] = old_count
            continue

        # Scale within the head region only
        scaled = (old_count - keep_below) / (top_count - keep_below)

        # Log compression: strongest effect on biggest classes
        compressed = math.log(1 + log_k * scaled) / math.log(1 + log_k)

        new_count = round(keep_below + (top_cap - keep_below) * compressed)
        new_count = min(new_count, old_count)
        targets[action] = new_count

    return targets


def print_distribution(title: str, counter: Counter, total: int):
    print(f"\n=== {title} ===")
    print(f"Total rows: {total}")
    print(f"Unique actions: {len(counter)}")
    for i, (action, count) in enumerate(counter.most_common(), start=1):
        pct = (count / total * 100) if total else 0.0
        print(f"{i:2d}) {action:20s} {count:6d}  {pct:6.2f}%")


def main():
    random.seed(SEED)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH.resolve()}")

    print(f"Loading {INPUT_PATH} ...")
    rows = load_rows(INPUT_PATH)
    print(f"Loaded {len(rows)} rows")

    grouped = defaultdict(list)
    invalid_rows = 0

    for row in rows:
        try:
            action = extract_action(row)
            if action not in VALID_ACTIONS:
                invalid_rows += 1
                continue
            grouped[action].append(row)
        except Exception:
            invalid_rows += 1

    if invalid_rows:
        print(f"[WARN] Skipped invalid/unreadable rows: {invalid_rows}")

    old_counter = Counter({a: len(v) for a, v in grouped.items()})
    old_total = sum(old_counter.values())

    print_distribution("OLD DISTRIBUTION", old_counter, old_total)

    target_counts = compute_target_counts(
        old_counter,
        keep_below=KEEP_BELOW,
        top_cap=TOP_CAP,
        log_k=LOG_K,
    )

    print("\n=== TARGET COUNTS ===")
    for i, action in enumerate(
        sorted(target_counts.keys(), key=lambda a: old_counter[a], reverse=True), start=1
    ):
        old_count = old_counter[action]
        new_count = target_counts[action]
        changed = "YES" if old_count != new_count else "NO"
        print(f"{i:2d}) {action:20s} old={old_count:6d}  new={new_count:6d}  changed={changed}")

    sampled_rows = []
    new_counter = Counter()

    for action, action_rows in grouped.items():
        k = target_counts[action]
        if len(action_rows) <= k:
            chosen = action_rows
        else:
            chosen = random.sample(action_rows, k)
        sampled_rows.extend(chosen)
        new_counter[action] = len(chosen)

    random.shuffle(sampled_rows)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in sampled_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    new_total = sum(new_counter.values())

    print(f"\nSaved to: {OUTPUT_PATH}")
    print_distribution("NEW DISTRIBUTION", new_counter, new_total)

    print("\n=== SUMMARY ===")
    print(f"Old total: {old_total}")
    print(f"New total: {new_total}")
    print(f"Reduction: {old_total - new_total}")
    print(f"Kept: {(new_total / old_total * 100):.2f}%")

    missing = [a for a in ALL_ACTIONS if new_counter[a] == 0]
    print(f"Missing actions after shrink: {len(missing)}")
    if missing:
        for i, action in enumerate(missing, start=1):
            print(f"{i:2d}) {action}")


if __name__ == "__main__":
    main()