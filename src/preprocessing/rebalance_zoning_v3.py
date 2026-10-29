import json
import random
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

# =========================
# Config
# =========================
INPUT_PATH = Path("cleaned_zoningdata.jsonl")
OUTPUT_PATH = Path("rebalanced_cleaned_zoningdata.jsonl")

SEED = 42

# Set top ranked action target count here
NEW_TOP_COUNT = 30000

HP_MIN = 1
HP_MAX = 400

ENERGY_MIN = 0
ENERGY_MAX = 300

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

# Head-compressed shape percentages
HEAD_SHAPE_PCTS = [
    6.82, 5.95, 4.97, 4.38, 3.75, 3.46, 3.42, 2.96, 2.78, 2.75,
    2.72, 2.68, 2.41, 2.39, 2.34, 2.31, 2.21, 1.96, 1.90, 1.89,
    1.87, 1.87, 1.86, 1.85, 1.84, 1.83, 1.83, 1.82, 1.82, 1.80,
    1.79, 1.78, 1.77, 1.76, 1.76, 1.75, 1.74, 1.74, 1.74, 1.73,
]

SELF_ENERGY_MIN_RULES = {
    "STAND_D_DF_FC": 100,
}
# =========================


def extract_action(row: dict) -> str:
    if "messages" in row:
        msgs = row["messages"]
        if len(msgs) < 3:
            raise ValueError("Row has fewer than 3 messages")
        return msgs[2]["content"].strip()

    if "output" in row:
        return str(row["output"]).strip()

    raise ValueError("Could not find action field")


def extract_state_json(row: dict) -> dict:
    if "messages" in row:
        msgs = row["messages"]
        if len(msgs) < 2:
            raise ValueError("Row has fewer than 2 messages")
        return json.loads(msgs[1]["content"])

    if "input" in row:
        if isinstance(row["input"], str):
            return json.loads(row["input"])
        return row["input"]

    raise ValueError("Could not find input state")


def write_state_json_back(row: dict, state: dict) -> dict:
    row = deepcopy(row)

    if "messages" in row:
        row["messages"][1]["content"] = json.dumps(state, ensure_ascii=False)
        return row

    if "input" in row:
        row["input"] = json.dumps(state, ensure_ascii=False)
        return row

    raise ValueError("Could not write state back to row")


def randomize_hp_and_energy_only(state: dict, action: str) -> dict:
    state = deepcopy(state)

    state["self"]["hp"] = random.randint(HP_MIN, HP_MAX)
    state["opponent"]["hp"] = random.randint(HP_MIN, HP_MAX)

    min_self_energy = SELF_ENERGY_MIN_RULES.get(action, ENERGY_MIN)
    state["self"]["energy"] = random.randint(min_self_energy, ENERGY_MAX)
    state["opponent"]["energy"] = random.randint(ENERGY_MIN, ENERGY_MAX)

    return state


def augment_row_no_x_jitter(row: dict) -> dict:
    action = extract_action(row)
    state = extract_state_json(row)
    state = randomize_hp_and_energy_only(state, action)
    return write_state_json_back(row, state)


def print_distribution(title: str, counter: Counter):
    total = sum(counter.values())
    print(f"\n=== {title} ===")
    print(f"Total rows: {total}")
    print(f"Unique actions: {len(counter)}")
    for i, (action, count) in enumerate(counter.most_common(), start=1):
        pct = (count / total * 100) if total else 0.0
        print(f"{i:2d}) {action:20s} {count:7d}  {pct:6.2f}%")


def main():
    random.seed(SEED)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH.resolve()}")

    rows = []
    invalid_rows = 0

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                action = extract_action(row)
                _ = extract_state_json(row)

                if action not in VALID_ACTIONS:
                    invalid_rows += 1
                    continue

                rows.append(row)
            except Exception as e:
                invalid_rows += 1
                print(f"[WARN] Skipping line {line_num}: {e}")

    print(f"Loaded valid rows: {len(rows)}")
    print(f"Invalid rows skipped: {invalid_rows}")

    grouped = defaultdict(list)
    for row in rows:
        grouped[extract_action(row)].append(row)

    current_counter = Counter({a: len(v) for a, v in grouped.items()})
    print_distribution("CURRENT CLEANED DATASET", current_counter)

    ranked_actions = [a for a, _ in current_counter.most_common()]
    shape_pcts = HEAD_SHAPE_PCTS[:len(ranked_actions)]

    if not ranked_actions:
        raise ValueError("No valid actions found in dataset")

    top_pct = shape_pcts[0] / 100.0
    total_target = round(NEW_TOP_COUNT / top_pct)

    print(f"\nTop ranked action: {ranked_actions[0]}")
    print(f"Chosen top count: {NEW_TOP_COUNT}")
    print(f"Top percentage: {shape_pcts[0]:.2f}%")
    print(f"Implied total target size: {total_target}")

    target_counts = {}
    for rank_idx, action in enumerate(ranked_actions):
        pct = shape_pcts[rank_idx] / 100.0
        target_counts[action] = round(total_target * pct)

    print("\n=== TARGET COUNTS BY CURRENT RANK ===")
    for i, action in enumerate(ranked_actions, start=1):
        old_count = current_counter[action]
        target = target_counts[action]
        delta = target - old_count
        print(
            f"{i:2d}) {action:20s} current={old_count:6d}  target={target:6d}  delta={delta:+6d}"
        )

    final_rows = []
    added_counter = Counter()
    dropped_counter = Counter()

    for action in ranked_actions:
        source_rows = grouped[action]
        current_count = len(source_rows)
        target = target_counts[action]

        if current_count > target:
            chosen = random.sample(source_rows, target)
            final_rows.extend(chosen)
            dropped_counter[action] = current_count - target

        elif current_count < target:
            final_rows.extend(source_rows)
            need = target - current_count
            for _ in range(need):
                base_row = random.choice(source_rows)
                new_row = augment_row_no_x_jitter(base_row)
                final_rows.append(new_row)
            added_counter[action] = need

        else:
            final_rows.extend(source_rows)

    random.shuffle(final_rows)

    final_counter = Counter(extract_action(r) for r in final_rows)
    print_distribution("FINAL REBALANCED DATASET", final_counter)

    print("\n=== DROPPED ROWS ===")
    if dropped_counter:
        for action, n in dropped_counter.most_common():
            print(f"{action:20s} -{n}")
    else:
        print("None")

    print("\n=== ADDED ROWS ===")
    if added_counter:
        for action, n in added_counter.most_common():
            print(f"{action:20s} +{n}")
    else:
        print("None")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in final_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved rebalanced dataset to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()