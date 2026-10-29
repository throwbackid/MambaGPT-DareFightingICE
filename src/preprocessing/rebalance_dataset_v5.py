import json
import random
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

# =========================
# Config
# =========================
INPUT_PATH = Path("rushdown_cleaned_v5.jsonl")
OUTPUT_PATH = Path("rebalanced_rushdown_v5.jsonl")

SEED = 42

# Anchor: current rank-1 action should become 6.82% with this count
NEW_TOP_COUNT = 14389

HP_MIN = 1
HP_MAX = 400

ENERGY_MIN = 0
ENERGY_MAX = 300

SAME_AS_OUTPUT_MAX_RATIO = 0.50

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


def is_same_as_output(row: dict) -> bool:
    action = extract_action(row)
    state = extract_state_json(row)
    return state["self"]["action"] == action


def print_distribution(title: str, counter: Counter):
    total = sum(counter.values())
    print(f"\n=== {title} ===")
    print(f"Total rows: {total}")
    print(f"Unique actions: {len(counter)}")
    for i, (action, count) in enumerate(counter.most_common(), start=1):
        pct = (count / total * 100) if total else 0.0
        print(f"{i:2d}) {action:20s} {count:7d}  {pct:6.2f}%")


def print_same_output_stats(title: str, grouped_rows: dict):
    print(f"\n=== {title} ===")
    for action, rows in grouped_rows.items():
        total = len(rows)
        same = sum(1 for r in rows if is_same_as_output(r))
        ratio = same / total if total else 0.0
        print(f"{action:20s} total={total:6d} same_as_output={same:6d} ratio={ratio:6.2%}")


def downsample_action_rows(rows: list[dict], target: int):
    """
    If current ratio > 50%, drop same rows first until ratio <= 50% or target reached.
    After that, downsample remaining rows proportionally/randomly without forcing 50%.
    """
    same_rows = [r for r in rows if is_same_as_output(r)]
    diff_rows = [r for r in rows if not is_same_as_output(r)]

    random.shuffle(same_rows)
    random.shuffle(diff_rows)

    total = len(rows)
    same = len(same_rows)
    to_drop = total - target
    dropped_same_first = 0

    if to_drop <= 0:
        return rows[:], dropped_same_first

    # Step 1: only if above ceiling, drop same first until <= 50% or no more drops needed
    while to_drop > 0 and total > 0 and (same / total) > SAME_AS_OUTPUT_MAX_RATIO and same_rows:
        same_rows.pop()
        same -= 1
        total -= 1
        to_drop -= 1
        dropped_same_first += 1

    remaining = same_rows + diff_rows
    if to_drop > 0:
        kept = random.sample(remaining, len(remaining) - to_drop)
    else:
        kept = remaining

    return kept, dropped_same_first


def upsample_action_rows(rows: list[dict], target: int):
    """
    Rule:
    - If current ratio <= 50%, keep it as-is and augment proportionally to current ratio.
    - If current ratio > 50%, first add only diff rows until ratio hits 50%.
    - After that, augment proportionally to the corrected ratio, not toward 50%.
    """
    same_rows = [r for r in rows if is_same_as_output(r)]
    diff_rows = [r for r in rows if not is_same_as_output(r)]

    final_rows = list(rows)

    same = len(same_rows)
    diff = len(diff_rows)
    total = same + diff
    need = target - total

    added_diff_first = 0
    added_same_after = 0
    added_diff_after = 0
    shortfall = 0

    if need <= 0:
        return final_rows, added_diff_first, added_same_after, added_diff_after, shortfall

    if not same_rows and not diff_rows:
        return final_rows, added_diff_first, added_same_after, added_diff_after, need

    # Step 1: if above ceiling, add diff only until ratio <= 50%
    if total > 0 and (same / total) > SAME_AS_OUTPUT_MAX_RATIO and diff_rows:
        needed_diff_first = max(0, 2 * same - total)
        diff_first_to_add = min(need, needed_diff_first)

        for _ in range(diff_first_to_add):
            base_row = random.choice(diff_rows)
            final_rows.append(augment_row_no_x_jitter(base_row))
            diff += 1
            total += 1
            need -= 1
            added_diff_first += 1

    # Step 2: continue augmentation while preserving current ratio
    # If still no diff rows exist, we can only add same rows.
    while need > 0:
        current_total = same + diff
        if current_total == 0:
            shortfall += need
            break

        same_ratio = same / current_total

        # preserve current ratio, but never exceed ceiling
        if same_ratio > SAME_AS_OUTPUT_MAX_RATIO:
            # should only happen if no diff rows available
            if diff_rows:
                base_row = random.choice(diff_rows)
                final_rows.append(augment_row_no_x_jitter(base_row))
                diff += 1
                added_diff_after += 1
            elif same_rows:
                base_row = random.choice(same_rows)
                final_rows.append(augment_row_no_x_jitter(base_row))
                same += 1
                added_same_after += 1
            else:
                shortfall += need
                break
            need -= 1
            continue

        # proportional augmentation
        # probability of choosing same should follow current corrected ratio
        choose_same = random.random() < same_ratio

        if choose_same and same_rows:
            # hard check to not exceed ceiling after adding same
            if (same + 1) / (current_total + 1) <= SAME_AS_OUTPUT_MAX_RATIO + 1e-12:
                base_row = random.choice(same_rows)
                final_rows.append(augment_row_no_x_jitter(base_row))
                same += 1
                added_same_after += 1
            elif diff_rows:
                base_row = random.choice(diff_rows)
                final_rows.append(augment_row_no_x_jitter(base_row))
                diff += 1
                added_diff_after += 1
            else:
                base_row = random.choice(same_rows)
                final_rows.append(augment_row_no_x_jitter(base_row))
                same += 1
                added_same_after += 1
        else:
            if diff_rows:
                base_row = random.choice(diff_rows)
                final_rows.append(augment_row_no_x_jitter(base_row))
                diff += 1
                added_diff_after += 1
            elif same_rows:
                if (same + 1) / (current_total + 1) <= SAME_AS_OUTPUT_MAX_RATIO + 1e-12:
                    base_row = random.choice(same_rows)
                    final_rows.append(augment_row_no_x_jitter(base_row))
                    same += 1
                    added_same_after += 1
                else:
                    shortfall += need
                    break
            else:
                shortfall += need
                break

        need -= 1

    return final_rows, added_diff_first, added_same_after, added_diff_after, shortfall


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
    print_same_output_stats("CURRENT SAME-AS-OUTPUT RATIOS", grouped)

    ranked_actions = [a for a, _ in current_counter.most_common()]
    if len(ranked_actions) != len(HEAD_SHAPE_PCTS):
        raise ValueError("Number of ranked actions does not match percentage template length")

    top_pct = HEAD_SHAPE_PCTS[0] / 100.0
    total_target = round(NEW_TOP_COUNT / top_pct)

    print(f"\nTop ranked action: {ranked_actions[0]}")
    print(f"Chosen top count: {NEW_TOP_COUNT}")
    print(f"Top percentage: {HEAD_SHAPE_PCTS[0]:.2f}%")
    print(f"Implied total target size: {total_target}")

    target_counts = {}
    for rank_idx, action in enumerate(ranked_actions):
        pct = HEAD_SHAPE_PCTS[rank_idx] / 100.0
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
    final_grouped = defaultdict(list)

    added_counter = Counter()
    added_diff_first_counter = Counter()
    added_same_after_counter = Counter()
    added_diff_after_counter = Counter()
    shortfall_counter = Counter()

    dropped_counter = Counter()
    dropped_same_first_counter = Counter()

    for action in ranked_actions:
        source_rows = grouped[action]
        current_count = len(source_rows)
        target = target_counts[action]

        if current_count > target:
            chosen, dropped_same_first = downsample_action_rows(source_rows, target)
            final_rows.extend(chosen)
            final_grouped[action].extend(chosen)
            dropped_counter[action] = current_count - len(chosen)
            dropped_same_first_counter[action] = dropped_same_first

        elif current_count < target:
            grown_rows, added_diff_first, added_same_after, added_diff_after, shortfall = upsample_action_rows(
                source_rows, target
            )
            final_rows.extend(grown_rows)
            final_grouped[action].extend(grown_rows)
            added_counter[action] = len(grown_rows) - current_count
            added_diff_first_counter[action] = added_diff_first
            added_same_after_counter[action] = added_same_after
            added_diff_after_counter[action] = added_diff_after
            shortfall_counter[action] = shortfall

        else:
            final_rows.extend(source_rows)
            final_grouped[action].extend(source_rows)

    random.shuffle(final_rows)

    final_counter = Counter(extract_action(r) for r in final_rows)
    print_distribution("FINAL REBALANCED DATASET", final_counter)
    print_same_output_stats("FINAL SAME-AS-OUTPUT RATIOS", final_grouped)

    print("\n=== DROPPED ROWS ===")
    if dropped_counter:
        for action, n in dropped_counter.most_common():
            same_first = dropped_same_first_counter[action]
            print(f"{action:20s} -{n}  (same-first dropped: {same_first})")
    else:
        print("None")

    print("\n=== ADDED ROWS ===")
    if added_counter:
        for action, n in added_counter.most_common():
            print(
                f"{action:20s} +{n}  "
                f"(diff-first: {added_diff_first_counter[action]}, "
                f"same-after: {added_same_after_counter[action]}, "
                f"diff-after: {added_diff_after_counter[action]}, "
                f"shortfall: {shortfall_counter[action]})"
            )
    else:
        print("None")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in final_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved rebalanced dataset to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()