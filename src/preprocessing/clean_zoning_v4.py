import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# =========================
# Config
# =========================
INPUT_PATH = Path("repaired_zoning_v4.jsonl")
OUTPUT_PATH = Path("cleaned_zoningv4data.jsonl")

SEED = 42
MAX_KEEP_PER_KEY = 1

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
# =========================


def normalize_action_text(text: str) -> str:
    text = text.strip()
    if text.startswith("action:"):
        text = text[len("action:"):].strip()
    if text.startswith("Action:"):
        text = text[len("Action:"):].strip()
    return text


def extract_action(row: dict) -> str:
    if "messages" in row:
        msgs = row["messages"]
        if len(msgs) < 3:
            raise ValueError("Row has fewer than 3 messages")
        return normalize_action_text(msgs[2]["content"])

    if "output" in row:
        return normalize_action_text(str(row["output"]))

    raise ValueError("Could not find action field")


def extract_state_json(row: dict) -> dict:
    if "messages" in row:
        msgs = row["messages"]
        if len(msgs) < 2:
            raise ValueError("Row has fewer than 2 messages")

        user_content = msgs[1]["content"]
        if isinstance(user_content, str):
            return json.loads(user_content)
        if isinstance(user_content, dict):
            return user_content
        raise ValueError("messages[1]['content'] is neither str nor dict")

    if "input" in row:
        if isinstance(row["input"], str):
            return json.loads(row["input"])
        if isinstance(row["input"], dict):
            return row["input"]
        raise ValueError("input is neither str nor dict")

    raise ValueError("Could not find input state")


def build_exact_key(row: dict) -> str:
    return json.dumps(row, sort_keys=True, ensure_ascii=False)


def center_x(position: dict) -> float:
    return (position["left"] + position["right"]) / 2.0


def bucket_distance(d: float) -> str:
    if d < 80:
        return "d0"
    if d < 160:
        return "d1"
    if d < 260:
        return "d2"
    return "d3"


def bucket_projectiles(projectiles) -> str:
    if not isinstance(projectiles, list):
        return "p0"
    n = len(projectiles)
    if n == 0:
        return "p0"
    if n <= 2:
        return "p1"
    return "p2"


def build_near_duplicate_key(row: dict):
    state = extract_state_json(row)
    out_action = extract_action(row)

    self_data = state["self"]
    opp_data = state["opponent"]

    self_cx = center_x(self_data["position"])
    opp_cx = center_x(opp_data["position"])
    dist = abs(self_cx - opp_cx)

    return (
        out_action,
        self_data["action"],
        opp_data["action"],
        bucket_distance(dist),
        bucket_projectiles(self_data.get("projectiles", [])),
        bucket_projectiles(opp_data.get("projectiles", [])),
    )


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
    raw_invalid_json = 0

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raw_invalid_json += 1
                print(f"[WARN] Skipping invalid JSON line {line_num}: {e}")

    print(f"Loaded rows: {len(rows)}")
    print(f"Invalid raw JSON lines skipped: {raw_invalid_json}")

    before_counter = Counter()
    valid_rows = []
    invalid_rows = 0

    for idx, row in enumerate(rows, start=1):
        try:
            action = extract_action(row)
            if action not in VALID_ACTIONS:
                invalid_rows += 1
                continue

            _ = extract_state_json(row)
            before_counter[action] += 1
            valid_rows.append(row)

        except Exception as e:
            invalid_rows += 1
            print(f"[WARN] Skipping invalid parsed row {idx}: {e}")

    print(f"Invalid rows skipped after parsing/action check: {invalid_rows}")
    print_distribution("BEFORE CLEANING", before_counter)

    # -------------------------
    # Step 1: Exact duplicate removal
    # -------------------------
    exact_seen = set()
    deduped_rows = []
    exact_removed = 0

    for row in valid_rows:
        key = build_exact_key(row)
        if key in exact_seen:
            exact_removed += 1
            continue
        exact_seen.add(key)
        deduped_rows.append(row)

    print(f"\nExact duplicates removed: {exact_removed}")
    after_exact_counter = Counter(extract_action(r) for r in deduped_rows)
    print_distribution("AFTER EXACT DEDUPE", after_exact_counter)

    # -------------------------
    # Step 2: Hard transition filter
    # -------------------------
    filtered_rows = []
    dropped_backstep_selfloop = 0

    for row in deduped_rows:
        state = extract_state_json(row)
        output_action = extract_action(row)
        self_action = state["self"]["action"]

        if output_action == "BACK_STEP" and self_action == "BACK_STEP":
            dropped_backstep_selfloop += 1
            continue

        filtered_rows.append(row)

    print(f"\nDropped BACK_STEP self-loop rows: {dropped_backstep_selfloop}")
    after_hard_filter_counter = Counter(extract_action(r) for r in filtered_rows)
    print_distribution("AFTER HARD FILTER", after_hard_filter_counter)

    # -------------------------
    # Step 3: Minimal-key aggressive dedupe
    # -------------------------
    grouped = defaultdict(list)

    for row in filtered_rows:
        action = extract_action(row)
        key = build_near_duplicate_key(row)
        grouped[(action, key)].append(row)

    kept_rows = []
    dropped_near_duplicates = 0

    for (_, _), group_rows in grouped.items():
        random.shuffle(group_rows)
        chosen = group_rows[:MAX_KEEP_PER_KEY]
        kept_rows.extend(chosen)
        dropped_near_duplicates += max(0, len(group_rows) - len(chosen))

    random.shuffle(kept_rows)

    after_counter = Counter(extract_action(r) for r in kept_rows)
    print(f"\nDropped near-duplicates: {dropped_near_duplicates}")
    print_distribution("AFTER MINIMAL-KEY DEDUPE", after_counter)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in kept_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved cleaned dataset to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()