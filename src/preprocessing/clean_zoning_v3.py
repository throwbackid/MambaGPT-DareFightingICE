import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# =========================
# Config
# =========================
INPUT_PATH = Path("zoningv4data.jsonl")
OUTPUT_PATH = Path("cleaned_zoningv4data.jsonl")

SEED = 42

LOOP_PRONE_ACTIONS = {
    "BACK_STEP",
    "FORWARD_WALK",
    "CROUCH",
    "STAND_GUARD",
    "CROUCH_GUARD",
    "DASH",
}

MAX_KEEP_PER_KEY_LOOP = 1
MAX_KEEP_PER_KEY_NORMAL = 2

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

STAGE_LEFT = 0
STAGE_RIGHT = 960
CORNER_MARGIN = 40
X_BUCKET_SIZE = 50
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


def build_exact_key(row: dict) -> str:
    return json.dumps(row, sort_keys=True, ensure_ascii=False)


def bucket_distance(d: float) -> str:
    if d < 40:
        return "d0"
    if d < 80:
        return "d1"
    if d < 120:
        return "d2"
    if d < 160:
        return "d3"
    if d < 220:
        return "d4"
    return "d5"


def bucket_energy(e: int) -> str:
    if e == 0:
        return "e0"
    if e < 50:
        return "e1"
    if e < 100:
        return "e2"
    if e < 150:
        return "e3"
    return "e4"


def bucket_x(center_x: float) -> str:
    return f"x{int(center_x // X_BUCKET_SIZE)}"


def bucket_speed(v: float) -> str:
    if v <= -8:
        return "n3"
    if v <= -3:
        return "n2"
    if v < 0:
        return "n1"
    if v == 0:
        return "z"
    if v < 3:
        return "p1"
    if v < 8:
        return "p2"
    return "p3"


def corner_flag(left: int, right: int) -> str:
    near_left = left <= STAGE_LEFT + CORNER_MARGIN
    near_right = right >= STAGE_RIGHT - CORNER_MARGIN

    if near_left and near_right:
        return "both"
    if near_left:
        return "left"
    if near_right:
        return "right"
    return "mid"


def bucket_projectiles(projectiles: list) -> str:
    n = len(projectiles)
    if n == 0:
        return "p0"
    if n == 1:
        return "p1"
    return "p2"


def center_x(position: dict) -> float:
    return (position["left"] + position["right"]) / 2.0


def build_near_duplicate_key(row: dict, strict_loop: bool = False):
    state = extract_state_json(row)
    out_action = extract_action(row)

    self_data = state["self"]
    opp_data = state["opponent"]

    self_pos = self_data["position"]
    opp_pos = opp_data["position"]

    self_cx = center_x(self_pos)
    opp_cx = center_x(opp_pos)
    dist = abs(self_cx - opp_cx)

    key_core = (
        out_action,
        self_data["action"],
        self_data["state"],
        opp_data["action"],
        opp_data["state"],
        bucket_distance(dist),
        bucket_energy(self_data["energy"]),
        bucket_energy(opp_data["energy"]),
        bucket_speed(self_data["speed"]["x"]),
        bucket_speed(self_data["speed"]["y"]),
        bucket_speed(opp_data["speed"]["x"]),
        bucket_speed(opp_data["speed"]["y"]),
        corner_flag(self_pos["left"], self_pos["right"]),
        corner_flag(opp_pos["left"], opp_pos["right"]),
        bucket_projectiles(self_data.get("projectiles", [])),
        bucket_projectiles(opp_data.get("projectiles", [])),
    )

    if strict_loop:
        return key_core

    key_extra = (
        bucket_x(self_cx),
        bucket_x(opp_cx),
    )

    return key_core + key_extra


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
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] Skipping invalid JSON line {line_num}: {e}")

    print(f"Loaded rows: {len(rows)}")

    before_counter = Counter()
    valid_rows = []
    invalid_rows = 0

    for row in rows:
        try:
            action = extract_action(row)
            if action not in VALID_ACTIONS:
                invalid_rows += 1
                continue
            _ = extract_state_json(row)
            before_counter[action] += 1
            valid_rows.append(row)
        except Exception:
            invalid_rows += 1

    print(f"Invalid rows skipped: {invalid_rows}")
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
    # Step 2: Hard transition filters
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
    # Step 3: Near-duplicate filtering
    # -------------------------
    grouped = defaultdict(list)

    for row in filtered_rows:
        action = extract_action(row)
        strict_loop = action in LOOP_PRONE_ACTIONS
        key = build_near_duplicate_key(row, strict_loop=strict_loop)
        grouped[(action, key)].append(row)

    kept_rows = []
    dropped_near_duplicates = 0

    for (action, key), group_rows in grouped.items():
        random.shuffle(group_rows)

        keep_n = MAX_KEEP_PER_KEY_LOOP if action in LOOP_PRONE_ACTIONS else MAX_KEEP_PER_KEY_NORMAL

        chosen = group_rows[:keep_n]
        kept_rows.extend(chosen)
        dropped_near_duplicates += max(0, len(group_rows) - len(chosen))

    random.shuffle(kept_rows)

    after_counter = Counter(extract_action(r) for r in kept_rows)

    print(f"\nDropped near-duplicates: {dropped_near_duplicates}")
    print_distribution("AFTER DEDUPE", after_counter)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in kept_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved cleaned dataset to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()