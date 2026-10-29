import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# =========================
# Config
# =========================
INPUT_PATH = Path("zipf_augmented.jsonl")
OUTPUT_PATH = Path("rushdown_cleaned_v5.jsonl")

SEED = 42

LOOP_PRONE_ACTIONS = {
    "BACK_STEP",
    "FORWARD_WALK",
    "CROUCH",
    "STAND_GUARD",
    "CROUCH_GUARD",
    "DASH",
}

MAX_KEEP_PER_KEY = 3
CONTINUATION_RATIO_CAP = 1.0
CONTINUATION_MIN_FLOOR = 30

STAGE_LEFT = 0
STAGE_RIGHT = 960
CORNER_MARGIN = 40
X_BUCKET_SIZE = 50

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


def center_x(position: dict) -> float:
    return (position["left"] + position["right"]) / 2.0


def bucket_x(center: float) -> str:
    return f"x{int(center // X_BUCKET_SIZE)}"


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


def bucket_projectiles(projectiles: list) -> str:
    n = len(projectiles)
    if n == 0:
        return "p0"
    if n == 1:
        return "p1"
    return "p2"


def is_backstep_into_wall(state: dict) -> bool:
    """
    True only when BACK_STEP continuation is likely harmful:
    already backstepping and already near own back wall.
    """
    self_data = state["self"]
    opp_data = state["opponent"]

    self_pos = self_data["position"]
    opp_pos = opp_data["position"]

    self_cx = center_x(self_pos)
    opp_cx = center_x(opp_pos)

    # Infer side: if self is left of opponent, back is left wall.
    if self_cx < opp_cx:
        return self_pos["left"] <= STAGE_LEFT + CORNER_MARGIN
    else:
        return self_pos["right"] >= STAGE_RIGHT - CORNER_MARGIN


def build_near_duplicate_key(row: dict):
    state = extract_state_json(row)
    out_action = extract_action(row)

    self_data = state["self"]
    opp_data = state["opponent"]

    self_pos = self_data["position"]
    opp_pos = opp_data["position"]

    self_cx = center_x(self_pos)
    opp_cx = center_x(opp_pos)
    dist = abs(self_cx - opp_cx)

    key = (
        out_action,

        self_data["action"],
        self_data["state"],
        opp_data["action"],
        opp_data["state"],

        bucket_x(self_cx),
        bucket_x(opp_cx),
        bucket_distance(dist),

        bucket_speed(self_data["speed"]["x"]),
        bucket_speed(self_data["speed"]["y"]),
        bucket_speed(opp_data["speed"]["x"]),
        bucket_speed(opp_data["speed"]["y"]),

        bucket_projectiles(self_data.get("projectiles", [])),
        bucket_projectiles(opp_data.get("projectiles", [])),

        bucket_energy(self_data["energy"]),
        bucket_energy(opp_data["energy"]),
    )
    return key


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

    before_counter = Counter(extract_action(r) for r in rows)
    print_distribution("BEFORE CLEANING", before_counter)

    # -------------------------
    # Step 1: exact dedupe
    # -------------------------
    exact_seen = set()
    exact_deduped = []
    exact_dropped = 0

    for row in rows:
        k = build_exact_key(row)
        if k in exact_seen:
            exact_dropped += 1
            continue
        exact_seen.add(k)
        exact_deduped.append(row)

    print(f"\nExact duplicates dropped: {exact_dropped}")

    after_exact_counter = Counter(extract_action(r) for r in exact_deduped)
    print_distribution("AFTER EXACT DEDUPE", after_exact_counter)

    # -------------------------
    # Step 2: wall-aware BACK_STEP filter
    # -------------------------
    filtered_rows = []
    dropped_bad_backstep = 0

    for row in exact_deduped:
        state = extract_state_json(row)
        output_action = extract_action(row)
        self_action = state["self"]["action"]

        if output_action == "BACK_STEP" and self_action == "BACK_STEP":
            if is_backstep_into_wall(state):
                dropped_bad_backstep += 1
                continue

        filtered_rows.append(row)

    print(f"\nDropped wall-aware BACK_STEP continuation rows: {dropped_bad_backstep}")

    after_backstep_counter = Counter(extract_action(r) for r in filtered_rows)
    print_distribution("AFTER BACK_STEP FILTER", after_backstep_counter)

    # -------------------------
    # Step 3: continuation vs transition control
    # -------------------------
    by_action = defaultdict(list)
    for row in filtered_rows:
        by_action[extract_action(row)].append(row)

    continuation_processed = []
    continuation_dropped = 0

    for action, action_rows in by_action.items():
        if action not in LOOP_PRONE_ACTIONS:
            continuation_processed.extend(action_rows)
            continue

        continuation_rows = []
        transition_rows = []

        for row in action_rows:
            state = extract_state_json(row)
            self_action = state["self"]["action"]
            if self_action == action:
                continuation_rows.append(row)
            else:
                transition_rows.append(row)

        keep_cont = max(CONTINUATION_MIN_FLOOR, round(len(transition_rows) * CONTINUATION_RATIO_CAP))
        keep_cont = min(keep_cont, len(continuation_rows))

        random.shuffle(continuation_rows)
        kept_cont_rows = continuation_rows[:keep_cont]

        continuation_dropped += max(0, len(continuation_rows) - len(kept_cont_rows))
        continuation_processed.extend(transition_rows)
        continuation_processed.extend(kept_cont_rows)

    print(f"\nDropped continuation rows by ratio cap: {continuation_dropped}")

    after_cont_counter = Counter(extract_action(r) for r in continuation_processed)
    print_distribution("AFTER CONTINUATION CAP", after_cont_counter)

    # -------------------------
    # Step 4: light near-duplicate cap
    # -------------------------
    grouped = defaultdict(list)
    for row in continuation_processed:
        action = extract_action(row)
        key = build_near_duplicate_key(row)
        grouped[(action, key)].append(row)

    final_rows = []
    near_dropped = 0

    for (_, _), group_rows in grouped.items():
        random.shuffle(group_rows)
        chosen = group_rows[:MAX_KEEP_PER_KEY]
        final_rows.extend(chosen)
        near_dropped += max(0, len(group_rows) - len(chosen))

    random.shuffle(final_rows)

    after_counter = Counter(extract_action(r) for r in final_rows)
    print(f"\nDropped near-duplicates by light cap: {near_dropped}")
    print_distribution("AFTER V5 CLEANING", after_counter)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in final_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved cleaned dataset to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()