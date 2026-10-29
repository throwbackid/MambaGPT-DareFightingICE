import json
from collections import Counter
from pathlib import Path

DATA_PATH = Path("sanitize.jsonl")

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


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"File not found: {DATA_PATH.resolve()}")

    counter = Counter()
    valid_counter = Counter()
    invalid_counter = Counter()

    total = 0
    valid_total = 0
    invalid_total = 0

    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
                action = extract_action(row)

                counter[action] += 1
                total += 1

                if action in VALID_ACTIONS:
                    valid_counter[action] += 1
                    valid_total += 1
                else:
                    invalid_counter[action] += 1
                    invalid_total += 1

            except Exception as e:
                print(f"[WARN] Skipping line {line_num}: {e}")

    seen_valid_actions = set(valid_counter.keys())
    missing_valid_actions = [action for action in ALL_ACTIONS if action not in seen_valid_actions]

    print(f"File: {DATA_PATH}")
    print(f"Total rows counted: {total}")
    print(f"Unique actions (all): {len(counter)}")
    print(f"Unique valid actions: {len(valid_counter)}")
    print(f"Unique invalid actions: {len(invalid_counter)}")
    print()

    print("=== SUMMARY ===")
    print(f"Valid actions total:   {valid_total}  ({(valid_total / total * 100) if total else 0.0:.2f}%)")
    print(f"Invalid actions total: {invalid_total}  ({(invalid_total / total * 100) if total else 0.0:.2f}%)")
    print(f"Missing valid actions: {len(missing_valid_actions)} / {len(ALL_ACTIONS)}")
    print()

    print("=== VALID ACTION DISTRIBUTION ===")
    if valid_counter:
        for i, (action, count) in enumerate(valid_counter.most_common(), start=1):
            pct_total = (count / total * 100) if total else 0.0
            pct_valid = (count / valid_total * 100) if valid_total else 0.0
            print(f"{i:2d}) {action:20s} {count:6d}  total={pct_total:6.2f}%  valid={pct_valid:6.2f}%")
    else:
        print("None")

    print()

    print(f"=== INVALID ACTIONS ({len(invalid_counter)}) ===")
    if invalid_counter:
        for i, (action, count) in enumerate(invalid_counter.most_common(), start=1):
            pct_total = (count / total * 100) if total else 0.0
            print(f"{i:2d}) {action:20s} {count:6d}  {pct_total:6.2f}%")
    else:
        print("None")

    print()

    print(f"=== MISSING VALID ACTIONS ({len(missing_valid_actions)}) ===")
    if missing_valid_actions:
        for i, action in enumerate(missing_valid_actions, start=1):
            print(f"{i:2d}) {action}")
    else:
        print("None")


if __name__ == "__main__":
    main()