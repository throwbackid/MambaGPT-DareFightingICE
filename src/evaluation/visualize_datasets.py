import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


DATASETS = {
    "zoningv4data": Path("zoningv4data.jsonl"),
}

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


def count_actions(path: Path) -> Counter:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    counter = Counter()
    skipped = 0

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                action = extract_action(row)
                if action in VALID_ACTIONS:
                    counter[action] += 1
                else:
                    skipped += 1
            except Exception:
                skipped += 1

    if skipped:
        print(f"[WARN] {path.name}: skipped {skipped} invalid rows")

    return counter


def plot_single_dataset(name: str, counter: Counter, out_path: Path):
    actions = [action for action, _ in counter.most_common()]
    counts = [count for _, count in counter.most_common()]
    total = sum(counts)

    plt.figure(figsize=(16, 7))
    plt.bar(range(len(actions)), counts)

    plt.xticks(range(len(actions)), actions, rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title(f"{name} Action Distribution (total={total})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def print_summary(name: str, counter: Counter):
    total = sum(counter.values())
    print(f"\n=== {name} ===")
    print(f"Total rows: {total}")
    print(f"Unique actions: {len(counter)}")
    for i, (action, count) in enumerate(counter.most_common(), start=1):
        pct = count / total * 100 if total else 0.0
        print(f"{i:2d}) {action:20s} {count:7d}  {pct:6.2f}%")


def main():
    for name, path in DATASETS.items():
        print(f"Counting {path} ...")
        counter = count_actions(path)
        print_summary(name, counter)

        output_name = f"{name}_barchart.png"
        plot_single_dataset(name, counter, Path(output_name))
        print(f"Saved: {output_name}")


if __name__ == "__main__":
    main()