import json
from datasets import load_dataset

DATASET_NAME = "Simonkami/MctsAi_Zoning"
OUTPUT_PATH = "mctsai_zoning_converted.jsonl"

TARGET_INSTRUCTION = "Predict the next FightingICE move."

ds = load_dataset(DATASET_NAME, split="train")

print("Columns:", ds.column_names)
print("First row:", ds[0])

bad_rows = 0

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for i, row in enumerate(ds):
        raw_input = row["input"].strip()
        output = row["output"].strip()

        # Keep only the JSON part.
        # The HF dataset has:
        # {"opponent": ..., "self": ...}, Reply with only one move.
        if ", Reply with only one move." in raw_input:
            raw_input = raw_input.split(", Reply with only one move.")[0].strip()

        try:
            game_state = json.loads(raw_input)
        except Exception as e:
            bad_rows += 1
            print(f"Skipping row {i}: could not parse input")
            print("Error:", e)
            print(raw_input[:500])
            continue

        example = {
            "instruction": TARGET_INSTRUCTION,
            "input": json.dumps(game_state, ensure_ascii=False),
            "output": output
        }

        f.write(json.dumps(example, ensure_ascii=False) + "\n")

print(f"Saved {len(ds) - bad_rows} rows to {OUTPUT_PATH}")
print(f"Skipped {bad_rows} bad rows")