# Running the workflow

The repository includes dataset samples, preprocessing utilities, a Qwen training pipeline, evaluation tools, model export, and the inference patch. The samples demonstrate the mechanics; the development scores used full datasets.

## 1. Environment

Use Python 3.10 or later; Python 3.12 is recommended. Preprocessing and CSV analysis use the standard library. Model training requires a compatible CUDA GPU environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Dependencies are listed by function rather than pinned as a historical lockfile. GPU/CUDA package compatibility should be checked for your machine. Runtime dependencies are separate in `requirements-runtime.txt`.

## 2. Inspect the datasets

See [datasets/README.md](../datasets/README.md) for raw/intermediate files and split sizes. Prepared Rushdown v3 examples are in `datasets/samples/project/rushdownv3data`; Zoning v3 examples are in `datasets/samples/project/zoningv3data`.

To inspect class counts, set `DATA_PATH` in `src/evaluation/count_action_labels.py` to the desired JSONL file, or use the visualization utility after selecting its input/output constants.

Preprocessing scripts resolve their relative input/output constants from the current working directory. Work on a copy:

```bash
mkdir -p work
cp -R datasets/samples/project work/data
```

The [dataset workflow guide](../datasets/README.md) maps scripts to their stage files. Run the required scripts from `work/data` using their repository paths. Full-data class rebalancers expect the complete action vocabulary; a small random sample may not contain every class. Prepared sample splits can be used directly for training mechanics without rerunning every historical rebalancing stage.

## 3. Prepare the action-token base

Run from the repository root:

```bash
python src/training/build_qwen_plain_action_base.py \
  --base-model Qwen/Qwen2.5-1.5B \
  --output-dir models/qwen-action-base
```

This downloads/loads the base model, registers the action tokens, resizes the embeddings, and saves the local action-token base. It is a model operation and needs the model dependencies.

## 4. Fine-tune

Rushdown example:

```bash
python src/training/train_qwen_repo_causal_lm.py \
  --base-model-dir models/qwen-action-base \
  --data-dir datasets/samples/project/rushdownv3data \
  --output-dir outputs/rushdown-v3 \
  --batch-size 16 --epochs 3
```

For Zoning, use `--data-dir datasets/samples/project/zoningv3data` and a separate output directory. For another Rushdown version, select its corresponding split directory. For a full-data experiment, provide a directory containing the full `train.jsonl` and `valid.jsonl`.

By default there is **no optimizer-step cap**. The script completes three epochs and saves `checkpoint-epoch-1`, `checkpoint-epoch-2`, and `checkpoint-epoch-3`, each with its tokenizer. It also retains periodic step snapshots and saves the final adapter at the output root. Compare checkpoints rather than assuming the last epoch is strongest.

For a short pipeline check, explicitly add `--max-steps 5`. A capped partial epoch is not saved as a completed-epoch checkpoint. Use `--help` for path and recipe options.

## 5. Evaluate and export

Offline action evaluation of an epoch checkpoint:

```bash
python src/evaluation/evaluate_qwen_repo_adapter.py \
  --base models/qwen-action-base \
  --adapter outputs/rushdown-v3/checkpoint-epoch-1 \
  --data datasets/samples/project/rushdownv3data/valid.jsonl \
  --limit 1000 --load_4bit
```

The epoch above is an example; evaluate the checkpoints you trained. Metrics include generated exact match and argmax within the fixed action vocabulary. These are action-prediction metrics, not game-state legality or gameplay win rates.

Merge a selected adapter:

```bash
python src/training/export_qwen_repo_merged_untied.py \
  --base models/qwen-action-base \
  --adapter outputs/rushdown-v3/checkpoint-epoch-1 \
  --output models/rushdown-export

python src/evaluation/evaluate_qwen_repo_merged.py \
  --model models/rushdown-export \
  --data datasets/samples/project/rushdownv3data/valid.jsonl --limit 1000
```

The exporter copies saved embedding/head modules, merges LoRA, and saves untied embeddings. Compare adapter and exported predictions before deployment.

## 6. Prepare DAgger data

The sample workflow validates learner/expert disagreement records and merges a smaller v3 sample:

```bash
python src/evaluation/analyze_v8.py \
  datasets/samples/project/dagger_blackmamba_disagreements.jsonl \
  --output-dir work/dagger_analysis

python src/preprocessing/clean_rushdown_v8.py \
  --dagger-file work/dagger_analysis/cleaned.jsonl \
  --v3-file datasets/samples/project/rushdownv3data/train.jsonl \
  --output-file work/rushdown_v8_train.jsonl \
  --sample-size 500 --seed 42
```

The full v8 preparation used 60,000 v3 examples. The command uses 500 to fit the bundled reduced dataset. It demonstrates labeling-data analysis and merging, rather than running the learner/expert collection loop.

## 7. Run in the game

Use the [inference guide](inference.md) to apply the runtime changes, select a Hugging Face model or your merged export, and connect to DareFightingICE.

Local match summaries use a headered CSV with `p1_model`, `p2_model`, `p1_hp`, `p2_hp`, and a game identifier (`game_index` or `game`). `winner` is optional. Keep the same player/model positions within a game identifier.

```bash
python src/evaluation/evaluate_match_csvs.py /path/to/matches.csv --no_by_file
```

The utility counts round wins by the winner field or final HP, and game wins by round wins then summed HP. It strips model paths to basenames. Missing game identifiers group rows into one game per file, and duplicate rows are not removed. The organizer's headerless `HPMode` CSVs and the repository's aggregated summary use different schemas and should not be passed directly to this command.

## Checks

```bash
python -m unittest discover -s tests -v
```

The tests cover sample selection, byte limits, config parsing, and training CLI behavior without downloading models. Utility smoke checks and runtime-patch application are also verified during repository preparation. GPU training, checkpoint export, and live-game evaluations require the corresponding hardware, weights, and game installation.
