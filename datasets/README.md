# Dataset samples and processing workflow

This directory contains reduced versions of all 50 raw, intermediate, analysis, and versioned JSONL files used to document the project workflow. The samples contain **46,438 original physical lines**, total **30.62 MiB**, and retain the original record bytes. The largest sample is under 1 MiB.

Samples are selected with seed 42, up to 1,000 lines per file, using uniform reservoir sampling. A 5 MiB per-file ceiling keeps the package small. [manifest.json](manifest.json) records each original file's size, line count, hash, selected line indices, and sample hash/count.

**These are workflow samples.** The experiment scores in the project documentation used full datasets. Totals across stages include transformed copies of the same gameplay observations and should not be treated as independent examples.

## Layout

| Directory | Contents |
|---|---|
| [samples/early_collection](samples/early_collection/) | Initial expert logs and early cleaning variants |
| [samples/project](samples/project/) | Raw gameplay, cleaned/rebalanced intermediates, and split directories |
| [samples/project/dagger_analysis](samples/project/dagger_analysis/) | DAgger valid rows, duplicates, warnings, and cleaned examples |

The `project` directory keeps the original relative filenames so the stage inputs and outputs remain easy to follow. The empty `malformed_rows.jsonl` file is an actual zero-row analysis output, not a placeholder.

## Full-data split sizes

The table describes the full files from which the included samples were selected.

| Split | Training rows | Validation rows |
|---|---:|---:|
| Rushdown v3 | 151,861 | 16,878 |
| Rushdown v4 | 173,755 | 19,308 |
| Rushdown v5 | 189,887 | 21,094 |
| Rushdown v6 | 151,861 | 16,878 |
| Rushdown v8 | 90,177 | 16,878 |
| Zoning v3 | 389,050 | 43,226 |
| Zoning v4 | 143,002 | 15,888 |

v7 continues from the v3 model using v6 data. The `ftg_messages_plain` files match the v3 Rushdown split. v8 reuses v3 validation. The early `combined_database.jsonl` matches `rushdown_original_v3.jsonl`; both filenames are retained in the sample collection.

## Record formats

Most Rushdown and later Zoning examples use three messages:

```text
system: instruction
user: JSON containing self and opponent
assistant: expert action label
```

The original Zoning v3 files use `instruction`, `input`, and `output` fields. The Qwen trainer supports both formats. Rushdown uses `Predict the next FightingICE action.`; Zoning v3 uses `Predict the next FightingICE move.`

Game entities contain `remaining_frames`, `hp`, `action`, `state`, `position`, `speed`, `energy`, and `projectiles`. DAgger records additionally identify the learner/expert actions, frame number, source, and BlackMamba action index. Diagnostic wrapper rows contain an original row plus duplicate/warning information; they are analysis outputs, not direct training examples.

Raw samples preserve the formatting problems that repair/cleaning scripts address. The manifest flags malformed outer JSON and malformed nested state JSON. Prepared train/validation samples are valid records. No original rows were silently repaired during sampling.

## Processing stages

| Branch / stage | Input | Script | Output |
|---|---|---|---|
| Head-class compression | `zipf_rebalanced_v2.jsonl` | `powerlaw_shrink_jsonl.py` | `head_compressed.jsonl` |
| Rushdown v4 cleaning | `rushdown_v4.jsonl` | `clean_rushdown_v3.py` | `cleaned_rushdown_v4.jsonl` |
| Rushdown v5 cleaning | `zipf_augmented.jsonl` | `clean_dataset_v5.py` | `rushdown_cleaned_v5.jsonl` |
| Rushdown v5 balancing | `rushdown_cleaned_v5.jsonl` | `rebalance_dataset_v5.py` | `rebalanced_rushdown_v5.jsonl` |
| Rushdown v6 cleaning | `zipf_augmented.jsonl` | `clean_rushdown_v6.py` | `cleaned_rushdown_v6.jsonl` |
| Rushdown v6 balancing | `cleaned_rushdown_v6.jsonl` | `rebalance_rushdown_v3.py` | `rebalanced_rushdown_v6.jsonl` |
| Rushdown v6 split | `rebalanced_rushdown_v6.jsonl` | `split_database_v3.py` | `rushdownv6data/train.jsonl`, `valid.jsonl` |
| Zoning JSON repair | `zoningv4data.jsonl` | `repair_zoning_v4.py` | `repaired_zoning_v4.jsonl` |
| Zoning v4 cleaning | `repaired_zoning_v4.jsonl` | `clean_zoning_v4.py` | `cleaned_zoningv4data.jsonl` |
| Zoning near-duplicate cleaner | `zoningv4data.jsonl` | `clean_zoning_v3.py` | `cleaned_zoningv4data.jsonl` |
| Zoning rebalancing | `cleaned_zoningdata.jsonl` | `rebalance_zoning_v3.py` | `rebalanced_cleaned_zoningdata.jsonl` |
| DAgger analysis | Disagreement JSONL | `analyze_v8.py` | `cleaned.jsonl` and analysis outputs |
| DAgger/v3 merge | Cleaned DAgger + v3 training | `clean_rushdown_v8.py` | Mixed training JSONL |

Preprocessing scripts are under [src/preprocessing](../src/preprocessing/); analysis tools are under [src/evaluation](../src/evaluation/). Shared utilities are reused across versions; select their input/output constants for the stage you want to run. `rebalanced_zoning_v3.jsonl` is included as a prepared intermediate, while the zoning rebalancer's current output name is `rebalanced_cleaned_zoningdata.jsonl`.

Cleaning checks labels and state JSON, removes exact/near duplicates, and applies action-continuation rules. Rebalancing caps head classes and augments HP/energy without position jitter. Splitting is stratified by action. Some fixed-distribution rebalancers require all expected action classes, which random samples may not contain; use prepared sample splits to exercise training and use full-class input data for those rebalancers.

## Regenerate samples from full local data

The sampler accepts local source roots; those paths are not stored in the public manifest.

```bash
python tools/sample_datasets.py \
  --project-root /path/to/PBL5-Python \
  --early-root /path/to/early-collection \
  --output-dir datasets --max-rows 1000 --seed 42
```

`--project-root` contains the project JSONL files. The early collection reads only the seven explicitly named files listed in the manifest. See `--help` for the byte limit. Outputs must be separate from source data.
