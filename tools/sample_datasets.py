#!/usr/bin/env python3
"""Create reproducible, byte-preserving excerpts of local gameplay JSONL files.

Run with --project-root and --early-root pointing at the original collections.
Only the seven explicitly named early collection files are considered. Project
JSONLs are discovered recursively. Outputs and metadata contain logical paths,
never the supplied local source roots. No third-party dependencies are needed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

EARLY_FILENAMES = (
    "blackmamba_dataset 1.jsonl",
    "combined_database.jsonl",
    "pruned_database.jsonl",
    "sanitize.jsonl",
    "sanitized-2.jsonl",
    "sanitized-3.jsonl",
    "sanitized.jsonl",
)
DEFAULT_MAX_ROWS = 1000
DEFAULT_MAX_BYTES = 5 * 1024 * 1024


def sample_file(source, target, *, max_rows=DEFAULT_MAX_ROWS,
                max_bytes=DEFAULT_MAX_BYTES, seed=42):
    """Sample physical lines without decoding, rewriting, or reordering them.

    Algorithm R selects a uniform reservoir among lines no larger than the byte
    limit. After reading, uniformly subsample to at most floor(max_bytes / the
    largest eligible line), which guarantees the aggregate cap. This may retain
    fewer than max_rows even when a particular larger sample could fit. Line
    indices are one-based; a final line without a newline remains byte-identical.
    """
    if max_rows <= 0 or max_bytes <= 0:
        raise ValueError("max_rows and max_bytes must both be positive")
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve():
        raise ValueError("source and target must differ")
    rng = random.Random(seed)
    source_hash = hashlib.sha256()
    reservoir = []
    source_rows = source_bytes = eligible_rows = skipped = largest_line = 0
    with source.open("rb") as stream:
        for source_rows, line in enumerate(stream, start=1):
            source_hash.update(line)
            source_bytes += len(line)
            if len(line) > max_bytes:
                skipped += 1
                continue
            eligible_rows += 1
            largest_line = max(largest_line, len(line))
            item = (source_rows, line)
            if len(reservoir) < max_rows:
                reservoir.append(item)
            else:
                slot = rng.randrange(eligible_rows)
                if slot < max_rows:
                    reservoir[slot] = item
    safe_rows = min(max_rows, max_bytes // largest_line) if largest_line else 0
    if len(reservoir) > safe_rows:
        reservoir = rng.sample(reservoir, safe_rows)
    reservoir.sort(key=lambda item: item[0])
    target.parent.mkdir(parents=True, exist_ok=True)
    sample_hash = hashlib.sha256()
    sample_bytes = 0
    validation = {"valid_json_rows": 0, "invalid_json_rows": 0, "schema_rows": {},
                  "invalid_user_state_json_rows": 0,
                  "invalid_json_source_line_indices": [],
                  "unexpected_schema_source_line_indices": []}
    with target.open("wb") as stream:
        for source_index, line in reservoir:
            stream.write(line)
            sample_hash.update(line)
            sample_bytes += len(line)
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                validation["invalid_json_rows"] += 1
                validation["invalid_json_source_line_indices"].append(source_index)
                continue
            validation["valid_json_rows"] += 1
            schema = "unexpected"
            if isinstance(row, dict):
                if isinstance(row.get("messages"), list):
                    schema = "messages"
                    for message in row["messages"]:
                        if (isinstance(message, dict) and message.get("role") == "user"
                                and isinstance(message.get("content"), str)):
                            try:
                                json.loads(message["content"])
                            except ValueError:
                                validation["invalid_user_state_json_rows"] += 1
                elif {"input", "instruction", "output"} <= row.keys():
                    schema = "instruction_input_output"
                elif "row" in row and {"warnings", "duplicate_hash", "line_number"} & row.keys():
                    schema = "diagnostic_wrapper"
            validation["schema_rows"][schema] = validation["schema_rows"].get(schema, 0) + 1
            if schema == "unexpected":
                validation["unexpected_schema_source_line_indices"].append(source_index)
    return {
        "source_sha256": source_hash.hexdigest(),
        "source_rows": source_rows,
        "source_bytes": source_bytes,
        "source_empty": source_bytes == 0,
        "eligible_rows": eligible_rows,
        "skipped_oversized_lines": skipped,
        "largest_eligible_line_bytes": largest_line,
        "sample_sha256": sample_hash.hexdigest(),
        "sample_rows": len(reservoir),
        "sample_bytes": sample_bytes,
        "selected_line_indices": [index for index, _ in reservoir],
        "sample_validation": validation,
    }


def build_samples(project_root, early_root, output_dir, *, max_rows=DEFAULT_MAX_ROWS,
                  max_bytes=DEFAULT_MAX_BYTES, seed=42):
    """Write samples and a portable manifest for the two source collections."""
    if max_rows <= 0 or max_bytes <= 0:
        raise ValueError("max_rows and max_bytes must both be positive")
    project_root, early_root, output_dir = map(Path, (project_root, early_root, output_dir))
    if not project_root.is_dir() or not early_root.is_dir():
        raise ValueError("both source roots must be existing directories")
    resolved_output = output_dir.resolve()
    for source_root in (project_root, early_root):
        resolved_source = source_root.resolve()
        if resolved_output == resolved_source or resolved_source in resolved_output.parents:
            raise ValueError("output directory must be outside both source roots")
    sources = [("project", path.relative_to(project_root), path)
               for path in project_root.rglob("*.jsonl")
               if path.is_file() and not path.is_symlink()]
    sources.extend(("early_collection", Path(name), early_root / name)
                   for name in EARLY_FILENAMES
                   if (early_root / name).is_file() and not (early_root / name).is_symlink())
    sources.sort(key=lambda item: (item[0], item[1].as_posix()))
    records = []
    for group, relative, source in sources:
        logical = Path(group) / relative
        sample = Path("samples") / logical
        info = sample_file(source, output_dir / sample, max_rows=max_rows,
                           max_bytes=max_bytes, seed=seed)
        records.append({"group": group, "source": logical.as_posix(),
                        "sample": sample.as_posix(), **info})
    manifest = {
        "format_version": 1,
        "description": "Reduced original-line excerpts for inspection and smoke tests; not full training datasets or reproduced benchmark results.",
        "sampling": {
            "algorithm": "uniform Algorithm R reservoir with uniform byte-safe subsampling",
            "seed": seed,
            "max_rows_per_file": max_rows,
            "max_bytes_per_file": max_bytes,
            "line_indices": "one-based physical original lines, retained in original order",
            "byte_policy": "Preserve exact original bytes. Skip lines larger than max_bytes. Uniformly reduce the reservoir to at most floor(max_bytes / largest_eligible_line_bytes).",
            "source_discovery": "All recursive project *.jsonl files and only the named early collection files that exist.",
            "validation": "Counts describe sampled physical rows only. Original malformed JSON and malformed user-state strings are preserved without repair; diagnostic wrappers are not training examples.",
            "early_collection_filenames": list(EARLY_FILENAMES),
        },
        "totals": {
            "files": len(records),
            **{field: sum(record[field] for record in records)
               for field in ("source_rows", "source_bytes", "sample_rows", "sample_bytes", "skipped_oversized_lines")},
        },
        "files": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--early-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "datasets")
    parser.add_argument("--max-rows", type=positive_int, default=DEFAULT_MAX_ROWS)
    parser.add_argument("--max-bytes", type=positive_int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        manifest = build_samples(args.project_root, args.early_root, args.output_dir,
                                 max_rows=args.max_rows, max_bytes=args.max_bytes, seed=args.seed)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(manifest["totals"], sort_keys=True))


if __name__ == "__main__":
    main()
