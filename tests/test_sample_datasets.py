"""Behavioral tests for the byte-preserving, bounded public dataset sampler."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "sample_datasets.py"


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def sample(self, content, **kwargs):
        self.assertTrue(SCRIPT.exists(), "the sampler must be implemented")
        spec = importlib.util.spec_from_file_location("sample_datasets", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source, target = self.root / "source.jsonl", self.root / "sample.jsonl"
        source.write_bytes(content)
        return module.sample_file(source, target, **kwargs), target.read_bytes()

    def test_preserves_exact_original_bytes_and_order(self):
        lines = [b'{"n":1}\r\n', b'{"n": 2}\n', b'{"n":3}']
        info, output = self.sample(b"".join(lines), max_rows=10)
        self.assertEqual(output, b"".join(lines))
        self.assertEqual(info["selected_line_indices"], [1, 2, 3])
        self.assertEqual(info["source_rows"], 3)
        self.assertEqual(info["source_sha256"], hashlib.sha256(output).hexdigest())
        self.assertEqual(info["sample_sha256"], hashlib.sha256(output).hexdigest())

    def test_reservoir_is_deterministic_and_selects_across_the_file(self):
        original = [f'{{"n":{n}}}\n'.encode() for n in range(100)]
        first, data = self.sample(b"".join(original), max_rows=7, seed=42)
        second, again = self.sample(b"".join(original), max_rows=7, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(data, again)
        self.assertEqual(first["sample_rows"], 7)
        indices = first["selected_line_indices"]
        self.assertEqual(indices, sorted(set(indices)))
        self.assertGreater(max(indices), 50)
        self.assertEqual(data, b"".join(original[i - 1] for i in indices))

    def test_bounds_rows_and_bytes_and_reports_oversized_lines(self):
        info, output = self.sample(b"x" * 31 + b"\n" + b"a\n" * 20, max_rows=8, max_bytes=20)
        self.assertLessEqual(len(output), 20)
        self.assertEqual(info["skipped_oversized_lines"], 1)
        self.assertEqual(info["source_rows"], 21)
        self.assertEqual(info["sample_rows"], 8)
        self.assertNotIn(1, info["selected_line_indices"])

    def test_aggregate_cap_keeps_whole_original_lines(self):
        info, output = self.sample(b"123456789\n" * 100, max_rows=20, max_bytes=35)
        self.assertEqual(info["sample_rows"], 3)
        self.assertEqual(len(output), 30)
        self.assertEqual(output, b"123456789\n" * 3)

    def test_empty_source_creates_empty_sample_and_explicit_metadata(self):
        info, output = self.sample(b"")
        self.assertEqual(output, b"")
        self.assertEqual(info["source_rows"], 0)
        self.assertEqual(info["sample_rows"], 0)
        self.assertEqual(info["source_bytes"], 0)
        self.assertEqual(info["sample_bytes"], 0)
        self.assertEqual(info["selected_line_indices"], [])
        self.assertTrue(info["source_empty"])

    def test_flags_original_json_and_state_errors_without_cleaning(self):
        original = (b'{"messages":[{"role":"user","content":"{bad"}]}\n'
                    b'{broken\n'
                    b'{"row":{},"line_number":1,"warnings":[]}\n'
                    b'{"unexpected":true}\n')
        info, output = self.sample(original)
        self.assertEqual(output, original)
        self.assertEqual(info.get("sample_validation"), {
            "valid_json_rows": 3,
            "invalid_json_rows": 1,
            "schema_rows": {"messages": 1, "diagnostic_wrapper": 1, "unexpected": 1},
            "invalid_user_state_json_rows": 1,
            "invalid_json_source_line_indices": [2],
            "unexpected_schema_source_line_indices": [4],
        })

    def test_zero_or_negative_limits_are_rejected_without_output(self):
        for option in ("max_rows", "max_bytes"):
            for number in (0, -1):
                with self.subTest(option=option, value=number):
                    with self.assertRaises(ValueError):
                        self.sample(b"{}\n", **{option: number})
                    self.assertFalse((self.root / "sample.jsonl").exists())

    def test_rejects_outputs_in_either_source_before_writing_anything(self):
        cases = [(source, kind) for source in ("project", "early")
                 for kind in ("equal", "nested", "symlink_nested")]
        for source_name, kind in cases:
            with self.subTest(source=source_name, output=kind):
                case_root = self.root / (source_name + "_" + kind)
                project, early = case_root / "project", case_root / "early"
                project.mkdir(parents=True)
                early.mkdir()
                (project / "train.jsonl").write_bytes(b'{"messages":[]}\n')
                (early / "sanitized.jsonl").write_bytes(b'{"messages":[]}\n')
                source = case_root / source_name
                if kind == "equal":
                    output = source
                elif kind == "nested":
                    output = source / "new" / "outputs"
                else:
                    alias = case_root / "source_alias"
                    alias.symlink_to(source, target_is_directory=True)
                    output = alias / "new" / "outputs"
                before = {path.relative_to(case_root): path.read_bytes()
                          for path in case_root.rglob("*") if path.is_file()}
                result = subprocess.run([
                    sys.executable, str(SCRIPT), "--project-root", str(project),
                    "--early-root", str(early), "--output-dir", str(output),
                ], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0, "unsafe output must be rejected")
                after = {path.relative_to(case_root): path.read_bytes()
                         for path in case_root.rglob("*") if path.is_file()}
                self.assertEqual(after, before, "rejection must leave all source files unchanged")
                if kind != "equal":
                    self.assertFalse(output.exists(), "rejection must not create output directories")

    def test_cli_preserves_recursive_paths_and_only_whitelisted_early_files(self):
        project, early, output = [self.root / name for name in ("project", "early", "out")]
        (project / "nested").mkdir(parents=True)
        early.mkdir()
        (project / "nested" / "train.jsonl").write_bytes(b'{"messages":[]}\n')
        (early / "sanitized.jsonl").write_bytes(b'{"input":{},"instruction":"a","output":"b"}\n')
        (early / "unrelated.jsonl").write_bytes(b'private\n')
        self.assertTrue(SCRIPT.exists(), "the sampler must be implemented")
        result = subprocess.run([sys.executable, str(SCRIPT), "--project-root", str(project), "--early-root", str(early), "--output-dir", str(output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual([x["source"] for x in manifest["files"]], ["early_collection/sanitized.jsonl", "project/nested/train.jsonl"])
        self.assertTrue((output / "samples" / "project" / "nested" / "train.jsonl").is_file())
        self.assertNotIn(str(self.root), (output / "manifest.json").read_text())
        self.assertEqual(manifest["totals"]["files"], 2)


if __name__ == "__main__":
    unittest.main()
