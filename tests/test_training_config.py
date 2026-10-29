"""Exercise CLI recipe selection and path checks without model dependencies."""
import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "src" / "training" / "config.py"


class TrainingConfigTests(unittest.TestCase):
    def load_config(self):
        self.assertTrue(CONFIG_PATH.is_file(), "A lightweight training config module is required")
        spec = importlib.util.spec_from_file_location("training_config", CONFIG_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_default_recipe_runs_three_epochs_without_step_cap(self):
        config = self.load_config()
        args = config.parse_training_args([])
        self.assertEqual((args.batch_size, args.epochs, args.max_steps), (16, 3, None))
        self.assertEqual(args.base_model_dir, Path("models/qwen-action-base"))
        self.assertEqual(args.data_dir, Path("datasets/samples/project/ftg_messages_plain"))
        self.assertEqual(config.parse_builder_args([]).output_dir, args.base_model_dir)

    def test_cli_selects_alternate_data_base_and_output_and_optional_smoke_cap(self):
        config = self.load_config()
        args = config.parse_training_args([
            "--base-model-dir", "models/custom", "--data-dir", "datasets/rushdownv3data",
            "--output-dir", "outputs/rushdown-v3", "--batch-size", "8", "--epochs", "2",
            "--max-steps", "5",
        ])
        self.assertEqual(args.base_model_dir, Path("models/custom"))
        self.assertEqual(args.data_dir, Path("datasets/rushdownv3data"))
        self.assertEqual(args.output_dir, Path("outputs/rushdown-v3"))
        self.assertEqual((args.batch_size, args.epochs, args.max_steps), (8, 2, 5))
        builder = config.parse_builder_args(["--output-dir", "models/custom"])
        self.assertEqual(builder.output_dir, args.base_model_dir)
        self.assertEqual(builder.base_model, "Qwen/Qwen2.5-1.5B")

    def test_nonpositive_batch_epoch_or_cap_is_rejected(self):
        config = self.load_config()
        for flag in ("--batch-size", "--epochs", "--max-steps"):
            for value in ("0", "-1"):
                with self.subTest(flag=flag, value=value):
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                        config.parse_training_args([flag, value])
                    self.assertEqual(error.exception.code, 2)

    def test_path_validation_requires_local_base_and_both_splits(self):
        config = self.load_config()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, data, output = root / "base", root / "data", root / "output"
            args = config.parse_training_args([
                "--base-model-dir", str(base), "--data-dir", str(data), "--output-dir", str(output)
            ])
            with self.assertRaisesRegex(ValueError, "base-model-dir"):
                config.validate_training_paths(args)
            base.mkdir()
            (base / "config.json").write_text("{}")
            data.mkdir()
            (data / "train.jsonl").write_text('{}\n')
            with self.assertRaisesRegex(ValueError, "valid.jsonl"):
                config.validate_training_paths(args)
            (data / "valid.jsonl").write_text('{}\n')
            config.validate_training_paths(args)
            self.assertFalse(output.exists(), "Validation must not create a training output")
            output.write_text("occupied")
            with self.assertRaisesRegex(ValueError, "output-dir"):
                config.validate_training_paths(args)

    def test_both_entry_points_offer_help_without_model_packages(self):
        for name in ("train_qwen_repo_causal_lm.py", "build_qwen_plain_action_base.py"):
            with self.subTest(script=name):
                result = subprocess.run(
                    [sys.executable, "-S", str(ROOT / "src" / "training" / name), "--help"],
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--output-dir", result.stdout)


if __name__ == "__main__":
    unittest.main()
