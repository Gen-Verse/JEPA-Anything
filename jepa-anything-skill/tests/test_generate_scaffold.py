from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = SKILL_ROOT / "scripts"
FIXTURE = SKILL_ROOT / "tests" / "fixtures" / "valid_design.json"


class GenerateScaffoldTests(unittest.TestCase):
    def test_generates_importable_contract_without_training_loop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "generated"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "generate_scaffold.py"),
                    str(FIXTURE),
                    "--output-dir",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertFalse(payload["contains_training_loop"])
            manifest = json.loads(
                (output / "scaffold-manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(manifest["contains_training_loop"])
            self.assertTrue((output / "config" / "design.json").is_file())

            contract_test = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    str(output / "tests"),
                    "-v",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(contract_test.returncode, 0, contract_test.stderr)
            self.assertIn("Ran 10 tests", contract_test.stderr)

    def test_invalid_design_is_blocked_before_destination_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            config = json.loads(FIXTURE.read_text(encoding="utf-8"))
            config["model"]["state"]["d"] = 99
            invalid = temp / "invalid.json"
            invalid.write_text(json.dumps(config), encoding="utf-8")
            output = temp / "must-not-exist"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "generate_scaffold.py"),
                    str(invalid),
                    "--output-dir",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertFalse(output.exists())
            self.assertFalse(json.loads(result.stdout)["valid"])

    def test_nonempty_destination_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "occupied"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "generate_scaffold.py"),
                    str(FIXTURE),
                    "--output-dir",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_undeclared_output_format_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            config = json.loads(FIXTURE.read_text(encoding="utf-8"))
            config["outputs"]["config_formats"] = ["json"]
            json_only = temp / "json-only.json"
            json_only.write_text(json.dumps(config), encoding="utf-8")
            output = temp / "must-not-exist"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "generate_scaffold.py"),
                    str(json_only),
                    "--output-dir",
                    str(output),
                    "--config-format",
                    "yaml",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertFalse(output.exists())
            self.assertIn("not declared", result.stderr)

    def test_generated_contract_tests_support_all_three_use_modes(self) -> None:
        variants = {
            "terminal_readout": {
                "mode": "terminal_readout",
                "readout": {"tasks": ["probe"], "metrics": ["probe_mse"]},
            },
            "factor_analysis": {
                "mode": "factor_analysis",
                "analysis": {
                    "probes": ["coordinate_probe"],
                    "interventions": ["coordinate_ablation"],
                    "coordinate_reporting": "prediction_coordinates_only",
                },
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            for mode, usage in variants.items():
                with self.subTest(mode=mode):
                    config = json.loads(FIXTURE.read_text(encoding="utf-8"))
                    config["usage"] = usage
                    config["experiments"][0]["usage_mode"] = mode
                    config["experiments"][0]["horizons"] = []
                    config["claims"][0]["usage_mode"] = mode
                    config["claims"][0]["horizons"] = []
                    config_path = temp / f"{mode}.json"
                    config_path.write_text(json.dumps(config), encoding="utf-8")
                    output = temp / f"generated-{mode}"
                    generated = subprocess.run(
                        [
                            sys.executable,
                            str(SCRIPT_DIR / "generate_scaffold.py"),
                            str(config_path),
                            "--output-dir",
                            str(output),
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(generated.returncode, 0, generated.stderr)
                    contract_test = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "unittest",
                            "discover",
                            "-s",
                            str(output / "tests"),
                            "-v",
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(contract_test.returncode, 0, contract_test.stderr)

    def test_generated_contract_supports_qr_retraction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            config = json.loads(FIXTURE.read_text(encoding="utf-8"))
            analysis = config["model"]["state"]["analysis"]
            analysis["orthogonality_mode"] = "qr_retraction"
            analysis["qr_retraction"] = {
                "timing": "after_optimizer_step",
                "frequency": 2,
                "sign_canonicalization": "positive_r_diagonal",
                "training_compute_accounting": "report_separately",
            }
            config_path = temp / "qr-retraction.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = temp / "generated"
            generated = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "generate_scaffold.py"),
                    str(config_path),
                    "--output-dir",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(generated.returncode, 0, generated.stderr)
            contract_test = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    str(output / "tests"),
                    "-v",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(contract_test.returncode, 0, contract_test.stderr)

    def test_generated_runtime_contract_is_not_removed_by_python_optimization(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "generated"
            generated = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "generate_scaffold.py"),
                    str(FIXTURE),
                    "--output-dir",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(generated.returncode, 0, generated.stderr)
            probe = (
                "import sys; "
                f"sys.path.insert(0, {str(output / 'src')!r}); "
                "import jepa_task.contracts as contracts; "
                "contracts.STATE_D += 1; contracts.assert_contract()"
            )
            optimized = subprocess.run(
                [sys.executable, "-O", "-c", probe],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(optimized.returncode, 0)
            self.assertIn("ValueError", optimized.stderr)


if __name__ == "__main__":
    unittest.main()
