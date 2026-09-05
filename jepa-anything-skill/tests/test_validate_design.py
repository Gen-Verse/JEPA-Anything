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
sys.path.insert(0, str(SCRIPT_DIR))

from validate_design import validate_config  # noqa: E402


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def failed_codes(report: dict) -> set[str]:
    return {
        item["code"] for item in report["checks"] if item["status"] == "fail"
    }


class ValidateDesignTests(unittest.TestCase):
    def test_valid_fixture_passes_stable_checks(self) -> None:
        report = validate_config(load_fixture())
        self.assertTrue(report["valid"])
        self.assertEqual(report["schema_version"], "2.0")
        pass_codes = {
            item["code"] for item in report["checks"] if item["status"] == "pass"
        }
        self.assertTrue(
            {
                "SYSTEM_IDENTITY_VALID",
                "TARGET_LEAKAGE_CONTROLS_VALID",
                "STATE_COMPOSITION_VALID",
                "FACTOR_ACTIVITY_CONFIGURED",
                "BASELINES_CAPACITY_MATCHED",
                "USAGE_MODE_VALID",
                "CLAIM_COVERAGE_COMPLETE",
                "OUTPUT_BOUNDARY_VALID",
            }
            <= pass_codes
        )

    def test_rejects_different_underlying_system(self) -> None:
        config = load_fixture()
        config["observation"]["target"]["system_id"] = "other-system"
        self.assertIn(
            "SYSTEM_IDENTITY_MISMATCH", failed_codes(validate_config(config))
        )

    def test_rejects_missing_or_inconsistent_instance_identity(self) -> None:
        config = load_fixture()
        del config["task"]["identity_keys"]
        del config["observation"]["context"]["identity_keys"]
        del config["observation"]["target"]["identity_keys"]
        self.assertIn(
            "SYSTEM_IDENTITY_MISMATCH", failed_codes(validate_config(config))
        )

        config = load_fixture()
        config["observation"]["target"]["identity_keys"] = ["other_id"]
        self.assertIn(
            "SYSTEM_IDENTITY_MISMATCH", failed_codes(validate_config(config))
        )

    def test_rejects_temporal_and_derived_target_leakage(self) -> None:
        config = load_fixture()
        config["observation"]["target"]["window"]["start"] = 0
        config["observation"]["context"]["target_derived_fields"] = ["future_mean"]
        codes = failed_codes(validate_config(config))
        self.assertIn("LEAKAGE_TEMPORAL_OVERLAP", codes)
        self.assertIn("LEAKAGE_TARGET_DERIVED_CONTEXT", codes)

    def test_rejects_adapter_fields_without_declared_lineage(self) -> None:
        config = load_fixture()
        config["observation"]["adapter"]["context_input_fields"] = [
            "observation",
            "future_label",
        ]
        config["observation"]["adapter"]["target_input_fields"] = ["future_label"]
        self.assertIn("LEAKAGE_SOURCE_UNKNOWN", failed_codes(validate_config(config)))

        config = load_fixture()
        config["observation"]["target"]["fields"] = ["second", "observation"]
        config["observation"]["adapter"]["target_input_fields"] = [
            "observation",
            "second",
        ]
        config["observation"]["context"]["sources"][0]["fields"].append("second")
        self.assertIn("LEAKAGE_SOURCE_UNKNOWN", failed_codes(validate_config(config)))

    def test_rejects_target_field_without_same_system_source_lineage(self) -> None:
        config = load_fixture()
        config["observation"]["target"]["fields"] = ["invented_future_label"]
        config["observation"]["adapter"]["target_input_fields"] = [
            "invented_future_label"
        ]

        self.assertIn("LEAKAGE_SOURCE_UNKNOWN", failed_codes(validate_config(config)))

    def test_rejects_unavailable_or_unknown_conditioning_lineage(self) -> None:
        config = load_fixture()
        descriptor = config["observation"]["target_descriptor"]["fields"][0]
        descriptor["lineage"]["available_through"] = 1
        exogenous = config["observation"]["exogenous_inputs"]["fields"][0]
        exogenous["lineage"] = {
            "source_id": "trajectory",
            "source_type": "context_source",
            "available_through": 0,
        }
        codes = failed_codes(validate_config(config))
        self.assertIn("LEAKAGE_INPUT_UNAVAILABLE", codes)
        self.assertIn("LEAKAGE_SOURCE_UNKNOWN", codes)

    def test_rejects_context_lineage_beyond_source_cutoff(self) -> None:
        config = load_fixture()
        config["observation"]["context"]["sources"][0]["available_through"] = -1
        descriptor = config["observation"]["target_descriptor"]["fields"][0]
        descriptor.update(
            {
                "name": "observation",
                "availability": "observed_by_context_end",
                "role": "conditioning",
                "lineage": {
                    "source_id": "trajectory",
                    "source_type": "context_source",
                    "available_through": 0,
                },
            }
        )
        codes = failed_codes(validate_config(config))
        self.assertIn("LEAKAGE_SOURCE_UNAVAILABLE", codes)
        self.assertIn("LEAKAGE_TARGET_DERIVED_CONTEXT", codes)

    def test_rejects_unknown_or_duplicate_target_source(self) -> None:
        config = load_fixture()
        config["observation"]["target"]["source_ids"] = ["unknown", "unknown"]
        report = validate_config(config)
        self.assertIn("LEAKAGE_SOURCE_UNKNOWN", failed_codes(report))
        self.assertTrue(
            any(
                item["code"] == "FIELD_INVALID"
                and item["path"] == "$.observation.target.source_ids"
                for item in report["checks"]
            )
        )

    def test_rejects_bad_state_shapes_analysis_and_synthesis(self) -> None:
        config = load_fixture()
        config["model"]["state"]["factors"].pop()
        self.assertIn(
            "COORDINATE_CONTRACT_INVALID", failed_codes(validate_config(config))
        )

        config = load_fixture()
        config["model"]["state"]["d"] = 7
        self.assertIn("STATE_DIMENSION_MISMATCH", failed_codes(validate_config(config)))

        config = load_fixture()
        config["model"]["state"]["analysis"]["kind"] = "fixed_slices"
        config["model"]["state"]["synthesis"]["kind"] = "concatenate"
        self.assertIn("STATE_COMPOSITION_INVALID", failed_codes(validate_config(config)))

    def test_accepts_optional_qr_orthogonality_modes(self) -> None:
        config = load_fixture()
        config["model"]["state"]["analysis"]["orthogonality_mode"] = "qr_init"
        self.assertTrue(validate_config(config)["valid"])

        config = load_fixture()
        analysis = config["model"]["state"]["analysis"]
        analysis["orthogonality_mode"] = "qr_retraction"
        analysis["qr_retraction"] = {
            "timing": "after_optimizer_step",
            "frequency": 2,
            "sign_canonicalization": "positive_r_diagonal",
            "training_compute_accounting": "report_separately",
        }
        self.assertTrue(validate_config(config)["valid"])

    def test_rejects_invalid_qr_orthogonality_contracts(self) -> None:
        config = load_fixture()
        config["model"]["state"]["analysis"]["orthogonality_mode"] = "forward_qr"
        self.assertIn(
            "ORTHOGONALITY_STRATEGY_INVALID",
            failed_codes(validate_config(config)),
        )

        config = load_fixture()
        analysis = config["model"]["state"]["analysis"]
        analysis["orthogonality_mode"] = "qr_retraction"
        analysis["qr_retraction"] = {
            "timing": "inside_forward",
            "frequency": 0,
            "sign_canonicalization": "none",
            "training_compute_accounting": "omit",
        }
        self.assertIn(
            "ORTHOGONALITY_STRATEGY_INVALID",
            failed_codes(validate_config(config)),
        )

        config = load_fixture()
        config["model"]["state"]["analysis"]["qr_retraction"] = {
            "timing": "after_optimizer_step",
            "frequency": 1,
            "sign_canonicalization": "positive_r_diagonal",
            "training_compute_accounting": "report_separately",
        }
        self.assertIn(
            "ORTHOGONALITY_STRATEGY_INVALID",
            failed_codes(validate_config(config)),
        )

    def test_rejects_semantic_factor_names_and_claim_prose(self) -> None:
        config = load_fixture()
        config["model"]["state"]["factors"][0]["meaning"] = "speed"
        config["claims"][0]["statement"] = "pc_000 represents speed."
        config["task"]["summary"] = "pc_000 is the speed factor."
        config["task"]["metadata"] = {
            "coordinate_meanings": {"pc_000": "speed"}
        }
        report = validate_config(config)
        self.assertIn("COORDINATE_CONTRACT_INVALID", failed_codes(report))
        self.assertIn("SEMANTIC_FACTOR_NAMING_FORBIDDEN", failed_codes(report))
        self.assertTrue(
            any(
                item["code"] == "FIELD_INVALID"
                and item["path"] == "$.claims[0]"
                for item in report["checks"]
            )
        )

    def test_rejects_incomplete_or_disabled_loss_contracts(self) -> None:
        config = load_fixture()
        config["losses"]["projector_orthogonality"]["components"].pop()
        config["losses"]["factor_activity"]["coordinates"].pop()
        config["losses"]["online_encoder_activity"]["dimensions"].pop()
        config["losses"]["factor_prediction"]["weight"] = 0
        codes = failed_codes(validate_config(config))
        self.assertIn("FIELD_INVALID", codes)
        self.assertIn("FACTOR_ACTIVITY_INCOMPLETE", codes)

    def test_rejects_missing_or_capacity_mismatched_baselines(self) -> None:
        config = load_fixture()
        config["baselines"] = [config["baselines"][0]]
        self.assertIn("BASELINE_REQUIRED_MISSING", failed_codes(validate_config(config)))

        config = load_fixture()
        match = config["baselines"][0]["capacity_match"]
        match["full_model_trainable_parameters"] = 20000
        match["predictor_flops"] = 60000
        del match["parameter_tolerance"]
        codes = failed_codes(validate_config(config))
        self.assertIn("BASELINE_CAPACITY_MISMATCH", codes)
        self.assertIn("BASELINE_FLOPS_MISMATCH", codes)

        config = load_fixture()
        config["baselines"][0]["id"] = config["model"]["id"]
        config["audits"]["capacity_matching"]["baseline_ids"][0] = config["model"]["id"]
        config["experiments"][0]["baselines"][0] = config["model"]["id"]
        config["claims"][0]["baseline_ids"][0] = config["model"]["id"]
        self.assertIn(
            "BASELINE_CAPACITY_MISMATCH", failed_codes(validate_config(config))
        )

    def test_rejects_wrong_use_mode_detail_and_horizon_order(self) -> None:
        config = load_fixture()
        config["usage"]["readout"] = {"tasks": ["probe"], "metrics": ["mse"]}
        config["usage"]["transition"]["rollout_horizons"] = [4, 1, 1]
        self.assertIn("USAGE_MODE_CONTRACT_INVALID", failed_codes(validate_config(config)))

    def test_all_three_use_modes_have_valid_contracts(self) -> None:
        for mode, usage in (
            (
                "terminal_readout",
                {
                    "mode": "terminal_readout",
                    "readout": {"tasks": ["terminal_probe"], "metrics": ["probe_mse"]},
                },
            ),
            (
                "factor_analysis",
                {
                    "mode": "factor_analysis",
                    "analysis": {
                        "probes": ["coordinate_probe"],
                        "interventions": ["coordinate_ablation"],
                        "coordinate_reporting": "prediction_coordinates_only",
                    },
                },
            ),
        ):
            config = load_fixture()
            config["usage"] = usage
            config["experiments"][0]["usage_mode"] = mode
            config["experiments"][0]["horizons"] = []
            config["claims"][0]["usage_mode"] = mode
            config["claims"][0]["horizons"] = []
            self.assertTrue(validate_config(config)["valid"], mode)

    def test_rejects_claim_without_exact_metric_coverage(self) -> None:
        config = load_fixture()
        config["experiments"][0]["metrics"] = [
            {"id": "coordinate_std", "direction": "higher_is_better"}
        ]
        self.assertIn("CLAIM_METRIC_UNCOVERED", failed_codes(validate_config(config)))

    def test_rejects_adversarial_accuracy_and_100_step_claim(self) -> None:
        config = load_fixture()
        claim = config["claims"][0]
        claim["metric_id"] = "accuracy"
        claim["direction"] = "higher_is_better"
        claim["horizons"] = [100]
        codes = failed_codes(validate_config(config))
        self.assertIn("CLAIM_METRIC_UNCOVERED", codes)
        self.assertIn("CLAIM_HORIZON_MISMATCH", codes)

    def test_rejects_claim_direction_split_baselines_and_audits_mismatch(self) -> None:
        config = load_fixture()
        claim = config["claims"][0]
        claim["direction"] = "higher_is_better"
        claim["split"] = "validation"
        claim["baseline_ids"] = ["standard_jepa"]
        claim["audit_ids"] = ["geometry"]
        codes = failed_codes(validate_config(config))
        self.assertIn("CLAIM_DIRECTION_MISMATCH", codes)
        self.assertIn("CLAIM_SPLIT_MISMATCH", codes)
        self.assertIn("CLAIM_BASELINE_UNCOVERED", codes)
        self.assertIn("CLAIM_AUDIT_UNCOVERED", codes)

    def test_rejects_claim_that_omits_required_audit_evidence(self) -> None:
        config = load_fixture()
        config["claims"][0]["audit_ids"] = ["claim_coverage"]
        config["experiments"][0]["audits"] = ["claim_coverage"]

        self.assertIn("CLAIM_AUDIT_UNCOVERED", failed_codes(validate_config(config)))

    def test_rejects_one_way_experiment_to_claim_reference(self) -> None:
        config = load_fixture()
        extra = json.loads(json.dumps(config["experiments"][0]))
        extra["id"] = "uncontracted-extra-experiment"
        config["experiments"].append(extra)

        self.assertIn("CLAIM_RECIPROCITY_MISSING", failed_codes(validate_config(config)))

    def test_rejects_result_or_conclusion_payload_inside_planned_experiment(self) -> None:
        config = load_fixture()
        config["experiments"][0]["result"] = {
            "supported": True,
            "conclusion": "reference wins",
        }

        self.assertIn("FIELD_INVALID", failed_codes(validate_config(config)))

    def test_rejects_non_string_claim_and_experiment_references(self) -> None:
        config = load_fixture()
        config["claims"][0]["experiment_ids"] = [123]
        config["experiments"][0]["tests_claims"] = [456]
        report = validate_config(config)
        invalid_paths = {
            item["path"] for item in report["checks"] if item["code"] == "FIELD_INVALID"
        }
        self.assertIn("$.claims[0].experiment_ids[0]", invalid_paths)
        self.assertIn("$.experiments[0].tests_claims[0]", invalid_paths)

    def test_rejects_training_loop_output(self) -> None:
        config = load_fixture()
        config["outputs"]["scaffold"]["include_training_loop"] = True
        self.assertIn("OUTPUT_TRAINING_FORBIDDEN", failed_codes(validate_config(config)))

    def test_cli_returns_one_for_invalid_design_and_two_for_parse_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            invalid_path = temp / "invalid.json"
            invalid = load_fixture()
            invalid["model"]["state"]["d"] = 9
            invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(SCRIPT_DIR / "validate_design.py"), str(invalid_path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertFalse(json.loads(result.stdout)["valid"])

            malformed = temp / "malformed.json"
            malformed.write_text("{", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(SCRIPT_DIR / "validate_design.py"), str(malformed)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
