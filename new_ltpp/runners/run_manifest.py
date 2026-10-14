"""Persist effective configuration and provenance before executing a phase."""

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path

import torch

from new_ltpp.configs.model_config import ThinningConfig
from new_ltpp.evaluation.statistical_testing.point_process_kernels.utils import (
    SIGNATURE_PATH_PREPARATION,
    SIGNATURE_PATH_REPRESENTATION,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(root, args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args],
            stderr=subprocess.DEVNULL,
            timeout=5,
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


class RunManifest:
    def __init__(self, config):
        self.config = config
        self.path = config.base_dir / "manifest.json"
        raw_config = config.get_yaml_config()
        encoded = json.dumps(raw_config, sort_keys=True).encode()
        config_hash = hashlib.sha256(encoded).hexdigest()
        if self.path.exists():
            self.value = json.loads(self.path.read_text(encoding="utf-8"))
            if self.value["config_sha256"] != config_hash:
                raise ValueError(
                    "Existing run has a different configuration; use a new run_id"
                )
            # Enrich compatible older manifests without replacing their phases,
            # recorded configuration or historical code/checkpoint provenance.
            self.value.setdefault(
                "simulation_contract", config.simulation_config.execution_contract()
            )
            self.value["numerics"].setdefault(
                "signature_path_representation", SIGNATURE_PATH_REPRESENTATION
            )
            self.save()
            return
        root = Path(__file__).resolve().parents[2]
        dataset = config.data_config
        files = {}
        for split, source in (
            ("train", dataset.train_dir),
            ("dev", dataset.valid_dir),
            ("test", dataset.test_dir),
        ):
            if dataset.data_format != "hf" and Path(source).is_file():
                files[split] = {"path": source, "sha256": sha256_file(source)}
        self.value = {
            "schema_version": 1,
            "run_id": config.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": raw_config,
            "config_sha256": config_hash,
            "code": {
                "commit": _git_value(root, ["rev-parse", "HEAD"]),
                "dirty": _git_value(root, ["status", "--porcelain"]) not in (None, ""),
            },
            "lock_sha256": (
                sha256_file(root / "uv.lock") if (root / "uv.lock").is_file() else None
            ),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": sorted(
                    (d.metadata["Name"], d.version)
                    for d in distributions()
                    if d.metadata["Name"]
                ),
            },
            "numerics": {
                "torch_cuda": torch.version.cuda,
                "deterministic": config.training_config.deterministic,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "mmd_estimator": "unbiased_off_diagonal_v2",
                "mmd_observation_protocol": "pairwise_truncation_recomputed_v1",
                "mmd_p_value_protocol": "plus_one_float64_conservative_ties_v1",
                "mmd_null_method": "permutation",
                "mmd_rng_policy": "independent_resampling_stream_v1",
                "likelihood_observation": "conditional_transitions_plus_explicit_known_end_v1",
                "rmtpp_compensator": "analytic_capped_exponential_zero_slope_gradients_v1",
                "training_resume": "completed_epoch_v1; mid_epoch_without_loader_cursor_rejected",
                "signature_path_preparation": SIGNATURE_PATH_PREPARATION,
                "signature_path_representation": SIGNATURE_PATH_REPRESENTATION,
                "neural_event_sampling": {
                    "algorithm": "windowed_thinning_unbounded_v1",
                    "waiting_time_cap": None,
                    "max_sampling_rounds": ThinningConfig.model_validate(
                        dict(raw_config["model_config"]["thinning_config"])
                    ).max_sampling_rounds,
                    "envelope": "sampled_local_windows_checked_at_proposals",
                    "certified_envelope_for_arbitrary_models": False,
                    "integrated_hazard_models": ["FullyNN", "RMTPP"],
                    "integrated_hazard_algorithm": "inverse_compensator_with_atom_at_infinity_v1",
                    "one_step_prediction": "conditional_on_next_event_with_explicit_absence_probability",
                    "fullynn_derivative": "pointwise_sum_vjp_zero_centered_compensator_v1",
                },
            },
            "simulation_contract": config.simulation_config.execution_contract(),
            "data": {
                "format": dataset.data_format,
                "revision": dataset.revision,
                "local_files": files,
            },
            "phases": {},
        }
        if config.simulation_config.conditional_evaluation is not None:
            self.value["numerics"].update(
                mmd_estimator="paired_joint_off_diagonal_v1",
                mmd_observation_protocol="paired_joint_history_future_u_mmd_v1",
                mmd_null_method="within_history_pair_swaps_joint_kernel_v1",
            )
        from new_ltpp.evaluation.statistical_testing.statistical_tests.stopped_mmd_test import (
            STOPPED_OBSERVATION_PROTOCOL,
            uses_stopped_event_budget,
        )

        if uses_stopped_event_budget(config.model_id):
            self.value["numerics"][
                "mmd_observation_protocol"
            ] = STOPPED_OBSERVATION_PROTOCOL
            self.value["numerics"][
                "terminal_flag_kernel"
            ] = "unit_bandwidth_gaussian_v1"
        config.save_to_yaml_file(config.base_dir / "effective_config.yaml")
        self.save()

    def save(self):
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.value, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def start(self, phase):
        self.value["phases"][phase] = {
            "status": "running",
            "statistical_protocol": {
                "observation": "pairwise_truncation_recomputed_v1",
                "p_value": "plus_one_float64_conservative_ties_v1",
                "null_method": "permutation",
                "rng": "independent_resampling_stream_v1",
            },
        }
        if phase == "predict":
            self.value["phases"][phase]["statistical_interpretation"] = {
                "conditional_gof_calibrated": False,
                "comparison": "history_vs_its_continuation_diagnostic",
                "calibrated_api": "ConditionalMMDTwoSampleTest_with_heldout_future",
            }
            conditional = self.config.simulation_config.conditional_evaluation
            if conditional is not None:
                self.value["phases"][phase]["statistical_protocol"].update(
                    observation="paired_joint_history_future_u_mmd_v1",
                    null_method="within_history_pair_swaps_joint_kernel_v1",
                    aggregation="sum_statistics_product_randomization_v1",
                )
                self.value["phases"][phase]["statistical_interpretation"] = {
                    "comparison": "heldout_future_vs_same_history_continuation",
                    "calibration": "validated_protocol_under_conditional_H0",
                    "requires_independent_fit_and_disjoint_sequences": True,
                    "fit_independence": "not_verified_by_pipeline",
                    "identity_guard": "stable_sequence_ids_no_reuse",
                    **conditional.model_dump(),
                }
            from new_ltpp.evaluation.statistical_testing.statistical_tests.stopped_mmd_test import (
                STOPPED_OBSERVATION_PROTOCOL,
                uses_stopped_event_budget,
            )

            if uses_stopped_event_budget(self.config.model_id):
                self.value["phases"][phase]["statistical_protocol"][
                    "observation"
                ] = STOPPED_OBSERVATION_PROTOCOL
                self.value["phases"][phase]["statistical_interpretation"].update(
                    observed_status="quota_reached_or_explicit_no_next_event",
                    terminal_status_inferred_from_quiet_time=False,
                )
        self.save()

    def finish(self, phase, checkpoint=None, error=None):
        result = {
            "status": "failed" if error is not None else "completed",
            "statistical_protocol": self.value["phases"]
            .get(phase, {})
            .get("statistical_protocol"),
        }
        interpretation = (
            self.value["phases"].get(phase, {}).get("statistical_interpretation")
        )
        if interpretation is not None:
            result["statistical_interpretation"] = interpretation
        if error is not None:
            result["error"] = str(error)
        if checkpoint and Path(checkpoint).is_file():
            result["checkpoint"] = {
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
            }
        self.value["phases"][phase] = result
        self.value["numerics"][
            "float32_matmul_precision"
        ] = torch.get_float32_matmul_precision()
        self.save()
