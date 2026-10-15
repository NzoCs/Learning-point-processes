"""Scientific fitted states can be replayed without expiring CI checkpoints."""

import pytest
import torch

from scripts.independent_simulator_validation import create_model
from scripts.likelihood_calibration_validation import archive_fitted_parameters
from scripts.stopped_conditional_calibration import load_frozen_model


@pytest.mark.parametrize("name", ["FullyNN", "RMTPP"])
def test_archived_fitted_parameters_reload_exactly(tmp_path, name):
    model = create_model(name, tmp_path)
    checkpoint, archive = tmp_path / "fit.ckpt", tmp_path / "frozen.json"
    torch.save({"state_dict": model.state_dict()}, checkpoint)
    archive_fitted_parameters(name, checkpoint, archive)
    restored, fit = load_frozen_model(name, tmp_path, archive)
    original, original_fit = load_frozen_model(name, tmp_path, checkpoint)
    assert fit["model_state_sha256"] == original_fit["model_state_sha256"]
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
        torch.testing.assert_close(original.state_dict()[key], value, rtol=0, atol=0)
    with pytest.raises(ValueError, match="another model"):
        load_frozen_model(
            "RMTPP" if name == "FullyNN" else "FullyNN", tmp_path, archive
        )
