"""A mid-epoch checkpoint must not silently promise exact shuffled replay."""

import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from new_ltpp.runners.rng_callback import ValidationRNGCallback
from scripts.independent_simulator_validation import constant_poisson_model, history


def test_actual_mid_epoch_checkpoint_is_explicitly_rejected(tmp_path):
    path = tmp_path / "partial.ckpt"

    class Interrupt(pl.Callback):
        def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
            if batch_idx == 0:
                trainer.save_checkpoint(path)
                raise RuntimeError("injected interruption")

    def trainer(callbacks):
        return pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=callbacks,
        )

    model, _ = constant_poisson_model("RMTPP", tmp_path)
    loader = DataLoader([history() for _ in range(4)], batch_size=None, shuffle=True)
    with pytest.raises(RuntimeError, match="injected interruption"):
        trainer([ValidationRNGCallback(), Interrupt()]).fit(model, loader)
    saved = torch.load(path, weights_only=False)
    assert saved["global_step"] == 1
    assert saved["training_resume_contract"]["epoch_complete"] is False
    with pytest.raises(ValueError, match="Mid-epoch"):
        trainer([ValidationRNGCallback()]).fit(model, loader, ckpt_path=path)
