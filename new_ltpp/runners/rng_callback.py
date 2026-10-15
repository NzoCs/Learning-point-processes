"""Keep validation sampling from advancing training random streams."""

import random

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.trainer.states import TrainerFn


class ValidationRNGCallback(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_complete = False

    def on_train_epoch_end(self, trainer, pl_module):
        self._epoch_complete = True

    def on_validation_start(self, trainer, pl_module):
        self.state = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        )

    def on_validation_end(self, trainer, pl_module):
        python_state, numpy_state, torch_state, cuda_state = self.state
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["training_resume_contract"] = {
            "version": 1,
            "epoch_complete": getattr(self, "_epoch_complete", False),
        }
        checkpoint["training_rng_state"] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        }

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if trainer.state.fn != TrainerFn.FITTING:
            return
        position = checkpoint.get("training_resume_contract")
        if position is not None and not position["epoch_complete"]:
            raise ValueError(
                "Mid-epoch training resume is unsupported: the shuffled loader cursor "
                "is not checkpointed. Resume from a completed epoch checkpoint."
            )
        state = checkpoint.get("training_rng_state")
        if state is not None:
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            torch.set_rng_state(state["torch"].cpu())
            if state["cuda"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
