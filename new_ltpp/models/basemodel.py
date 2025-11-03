"""Base model with common functionality using PyTorch Lightning"""

import os
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Literal, Optional, Union

import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from new_ltpp.configs import ModelConfig
from new_ltpp.evaluation.metrics_helper import EvaluationMode, MetricsHelper
from new_ltpp.models.thinning import EventSampler
from new_ltpp.utils import format_multivariate_simulations, logger, save_json

from .model_registry import RegistryMeta


class Model(pl.LightningModule, ABC, metaclass=RegistryMeta):
    """Base model class for all TPP models."""

    def __init__(
        self,
        model_config: ModelConfig,
        *,
        num_event_types: int,
        dtime_max: float,
    ):
        """Initialize the Model

        Args:
            model_config (new_ltpp.ModelConfig): model spec of configs
        """
        super(Model, self).__init__()

        # Save hyperparameters for later use
        self.save_hyperparameters()

        self.dtime_max = dtime_max

        # Load model configuration
        pretrain_model_path = model_config.pretrain_model_path
        self.scheduler_config = model_config.scheduler_config

        # Load training configuration
        self.compute_simulation = model_config.compute_simulation

        # Load data and model specifications
        self.num_event_types = num_event_types
        self.pad_token_id = num_event_types

        self.loss_integral_num_sample_per_step = (
            model_config.thinning_config.loss_integral_num_sample_per_step
        )

        self.eps = torch.finfo(torch.float32).eps

        # Model prediction configuration
        self.gen_config = model_config.thinning_config
        self.use_mc_samples = self.gen_config.use_mc_samples
        self._device = model_config.device
        self.num_step_gen = self.gen_config.num_steps
        self.dtime_max = self.gen_config.dtime_max

        simulation_config = model_config.simulation_config

        # Simulation from the model configuration
        if simulation_config is not None:
            self.seed = simulation_config.seed
            self.simulation_batch_size = simulation_config.batch_size
            self.simulation_start_time = simulation_config.start_time
            self.simulation_end_time = simulation_config.end_time
            self.max_simul_events = simulation_config.max_sim_events

        self.sim_events_counter = 0
        self.simulations = []

        # Cache for event samplers to avoid reconstruction
        self._event_sampler_cache = {}

        # Load pretrained model if path is provided
        if pretrain_model_path is not None:
            checkpoint = torch.load(
                pretrain_model_path, map_location=self.device, weights_only=False
            )
            # Adjust keys if necessary, e.g., remove prefix if saved with DDP
            state_dict = checkpoint["state_dict"]
            # Example key adjustment (if needed):
            # state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            self.load_state_dict(
                state_dict, strict=False
            )  # Use strict=False if some layers are different
            logger.info(
                f"Successfully loaded pretrained model from: {pretrain_model_path}"
            )

    # Implement for the models based on intensity (not implemented in intensity free)
    def compute_intensities_at_sample_times(
        self,
        time_seqs: torch.Tensor,
        time_delta_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        sample_dtimes: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the intensity at sampled times, not only event times.

        Args:
            time_seqs (tensor): [batch_size, seq_len], times seqs.
            time_delta_seqs (tensor): [batch_size, seq_len], time delta seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.
            sample_dtimes (tensor): [batch_size, seq_len, num_sample], sampled inter-event timestamps.

        Returns:
            tensor: [batch_size, num_times, num_mc_sample, num_event_types],
                    intensity at each timestamp for each event type."""
        pass

    @abstractmethod
    def loglike_loss(
        self,
        batch: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> tuple[torch.Tensor, int]:
        """Compute the log-likelihood loss for a batch of data.

        Args:
            batch: Contains time_seq, time_delta_seq, event_seq, batch_non_pad_mask, batch_attention_mask

        Returns:
            loss, number of events.
        """
        pass

    # Set up the event sampler if generation config is provided
    def event_sampler(self, num_sample=None, mode: Literal['train', 'simulation'] = 'train') -> EventSampler:
        """Get the event sampler for generating events with caching."""

        gen_config = self.gen_config

        if num_sample is None:
            num_sample = gen_config.num_sample

        # Use num_sample as cache key
        cache_key = num_sample

        # Check if we have a cached sampler for this num_sample
        if cache_key in self._event_sampler_cache:
            cached_sampler = self._event_sampler_cache[cache_key]
            # Verify that the cached sampler's device matches current device
            if cached_sampler.device == self._device:
                return cached_sampler
            else:
                # Device mismatch, remove from cache
                del self._event_sampler_cache[cache_key]

        # Create new event sampler
        event_sampler = EventSampler(
            num_sample=num_sample,
            num_exp=gen_config.num_exp,
            over_sample_rate=gen_config.over_sample_rate,
            num_samples_boundary=gen_config.num_samples_boundary,
            dtime_max=self.dtime_max,
            device=self._device,
            mode=mode
        )

        # Cache the new sampler
        self._event_sampler_cache[cache_key] = event_sampler

        return event_sampler

    @property
    def device(self):
        """Get the current device."""
        return self._device

    def to(self, *args, **kwargs):
        """Override to() method to update device-dependent components."""
        device = None
        if args and isinstance(args[0], (str, torch.device)):
            device = args[0]
        elif "device" in kwargs:
            device = kwargs["device"]

        # Call the parent's to() method first
        model = super().to(*args, **kwargs)

        # Update our stored device if a new one was specified
        if device is not None:
            model._device = device
            # Clear event sampler cache when device changes
            model._event_sampler_cache.clear()

        return model

    @staticmethod
    def get_logits_at_last_step(logits, batch_non_pad_mask, sample_len=None):
        """Retrieve the hidden states of last non-pad events.

        Args:
            logits (tensor): [batch_size, seq_len, hidden_dim], a sequence of logits
            batch_non_pad_mask (tensor): [batch_size, seq_len], a sequence of masks
            sample_len (tensor): default None, use batch_non_pad_mask to find out the last non-mask position

        ref: https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4

        Returns:
            tensor: retrieve the logits of EOS event
        """

        seq_len = batch_non_pad_mask.sum(dim=1)
        select_index = seq_len - 1 if sample_len is None else seq_len - 1 - sample_len
        # [batch_size, hidden_dim]
        select_index = select_index.unsqueeze(1).repeat(1, logits.size(-1))
        # [batch_size, 1, hidden_dim]
        select_index = select_index.unsqueeze(1)
        # [batch_size, hidden_dim]
        last_logits = torch.gather(logits, dim=1, index=select_index).squeeze(1)
        return last_logits

    def make_dtime_loss_samples(self, time_delta_seq):
        """Generate the time point samples for every interval.

        Args:
            time_delta_seq (tensor): [batch_size, seq_len].

        Returns:
            tensor: [batch_size, seq_len, n_samples]
        """
        # [1, 1, n_samples]
        dtimes_ratio_sampled = torch.linspace(
            start=0.0,
            end=1.0,
            steps=self.loss_integral_num_sample_per_step,
            device=self.device,
        )[None, None, :]

        # [batch_size, max_len, n_samples]
        sampled_dtimes = time_delta_seq[:, :, None] * dtimes_ratio_sampled

        return sampled_dtimes

    def compute_loglikelihood(
        self, time_delta_seq, lambda_at_event, lambdas_loss_samples, seq_mask, type_seq
    ):
        """Compute the loglikelihood of the event sequence based on Equation (8) of NHP paper.

        Args:
            time_delta_seq (tensor): [batch_size, seq_len], time_delta_seq from model input.
            lambda_at_event (tensor): [batch_size, seq_len, num_event_types], unmasked intensity at
            (right after) the event.
            lambdas_loss_samples (tensor): [batch_size, seq_len, num_sample, num_event_types],
            intensity at sampling times.
            seq_mask (tensor): [batch_size, seq_len], sequence mask vector to mask the padded events.
            type_seq (tensor): [batch_size, seq_len], sequence of mark ids, with padded events having a mark of self.pad_token_id

        Returns:
            tuple: event loglike, non-event loglike, intensity at event with padding events masked
        """

        # First, add an epsilon to every marked intensity for stability
        lambda_at_event = lambda_at_event + self.eps

        log_marked_event_lambdas = lambda_at_event.log()
        total_sampled_lambdas = lambdas_loss_samples.sum(dim=-1)

        # Compute event LL - [batch_size, seq_len]
        event_ll = -F.nll_loss(
            log_marked_event_lambdas.permute(
                0, 2, 1
            ),  # mark dimension needs to come second, not third to match nll_loss specs
            target=type_seq,
            ignore_index=self.pad_token_id,  # Padded events have a pad_token_id as a value
            reduction="none",  # Does not aggregate, and replaces what would have been the log(marked intensity) with 0.
        )

        # Compute non-event LL [batch_size, seq_len]
        # interval_integral = length_interval * average of sampled lambda(t)
        if self.use_mc_samples:
            non_event_ll = (
                total_sampled_lambdas.mean(dim=-1) * time_delta_seq * seq_mask
            )
        else:  # Use trapezoid rule
            non_event_ll = (
                0.5
                * (
                    total_sampled_lambdas[..., 1:] + total_sampled_lambdas[..., :-1]
                ).mean(dim=-1)
                * time_delta_seq
                * seq_mask
            )

        num_events = torch.masked_select(event_ll, event_ll.ne(0.0)).size()[0]
        return event_ll, non_event_ll, num_events

    def configure_optimizers(self):
        """Configure the optimizer for the model.

        Returns:
            optimizer: The optimizer to use for training.
        """

        # Use Adam optimizer with optional learning rate scheduler
        lr = self.scheduler_config.lr
        lr_scheduler = self.scheduler_config.lr_scheduler
        max_epochs = self.scheduler_config.max_epochs

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Use cosine decay scheduler instead
        if hasattr(self, "lr_scheduler") and lr_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_epochs,  # Total number of epochs
                eta_min=lr * 0.01,  # Minimum learning rate at the end of schedule
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": None}

        return optimizer

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """Training step for Lightning.

        Args:
            batch: Contains time_seq, time_delta_seq, event_seq, batch_non_pad_mask, batch_attention_mask
            batch_idx: Index of the batch

        Returns:
            STEP_OUTPUT: The output of the training step
        """

        # Convert dict to tuple for models that expect tuple format
        if not isinstance(batch, tuple):
            batch = tuple(batch.values())

        loss, num_events = self.loglike_loss(batch)
        avg_loss = loss / num_events
        self.log(
            "train_loss",
            avg_loss.item(),
            prog_bar=True,
            sync_dist=True,
            on_epoch=True,
            on_step=False,
        )

        return avg_loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """Validation step for Lightning.

        Args:
            batch: Contains time_seq, time_delta_seq, event_seq, batch_non_pad_mask, batch_attention_mask
            batch_idx: Index of the batch

        Returns:
            STEP_OUTPUT: The output of the validation step
        """
        # Fix: always convert dict.values() to tuple
        if not isinstance(batch, tuple):
            batch = tuple(batch.values())

        label_batch = [seq[:, 1:] for seq in batch]

        loss, num_events = self.loglike_loss(batch)
        avg_loss = loss / num_events

        self.log(
            "val_loss",
            avg_loss.item(),
            prog_bar=True,
            sync_dist=True,
            on_epoch=True,
            on_step=False,
        )

        # Compute some validation metrics
        pred = self.predict_one_step_at_every_event(batch)

        one_step_metrics_compute = MetricsHelper(
            num_event_types=self.num_event_types, mode=EvaluationMode.PREDICTION
        )

        one_step_metrics = one_step_metrics_compute.compute_all_metrics(
            batch=label_batch, pred=pred
        )

        for key in one_step_metrics:
            if key == "confusion_matrix":
                # Skip confusion matrix as it's a 2D tensor that can't be logged directly
                # We could save it separately or log derived metrics like accuracy
                continue
            self.log(f"{key}", one_step_metrics[key], prog_bar=False, sync_dist=True)

        return avg_loss

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """Test step for Lightning.

        Args:
            batch: Contains time_seq, time_delta_seq, event_seq, batch_non_pad_mask, batch_attention_mask
            batch_idx: Index of the batch

        Returns:
            STEP_OUTPUT: The output of the test step
        """
        # Fix: always convert dict.values() to tuple
        if not isinstance(batch, tuple):
            batch = tuple(batch.values())

        # Create label batch by removing first element from each sequence
        (
            time_seq,
            time_delta_seq,
            event_seq,
            batch_non_pad_mask,
            batch_attention_mask,
        ) = batch
        label_batch = (
            time_seq[:, 1:],
            time_delta_seq[:, 1:],
            event_seq[:, 1:],
            batch_non_pad_mask[:, 1:],
            batch_attention_mask[:, 1:] if batch_attention_mask is not None else None,
        )

        loss, num_events = self.loglike_loss(batch)
        avg_loss = loss / num_events
        self.log("test_loss", avg_loss.item(), prog_bar=True, sync_dist=True)

        # Compute some prediction metrics
        pred = self.predict_one_step_at_every_event(batch)

        one_step_metrics_compute = MetricsHelper(
            num_event_types=self.num_event_types, mode=EvaluationMode.PREDICTION
        )
        one_step_metrics = one_step_metrics_compute.compute_all_metrics(
            batch=label_batch, pred=pred
        )
        for key in one_step_metrics:
            if key == "confusion_matrix":
                # Skip confusion matrix as it's a 2D tensor that can't be logged directly
                # We could save it separately or log derived metrics like accuracy
                continue
            self.log(f"{key}", one_step_metrics[key], prog_bar=False, sync_dist=True)

        if self.compute_simulation:

            if self.sim_events_counter >= self.max_simul_events:
                logger.warning(
                    f"Simulation limit reached: {self.sim_events_counter} events generated, "
                    f"max is {self.max_simul_events}."
                )
                return avg_loss

            # Compute simulation metrics
            simulation = self.simulate(batch=batch)

            simulation_metrics_compute = MetricsHelper(
                num_event_types=self.num_event_types, mode=EvaluationMode.SIMULATION
            )

            simulation_metrics = simulation_metrics_compute.compute_all_metrics(
                batch=label_batch, pred=simulation
            )  # Log simulation metrics
            for key in simulation_metrics:
                if key == "confusion_matrix":
                    # Skip confusion matrix as it's a 2D tensor that can't be logged directly
                    # We could save it separately or log derived metrics like accuracy
                    continue
                self.log(
                    f"sim_{key}",
                    simulation_metrics[key],
                    prog_bar=False,
                    sync_dist=True,
                )

            simul_time_seq, simul_time_delta_seq, simul_event_seq, simul_mask = (
                simulation
            )

            # Convert tensors to CPU in batch to avoid doing it in the loop
            simul_time_seq_cpu = simul_time_seq.detach().cpu()
            simul_time_delta_seq_cpu = simul_time_delta_seq.detach().cpu()
            simul_event_seq_cpu = simul_event_seq.detach().cpu()
            simul_mask_cpu = simul_mask.detach().cpu()

            batch_size = simul_time_seq_cpu.size(0)

            # Increment global event counter
            n_events_generated = int(simul_mask_cpu.sum().item())
            self.sim_events_counter += n_events_generated

            # Build list of new dictionaries to add to self.simulations
            nouveaux = []
            for i in range(batch_size):
                mask_i = simul_mask_cpu[i]
                if not mask_i.any():
                    continue

                # Index CPU vectors directly with boolean mask
                ts_i = simul_time_seq_cpu[i][mask_i]
                dts_i = simul_time_delta_seq_cpu[i][mask_i]
                evs_i = simul_event_seq_cpu[i][mask_i]

                # Add clear dict without re-cloning/detaching (already on CPU and detached)
                nouveaux.append(
                    {
                        "time_seq": ts_i,
                        "time_delta_seq": dts_i,
                        "event_seq": evs_i,
                    }
                )

            # Extend existing list (avoids calling append each time)
            self.simulations.extend(nouveaux)

        return avg_loss

    def predict_step(self, batch, batch_idx, **kwargs) -> STEP_OUTPUT:
        """Prediction step for Lightning.

        Args:
            batch: Contient time_seq, time_delta_seq, type_seq, batch_non_pad_mask, batch_attention_mask
            batch_idx: Index du batch (non utilisé ici, mais requis par Lightning)
        Returns:
            STEP_OUTPUT: Liste de dictionnaires de simulations produites
        """

        # 1) Si on a déjà généré assez d'événements, on s'arrête immédiatement
        if self.sim_events_counter >= self.max_simul_events:
            logger.warning(
                f"Simulation limit reached: {self.sim_events_counter} events generated, "
                f"max is {self.max_simul_events}."
            )
            return self.simulations

        # 2) On s'assure que batch est un tuple (Lightning peut lui passer un dict)
        if not isinstance(batch, tuple):
            batch = tuple(batch.values())

        # 4) Appel à la simulation « vectorisée » (retourne tout sur le device GPU/CPU interne)
        simul_time_seq, simul_time_delta_seq, simul_event_seq, simul_mask = (
            self.simulate(batch=batch)
        )

        # simul_time_seq, simul_time_delta_seq, simul_event_seq  sont des tenseurs sur self.device,
        # simul_mask aussi. On convertit l'ensemble en CPU EN UN SEUL BATCH,
        # pour ne pas le faire dans la boucle ci-dessous.
        simul_time_seq_cpu = simul_time_seq.detach().cpu()
        simul_time_delta_seq_cpu = simul_time_delta_seq.detach().cpu()
        simul_event_seq_cpu = simul_event_seq.detach().cpu()
        simul_mask_cpu = simul_mask.detach().cpu()

        batch_size = simul_time_seq_cpu.size(0)

        # 5) On incrémente le compteur global d'événements simulés
        #    simul_mask_cpu.sum() est un scalaire Python
        n_events_generated = int(simul_mask_cpu.sum().item())
        self.sim_events_counter += n_events_generated

        # 6) On construit la liste des nouveaux dictionnaires à ajouter à self.simulations.
        #    Pour chaque i dans [0..batch_size-1], si simul_mask_cpu[i] comporte
        #    au moins un True, on extrait en une fois « time_seq[i][mask_i] », etc.
        nouveaux = []
        for i in range(batch_size):
            mask_i = simul_mask_cpu[i]
            if not mask_i.any():
                continue

            # On indexe directement les vecteurs CPU avec le booléen mask_i
            ts_i = simul_time_seq_cpu[i][mask_i]
            dts_i = simul_time_delta_seq_cpu[i][mask_i]
            evs_i = simul_event_seq_cpu[i][mask_i]

            # On ajoute un dict clair, sans refaire clone/detach (déjà sur CPU et detaché)
            nouveaux.append(
                {
                    "time_seq": ts_i,
                    "time_delta_seq": dts_i,
                    "event_seq": evs_i,
                }
            )

        # On étend la liste existante (évite d’appeler append à chaque fois)
        self.simulations.extend(nouveaux)

        return self.simulations

    def format_and_save_simulations(self, save_dir: Union[str, Path]) -> list[dict]:
        """
        Formats the raw simulation results into a list of dictionaries, one per sequence.

        Each dictionary follows a structure similar to Hugging Face datasets,
        containing event times, time deltas, event types, etc.

        Args:
            simulations (List[Dict]): A list where each dict contains tensors
                                      ('time_seq', 'time_delta_seq', 'event_seq')
                                      for a single simulated sequence.
            dim_process (Optional[int]): The number of event types (dimensionality) in the process.

        Returns:
            List[Dict]: A list of dictionaries, each representing a formatted sequence.
        """

        if self.simulations == []:
            logger.warning("No simulations to format.")
            return None
        formatted_data = format_multivariate_simulations(
            simulations=self.simulations, dim_process=self.num_event_types
        )

        if isinstance(save_dir, str):
            save_dir = Path(save_dir)

        save_data_path = save_dir / "simulations.json"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_json(formatted_data, save_data_path)

        return formatted_data

    def save_metadata(self, save_dir: Union[str, Path], formatted_data: list[dict]) -> None:
        """
        Saves metadata about the simulation run, including configuration details
        and total event counts.

        Args:
            formatted_data (List[Dict]): The list of all formatted sequences (used for stats).
        """
        total_events = sum(item.get("seq_len", 0) for item in formatted_data)
        avg_seq_len = total_events / len(formatted_data) if formatted_data else 0

        metadata = {
            "simulation_summary": {
                "total_sequences_generated": len(formatted_data),
                "total_events_generated": total_events,
                "average_sequence_length": round(avg_seq_len, 2),
                "dimension": (
                    self.num_event_types
                    if self.num_event_types is not None
                    else "Unknown"
                ),
                "simulation_time_interval": [
                    self.simulation_start_time,
                    self.simulation_end_time,
                ],
                "generating_model": self.__class__.__name__,
                "seed_used": self.seed,
            }
        }

        if isinstance(save_dir, str):
            save_dir = Path(save_dir)

        meta_filepath = save_dir / "metadata.json"
        save_json(metadata, meta_filepath)
        logger.info(f"Metadata saved to {meta_filepath}")

    def predict_one_step_at_every_event(
        self,
        batch: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One-step prediction for every event in the sequence.

        Args:
            batch: The batch of data

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        time_seq, time_delta_seq, event_seq, batch_non_pad_mask, _ = batch

        # remove the last event, as the prediction based on the last event has no label
        # note: the first dts is 0
        # [batch_size, seq_len]
        time_seq, time_delta_seq, event_seq = (
            time_seq[:, :-1],
            time_delta_seq[:, :-1],
            event_seq[:, :-1],
        )

        # [batch_size, seq_len]
        dtime_boundary = torch.max(
            time_delta_seq * self.dtime_max, time_delta_seq + self.dtime_max
        )

        # [batch_size, seq_len, num_sample]
        accepted_dtimes, weights = self.event_sampler().draw_next_time_one_step(
            time_seq,
            time_delta_seq,
            event_seq,
            dtime_boundary,
            self.compute_intensities_at_sample_times,
            compute_last_step_only=False,
        )  # make it explicit

        # We should condition on each accepted time to sample event mark, but not conditioned on the expected event time.
        # 1. Use all accepted_dtimes to get intensity.
        # [batch_size, seq_len, num_sample, num_marks]
        intensities_at_times = self.compute_intensities_at_sample_times(
            time_seq, time_delta_seq, event_seq, accepted_dtimes
        )

        # 2. Normalize the intensity over last dim and then compute the weighted sum over the `num_sample` dimension.
        # Each of the last dimension is a categorical distribution over all marks.
        # [batch_size, seq_len, num_sample, num_marks]
        intensities_normalized = intensities_at_times / intensities_at_times.sum(
            dim=-1, keepdim=True
        )

        # 3. Compute weighted sum of distributions and then take argmax.
        # [batch_size, seq_len, num_marks]
        intensities_weighted = torch.einsum(
            "...s,...sm->...m", weights, intensities_normalized
        )

        # [batch_size, seq_len]
        types_pred = torch.argmax(intensities_weighted, dim=-1)

        # [batch_size, seq_len]
        dtimes_pred = torch.sum(
            accepted_dtimes * weights, dim=-1
        )  # compute the expected next event time

        return dtimes_pred, types_pred

    def predict_multi_step_since_last_event(
        self,
        batch: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
        forward=False,
        num_step: int = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Multi-step prediction since last event in the sequence.

        Args:
            batch : Contains time_seq, time_delta_seq, event_seq, batch_non_pad_mask, batch_attention_mask
            num_step : the number of steps to take
            forward : wheter to predict after the last event or to go back and predict events that occured

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, num_step].
        """

        time_seq_label, time_delta_seq_label, event_seq_label, _, _ = batch

        if num_step is None:
            num_step = self.num_step_gen

        batch_size = time_seq_label.size(0)

        if not forward:
            initial_seq = time_seq_label[:, :-num_step]
            initial_delta = time_delta_seq_label[:, :-num_step]
            initial_event = event_seq_label[:, :-num_step]
        else:
            initial_seq = time_seq_label
            initial_delta = time_delta_seq_label
            initial_event = event_seq_label

        initial_len = initial_seq.size(1)
        total_len = initial_len + num_step

        # Pré-allocation des buffers pour éviter les torch.cat
        time_buffer = torch.zeros(
            batch_size, total_len, dtype=initial_seq.dtype, device=initial_seq.device
        ).contiguous()
        time_delta_buffer = torch.zeros(
            batch_size,
            total_len,
            dtype=initial_delta.dtype,
            device=initial_delta.device,
        ).contiguous()
        event_buffer = torch.zeros(
            batch_size,
            total_len,
            dtype=initial_event.dtype,
            device=initial_event.device,
        ).contiguous()

        # Copier les séquences initiales dans les buffers
        time_buffer[:, :initial_len].copy_(initial_seq)
        time_delta_buffer[:, :initial_len].copy_(initial_delta)
        event_buffer[:, :initial_len].copy_(initial_event)

        current_len = initial_len

        # Boucle de prédiction avec indexation directe sur les buffers
        for _ in range(num_step):
            current_len += 1

            # Obtenir les vues actuelles des séquences (pas de copie)
            current_time_seq = time_buffer[:, :current_len]
            current_time_delta = time_delta_buffer[:, :current_len]
            current_event_seq = event_buffer[:, :current_len]

            # Utiliser predict_one_step pour éviter la duplication de code
            dtimes_pred, types_pred = self.predict_one_step(
                current_time_seq, current_time_delta, current_event_seq
            )

            # Calcul du nouveau temps
            time_pred_step = current_time_seq[:, -1:] + dtimes_pred

            # Écriture directe dans les buffers (pas de concatenation)
            time_buffer[:, current_len] = time_pred_step.squeeze(-1)
            time_delta_buffer[:, current_len] = dtimes_pred.squeeze(-1)
            event_buffer[:, current_len] = types_pred.squeeze(-1)

        # Extraction des résultats finaux
        return (
            time_delta_buffer[:, -num_step - 1 :],
            event_buffer[:, -num_step - 1 :],
            time_delta_seq_label[:, -num_step - 1 :],
            event_seq_label[:, -num_step - 1 :],
        )

    def simulate(
        self,
        batch: Optional[tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        batch_size: Optional[int] = None,
        max_events: int = 1000,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Version encore plus optimisée avec approche vectorisée avancée.

        Cette version essaie de traiter plusieurs événements en parallèle
        pour chaque séquence du batch quand c'est possible.
        """

        if start_time is None:
            start_time = self.simulation_start_time
        if end_time is None:
            end_time = self.simulation_end_time
        if batch_size is None:
            batch_size = self.simulation_batch_size

        # Initialize sequences
        if batch is None:
            batch = (
                torch.zeros(batch_size, 2, device=self.device, dtype=torch.float32),
                torch.zeros(batch_size, 2, device=self.device, dtype=torch.float32),
                torch.zeros(batch_size, 2, device=self.device, dtype=torch.long),
                torch.zeros(1, 2, device=self.device, dtype=torch.float32),
                torch.zeros(1, 2, device=self.device, dtype=torch.float32),
            )
        else:
            batch_size = batch[0].size(0)

        time_seq, time_delta_seq, event_seq, _, _ = batch
        num_mark = self.num_event_types

        # Pré-allocation avec mémoire contigüe
        max_seq_len = max_events + time_seq.size(1)

        # Utiliser des tenseurs contigus pour de meilleures performances
        time_buffer = torch.zeros(
            batch_size, max_seq_len, device=self.device, dtype=torch.float32
        ).contiguous()
        time_delta_buffer = torch.zeros(
            batch_size, max_seq_len, device=self.device, dtype=torch.float32
        ).contiguous()
        event_buffer = torch.zeros(
            batch_size, max_seq_len, device=self.device, dtype=torch.long
        ).contiguous()

        # Copie initiale
        initial_len = time_seq.size(1)
        time_buffer[:, :initial_len].copy_(time_seq)
        time_delta_buffer[:, :initial_len].copy_(time_delta_seq)
        event_buffer[:, :initial_len].copy_(event_seq)

        # Calcul vectorisé optimisé de last_event_time
        last_event_time = torch.zeros(
            (batch_size, num_mark), device=self.device, dtype=torch.float32
        )

        # Utilisation de advanced indexing pour l'optimisation
        for mark in range(num_mark):
            mark_mask = event_seq == mark  # [batch_size, seq_len]
            if mark_mask.any():
                # Opération vectorisée avec masquage efficace
                masked_times = time_seq.masked_fill(~mark_mask, float("-inf"))
                max_times, _ = masked_times.max(dim=1)
                valid_mask = max_times != float("-inf")
                last_event_time[valid_mask, mark] = max_times[valid_mask]

        current_time = start_time

        # Simulation avec moins d'allocations mémoire
        with torch.no_grad():  # Pas besoin de gradients pour la simulation
            pbar = tqdm(total=end_time, desc="Simulation", leave=False)

            step_count = 0
            batch_active = torch.ones(batch_size, dtype=torch.bool, device=self.device)

            while current_time < end_time and step_count < max_seq_len - 1:
                if not batch_active.any():
                    break

                # Extraire seulement les séquences actives
                active_indices = batch_active.nonzero(as_tuple=True)[0]

                if len(active_indices) == 0:
                    break

                current_len = initial_len + step_count

                # Prédiction sur les séquences actives seulement
                active_time_seq = time_buffer[active_indices, :current_len]
                active_time_delta = time_delta_buffer[active_indices, :current_len]
                active_event_seq = event_buffer[active_indices, :current_len]

                try:
                    dtimes_pred, type_pred = self.predict_one_step(
                        active_time_seq,
                        active_time_delta,
                        active_event_seq,
                        num_sample=1,
                        mode="simulation"
                    )

                    # Calcul des nouveaux temps
                    new_times = active_time_seq[:, -1:] + dtimes_pred

                    # Mise à jour vectorisée de last_event_time pour les séquences actives
                    active_batch_size = len(active_indices)

                    # Vectorisation complète : utiliser advanced indexing pour mettre à jour last_event_time
                    type_pred_flat = type_pred.squeeze(-1)  # [active_batch_size]
                    last_times_flat = active_time_seq[:, -1]  # [active_batch_size]

                    # Mise à jour vectorisée avec scatter_
                    last_event_time[active_indices, type_pred_flat] = last_times_flat

                    # Recalcul des deltas
                    batch_indices_active = torch.arange(
                        active_batch_size, device=self.device
                    )
                    type_pred_flat = type_pred.squeeze(-1)
                    last_events_active = last_event_time[active_indices][
                        batch_indices_active, type_pred_flat
                    ]
                    dtimes_corrected = new_times.squeeze(-1) - last_events_active

                    # Mise à jour des buffers pour les séquences actives
                    time_buffer[active_indices, current_len] = new_times.squeeze(-1)
                    time_delta_buffer[active_indices, current_len] = dtimes_corrected
                    event_buffer[active_indices, current_len] = type_pred.squeeze(-1)

                    # Mise à jour du temps courant et des séquences actives
                    current_time = new_times.min().item()

                    # Désactiver les séquences qui ont dépassé end_time
                    exceed_time_mask = new_times.squeeze(-1) >= end_time
                    if exceed_time_mask.any():
                        exceed_indices = active_indices[exceed_time_mask]
                        batch_active[exceed_indices] = False

                    step_count += 1

                    if step_count % 50 == 0:
                        pbar.n = min(current_time, end_time)
                        pbar.refresh()

                except Exception as e:
                    logger.error(f"Error in vectorized simulation: {e}")
                    break

            pbar.close()

        # Extraction des résultats
        first_pred_idx = initial_len
        final_time_seq = time_buffer[:, first_pred_idx:current_len]
        final_time_delta = time_delta_buffer[:, first_pred_idx:current_len]
        final_event_seq = event_buffer[:, first_pred_idx:current_len]

        # Masque final
        simul_mask = torch.logical_and(
            final_time_seq >= start_time, final_time_seq <= end_time
        )

        return final_time_seq, final_time_delta, final_event_seq, simul_mask

    def intensity_graph(
        self,
        start_time: float = 100.0,
        end_time: float = 200.0,
        precision: int = 100,
        plot: bool = False,
        save_plot: bool = True,
        save_data: bool = True,
        save_dir: str = "./",
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
        """
        Génère et affiche la courbe d'intensité du modèle pour une séquence donnée.

        Cette fonction calcule les intensités du modèle aux instants échantillonnés et
        permet de visualiser leur évolution en fonction du temps, séparément pour chaque type d'événement.

        Args:
            start_time (float): Temps de début de la simulation.
            end_time (float): Temps de fin de la simulation.
            precision (int, optionnel): Nombre de points interpolés entre deux événements
                pour lisser la courbe d'intensité. Par défaut à 100.
            plot (bool, optionnel): Indique s'il faut afficher le graphique des intensités.
                Par défaut à False.
            save_plot (bool, optionnel): Indique s'il faut sauvegarder le graphique.
                Par défaut à False.
            save_data (bool, optionnel): Indique s'il faut sauvegarder les données d'intensité.
                Par défaut à False.
            save_dir (str): Répertoire de sauvegarde du graphique et des données.

        Returns:
            tuple:
                - torch.Tensor: Matrice des intensités calculées pour chaque type d'événement [num_sample_points, num_event_types].
                - torch.Tensor: Points de temps correspondant aux échantillons d'intensité [num_sample_points].
                - dict[int, torch.Tensor]: Dictionnaire des instants où chaque type d'événement est observé.
        """

        num_mark = self.num_event_types

        # Simulate data

        if self.simulations is None:

            time_seq, time_delta_seq, type_seq, simul_mask = self.simulate(
                start_time=start_time, end_time=end_time, batch_size=1
            )
            # Extract the first (and only) sequence from batch dimension
            time_seq = time_seq[0][simul_mask[0]]  # [seq_len]
            time_delta_seq = time_delta_seq[0][simul_mask[0]]  # [seq_len]
            type_seq = type_seq[0][simul_mask[0]]  # [seq_len]
        else:

            # Use the first simulation in the batch
            time_seq = self.simulations[0]["time_seq"]
            time_delta_seq = self.simulations[0]["time_delta_seq"]
            type_seq = self.simulations[0]["event_seq"]
            # Note: simulations are already filtered, so no additional mask needed

        # Add batch dimension back for processing
        time_seq = time_seq.unsqueeze(0)  # [1, seq_len]
        time_delta_seq = time_delta_seq.unsqueeze(0)  # [1, seq_len]
        type_seq = type_seq.unsqueeze(0)  # [1, seq_len]

        ratios = (
            torch.linspace(start=0.0, end=1.0, steps=precision, device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )  # [1, 1, precision]

        # Reconstruct actual time points from intervals
        # For each interval, create time points: start_time + delta * ratio
        time_starts = time_seq[:, :-1]  # [1, seq_len-1] - exclude last event
        time_deltas = time_delta_seq[
            :, 1:
        ]  # [1, seq_len-1] - exclude first delta (which is 0)

        time_deltas_sample = time_deltas.unsqueeze(-1) * ratios

        # Calculate intensities on augmented intervals
        # [1, seq_len, precision, num_event_types]
        # Run the intensity computations in a no-grad context to avoid creating
        # "inference tensors" that PyTorch disallows being saved for backward.
        # We then return a cloned/detached tensor so downstream code that may
        # accidentally run under autograd won't try to save inference-only tensors.
        with torch.no_grad():
            intensities = self.compute_intensities_at_sample_times(
                time_seq[:, 1:],  # [1, seq_len-1]
                time_delta_seq[:, 1:],  # [1, seq_len-1]
                type_seq[:, 1:],  # [1, seq_len-1]
                time_deltas_sample,
            )

        # Ensure we return a regular Tensor (detach + clone) so it can be
        # used safely by code paths that expect tensors requiring grad or that
        # might be captured by autograd later. This avoids the RuntimeError:
        # "Inference tensors cannot be saved for backward..."
        intensities = intensities.detach().clone()

        time_diffs = time_seq.diff()

        # Calculate actual time points: [1, seq_len-1, precision]
        time_points = time_starts.unsqueeze(-1) + time_diffs.unsqueeze(-1) * ratios

        # Flatten time points and intensities for plotting
        # [total_points] where total_points = (seq_len-1) * precision
        time_flat = time_points.view(-1)

        # Remove batch dimension and flatten: [total_points, num_event_types]
        intensities_flat = intensities[0, ...].view(-1, num_mark)

        # Collect marked times for each event type
        marked_times = defaultdict(list)
        time_seq_flat = time_seq.squeeze(0)  # Remove batch dimension
        type_seq_flat = type_seq.squeeze(0)  # Remove batch dimension

        for i in range(num_mark):
            mask = type_seq_flat == i
            if mask.any():
                marked_times[i] = time_seq_flat[mask]
            else:
                marked_times[i] = torch.empty(0, device=self.device)

        # Sauvegarder les données d'intensité si demandé
        if save_data:
            os.makedirs(save_dir, exist_ok=True)

            # Sauvegarder les intensités et les points temporels
            intensity_data = {
                "time_points": time_flat.cpu().detach().numpy().tolist(),
                "intensities": intensities_flat.cpu().detach().numpy().tolist(),
                "marked_times": {
                    str(dim): times.cpu().detach().numpy().tolist()
                    for dim, times in marked_times.items()
                },
                "metadata": {
                    "precision": precision,
                    "start_time": start_time,
                    "end_time": end_time,
                    "num_event_types": num_mark,
                    "model_type": self.__class__.__name__,
                },
            }

            data_file = f"{self.__class__.__name__}_intensity_data.json"
            data_file = os.path.join(save_dir, data_file)

            save_json(intensity_data, data_file)
            logger.info(f"Données d'intensité sauvegardées dans {data_file}")

        # Affichage et/ou sauvegarde du graphe si demandé
        if plot or save_plot:
            # Create directory if it doesn't exist
            if save_plot:
                os.makedirs(save_dir, exist_ok=True)

            # Liste de marqueurs pour distinguer les événements
            markers = ["o", "D", ",", "x", "+", "^", "v", "<", ">", "s", "p", "*"]

            for i in range(num_mark):
                # Créer une nouvelle figure pour chaque type d'événement
                fig, ax = plt.subplots(1, 1, figsize=(12, 6))

                # Tracé de l'intensité en fonction du temps
                ax.plot(
                    time_flat.cpu().detach().numpy(),
                    intensities_flat[:, i].cpu().detach().numpy(),
                    color=f"C{i}",
                    linewidth=2,
                    label=f"Intensity Mark {i}",
                )

                # Ajout des événements observés sous forme de points
                if len(marked_times[i]) > 0:
                    ax.scatter(
                        marked_times[i].cpu().detach().numpy(),
                        torch.zeros_like(marked_times[i]).cpu().detach().numpy()
                        - 0.05 * intensities_flat[:, i].max().item(),
                        s=30,
                        color=f"C{i}",
                        marker=markers[i % len(markers)],
                        label=f"Events Mark {i}",
                        alpha=0.8,
                    )

                ax.set_title(f"Intensité pour la marque {i}")
                ax.set_xlabel("Temps")
                ax.set_ylabel("Intensité")
                ax.legend()
                ax.grid(True, alpha=0.3)

                plt.tight_layout()

                # Sauvegarder le graphique si demandé
                if save_plot:
                    save_file = (
                        f"{self.__class__.__name__}_intensity_graph_mark_{i}.png"
                    )
                    save_file = os.path.join(save_dir, save_file)
                    plt.savefig(save_file, dpi=150, bbox_inches="tight")
                    logger.info(
                        f"Graphique d'intensité pour la marque {i} sauvegardé dans {save_file}"
                    )

                # Afficher le graphique si demandé
                if plot:
                    plt.show()
                else:
                    plt.close()

        return intensities_flat, time_flat, marked_times

    def get_model_metadata(self):
        """
        Get metadata about the model for simulation purposes.

        Returns:
            dict: Dictionary containing model metadata
        """
        metadata = {
            "model_type": self.__class__.__name__,
            "hidden_size": self.hidden_size,
            "num_event_types": self.num_event_types,
            "lr": self.lr,
        }

        # Add additional parameters that are specific to the model
        if hasattr(self, "hparams"):
            # Extract hyperparameters but exclude complex objects like modules
            for key, value in self.hparams.items():
                if (
                    isinstance(value, (int, float, str, bool, list, dict))
                    or value is None
                ):
                    if key not in metadata:  # Avoid overriding existing keys
                        metadata[key] = value

        return metadata

    def predict_one_step(
        self,
        time_seq: torch.Tensor,
        time_delta_seq: torch.Tensor,
        event_seq: torch.Tensor,
        mode: Literal["train", "simulation"] = "train",
        num_sample: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Utility method to predict the next time delta using the event sampler.

        Args:
            time_seq (torch.Tensor): Time sequence [batch_size, seq_len]
            time_delta_seq (torch.Tensor): Time delta sequence [batch_size, seq_len]
            event_seq (torch.Tensor): Event type sequence [batch_size, seq_len]
            compute_last_step_only (bool): Whether to compute only the last step

        Returns:
            torch.Tensor: Predicted time deltas [batch_size, 1] if compute_last_step_only=True else [batch_size, seq_len]
            torch.Tensor: Predicted event types [batch_size, 1] if compute_last_step_only=True else [batch_size, seq_len]
        """
        # Determine possible event times
        dtime_boundary = time_delta_seq + self.dtime_max
        accepted_dtimes, weights = self.event_sampler(
            num_sample=num_sample,
            mode=mode
        ).draw_next_time_one_step(
            time_seq,
            dtime_boundary,
            event_seq,
            dtime_boundary,
            self.compute_intensities_at_sample_times,
            compute_last_step_only=True,
        )

        # Estimate next time delta
        dtimes_pred = torch.sum(accepted_dtimes * weights, dim=-1)  # [batch_size, 1]
        batch_size, num_mark = time_seq.size(0), self.num_event_types

        # Select next event type based on intensities
        intensities_at_times = self.compute_intensities_at_sample_times(
            time_seq,
            time_delta_seq,
            event_seq,
            dtimes_pred[:, :, None],
            compute_last_step_only=True,
        ).view(
            batch_size, num_mark
        )  # [batch_size, num_event_types]

        total_intensities = intensities_at_times.sum(dim=-1)

        if torch.any(total_intensities == 0):
            raise ValueError("Total intensities is null, simulation stops.")

        probs = intensities_at_times / total_intensities[:, None]
        type_pred = torch.multinomial(probs, num_samples=1)

        return dtimes_pred, type_pred
