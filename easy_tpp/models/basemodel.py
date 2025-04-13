""" Base model with common functionality using PyTorch Lightning """
from collections import defaultdict
from abc import ABC, abstractmethod
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from matplotlib import pyplot as plt
import os


from easy_tpp.models.thinning import EventSampler
from easy_tpp.config_factory import ModelConfig
from easy_tpp.evaluate import TPPMetricsCompute, EvaluationMode


class BaseModel(pl.LightningModule, ABC):
    
    def __init__(self, model_config : ModelConfig, **kwargs):
        """Initialize the BaseModel

        Args:
            model_config (EasyTPP.ModelConfig): model spec of configs
        """
        super(BaseModel, self).__init__()
        
        #Paramètre du modele
        self.save_hyperparameters()
        
        self.lr = kwargs.get('lr')
        self.lr_scheduler = kwargs.get("lr_scheduler")
        self.max_epochs = kwargs.get("max_epochs")
        self.num_event_types = kwargs.get('num_event_types')  # not include [PAD], [BOS], [EOS]
        
        if isinstance(self.num_event_types, int):
            self.num_event_types_pad = kwargs.get('num_event_types_pad', self.num_event_types + 1) # include [PAD], [BOS], [EOS]
            
        self.pad_token_id = kwargs.get('pad_token_id', self.num_event_types)
        
        
        self.loss_integral_num_sample_per_step = model_config.loss_integral_num_sample_per_step
        self.hidden_size = model_config.hidden_size
        
        self.eps = torch.finfo(torch.float32).eps

        self.layer_type_emb = nn.Embedding(
            self.num_event_types_pad,  # have padding
            self.hidden_size,
            padding_idx = self.pad_token_id)

        #Paramètre de la génération de données
        self.gen_config = model_config.thinning
        self.event_sampler = None
        self.use_mc_samples = model_config.use_mc_samples
        
        # Set up the event sampler if generation config is provided
        if self.gen_config:
            self.event_sampler = EventSampler(num_sample = self.gen_config.num_sample,
                                              num_exp = self.gen_config.num_exp,
                                              over_sample_rate = self.gen_config.over_sample_rate,
                                              patience_counter = self.gen_config.patience_counter,
                                              num_samples_boundary = self.gen_config.num_samples_boundary,
                                              dtime_max = self.gen_config.dtime_max,
                                              device = self.device)
        
        simulation_config = model_config.simulation_config
        self.simulation_start_time = simulation_config.get('start_time')
        self.simulation_end_time = simulation_config.get('end_time')
    
    @staticmethod
    def generate_model_from_config(model_config : ModelConfig, **kwargs):
        """Generate the model in derived class based on model config.

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.
        """
        model_id = model_config.model_id

        for subclass in BaseModel.__subclasses__():
            if subclass.__name__ == model_id:
                return subclass(model_config, **kwargs)

        raise RuntimeError('No model named ' + model_id)

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
        dtimes_ratio_sampled = torch.linspace(start=0.0,
                                              end=1.0,
                                              steps=self.loss_integral_num_sample_per_step,
                                              device=self.device)[None, None, :]

        # [batch_size, max_len, n_samples]
        sampled_dtimes = time_delta_seq[:, :, None] * dtimes_ratio_sampled

        return sampled_dtimes
    
    def compute_loglikelihood(self, time_delta_seq, lambda_at_event, lambdas_loss_samples, seq_mask, type_seq):
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
        lambdas_loss_samples = lambdas_loss_samples + self.eps

        log_marked_event_lambdas = lambda_at_event.log()
        total_sampled_lambdas = lambdas_loss_samples.sum(dim=-1)

        # Compute event LL - [batch_size, seq_len]
        event_ll = -F.nll_loss(
            log_marked_event_lambdas.permute(0, 2, 1),  # mark dimension needs to come second, not third to match nll_loss specs
            target=type_seq,
            ignore_index=self.pad_token_id,  # Padded events have a pad_token_id as a value
            reduction='none', # Does not aggregate, and replaces what would have been the log(marked intensity) with 0.
        )

        # Compute non-event LL [batch_size, seq_len]
        # interval_integral = length_interval * average of sampled lambda(t)
        if self.use_mc_samples:
            non_event_ll = total_sampled_lambdas.mean(dim=-1) * time_delta_seq * seq_mask
        else: # Use trapezoid rule
            non_event_ll = 0.5 * (total_sampled_lambdas[..., 1:] + total_sampled_lambdas[..., :-1]).mean(dim=-1) * time_delta_seq * seq_mask

        num_events = torch.masked_select(event_ll, event_ll.ne(0.0)).size()[0]
        return event_ll, non_event_ll, num_events
    
    @abstractmethod
    def loglike_loss(
        self,
        batch : tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,]
        )-> tuple[torch.Tensor, int]:
        
        """Compute the log-likelihood loss for a batch of data.
        
        Args:
            batch: Contains time_seq, time_delta_seq, event_seq, batch_non_pad_mask, batch_attention_mask
            
        Returns:
            loss, number of events.
        """
        pass
        
    @abstractmethod
    def compute_intensities_at_sample_times(self, **kwargs):
        """Compute intensities at sample times.
        
        Implementation should be provided by subclasses.
        """
        pass
    
    def configure_optimizers(self):
        """Configure the optimizer for the model.
            
        Returns:
            optimizer: The optimizer to use for training.
        """
        
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
            
        # Use cosine decay scheduler instead
        if hasattr(self, 'lr_scheduler') and self.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.max_epochs,  # Total number of epochs
                eta_min=self.lr * 0.01  # Minimum learning rate at the end of schedule
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": None
                }
            
        return optimizer
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """Training step for Lightning.
        
        Args:
            batch: Contains time_seq, time_delta_seq, event_seq, batch_non_pad_mask, batch_attention_mask
            batch_idx: Index of the batch
            
        Returns:
            STEP_OUTPUT: The output of the training step
        """
        
        batch = batch.values()
        loss,_ = self.loglike_loss(batch)
        avg_loss = loss/batch[0].numel()
        self.log('avg_train_loss', avg_loss.item(), prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """Validation step for Lightning.
        
        Args:
            batch: Contains time_seq, time_delta_seq, event_seq, batch_non_pad_mask, batch_attention_mask
            batch_idx: Index of the batch
            
        Returns:
            STEP_OUTPUT: The output of the validation step
        """
        
        batch = batch.values()
        
        label_batch = [seq[:,1:] for seq in batch]
        
        loss,_ = self.loglike_loss(batch)
        avg_loss = loss/batch[0].numel()
        #Compute some validation metrics
        pred = self.predict_one_step_at_every_event(batch)
        
        one_step_metrics_compute = TPPMetricsCompute(
            num_event_types = self.num_event_types,
            mode = EvaluationMode.PREDICTION
            )
            
        one_step_metrics = one_step_metrics_compute.compute_all_metrics(
            batch = label_batch,
            pred = pred)
        
        self.log('avg_val_loss', avg_loss.item(), prog_bar=True)
        
        for key in one_step_metrics : 
            
            self.log(f"avg_{key}", one_step_metrics[key]/batch[0].numel(), prog_bar=True)
            
        
        return loss
    
    def test_step(
        self,
        batch, 
        batch_idx,
        **kwargs
        ) -> STEP_OUTPUT:
        """Test step for Lightning.
        
        Args:
            batch: Contains time_seq, time_delta_seq, event_seq, batch_non_pad_mask, batch_attention_mask
            batch_idx: Index of the batch
            **kwargs: Additional keyword arguments for customizing test behavior
            
        Returns:
            STEP_OUTPUT: The output of the test step
        """
        
        batch = batch.values()
        label_batch = [seq[:, 1:] for seq in batch]
        
        loss,_ = self.loglike_loss(batch)
        avg_loss = loss/batch[0].numel()
        
        # Override simulation parameters if provided in kwargs
        start_time = kwargs.get('start_time', self.simulation_end_time)
        end_time = kwargs.get('end_time', self.simulation_start_time)
        
        # Compute some prediction metrics
        pred = self.predict_one_step_at_every_event(batch)
        
        one_step_metrics_compute = TPPMetricsCompute(
            num_event_types = self.num_event_types,
            mode = EvaluationMode.PREDICTION
            )
        one_step_metrics = one_step_metrics_compute.compute_all_metrics(
            batch = label_batch,
            pred = pred)

        # Compute simulation metrics
        simulation = self.simulate(
            start_time = start_time,
            end_time = end_time,
            batch = batch,
            batch_size = batch[0].size(0)
        )
        simulation_metrics_compute = TPPMetricsCompute(
            num_event_types = self.num_event_types,
            mode = EvaluationMode.SIMULATION
            )
        simulation_metrics = simulation_metrics_compute.compute_all_metrics(
            batch = label_batch,
            pred = simulation
        )
        
        self.log('avg_test_loss', avg_loss.item(), prog_bar=True)
        
        for key in one_step_metrics : 
            
            self.log(f"avg_{key}", one_step_metrics[key]/batch[0].numel(), prog_bar=True)
        
        for key in simulation_metrics :
            self.log(f"avg_{key}", simulation_metrics[key]/batch[0].numel(), prog_bar=True)
        
        return loss
    

    def predict_one_step_at_every_event(
        self,
        batch : tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
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
        time_seq, time_delta_seq, event_seq = time_seq[:, :-1], time_delta_seq[:, :-1], event_seq[:, :-1]

        # [batch_size, seq_len]
        dtime_boundary = torch.max(time_delta_seq * self.event_sampler.dtime_max,
                                   time_delta_seq + self.event_sampler.dtime_max)

        # [batch_size, seq_len, num_sample]
        accepted_dtimes, weights = self.event_sampler.draw_next_time_one_step(
            time_seq,
            time_delta_seq,
            event_seq,
            dtime_boundary,
            self.compute_intensities_at_sample_times,
            compute_last_step_only=False
        )  # make it explicit

        # We should condition on each accepted time to sample event mark, but not conditioned on the expected event time.
        # 1. Use all accepted_dtimes to get intensity.
        # [batch_size, seq_len, num_sample, num_marks]
        intensities_at_times = self.compute_intensities_at_sample_times(
            time_seq,
            time_delta_seq,
            event_seq,
            accepted_dtimes
        )

        # 2. Normalize the intensity over last dim and then compute the weighted sum over the `num_sample` dimension.
        # Each of the last dimension is a categorical distribution over all marks.
        # [batch_size, seq_len, num_sample, num_marks]
        intensities_normalized = intensities_at_times / intensities_at_times.sum(dim=-1, keepdim=True)

        # 3. Compute weighted sum of distributions and then take argmax.
        # [batch_size, seq_len, num_marks]
        intensities_weighted = torch.einsum('...s,...sm->...m', weights, intensities_normalized)

        # [batch_size, seq_len]
        types_pred = torch.argmax(intensities_weighted, dim=-1)

        # [batch_size, seq_len]
        dtimes_pred = torch.sum(accepted_dtimes * weights, dim=-1)  # compute the expected next event time
        
        return dtimes_pred, types_pred

    def predict_multi_step_since_last_event(
        self,
        batch : tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        forward = False,
        num_step : int = None
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
            num_step = self.gen_config.num_step_gen

        if not forward:
            time_seq = time_seq_label[:, :-num_step]
            time_delta_seq = time_delta_seq_label[:, :-num_step]
            event_seq = event_seq_label[:, :-num_step]
        else:
            time_seq, time_delta_seq, event_seq = time_seq_label, time_delta_seq_label, event_seq_label

        for i in range(num_step):
            # [batch_size, seq_len]
            dtime_boundary = time_delta_seq + self.event_sampler.dtime_max

            # [batch_size, 1, num_sample]
            accepted_dtimes, weights = \
                self.event_sampler.draw_next_time_one_step(time_seq,
                                                           time_delta_seq,
                                                           event_seq,
                                                           dtime_boundary,
                                                           self.compute_intensities_at_sample_times,
                                                           compute_last_step_only=True)

            # [batch_size, 1]
            dtimes_pred = torch.sum(accepted_dtimes * weights, dim=-1)

            # [batch_size, seq_len, 1, event_num]
            intensities_at_times = self.compute_intensities_at_sample_times(time_seq,
                                                                            time_delta_seq,
                                                                            event_seq,
                                                                            dtimes_pred[:, :, None],
                                                                            max_steps=event_seq.size()[1])

            # [batch_size, seq_len, event_num]
            intensities_at_times = intensities_at_times.squeeze(dim=-2)

            # [batch_size, seq_len]
            types_pred = torch.argmax(intensities_at_times, dim=-1)

            # [batch_size, 1]
            types_pred_ = types_pred[:, -1:]
            dtimes_pred_ = dtimes_pred[:, -1:]
            time_pred_ = time_seq[:, -1:] + dtimes_pred_

            # concat to the prefix sequence
            time_seq = torch.cat([time_seq, time_pred_], dim=-1)
            time_delta_seq = torch.cat([time_delta_seq, dtimes_pred_], dim=-1)
            event_seq = torch.cat([event_seq, types_pred_], dim=-1)

        return time_delta_seq[:, -num_step - 1:], event_seq[:, -num_step - 1:], \
               time_delta_seq_label[:, -num_step - 1:], event_seq_label[:, -num_step - 1:]
            
    def simulate(
        self,
        start_time: float,
        end_time: float,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = None,
        batch_size: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        
        Attention pad the sequence on the right side or the predictions will be wrong.
        
        Simulate a sequence of events from the model taking into account batch.
        Returns only events between `start_time` and `end_time`.

        Args:
            start_time (float): Start time of the generated sequence.
            end_time (float): End time of the generated sequence.
            batch (dict[str, torch.Tensor], optional): 
                batch of event sequences, in the form of a dictionary with keys:
                - 'time_seqs': Tensor of dimension [batch_size, seq_len] representing timestamps of past events.
                - 'time_delta_seqs': Tensor [batch_size, seq_len] of time differences between events.
                - 'type_seqs': Tensor [batch_size, seq_len] of event types.

                If `None`, batch is initialized to an empty sequence.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing the generated sequence, with the same keys as `batch`:
                - 'time_seqs': Tensor containing timestamps of generated events.
                - 'time_delta_seqs': Tensor of time differences between events.
                - 'type_seqs': Tensor of generated event types.
        """

        # Initialize sequences
        if batch is None :
            batch = [torch.zeros(batch_size, 2) for _ in range(3)] + [None, None]
        else :
            batch_size = batch[0].size(0)
        
        time_seq_label, time_delta_seq_label, event_seq_label, non_pad_mask, _ = batch
        
        time_seq = time_seq_label.clone()
        time_delta_seq = time_seq_label.clone()
        event_seq = event_seq_label.clone()
        
        num_mark = self.num_event_types
        num_step = 0
        seq_len = 0
        
        last_event_time = torch.zeros(batch_size, num_mark).to(self.device)
        
        for mark in range(num_mark):
            # Create a mask for each mark separately to avoid broadcasting issues
            mark_mask = (event_seq_label == mark)
            masked_time_seq = torch.where(mark_mask, time_seq_label, torch.tensor(0.0).to(self.device))
            marked_last_time_label, _ = masked_time_seq.max(dim=1)
            last_event_time[:,mark] = marked_last_time_label
                    
        min_time = torch.min(last_event_time)
        current_time = min_time.item()
        pbar = tqdm(total = end_time, desc="Simulating sequences", unit="time", leave=False)
        
        while current_time < end_time:
            num_step += 1

            # Determine possible event times
            dtime_boundary = time_delta_seq + self.event_sampler.dtime_max
            accepted_dtimes, weights = self.event_sampler.draw_next_time_one_step(
                time_seq, time_delta_seq, event_seq, dtime_boundary,
                self.compute_intensities_at_sample_times, compute_last_step_only=True
            )

            # Estimate next time delta
            dtimes_pred = torch.sum(accepted_dtimes * weights, dim=-1) #[batch_size, 1]
            time_pred_ = time_seq[:, -1:] + dtimes_pred
            min_time = time_pred_.min()
            current_time = min_time.item()
            pbar.n = min(current_time, end_time)
            pbar.refresh()

            if current_time >= end_time:
                break

            seq_len += 1

            # Select next event type based on intensities
            intensities_at_times = self.compute_intensities_at_sample_times(
                time_seq,
                time_delta_seq,
                event_seq,
                dtimes_pred[:, :, None],
                compute_last_step_only=True
            ).squeeze(1).squeeze(2) #[batch_size, num_event_types]

            total_intensities = intensities_at_times.sum(dim=-1)
            
            if torch.any(total_intensities == 0):
                break
            
            probs = intensities_at_times / total_intensities[:,None]
            type_pred = torch.multinomial(probs, num_samples=1)

            # Update last event times for each mark
            for mark in range(num_mark):
                # Create a boolean mask matching the batch dimension
                mark_mask = (type_pred.squeeze(-1) == mark)
                # Update last event times for this mark where the mask is True
                marked_last_time = torch.where(
                    mark_mask, 
                    time_pred_.squeeze(-1),
                    last_event_time[:,mark]
                )
                last_event_time[:,mark] = marked_last_time
                
            # Get the last event times for the predicted types
            # Gather last event times for the specific types predicted
            batch_indices = torch.arange(batch_size, device=self.device)
            last_events = last_event_time[batch_indices, type_pred.squeeze(-1)]
            last_events = last_events.unsqueeze(-1)  # Add dimension to match time_pred_
            dtimes_pred = time_pred_ - last_events
            
            # Update generated sequences
            time_seq = torch.cat([time_seq, time_pred_], dim=-1)
            time_delta_seq = torch.cat([time_delta_seq, dtimes_pred], dim=-1)
            event_seq = torch.cat([event_seq, type_pred], dim=-1)

        pbar.close()

        first_pred_idx = time_delta_seq.size(1) - seq_len
        
        time_seq = time_seq[:, first_pred_idx:]
        time_delta_seq = time_delta_seq[:, first_pred_idx:]
        event_seq = event_seq[:, first_pred_idx:]
        
        # Create mask for events within the time range
        # Using separate comparisons and combining with logical_and
        simul_mask = torch.logical_and(
            time_seq >= start_time,
            time_seq <= end_time
        )
        
        return time_seq, time_delta_seq, event_seq, simul_mask
    
    
    def intensity_graph(
        self,
        start_time: float = 0.0,
        end_time: float = 30.0,
        precision: int = 20,
        plot: bool = False,
        save_dir: str = './',
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
        """
        Génère et affiche la courbe d'intensité du modèle pour une séquence donnée.

        Cette fonction calcule les intensités du modèle aux instants échantillonnés et 
        permet de visualiser leur évolution en fonction du temps, séparément pour chaque type d'événement.

        Args:
            time_seq (torch.Tensor): Séquence des temps d'événements, de taille [seq_len].
            time_delta_seq (torch.Tensor): Différences de temps entre les événements.
            type_seq (torch.Tensor): Séquence des types d'événements, de taille [seq_len].
            precision (int, optionnel): Nombre de points interpolés entre deux événements 
                pour lisser la courbe d'intensité. Par défaut à 20.
            plot (bool, optionnel): Indique s'il faut afficher le graphique des intensités. 
                Par défaut à False.

        Returns:
            tuple:
                - torch.Tensor: Matrice des intensités calculées pour chaque type d'événement.
                - torch.Tensor: Points de temps correspondant aux échantillons d'intensité.
                - dict[int, torch.Tensor]: Dictionnaire des instants où chaque type d'événement est observé.

        Exemple:
            ```python
            intensities, sample, marked_times = model.intensity_graph(time_seq, time_delta_seq, type_seq, precision=50, plot=True)
            ```
            Cela affichera les courbes d'intensité et retournera les valeurs numériques associées.

        Détails:
        - Les intensités sont calculées aux instants interpolés définis par `precision`.
        - La visualisation affiche une courbe par type d'événement, avec des marqueurs indiquant les événements observés.
        """
        
        num_mark = self.num_mark

        time_delta_seq, type_seq, simul_mask = self.simulate(
            start_time = start_time,
            end_time = end_time,
            history_batch = None,
            batch_size= 1
        )
        
        # Normalisation du temps pour commencer à zéro
        time_seq = time_seq - time_seq[0]

        sample = torch.linspace(
            time_seq[0].item(), time_seq[-1].item(), precision
        ).to(self.device).unsqueeze(0).unsqueeze(-1)

        intensities = self.compute_intensities_at_sample_times(
            time_seq,
            time_delta_seq,
            type_seq,
            sample
        ).squeeze()
        
        marked_times = defaultdict(list)
        for i in range(num_mark):
            marked_times[i] = time_seq[type_seq == i].squeeze()
        
        # Affichage du graphe si demandé
        if plot:
            
            save_file = f'{self._get_name}/intensity_graph.png'
            
            save_file = os.path.join(save_dir, save_file)
            
            fig, axes = plt.subplots(num_mark, 1, figsize=(10, 6))

            # Gestion du cas où num_mark == 1
            if num_mark == 1:
                axes = [axes]

            # Liste de marqueurs pour distinguer les événements
            markers = ['o', 'D', ',', 'x', '+', '^', 'v', '<', '>', 's', 'p', '*']

            for i in range(num_mark):
                ax = axes[i]

                # Tracé de l'intensité en fonction du temps
                ax.plot(sample.numpy(), intensities[:, i].numpy(), color=f'C{i}')

                # Ajout des événements observés sous forme de points
                ax.scatter(marked_times[i].numpy(),
                        torch.zeros_like(marked_times[i]).numpy() - 0.3,
                        s=20, color=f'C{i}',
                        marker=markers[i % len(markers)],
                        label=f'Mark {i}')

                ax.set_title(f"Intensité pour la marque {i}")
                ax.legend()

            plt.tight_layout()
            plt.savefig(save_file)
            plt.show()

        return intensities, sample, marked_times