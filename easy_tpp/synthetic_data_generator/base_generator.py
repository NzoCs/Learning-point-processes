import matplotlib.pyplot as plt
from collections import defaultdict
from torchtyping import TensorType
from typing import Optional
import os
from tqdm import tqdm
import torch
from abc import ABC, abstractmethod

from easy_tpp.utils import save_json, format_gen_data_to_hf, logger
from easy_tpp.synthetic_data_generator.synthetic_data_thinning import SynEventSampler
from easy_tpp.config_factory import SynGenConfig

class BaseGenerator(torch.nn.Module, ABC):
    
    def __init__(self, gen_config : SynGenConfig) -> None:

        super(BaseGenerator, self).__init__()
        
        os.makedirs(gen_config.save_dir, exist_ok=True)
        self.save_dir = gen_config.save_dir
        
        self.experiment_id = gen_config.experiment_id
        self.start_time = gen_config.start_time
        self.end_time = gen_config.end_time
        self.test_end_time = gen_config.test_end_time
        self.start_time = gen_config.start_time
        self.num_batch = gen_config.num_batch
        self.batch_size = gen_config.batch_size
        self.num_event_types = gen_config.num_mark
        self.device = gen_config.device
        
        sampler_config = gen_config.sampler_config
        # Set up the event sampler if generation config is provided
        self.event_sampler = SynEventSampler(**sampler_config, device = self.device)
        
        # Move model to the specified device
        self.to(self.device)
    
    def set_num_mark(self, num_mark : int):
        self.num_event_types = num_mark
    
    def set_start_time(self, start_time : float) : 
        self.start_time = start_time
    
    def set_test_end_time(self, test_end_time : float) :
        self.test_end_time = test_end_time
        
    def set_end_time(self, end_time : float) : 
        self.end_time = end_time
    
    def set_num_batch(self, num_batch : float) : 
        self.num_batch  = num_batch
    
    # Helper method to ensure tensors are on the correct device
    def ensure_tensor_on_device(self, tensor):
        if tensor is not None and isinstance(tensor, torch.Tensor) and tensor.device != self.device:
            return tensor.to(self.device)
        return tensor
        
    @abstractmethod
    def compute_intensities_at_sample_times(
        self,
        batch : tuple[TensorType["batch_size", "seq_len"], TensorType["batch_size", "seq_len"], TensorType["batch_size", "seq_len"], TensorType["batch_size", "seq_len"]],
        sample_batch : tuple[TensorType["batch_size", "sample_len", "num_sample"], TensorType["batch_size", "sample_len", "num_sample"]],
        **kwargs
        ) -> TensorType["batch_size", "sample_len", "num_sample", "num_mark"] : 
        pass
    
    @staticmethod
    def generate_model_from_config(gen_config : SynGenConfig) -> 'BaseGenerator':
        """Generate the model in derived class based on model config.

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.
        """
        model_id = gen_config.model_id

        for subclass in BaseGenerator.__subclasses__():
            if subclass.__name__ == model_id:
                return subclass(gen_config)

        raise RuntimeError('No model named ' + model_id)
    
    def sample(
        self,
        save_file : str = None,
        history_batch = None,
        save : bool = True
    ) -> dict[str, list[torch.Tensor]]:
        
        """
        Génère des séquences d'événements en échantillonnant un modèle de processus ponctuel.

        Cette méthode simule `num_seq` séquences d'événements temporels selon le modèle sous-jacent.
        Chaque séquence contient les types d'événements, les timestamps des événements et les 
        différences de temps entre événements successifs.

        Args:
            history (dict[str, torch.Tensor], optionnel) : 
                Un dictionnaire contenant un historique des événements sous forme de tenseurs PyTorch.
                Cela peut être utilisé pour conditionner la génération des séquences d'événements.
                Par défaut, `None`, ce qui signifie qu'aucune condition initiale n'est imposée.

        Returns:
            dict[str, list[torch.Tensor]] : 
                Un dictionnaire contenant :
                - `'type_seqs'` : une liste de tenseurs représentant les types d'événements générés.
                - `'time_seqs'` : une liste de tenseurs représentant les timestamps des événements générés.
                - `'time_delta_seqs'` : une liste de tenseurs représentant les temps inter-événements.
        """
        
        # Initialize sequences
        if history_batch is None :
            try :
                history_batch = [torch.zeros(self.batch_size, 2, device=self.device) for _ in range(3)] + [None, None]
            except :
                raise AttributeError("batch_size must be provided in the config")
        else:
            # Ensure all tensors in history_batch are on the correct device
            history_batch = [self.ensure_tensor_on_device(tensor) for tensor in history_batch]
        
        type_seqs = []
        time_delta_seqs = []
        simul_masks = []
        time_seqs = []
        
        num_batch = self.num_batch

        pbar = tqdm(total=num_batch, desc="Simulating events", unit="seq", leave=False)
        
        for _ in range(num_batch):
            pbar.update(1)
            events = self.simulate(history_batch = history_batch)
            
            time_seq, time_delta_seq_, type_seq_, simul_mask_ = events
            
            time_seqs.append(time_seq)
            type_seqs.append(type_seq_)
            time_delta_seqs.append(time_delta_seq_)
            simul_masks.append(simul_mask_)
            
        pbar.close()
        
        sample = {
            'time_seqs': time_seqs,
            'type_seqs': type_seqs,
            'time_delta_seqs': time_delta_seqs,
            'simul_mask': simul_masks
        }

        if save : 
            
            if not save_file : 
                
                # Replace the undefined date_time() function
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Write to JSON file
                save_file = f"{self.experiment_id}/{timestamp}.json"
            
            filepath = os.path.join(self.save_dir, save_file)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            sample_format = format_gen_data_to_hf(
                sample,
                dim_process = self.num_event_types
            )  # Added missing closing parenthesis
            
            # Save the formatted data
            save_json(sample_format, filepath)
            
            message = f'sample successfully saved in {filepath}'
            
            logger.info(message)

        
        return sample


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
        # Ensure all tensors in the batch are on the correct device
        batch = [self.ensure_tensor_on_device(tensor) for tensor in batch]
        time_seq, time_delta_seq, event_seq, batch_non_pad_mask, _ = batch

        # remove the last event, as the prediction based on the last event has no label
        # note: the first dts is 0
        # [batch_size, seq_len]
        time_seq, time_delta_seq, event_seq = time_seq[:, :-1], time_delta_seq[:, :-1], event_seq[:, :-1]
        

        # [batch_size, seq_len, num_sample]
        accepted_dtimes, weights = self.event_sampler.draw_next_time_one_step(
            time_seq,
            time_delta_seq,
            event_seq,
            self.compute_intensities_at_sample_times,
            compute_last_step_only=False
        )  # make it explicit
        
        
        
        accepted_times = time_seq + accepted_dtimes

        # We should condition on each accepted time to sample event mark, but not conditioned on the expected event time.
        # 1. Use all accepted_dtimes to get intensity.
        # [batch_size, seq_len, num_sample, num_marks]
        intensities_at_times = self.compute_intensities_at_sample_times(
            time_seq,
            time_delta_seq,
            event_seq,
            accepted_times
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
        
        
        
    # def simulate(
    #     self,
    #     history_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = None,
    #     batch_size: Optional[int] = None,
    # ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """
        
    #     Attention pad the sequence on the right side or the predictions will be wrong.
        
    #     Simulate a sequence of events from the model taking into account batch.
    #     Returns only events between `start_time` and `end_time`.

    #     Args:
    #         start_time (float): Start time of the generated sequence.
    #         end_time (float): End time of the generated sequence.
    #         batch (dict[str, torch.Tensor], optional): 
    #             batch of event sequences, in the form of a dictionary with keys:
    #             - 'time_seqs': Tensor of dimension [batch_size, seq_len] representing timestamps of past events.
    #             - 'time_delta_seqs': Tensor [batch_size, seq_len] of time differences between events.
    #             - 'type_seqs': Tensor [batch_size, seq_len] of event types.

    #             If `None`, batch is initialized to an empty sequence.

    #     Returns:
    #         dict[str, torch.Tensor]: Dictionary containing the generated sequence, with the same keys as `batch`:
    #             - 'time_delta_seqs': Tensor of time differences between events.
    #             - 'type_seqs': Tensor of generated event types.
    #             - 'simul_mask': Tensor of booleans indicating if the event is in the simulation.
    #     """

    #     start_time= self.start_time
    #     end_time = self.end_time
                
    #     # Initialize sequences
    #     if history_batch is None:
    #         try:
    #             batch = [torch.zeros(batch_size, 2, device=self.device) for _ in range(3)] + [None, None]
    #         except:
    #             raise AttributeError("batch_size must be provided in the config")
    #     else:
    #         # Ensure all tensors in history_batch are on the correct device
    #         batch = [self.ensure_tensor_on_device(tensor) for tensor in history_batch]
    #         batch_size = batch[0].size(0)
            
        
    #     time_seq_label, time_delta_seq_label, event_seq_label, non_pad_mask, _ = batch
    #     current_time = time_seq_label[:, -1].min().item()
        
    #     time_seq = time_seq_label.clone()
    #     time_delta_seq = time_seq_label.clone()
    #     event_seq = event_seq_label.clone()
        
    #     num_mark = self.num_event_types
    #     seq_len = 0
        
    #     sim_events = {}
    #     last_time_label = {}
        
    #     # Récuperation pour chaque marque du dernier temps d'événement dans la sequence
    #     for mark in range(num_mark):
            
    #         for batch_idx in range(batch_size):
                
    #             sim_events[batch_idx, mark] = []
    #             last_time_label[batch_idx, mark] = 0
                
    #             # Parcourir la séquence de la fin vers le début pour trouver le dernier événement de type mark
    #             for inv_seq_idx in range(event_seq_label.size(1)):
    #                 cur_mark = event_seq_label[batch_idx, -1-inv_seq_idx]
    #                 if cur_mark == mark:
    #                     last_time_label[batch_idx, mark] = time_seq_label[batch_idx, -1-inv_seq_idx]
    #                     break
        
        
    #     pbar = tqdm(total=end_time - current_time, desc="Simulating events", unit="time", leave=False)

    #     while current_time < end_time:

    #         # Determine possible event times
    #         accepted_dtimes, weights = self.event_sampler.draw_next_time_one_step(
    #             time_seq, time_delta_seq, event_seq,
    #             self.compute_intensities_at_sample_times, compute_last_step_only=True
    #         )

    #         # Estimate next time delta
            
    #         dtimes_pred = torch.sum(accepted_dtimes * weights, dim=-1) #[batch_size, 1]
    #         time_pred_ = time_seq[:, -1:] + dtimes_pred
            
    #         min_time = torch.min(time_pred_).item()
    #         current_increment = min_time - current_time
    #         current_time = min_time
    #         pbar.update(min(current_increment, end_time - current_time))

    #         if current_time >= end_time:
    #             break

    #         seq_len += 1

    #         # Select next event type based on intensities
    #         intensities_at_times = self.compute_intensities_at_sample_times(
    #             time_seq,
    #             time_delta_seq,
    #             event_seq,
    #             time_pred_[:, :, None],
    #             max_steps=event_seq.size()[1],
    #             compute_last_step_only=True
    #         ).squeeze(1).squeeze(1) #[batch_size, num_event_types]

    #         total_intensities = intensities_at_times.sum(dim=-1)
            
    #         try:
    #             probs = intensities_at_times / total_intensities.unsqueeze(-1) # [batch_size, num_event_types]
                
    #         except ZeroDivisionError as e: 
    #             logger.warning(f"ZeroDivisionError: {e}. Setting probs to uniform distribution.")
    #             break
                
                
    #         type_pred = torch.multinomial(probs, num_samples=1, replacement=True)  # [batch_size, 1]
    #         marks = type_pred.squeeze(-1).detach().cpu().tolist()

    #         last_event_times = torch.zeros(batch_size, device=self.device)

    #         # delta temps par marque
    #         for batch_idx, mark in enumerate(marks):
                
    #             if len(sim_events[batch_idx, mark]) > 0:
    #                 last_event_times[batch_idx] = sim_events[batch_idx, mark][-1]
    #             else:
    #                 last_event_times[batch_idx] = last_time_label[batch_idx, mark]
                
    #             # Properly extract the scalar time value
    #             sim_events[batch_idx, mark].append(time_pred_[batch_idx, :])
            
    #         dtimes_pred = time_pred_ - last_event_times.unsqueeze(-1)  # [batch_size, 1]

    #         # Update generated sequences
    #         time_seq = torch.cat([time_seq, time_pred_], dim=-1)
    #         time_delta_seq = torch.cat([time_delta_seq, dtimes_pred], dim=-1)
    #         event_seq = torch.cat([event_seq, type_pred], dim=-1)

    #     pbar.close()

    #     first_pred_idx = time_delta_seq.size(1) - seq_len
        
    #     time_seq = time_seq[:, first_pred_idx:]
    #     time_delta_seq = time_delta_seq[:, first_pred_idx:]
    #     event_seq = event_seq[:, first_pred_idx:]
        
    #     simul_mask = (start_time <= time_seq) & (time_seq <= end_time)
        
    #     return time_seq, time_delta_seq, event_seq, simul_mask
        
    def generate_split(self):
        """
        Génère et sauvegarde des ensembles de données d'entraînement, de validation et de test.

        Cette fonction divise les données générées en trois ensembles :
        - 60% pour l'entraînement (`train.json`).
        - 20% pour la validation (`dev.json`).
        - 20% pour le test (`test.json`).

        Chaque ensemble est généré en ajustant dynamiquement `num_seq` et en enregistrant les 
        échantillons sous forme de fichiers JSON dans le répertoire spécifié.

        Args:
            target_dir (str): Chemin du répertoire où sauvegarder les fichiers générés.
            experiment_id (str): Identifiant unique de l'expérience, utilisé pour structurer les fichiers de sortie.

        Returns:
            None: Les fichiers sont directement sauvegardés sur le disque.

        Exemple:
            ```python
            model = MyEventModel()
            model.generate_split(target_dir="data", experiment_id="exp_001")
            ```
            Cela créera les fichiers suivants :
            - `data/exp_001/train.json`
            - `data/exp_001/dev.json`
            - `data/exp_001/test.json`
        """
        
        # Définition des proportions pour train, dev et test
        num_train_batch = int(self.num_batch * 0.6)
        num_dev_batch = int(self.num_batch * 0.2)

        # Génération et sauvegarde de l'ensemble d'entraînement
        self.set_num_batch(num_train_batch)
        self.sample(save_file='train.json')

        # Génération et sauvegarde de l'ensemble de validation
        self.set_num_batch(num_dev_batch)
        self.sample(save_file='dev.json')

        # Génération et sauvegarde de l'ensemble de test
        if self.test_end_time is None:
            self.set_test_end_time(self.end_time)
            
        self.set_end_time(self.test_end_time)
        self.sample(save_file='test.json')

        print(f'Data split successfully saved in {self.save_dir}')
        
        
    def intensity_graph(
        self,
        plot: bool = False,
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
        
        num_mark = self.num_event_types

        time_delta_seq, type_seq, _, _ = self.simulate(
            history_batch=None,
            batch_size=1
        )
        
        time_seq = time_delta_seq.cumsum(dim=-1)
        start = time_seq[:,0].item()
        end = time_seq[:,-1].item()
        
        precision = time_seq.size(1)*20
        
        sample_time = torch.ones(precision, device=self.device)*(end - start)/precision

        intensities = self.compute_intensities_at_sample_times(
            time_seq,
            time_delta_seq,
            type_seq,
            sample_time.unsqueeze(0).unsqueeze(-1)
        ).squeeze(0).squeeze(-2)
        
        marked_times = defaultdict(list)
        for i in range(num_mark):
            marked_times[i] = time_seq[type_seq == i].squeeze()
        
        # Affichage du graphe si demandé
        if plot:
            
            save_file = f'{self.experiment_id}/intensity_graph.png'
            save_file = os.path.join(self.save_dir, save_file)
            
            # Création du répertoire parent s'il n'existe pas
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            
            fig, axes = plt.subplots(num_mark, 1, figsize=(10, 6))

            # Gestion du cas où num_mark == 1
            if num_mark == 1:
                axes = [axes]

            # Liste de marqueurs pour distinguer les événements
            markers = ['o', 'D', ',', 'x', '+', '^', 'v', '<', '>', 's', 'p', '*']

            for i in range(num_mark):
                ax = axes[i]

                # Tracé de l'intensité en fonction du temps
                ax.plot(sample_dt.cumsum(0).numpy(), intensities[:, i].numpy(), color=f'C{i}')

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

        return intensities, sample_dt.cumsum(0), marked_times
