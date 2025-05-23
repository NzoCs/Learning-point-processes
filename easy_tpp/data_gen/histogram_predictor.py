import numpy as np
import torch
from easy_tpp.utils import logger
from easy_tpp.preprocess.data_loader import TPPDataModule

class HistogramPredictor:
    """
    Cette classe est dédiée à la création de prédictions par histogramme.
    Elle segmente la création des données (histogrammes des marques et temps)
    et l’évaluation par échantillonnage à partir de ces histogrammes.
    """
    
    def __init__(self, data_module: TPPDataModule, split='test'):
        self.data_module = data_module
        self.split = split
        self.num_event_types = data_module.num_event_types
        
        # Setup the data module for the specified split
        if split == 'test':
            self.data_module.setup('test')
        elif split in ['val', 'dev']:
            self.data_module.setup('fit')  # This loads both train and val
        elif split == 'train':
            self.data_module.setup('fit')

    @property
    def loader(self):
        return self.data_module.get_dataloader(split=self.split)
    
    def create_histograms(self) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """
        Crée des histogrammes pour les marques et les temps à partir du test_loader.

        Returns:
            tuple: (mark_hist, time_hist) - histogrammes calculés avec np.histogram()
        """
        all_marks = []
        all_times = []
        
        for batch in self.loader:
            _, time_delta_seqs, type_seqs, _, _ = batch
            
            if isinstance(type_seqs, torch.Tensor):
                type_seqs = type_seqs.cpu().numpy()
            if isinstance(time_delta_seqs, torch.Tensor):
                time_delta_seqs = time_delta_seqs.cpu().numpy()
            
            all_marks.extend(type_seqs.flatten())
            all_times.extend(time_delta_seqs.flatten())
        
        self.loader_exhausted = True
        all_marks = np.array(all_marks)
        all_times = np.array(all_times)
        
        logger.info(f"Création d'histogrammes à partir de {len(all_marks)} marques et {len(all_times)} temps.")
        
        # Histogramme pour les marques
        if len(all_marks) > 0:
            if np.issubdtype(all_marks.dtype, np.integer) or all_marks.dtype == bool:
                mark_min = np.min(all_marks)
                mark_max = np.max(all_marks)
                num_mark_bins = int(mark_max - mark_min + 1)
                mark_hist = np.histogram(all_marks, bins=num_mark_bins, range=(mark_min - 0.5, mark_max + 0.5))
                logger.info(f"Histogramme des marques créé avec {num_mark_bins} bins discrets.")
            else:
                num_mark_bins = int(np.ceil(np.log2(len(all_marks)) + 1))
                mark_hist = np.histogram(all_marks, bins=num_mark_bins)
                logger.info(f"Histogramme des marques créé avec {num_mark_bins} bins continus.")
        else:
            mark_hist = (np.array([]), np.array([0, 1]))
            logger.warning("Aucune marque trouvée pour créer l'histogramme.")
        
        # Histogramme pour les temps
        if len(all_times) > 0:
            num_time_bins = int(np.ceil(np.log2(len(all_times)) + 1))
            time_hist = np.histogram(all_times, bins=num_time_bins)
            logger.info(f"Histogramme des temps créé avec {num_time_bins} bins.")
        else:
            time_hist = (np.array([]), np.array([0, 1]))
            logger.warning("Aucun temps trouvé pour créer l'histogramme.")
        
        return mark_hist, time_hist
    
    def predict(self) -> dict:
        """
        Génère une prédiction basée sur les histogrammes.
        Pour chaque séquence du test_loader, échantillonne des valeurs de marque et de temps
        à partir des probabilités dérivées des histogrammes.

        Returns:
            dict: Contenant les listes des véritables valeurs et des valeurs échantillonnées
                  pour les marques et les temps.
        """
        mark_hist, time_hist = self.create_histograms()
        mark_counts, mark_bins = mark_hist
        time_counts, time_bins = time_hist
        
        # Normalisation pour obtenir des probabilités
        mark_probs = mark_counts / np.sum(mark_counts) if np.sum(mark_counts) > 0 else np.zeros_like(mark_counts)
        time_probs = time_counts / np.sum(time_counts) if np.sum(time_counts) > 0 else np.zeros_like(time_counts)
        
        all_true_marks = []
        all_true_times = []
        all_sampled_marks = []
        all_sampled_times = []
        
        for batch in self.loader:
            _, true_times, true_marks, _, _ = batch
            
            # Ignorer les batchs vides
            if true_marks.numel() == 0 or true_times.numel() == 0:
                continue
            
            batch_size = true_marks.size(0)
            for i in range(batch_size):
                seq_marks = true_marks[i]
                seq_times = true_times[i]
                
                # Filtrer les séquences vides ou en padding
                valid_indices = seq_marks != 0 if true_marks.dtype == torch.long else torch.ones_like(seq_marks, dtype=torch.bool)
                seq_marks = seq_marks[valid_indices]
                seq_times = seq_times[valid_indices]
                
                if seq_marks.numel() == 0 or seq_times.numel() == 0:
                    continue
                
                seq_marks_np = seq_marks.cpu().numpy()
                seq_times_np = seq_times.cpu().numpy()
                
                try:
                    mark_bin_indices = np.random.choice(len(mark_probs), size=seq_marks.numel(), p=mark_probs)
                    sampled_marks = np.array([np.mean([mark_bins[j], mark_bins[j+1]]) for j in mark_bin_indices])
                except Exception as e:
                    logger.error(f"Erreur lors de l'échantillonnage des marques: {e}")
                    continue
                
                try:
                    time_bin_indices = np.random.choice(len(time_probs), size=seq_times.numel(), p=time_probs)
                    sampled_times = np.array([np.mean([time_bins[j], time_bins[j+1]]) for j in time_bin_indices])
                except Exception as e:
                    logger.error(f"Erreur lors de l'échantillonnage des temps: {e}")
                    continue
                
                # Si les marques sont discrètes, on arrondit les valeurs échantillonnées
                if seq_marks.dtype in [torch.int32, torch.int64, torch.bool]:
                    sampled_marks = np.round(sampled_marks).astype(int)
                
                all_true_marks.append(seq_marks.cpu().tolist())
                all_true_times.append(seq_times.cpu().tolist())
                all_sampled_marks.append(sampled_marks.tolist())
                all_sampled_times.append(sampled_times.tolist())
        
        
        return {
            'sampled_marks': all_sampled_marks,
            'sampled_times': all_sampled_times
        }
