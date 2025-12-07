import torch
import torch.nn.functional as F
from torch import nn
from typing import Union, Tuple, Optional, List
from tqdm import tqdm

from new_ltpp.models.basemodel import Model
from new_ltpp.shared_types import Batch
from new_ltpp.models.mixins.simulation_mixin import Buffers, SimulationState

class SelfCorrecting(Model):
    """
    PyTorch implementation of the Self-Correcting Point Process model.
    Intensity for type i: lambda_i(t) = exp(mu_i + alpha_i * (t - N_i(t)))
    where N_i(t) is the number of events of type i occurred strictly before time t.
    """

    def __init__(
        self,
        *,
        mu: Union[List[float], torch.Tensor],
        alpha: Union[List[float], torch.Tensor],
        **kwargs,
    ):
        super(SelfCorrecting, self).__init__(**kwargs)

        # Validation et conversion des paramètres
        mu = torch.as_tensor(mu, dtype=torch.float32, device=self.device)
        alpha = torch.as_tensor(alpha, dtype=torch.float32, device=self.device)

        if mu.shape[0] != self.num_event_types or alpha.shape[0] != self.num_event_types:
            raise ValueError(
                f"Dimension mismatch. Expected ({self.num_event_types},). "
                f"Got mu: {mu.shape}, alpha: {alpha.shape}"
            )

        self.mu = nn.Parameter(mu.view(self.num_event_types))
        self.alpha = nn.Parameter(alpha.view(self.num_event_types))

    def _get_cumulative_counts(self, type_seq: torch.Tensor) -> torch.Tensor:
        """
        Calcule efficacement N_i(t) pour chaque pas de temps.
        Returns:
            torch.Tensor: [Batch, Seq_Len, Num_Types]
            counts[b, k, i] = nombre d'événements de type i dans type_seq[b, :k+1]
        """
        # [Batch, Seq, Num_Types]
        type_one_hot = F.one_hot(type_seq.long(), num_classes=self.num_event_types).float()
        
        # Cumsum le long de la dimension temps
        cumulative_counts = torch.cumsum(type_one_hot, dim=1)
        return cumulative_counts

    def compute_intensities_at_sample_times(
        self,
        *,
        time_seq: torch.Tensor,
        type_seq: torch.Tensor,
        sample_dtimes: torch.Tensor,
        compute_last_step_only: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Calcule lambda(t + delta) de manière vectorisée.
        """
        # 1. Récupération des compteurs historiques N_i(t)
        # counts_history[b, k] contient le compte INCLUANT l'événement k.
        counts_at_events = self._get_cumulative_counts(type_seq) # [B, L, D]

        if compute_last_step_only:
            # On prend le temps du dernier événement et les comptes associés
            base_time = time_seq[:, -1:].unsqueeze(-1) # [B, 1, 1]
            base_counts = counts_at_events[:, -1:, :].unsqueeze(2) # [B, 1, 1, D]
            
            # Si sample_dtimes est [B, 1, N_samples] ou [B, L, N_samples], on adapte
            if sample_dtimes.dim() == 3 and sample_dtimes.shape[1] != 1:
                sample_dtimes = sample_dtimes[:, -1:, :]
            
            # Temps absolu t = t_last + delta
            # [B, 1, N_samples] -> [B, 1, N_samples, 1]
            current_times = (base_time + sample_dtimes).unsqueeze(-1)
            
        else:
            # Pour toute la séquence
            base_time = time_seq.unsqueeze(-1) # [B, L, 1]
            base_counts = counts_at_events.unsqueeze(2) # [B, L, 1, D]
            
            # Temps absolu t = t_k + delta
            # [B, L, N_samples, 1]
            current_times = (base_time + sample_dtimes).unsqueeze(-1)

        # 2. Calcul de l'intensité
        # Formule : exp(mu + alpha * (t - N(t)))
        # N(t) ici est le nombre d'événements *strictement avant* t.
        # Puisque t > t_k (car delta > 0), N(t) inclut l'événement k.
        # Donc base_counts (qui est le cumsum incluant k) est correct.
        
        # [1, 1, 1, D]
        mu = self.mu.view(1, 1, 1, -1)
        alpha = self.alpha.view(1, 1, 1, -1)
        
        # exponent: [B, L, N_samples, D]
        # Broadcasting: t (..., 1) - N (..., D)
        exponent = mu + alpha * (current_times - base_counts)
        
        intensities = torch.exp(exponent)
        
        # Sécurité numérique optionnelle
        # intensities = torch.clamp(intensities, min=1e-9) 
        
        return intensities

    def loglike_loss(self, batch: Batch) -> Tuple[torch.Tensor, int]:
        """
        Calcule la log-vraisemblance exacte (analytique).
        LL = sum(log(lambda(t_i))) - int(lambda(t) dt)
        """
        time_seq = batch.time_seqs
        type_seq = batch.type_seqs
        
        # Masque (on ignore le padding et le premier événement qui sert juste d'ancrage t0)
        # [Batch, L-1]
        seq_mask = batch.seq_non_pad_mask[:, 1:]
        
        # --- Préparation des Données ---
        
        # 1. Compteurs N(t)
        # [Batch, L, D]
        all_counts = self._get_cumulative_counts(type_seq)
        
        # Pour prédire l'événement k (au temps t_k), on utilise l'historique jusqu'à k-1.
        # Donc N(t_k) = counts[k-1]
        # [Batch, L-1, D]
        N_prev = all_counts[:, :-1, :]
        
        # Temps des événements cibles t_1 ... t_N
        # [Batch, L-1, 1]
        t_target = time_seq[:, 1:].unsqueeze(-1)
        
        # Paramètres reshaped [1, 1, D]
        mu = self.mu.view(1, 1, -1)
        alpha = self.alpha.view(1, 1, -1)

        # --- A. Log-Vraisemblance des Événements (Event Log-Likelihood) ---
        
        # lambda(t_k) = exp(mu + alpha * (t_k - N(t_k)))
        # [Batch, L-1, D]
        exponent_at_event = mu + alpha * (t_target - N_prev)
        lambda_at_event = torch.exp(exponent_at_event)
        
        # On sélectionne l'intensité du type qui s'est RÉELLEMENT produit
        target_types = type_seq[:, 1:].long().unsqueeze(-1) # [Batch, L-1, 1]
        
        # [Batch, L-1]
        lambda_target = torch.gather(lambda_at_event, dim=-1, index=target_types).squeeze(-1)
        
        # Log(lambda)
        event_ll = torch.log(lambda_target + 1e-9)

        # --- B. Intégrale (Non-Event Log-Likelihood) ---
        
        # On intègre sur les intervalles [t_{k-1}, t_k].
        # Durée de l'intervalle : t_k - t_{k-1} (mais on utilise les temps absolus pour la formule)
        # Dans l'intervalle (t_{k-1}, t_k), le compte N(t) est constant et vaut N_prev (comptes jusqu'à k-1).
        
        t_start = time_seq[:, :-1].unsqueeze(-1) # t_{k-1}
        t_end   = time_seq[:, 1:].unsqueeze(-1)  # t_k
        
        # Calcul analytique de l'intégrale sur [t_start, t_end]
        # Int = (1/alpha) * [ lambda(t_end) - lambda(t_start) ]
        # Attention: lambda ici est calculé avec le N courant (N_prev)
        
        # Lambda au début de l'intervalle (juste après l'événement précédent)
        lambda_start = torch.exp(mu + alpha * (t_start - N_prev))
        
        # Lambda à la fin de l'intervalle (juste avant l'événement actuel)
        # C'est exactement `lambda_at_event` calculé plus haut !
        lambda_end = lambda_at_event
        
        # Intégrale par type: [Batch, L-1, D]
        # Gestion de la division par zéro si alpha ~ 0 (optionnel, ici on suppose alpha != 0)
        # Pour stabilité : alpha + epsilon
        alpha_safe = alpha + 1e-9 * torch.sign(alpha)
        
        integral_per_type = (lambda_end - lambda_start) / alpha_safe
        
        # Somme sur tous les types D
        # [Batch, L-1]
        non_event_ll = integral_per_type.sum(dim=-1)

        # --- C. Loss Totale ---
        
        # Somme masquée
        loss_event = (event_ll * seq_mask).sum()
        loss_non_event = (non_event_ll * seq_mask).sum()
        
        num_events = seq_mask.sum().item()
        
        # NLL
        loss = -(loss_event - loss_non_event)
        
        return loss, int(num_events)
    
    def _run_simulation_loop(
        self,
        buffers: Buffers,
        sim_state: SimulationState,
        start_times: torch.Tensor,
        end_times: torch.Tensor,
    ) -> None:
        """
        Simulation loop optimisée pour Self-Correcting.
        Maintient les compteurs N(t) au lieu de relire l'historique.
        """
        initial_len = buffers["initial_len"]
        max_seq_len = buffers["time"].size(1)
        batch_size = start_times.size(0)
        device = self.device

        # 1. Initialisation de l'état N(t)
        # [Batch, D]
        current_counts = torch.zeros(batch_size, self.num_event_types, device=device)
        
        if initial_len > 0:
            init_types = buffers["event"][:, :initial_len] # [B, L]
            # One hot sum
            # [B, L, D]
            one_hots = F.one_hot(init_types.long(), self.num_event_types).float()
            # Mask padding
            mask = init_types != self.pad_token_id
            one_hots = one_hots * mask.unsqueeze(-1)
            current_counts = one_hots.sum(dim=1)
            
            current_abs_time = buffers["time"][:, initial_len-1]
        else:
            current_abs_time = start_times

        absolute_end_times = end_times + (end_times - start_times)
        
        # Paramètres pour broadcasting [1, D]
        mu = self.mu.unsqueeze(0)
        alpha = self.alpha.unsqueeze(0)

        pbar = tqdm(total=absolute_end_times.max().item(), desc="Simulating (SCPP)", leave=False)

        while sim_state["batch_active"].any():
            active_idx = sim_state["batch_active"]
            
            # --- Ogata's Thinning pour SCPP ---
            # Intensité lambda(t) = exp(mu + alpha*t - alpha*N)
            # Si alpha > 0, l'intensité AUGMENTE avec le temps (entre deux événements N est constant).
            # Donc lambda(t_curr) est un MINORANT (pas bon pour thinning classique).
            # On doit majorer lambda sur un intervalle [t, t+delta].
            # Ou utiliser l'Inverse Transform Method si on peut intégrer lambda.
            
            # CAS SIMPLIFIÉ (Inverse Method approchée ou Thinning avec Lookahead) :
            # Comme SCPP "explose" si on attend trop, on majore lambda à t_curr + step_lookahead.
            # Supposons un horizon de sureté (ex: moyenne des dt précédents ou constante).
            # Prenons lambda_bar = lambda(t_curr + horizon).
            
            horizon = 1.0 # À tuner ou dynamique
            
            # Bound calculation
            # [B, D]
            exponent_bound = mu + alpha * ((current_abs_time.unsqueeze(1) + horizon) - current_counts)
            lambda_bound_vec = torch.exp(exponent_bound)
            lambda_bar = lambda_bound_vec.sum(dim=1) # [B]
            
            # Sampling candidat
            u = torch.rand(batch_size, device=device)
            dt_candidate = -torch.log(u) / (lambda_bar + 1e-9)
            
            # Si le candidat dépasse l'horizon, on avance juste le temps (pas d'event) et on update la borne
            # C'est une variante de l'algo de Lewis & Shedler pour processus non-homogènes.
            
            candidate_time = current_abs_time + dt_candidate
            
            # Rejet si on dépasse l'horizon (car la borne n'est plus valide)
            # Dans ce cas, on avance t à t + horizon (ou t + dt) et on ne génère rien (fictive rejection)
            # Mais pour simplifier ici : si dt > horizon, on rejette tout de suite.
            valid_bound_mask = dt_candidate <= horizon
            
            if not valid_bound_mask.all():
                # Pour ceux qui dépassent l'horizon, on avance le temps sans event
                # et on recommence la boucle (counts ne change pas, donc intensité monte)
                too_far = (~valid_bound_mask) & sim_state["batch_active"]
                current_abs_time[too_far] += horizon # Avance prudente
                # Pas de mise à jour de N, on boucle
                # (Attention boucle infinie si lambda très petit, mais SCPP lambda augmente avec t)
            
            # Pour ceux dans l'horizon, on fait le vrai test de rejet Ogata
            check_mask = valid_bound_mask & sim_state["batch_active"]
            
            if check_mask.any():
                # Vraie intensité à t_candidate
                exponent_true = mu + alpha * (candidate_time.unsqueeze(1) - current_counts)
                true_lambda_vec = torch.exp(exponent_true)
                true_lambda = true_lambda_vec.sum(dim=1)
                
                v = torch.rand(batch_size, device=device) * lambda_bar
                
                # Check Rejet et Fin de simulation
                accepted = (v < true_lambda) & check_mask
                over_time = candidate_time >= absolute_end_times
                accepted = accepted & (~over_time)
                
                # Désactivation finis
                sim_state["batch_active"][over_time & active_idx] = False
                
                # Mise à jour ACCEPTÉS
                if accepted.any():
                    # Sample Type
                    probs = true_lambda_vec[accepted]
                    types_sample = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    
                    idx_acc = torch.nonzero(accepted, as_tuple=True)[0]
                    
                    # Update Buffer
                    for i, batch_idx in enumerate(idx_acc):
                         pos = sim_state.get("seq_lens", torch.full((batch_size,), initial_len, device=device))[batch_idx]
                         if pos < max_seq_len:
                            buffers["time"][batch_idx, pos] = candidate_time[batch_idx]
                            buffers["time_delta"][batch_idx, pos] = dt_candidate[batch_idx]
                            buffers["event"][batch_idx, pos] = types_sample[i]
                            if "seq_lens" in sim_state: sim_state["seq_lens"][batch_idx] += 1

                    # UPDATE STATE : N(t) += 1 pour le type choisi
                    # [N_acc, D]
                    one_hot_inc = F.one_hot(types_sample, self.num_event_types).float()
                    current_counts[accepted] += one_hot_inc
                    
                    current_abs_time[accepted] = candidate_time[accepted]
                
                # Mise à jour REJETÉS (dans l'horizon)
                # Ils avancent le temps mais ne changent pas N
                rejected = check_mask & (~accepted) & sim_state["batch_active"]
                if rejected.any():
                    current_abs_time[rejected] = candidate_time[rejected]

            pbar.n = int(current_abs_time.min().item())
            pbar.refresh()
            
        pbar.close()