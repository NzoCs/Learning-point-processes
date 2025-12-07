from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from new_ltpp.models.basemodel import Model
from new_ltpp.models.mixins.simulation_mixin import Buffers, SimulationState
from new_ltpp.shared_types import Batch, SimulationResult


class Hawkes(Model):
    """
    PyTorch implementation of the Hawkes process model.
    Inherits from Model for integration with the framework, enabling
    methods like predict_one_step_at_every_event.
    """

    mu : torch.Tensor
    alpha : torch.Tensor
    beta : torch.Tensor

    def __init__(
        self,
        mu: list[float] | torch.Tensor,
        alpha: list[list[float]] | torch.Tensor,
        beta: list[list[float]] | torch.Tensor,
        **kwargs,
    ) -> None:
        """
        Initialize the Hawkes model.

        Args:
            model_config (new_ltpp.ModelConfig): Configuration object containing model specs.
                Expected specs: 'mu' (list), 'alpha' (list of lists), 'beta' (list of lists).
        """

        super(Hawkes, self).__init__(**kwargs)

        # Convert parameters to tensors and move to the correct device
        mu = torch.tensor(mu, dtype=torch.float32, device=self.device).view(self.num_event_types)
        alpha = torch.tensor(alpha, dtype=torch.float32, device=self.device).view(
            self.num_event_types, self.num_event_types
        )
        beta = torch.tensor(beta, dtype=torch.float32, device=self.device).view(
            self.num_event_types, self.num_event_types
        )

        if (
            mu.shape[0] != self.num_event_types
            or alpha.shape != (self.num_event_types, self.num_event_types)
            or beta.shape != (self.num_event_types, self.num_event_types)
        ):
            raise ValueError(
                f"Hawkes parameter dimensions mismatch. Expected mu: ({self.num_event_types},), "
                f"alpha/beta: ({self.num_event_types}, {self.num_event_types}). "
                f"Got mu: {mu.shape}, alpha: {alpha.shape}, beta: {beta.shape}"
            )

        # Ensure beta values are positive for numerical stability
        beta = torch.clamp(beta, min=self.eps)

        self.mu = nn.Parameter(mu)
        self.alpha = nn.Parameter(alpha)
        self.beta = nn.Parameter(beta)


    def compute_cumsum_dtime(
        self, 
        time_seqs: torch.Tensor,  # [Batch, Seq_Len] (Temps absolus t1, t2, ...)
        sample_dtimes: Optional[torch.Tensor] = None # [Batch, Seq_Len, N_samples] (Deltas relatifs)
    ) -> torch.Tensor:
        """
        Calcule la matrice des temps écoulés directement depuis les temps absolus.
        Retourne |(t_j + delta) - t_i| pour tout historique i et cible j.
        """

        # 1. Calcul de la matrice des différences brutes : t_j - t_i
        # time_seqs.unsqueeze(2) : [B, L, 1]  (Target j en lignes)
        # time_seqs.unsqueeze(1) : [B, 1, L]  (Source i en colonnes)
        # Résultat : [B, L, L] 
        # (Ligne j, Colonne i) contient t_j - t_i
        base_elapses = time_seqs.unsqueeze(2) - time_seqs.unsqueeze(1)

        # 2. Ajout des deltas d'échantillonnage (si fournis)
        # On calcule : (t_j + delta) - t_i  <=>  (t_j - t_i) + delta
        if sample_dtimes is not None:
            # base_elapses : [B, L, L] -> [B, L, L, 1]
            # sample_dtimes: [B, L, N] -> [B, L, 1, N] (On ajoute delta à la dimension Target j)
            
            # Résultat : [B, L, L, N_samples]
            elapsed_times = base_elapses.unsqueeze(-1) + sample_dtimes.unsqueeze(2)
        else:
            # Pas d'échantillons, on retourne juste la matrice carrée
            elapsed_times = base_elapses.unsqueeze(-1)  # [B, L, L, 1]

        # 3. Valeur absolue
        # Nécessaire car si i > j (futur), la différence est négative.
        # Le masque causal (tril) appliqué plus tard dans le code ignorera ces valeurs, 
        # mais abs() assure que l'exponentielle ne reçoit pas de valeurs aberrantes avant masquage.
        return torch.abs(elapsed_times)
    
    def compute_intensities_at_sample_dtimes(
        self,
        *,
        time_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        seq_non_pad_mask: torch.Tensor,  # [Batch, Seq_Len] (1/True pour event, 0/False pour padding)
        sample_dtimes: Optional[torch.Tensor] = None,
        compute_last_step_only: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Calcul vectorisé robuste de l'intensité du processus de Hawkes Multivarié avec gestion du padding.
        Lambda_k(t) = mu_k + Sum_{i < t} [ alpha_{k, m_i} * beta_{k, m_i} * exp(-beta_{k, m_i} * (t - t_i)) ]
        """

        # 1. Calcul des Delta de temps cumulés (tau)
        # Shape: [Batch, L_target, L_source, N_samples]
        cumsum_dtimes = self.compute_cumsum_dtime(
            time_seqs=time_seqs, sample_dtimes=sample_dtimes
        )

        # 2. Gestion du mode 'last step' et extraction de l'historique
        if compute_last_step_only:
            cumsum_dtimes = cumsum_dtimes[:, -1:, :, :] # [B, 1, L, N]
            target_seq_len = 1
        else:
            target_seq_len = time_seqs.shape[1]

        # 3. Extraction efficace des paramètres avec GESTION DU PADDING
        # A. Création d'indices sûrs (Safe Indices)
        # Si type_seqs contient du padding (ex: -1 ou num_classes), F.embedding plante.
        # On remplace temporairement le padding par 0.
        safe_type_seqs = type_seqs.long().clone()
        
        # On s'assure que le masque est booléen pour l'indexation
        is_padding = ~seq_non_pad_mask.bool()
        safe_type_seqs[is_padding] = 0 

        # B. Lookup (Embedding)
        # [Batch, L_source, K_target]
        alpha_hist = F.embedding(safe_type_seqs, self.alpha.t())
        beta_hist  = F.embedding(safe_type_seqs, self.beta.t())

        # C. Application du Masque (Zero-out padding influence)
        # On multiplie par 0 les paramètres récupérés pour les indices de padding.
        # non_pad_mask: [Batch, L_source] -> [Batch, L_source, 1] (Broadcasting sur K_target)
        valid_event_mask = seq_non_pad_mask.float().unsqueeze(-1)
        
        alpha_hist = alpha_hist * valid_event_mask
        beta_hist  = beta_hist * valid_event_mask

        # 4. Préparation des dimensions pour le broadcasting
        # On aligne tout pour avoir: [Batch, L_target, L_source, N_samples, K_target]
        
        # [Batch, 1, L_source, 1, K_target]
        alpha_hist = alpha_hist.unsqueeze(1).unsqueeze(3)
        beta_hist  = beta_hist.unsqueeze(1).unsqueeze(3)
        
        # [Batch, L_target, L_source, N_samples, 1]
        tau = cumsum_dtimes.unsqueeze(-1)

        # 5. Calcul du Noyau (Kernel)
        # Formule: alpha * beta * exp(-beta * tau)
        decay = torch.exp(-beta_hist * tau)
        excitation = (alpha_hist * beta_hist) * decay

        # 6. Masquage Causal (Futur -> Passé)
        # On doit s'assurer que l'événement j n'est pas influencé par i si i >= j.
        if not compute_last_step_only:
             L = time_seqs.shape[1]
             # Masque triangulaire : 1 si source < target, 0 sinon
             causal_mask = torch.tril(torch.ones(L, L, device=time_seqs.device), diagonal=-1)
             
             # Reshape pour le broadcast: [1, L, L, 1, 1]
             causal_mask = causal_mask.view(1, L, L, 1, 1)
             
             excitation = excitation * causal_mask

        # 7. Sommation sur l'historique (L_source)
        # On somme l'influence de tous les événements passés i valides.
        # Shape sortie: [Batch, L_target, N_samples, K_target]
        past_influence = excitation.sum(dim=2)

        # 8. Ajout de l'intensité de base (Mu)
        # self.mu: [K_target] -> [1, 1, 1, K_target]
        mu = self.mu.view(1, 1, 1, -1)
        
        lambda_t = mu + past_influence
        
        # Application du Softplus pour garantir une intensité positive
        return F.softplus(lambda_t)
    
    def loglike_loss(self, batch: Batch) -> tuple[torch.Tensor, int]:
        """
        Calcule la log-vraisemblance exacte (analytique) pour le modèle Hawkes.
        LL = sum(log(lambda(t_i))) - int(lambda(t) dt)
        """
        # [Batch, L]
        time_seq = batch.time_seqs
        type_seq = batch.type_seqs
        time_delta_seq = batch.time_delta_seqs
        
        # Masque pour ignorer le padding
        # [Batch, L-1] (car on prédit à partir du 2ème événement jusqu'à la fin)
        seq_non_pad_mask = batch.seq_non_pad_mask

        # --- 1. Terme Log-Intensité (Event Log-Likelihood) ---
        # On calcule lambda(t_i) pour chaque événement i.
        
        # Intensités à t_1 ... t_N (basé sur l'historique t_0 ... t_{N-1})
        # Note: compute_intensities_at_sample_dtimes peut être utilisé avec sample_dtimes=0
        # Mais ici, on suppose que vous avez une méthode qui calcule lambda au temps exact de l'event.
        # Si vous utilisez la méthode précédente, on peut passer sample_dtimes=0.
        
        
        # [Batch, L-1, 1, K]
        intensities_at_events = self.compute_intensities_at_sample_dtimes(
            time_seqs=time_seq[:, 1:], # Séquence complète
            type_seqs=type_seq[:, 1:], # Séquence complète
            seq_non_pad_mask=seq_non_pad_mask[:, 1:],
            sample_dtimes=None,
            compute_last_step_only=False
        ).squeeze(-2) # [Batch, L-1, K]

        
        safe_type_seqs = type_seq.long().clone()
        
        # On s'assure que le masque est booléen pour l'indexation
        is_padding = ~seq_non_pad_mask.bool()
        safe_type_seqs[is_padding] = 0 

        # Sélection de la lambda correspondant au VRAI type d'événement qui s'est produit
        # type_seq[:, 1:] contient les types cibles k_1 ... k_N
        target_types = safe_type_seqs[:, 1:].long() # [Batch, L-1]
        
        # Gather : on récupère lambda_{k_true}(t)
        # [Batch, L-1]
        lambda_target = torch.gather(intensities_at_events, dim=-1, index=target_types.unsqueeze(-1)).squeeze(-1)
        
        # Log-Vraisemblance des événements (+ epsilon pour stabilité num)
        event_ll = torch.log(lambda_target + 1e-9)

        # --- 2. Terme Intégrale (Non-Event Log-Likelihood) ---
        # Calcul analytique de l'intégrale sur chaque intervalle (t_j, t_{j+1})
        # [Batch, L-1]
        integral_term = self._compute_integral_analytical(
            time_seq=time_seq,
            time_delta_seq=time_delta_seq,
            type_seq=safe_type_seqs,
        )

        # --- 3. Perte Totale ---
        # Loss = - (Somme(Log_LL) - Somme(Int_LL))
        # On applique le masque pour ne pas compter le padding
        
        event_ll = (event_ll * seq_non_pad_mask[:, 1:]).sum()
        non_event_ll = (integral_term * seq_non_pad_mask[:, 1:]).sum()
        
        # Nombre total d'événements (pour monitoring)
        num_events = seq_non_pad_mask[:, 1:].sum().item()
        # NLL (Negative Log Likelihood)
        loss = -(event_ll - non_event_ll)

        return loss, int(num_events)

    def _compute_integral_analytical(
        self, 
        time_seq: torch.Tensor, 
        time_delta_seq: torch.Tensor, 
        type_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcule l'intégrale analytique de l'intensité multivariée sur chaque intervalle.
        Returns:
            torch.Tensor: Somme de l'intégrale sur toutes les dimensions K pour chaque intervalle.
                          Shape [Batch, L-1] (correspondant aux intervalles dt_1...dt_N)
        """
        # Nous intégrons sur les intervalles dt_1, dt_2, ...
        # L'intervalle j (dt_{j+1}) commence à t_j.
        # time_delta_seq[:, 1:] contient les durées d'intégration.
        interval_durations = time_delta_seq[:, 1:] # [Batch, L-1]
        
        # L_target correspond au nombre d'intervalles sur lesquels on intègre
        batch_size, seq_len = time_seq.shape
        L_target = seq_len - 1 

        # --- A. Partie Intensité de Base (Mu) ---
        # Intégrale de mu_k sur dt = mu_k * dt
        # Somme sur tous les types K: (Sum_k mu_k) * dt
        # [1] -> scalaire sum(mu)
        mu_sum = self.mu.sum() 
        integral_base = mu_sum * interval_durations # [Batch, L-1]

        # --- B. Partie Excitation (Hawkes) ---
        # Formule : Sum_{i <= j} Sum_{k} alpha_{k, mi} * (1 - exp(-beta_{k, mi} * dt_{j+1})) * exp(-beta_{k, mi} * (t_j - t_i))
        
        # 1. Matrice des temps écoulés (t_j - t_i) au DÉBUT de l'intervalle
        # On utilise compute_cumsum_dtime sur les données brutes.
        # [Batch, L, L] -> On prend le sous-bloc [:, 1:, :-1] car:
        # - Cibles (lignes) : t_1 ... t_{N-1} (débuts des intervalles)
        # - Sources (colonnes) : t_0 ... t_{N-2} (historique possible)
        # Mais pour simplifier, réutilisons la logique vectorielle complète et masquons ensuite.
        
        # Calcul de tous les t_j - t_i
        # [Batch, L, L, 1]
        dist_matrix = self.compute_cumsum_dtime(time_delta_seq, sample_dtimes=None)
        
        # On s'intéresse aux intervalles qui commencent à t_1, ..., t_{N-1}
        # Donc on prend les lignes 1 à la fin.
        # L'historique pour l'intervalle commençant à t_j inclut t_j lui-même (i <= j).
        # [Batch, L-1, L, 1]
        tau_start = dist_matrix[:, 1:, :, :] 

        # 2. Paramètres Alpha et Beta basés sur l'historique
        # type_seq: [Batch, L]
        # Extraction efficace comme vu précédemment
        # [Batch, L, K_target]
        alpha_hist = F.embedding(type_seq.long(), self.alpha.t())
        beta_hist  = F.embedding(type_seq.long(), self.beta.t())

        # 3. Calcul du terme dépendant de la durée d'intervalle (dt)
        # Terme : (1 - exp(-beta * dt))
        # interval_durations: [Batch, L-1] -> [Batch, L-1, 1, 1]
        dt_reshaped = interval_durations.unsqueeze(-1).unsqueeze(-1)
        
        # Alignement des betas pour le produit avec dt
        # beta_hist correspond aux sources (colonnes). On doit le broadcaster sur les lignes (cibles).
        # [Batch, 1, L, K_target]
        beta_hist_b = beta_hist.unsqueeze(1)
        
        # Facteur d'intégration temporelle F(dt)
        # [Batch, L-1, L, K_target]
        time_integral_factor = torch.tensor(1.0, device=self.device) - torch.exp(-beta_hist_b * dt_reshaped)

        # 4. Calcul de la décroissance à l'instant initial t_j
        # Terme : exp(-beta * (t_j - t_i))
        decay_at_start = torch.exp(-beta_hist_b * tau_start)

        # 5. Combinaison des termes
        # Intégrale_ij = alpha * F(dt) * decay_start
        # [Batch, L-1, L, K_target]
        # alpha_hist_b : [Batch, 1, L, K_target]
        alpha_hist_b = alpha_hist.unsqueeze(1)
        
        term_integral = alpha_hist_b * time_integral_factor * decay_at_start

        # 6. Masque Causal
        # On intègre sur l'intervalle [t_j, t_{j+1}]. L'historique pertinent est i <= j.
        # tau_start est construit sur time_seq[:, 1:].
        # Indices lignes (cibles) : 1..N-1 (soit j allant de 1 à N-1)
        # Indices colonnes (sources) : 0..N-1 (soit i allant de 0 à N-1)
        # Condition : i <= j (puisque j est l'index dans la séquence d'origine)
        # Le masque triangulaire inférieur doit inclure la diagonale.
        
        L = time_seq.shape[1]
        # Masque [L-1, L]
        # On veut que la ligne k (qui correspond à l'event k+1) voit les colonnes 0..k+1
        ones = torch.ones(L, L, device=time_seq.device)
        # tril avec diagonal=0 garde la diagonale
        full_mask = torch.tril(ones, diagonal=0) 
        # On garde les lignes 1: pour correspondre à tau_start
        causal_mask = full_mask[1:, :].view(1, L-1, L, 1)

        term_integral = term_integral * causal_mask

        # 7. Sommations
        # Somme sur les dimensions K (tous les processus cibles possibles)
        # [Batch, L-1, L]
        sum_over_K = term_integral.sum(dim=-1)
        
        # Somme sur l'historique i
        # [Batch, L-1]
        excitation_integral = sum_over_K.sum(dim=-1)

        return integral_base + excitation_integral


### --------------------------------------------------------------------------------
### ----------------- Simulation custom plus opti pour hawkes ----------------------

    def simulate_one_step(self, decay_state: torch.Tensor, active_mask: torch.Tensor):
        """
        Simule le prochain dt pour un batch donné.
        Gère la boucle de rejection (Thinning) : si rejeté, on avance le temps et on recommence.
        
        Args:
            decay_state: [B, K, K] État actuel du processus.
            active_mask: [B] Booléen, quels batchs sont encore en cours de simulation.
        
        Returns:
            final_dt: [B] Le temps écoulé TOTAL pour atteindre le prochain événement.
            final_type: [B] L'index du type d'événement accepté.
            current_decay: [B, K, K] L'état mis à jour juste AVANT le saut (decayé de final_dt).
        """
        B = decay_state.size(0)
        K = self.num_event_types
        device = self.device
        
        # Accumulateur de temps (si on rejette, on ajoute le dt rejeté ici)
        accumulated_dt = torch.zeros(B, device=device)
        
        # Masque local : qui cherche encore son prochain événement dans cette étape ?
        # On ne travaille que sur ceux qui sont 'active' globalement
        seeking_event = active_mask.clone()
        
        # Buffer pour stocker le type trouvé
        found_types = torch.zeros(B, dtype=torch.long, device=device)
        
        # Copie de l'état de travail qu'on va faire "decay" au fur et à mesure des rejets
        work_decay_state = decay_state.clone()

        while seeking_event.any():
            # Index des batchs qui cherchent encore
            idx_seeking = torch.nonzero(seeking_event, as_tuple=True)[0]
            
            # --- A. Borne Supérieure (Upper Bound) ---
            # λ_bound(t) = μ + sum(H(t)). Comme H décroît, c'est une borne valide pour l'intervalle.
            # shape: [N_seeking, K]
            current_intensity = self.mu.unsqueeze(0) + work_decay_state[idx_seeking].sum(dim=2)
            
            # Pour utiliser tes fonctions, on considère L=1
            # rate : [N_seeking, K] (Borne par type)
            rate = current_intensity 

            # --- B. Échantillonnage des candidats dt ---
            # Utilise ta fonction sample_exp_distribution
            # returns: [N_seeking, K, num_exp=1] -> on squeeze
            exp_jumps = self.get_event_sampler().sample_exp_distribution(rate).squeeze(-1) 
            
            # Note : Ici, on a un dt par type K. Dans une superposition, le prochain événement 
            # est le min(dt) parmi tous les K.
            # Pour simplifier l'usage de ton sample_accept vectorisé, on garde les K candidats.

            # --- C. Calcul de l'intensité réelle au temps candidat ---
            # intens(t + dt) = μ + H(t) * exp(-β * dt)
            # shape: [N_seeking, K, K]
            dt_view = exp_jumps.unsqueeze(-1) # [N, K, 1] pour broadcast sur la dimension source
            candidate_decay = work_decay_state[idx_seeking] * torch.exp(-self.beta.unsqueeze(0) * dt_view)
            
            # [N_seeking, K]
            true_intensity = self.mu.unsqueeze(0) + candidate_decay.sum(dim=2)

            # --- D. Acceptation Vectorisée ---
            # On prépare les tenseurs pour sample_accept (ajout dimension L=1, E=1)
            # u shape requis: [B, L, K, E]
            u = self.get_event_sampler().sample_uniform(rate.unsqueeze(1), num_samples=1) 
            
            # sample_accept retourne les dt acceptés, ou 0.0 si rejeté (none_accepted)
            # Input shapes ajustés: [N, 1, K, 1] etc.
            accepted_dts_per_type = self.get_event_sampler().sample_accept(
                u=u, 
                rate=rate.unsqueeze(1), 
                intens=true_intensity.unsqueeze(1).unsqueeze(-1), 
                exp_jumps=exp_jumps.unsqueeze(1).unsqueeze(-1)
            ).squeeze(1) # Résultat: [N_seeking, K] (contient le dt choisi ou 0)

            # --- E. Analyse du résultat ---
            # sample_accept a déjà sélectionné le "premier" type accepté (argmax).
            # Si tout est rejeté, sample_accept renvoie 0.0 (selon ton code).
            
            # On prend le max sur K pour récupérer le dt (un seul est non-nul grâce à ton argmax interne)
            # Si sum/max est 0, c'est que rien n'a été accepté (none_accepted était True)
            candidate_dt_final = accepted_dts_per_type.sum(dim=1) # [N_seeking]
            is_accepted = candidate_dt_final > 0

            # --- F. Mise à jour pour ceux qui ont ACCEPTÉ ---
            if is_accepted.any():
                # Indices globaux des acceptés
                idx_accepted_global = idx_seeking[is_accepted]
                
                # On récupère le dt accepté
                valid_dt = candidate_dt_final[is_accepted]
                
                # On récupère le type (argmax sur la dimension K des candidats acceptés)
                # Note: exp_jumps contenait les candidats. sample_accept a filtré.
                # On doit retrouver quel K a gagné.
                # Astuce: On peut réutiliser la logique ou refaire l'argmax sur le résultat de sample_accept
                valid_types = accepted_dts_per_type[is_accepted].argmax(dim=1)
                
                # Enregistrement
                accumulated_dt[idx_accepted_global] += valid_dt
                found_types[idx_accepted_global] = valid_types
                
                # Mise à jour de l'état "final" (decayé jusqu'au moment de l'event)
                # On doit appliquer le decay final correspondant exactement au dt accepté
                # Pour ça, on prend le candidate_decay calculé plus haut
                # (Attention : candidate_decay a shape [N_seeking, K, K], il faut filtrer)
                final_decay_subset = candidate_decay[is_accepted]
                
                # On ne met PAS ENCORE le saut (jump), ça se fera dans update_states
                # Mais on doit sauvegarder l'état decayé pour le retourner
                # work_decay_state est mis à jour temporairement pour la boucle, 
                # mais pour le return final, on veut l'état propre.
                # Ici, on écrase work_decay_state pour les acceptés car ils sortent de la boucle.
                work_decay_state[idx_accepted_global] = final_decay_subset
                
                # On sort ces indices de la boucle
                seeking_event[idx_accepted_global] = False

            # --- G. Mise à jour pour ceux qui ont REJETÉ (Thinning) ---
            # C'est la partie cruciale pour la boucle while : on avance le temps mais sans event.
            if (~is_accepted).any():
                idx_rejected_global = idx_seeking[~is_accepted]
                
                # De combien on avance ? 
                # Dans une superposition vectorisée, si on rejette, on avance normalement du min(dt_candidats).
                # Ici on va prendre le min des candidats générés (exp_jumps) pour avancer.
                rejected_dts = exp_jumps[~is_accepted].min(dim=1).values # [N_rejected]
                
                # 1. On accumule ce temps "perdu"
                accumulated_dt[idx_rejected_global] += rejected_dts
                
                # 2. On met à jour l'état interne (Decay SEULEMENT)
                # H_new = H_old * exp(-beta * dt_rejected)
                decay_factor = torch.exp(-self.beta.unsqueeze(0) * rejected_dts.view(-1, 1, 1))
                work_decay_state[idx_rejected_global] *= decay_factor
                
                # Ils restent dans seeking_event=True et on boucle.

        return accumulated_dt, found_types, work_decay_state
    
    def update_states(self, decay_state: torch.Tensor, event_types: torch.Tensor, active_mask: torch.Tensor):
        """
        Ajoute le saut (Jump) à l'état decayé.
        H = H_decayed + (Alpha * Beta)_k
        """
        # decay_state est déjà à jour temporellement (H * exp(-beta * dt)) grâce à simulate_one_step
        
        # Pré-calcul du terme de saut global : [1, K_target, K_source]
        jump_term = (self.alpha * self.beta).unsqueeze(0)
        
        # Création du masque One-Hot pour les types d'événements survenus : [B, 1, K_source]
        # On ne sélectionne que la colonne correspondant au type d'événement
        B = decay_state.size(0)
        K = self.num_event_types
        
        one_hot = F.one_hot(event_types, K).float().unsqueeze(1) # [B, 1, K]
        
        # Terme à ajouter : on diffuse jump_term et on masque
        # [B, K, K]
        jump_update = jump_term * one_hot
        
        # On applique le saut uniquement sur les batchs actifs
        # (Sinon on risque de polluer les états des séquences finies)
        mask_view = active_mask.view(-1, 1, 1).float()
        
        new_state = decay_state + (jump_update * mask_view)
        
        return new_state
    
    def update_buffers(self, buffers, sim_state, dt, event_types, active_mask):
        """
        Stocke dt et event_type dans les buffers à la position courante.
        """
        batch_indices = torch.nonzero(active_mask, as_tuple=True)[0]
        
        # Récupère la position d'écriture pour chaque batch (compteur)
        # Supposons que sim_state["step_idx"] est un tensor [B] contenant l'index actuel
        current_steps = sim_state["step_idx"][batch_indices]
        
        # Sécurité pour ne pas dépasser la taille du buffer
        max_len = buffers["time"].size(1)
        valid_write = current_steps < max_len
        
        final_indices = batch_indices[valid_write]
        final_steps = current_steps[valid_write]
        
        if final_indices.numel() > 0:
            # 1. Écriture du type d'événement
            buffers["event"][final_indices, final_steps] = event_types[final_indices].float()
            
            # 2. Écriture du dt
            buffers["time_delta"][final_indices, final_steps] = dt[final_indices]
            
            # 3. Calcul du temps absolu (optionnel si tu ne stockes que dt)
            # buffers["time"][final_indices, final_steps] = prev_time + dt ...
            
            # Incrément du compteur
            sim_state["step_idx"][final_indices] += 1
    
    def _run_simulation_loop(self, buffers, sim_state, start_times, end_times):
        
        batch_size = start_times.size(0)
        device = self.device
        
        # Initialisation decay_state (comme dans ton code initial)
        decay_state = torch.zeros(batch_size, self.num_event_types, self.num_event_types, device=device)
        # ... (Logique d'init decay_state avec historique si nécessaire) ...
        
        # Init compteurs
        if "step_idx" not in sim_state:
            sim_state["step_idx"] = torch.full((batch_size,), buffers["initial_len"], dtype=torch.long, device=device)
            
        current_abs_time = start_times.clone()
        
        # Boucle principale : on avance pas à pas
        # On s'arrête quand tout le monde a dépassé son end_time
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        while active_mask.any():
            
            # 1. SIMULATE ONE STEP (Boucle while interne pour le thinning)
            # Retourne le dt total accepté et l'état decayé à ce moment-là
            dt, types, decayed_state_at_t = self.simulate_one_step(decay_state, active_mask)
            
            # Mise à jour du temps absolu
            current_abs_time += dt
            
            # Vérification : A-t-on dépassé la fin ?
            is_within_horizon = current_abs_time < end_times
            
            # On ne garde que ceux qui sont encore valides (Actifs ET pas dépassés)
            valid_step_mask = active_mask & is_within_horizon
            
            # 2. UPDATE STATE (Ajout du saut)
            # On met à jour l'état global pour la prochaine itération
            # Note: on utilise decayed_state_at_t comme base
            decay_state = self.update_states(decayed_state_at_t, types, valid_step_mask)
            
            # 3. UPDATE BUFFERS
            self.update_buffers(buffers, sim_state, dt, types, valid_step_mask)
            
            # Mise à jour du masque global (ceux qui ont fini sortent)
            active_mask = valid_step_mask