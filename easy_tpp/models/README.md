# Models

Implémentations des modèles de processus ponctuels temporels.

## Contenu

Ce dossier contient les implémentations des différents modèles TPP disponibles dans EasyTPP.

### Modèles de base

- **basemodel.py** : Classe de base pour tous les modèles TPP avec interface commune
- **baselayer.py** : Couches de base réutilisables (attention, embeddings, etc.)

### Modèles neuraux modernes

- **nhp.py** : Neural Hawkes Process - Modèle neural de base pour TPP
- **thp.py** : Transformer Hawkes Process - Utilise l'architecture Transformer
- **rmtpp.py** : Recurrent Marked Temporal Point Process - Modèle RNN pour TPP
- **attnhp.py** : Attention-based Neural Hawkes Process - Mécanisme d'attention
- **sahp.py** : Self-Attentive Hawkes Process - Auto-attention pour capture de dépendances
- **anhn.py** : Additive Noise Hawkes Network - Modèle avec bruit additif

### Modèles classiques

- **hawkes.py** : Processus de Hawkes classique - Implémentation traditionnelle
- **self_correcting.py** : Processus auto-correcteur - Modèle avec mécanisme d'inhibition

### Modèles avancés

- **fullynn.py** : Fully Neural Network model - Architecture entièrement neuronale
- **ode_tpp.py** : ODE-based Temporal Point Process - Basé sur les équations différentielles
- **intensity_free.py** : Modèles sans fonction d'intensité - Approches alternatives

### Utilitaires

- **thinning.py** : Algorithmes de thinning pour la simulation et génération d'événements

## Utilisation

### Création d'un modèle

```python
from easy_tpp.models import NHP, THP, RMTPP
from easy_tpp.config_factory import ModelConfig

# Configuration du modèle - Instance de ModelConfig (pas un dictionnaire)
model_config = ModelConfig(
    model_id='NHP',
    num_event_types=5,
    specs={
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.1
    }
)

# Créer un modèle Neural Hawkes Process
model = NHP(model_config)

# Ou un Transformer Hawkes Process
model_config_thp = ModelConfig(
    model_id='THP',
    num_event_types=5,
    specs={
        'hidden_size': 128,
        'num_heads': 8,
        'num_layers': 4,
        'dropout': 0.1
    }
)
model = THP(model_config_thp)

# Ou un RMTPP
model_config_rmtpp = ModelConfig(
    model_id='RMTPP',
    num_event_types=5,
    specs={
        'hidden_size': 128,
        'rnn_type': 'LSTM',
        'num_layers': 2,
        'dropout': 0.1
    }
)
model = RMTPP(model_config_rmtpp)
```

### Entraînement d'un modèle

```python
import torch
from torch.optim import Adam

# Initialiser l'optimiseur
optimizer = Adam(model.parameters(), lr=1e-3)

# Boucle d'entraînement
model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        # Forward pass
        outputs = model(batch)
        loss = model.compute_loss(batch, outputs)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### Prédiction avec un modèle

```python
# Mode évaluation
model.eval()

with torch.no_grad():
    for batch in test_loader:
        # Prédiction des temps et types d'événements
        predictions = model.predict(batch)
        
        # Accès aux prédictions
        time_predictions = predictions['time_pred']
        type_predictions = predictions['type_pred']
        
        print(f"Predicted times: {time_predictions}")
        print(f"Predicted types: {type_predictions}")
```

### Simulation d'événements

```python
# Simuler de nouveaux événements
model.eval()
with torch.no_grad():
    # Paramètres de simulation
    history = initial_sequence  # Séquence d'événements initiale
    max_time = 10.0            # Temps maximum de simulation
    
    # Générer des événements
    simulated_events = model.simulate(
        history=history,
        max_time=max_time,
        num_samples=100
    )
    
    print(f"Simulated {len(simulated_events)} events")
```

## Architectures des modèles

### Neural Hawkes Process (NHP)

- **Architecture** : RNN avec fonction d'intensité neuronale
- **Avantages** : Simple, efficace, bien établi
- **Cas d'usage** : Baseline robuste pour la plupart des applications

### Transformer Hawkes Process (THP)

- **Architecture** : Transformer avec mécanisme d'attention
- **Avantages** : Capture les dépendances à long terme
- **Cas d'usage** : Séquences longues, patterns complexes

### Recurrent Marked TPP (RMTPP)

- **Architecture** : LSTM/GRU avec prédiction continue du temps
- **Avantages** : Modélisation directe de l'inter-temps
- **Cas d'usage** : Prédiction précise des temps d'arrivée

### Self-Attentive Hawkes Process (SAHP)

- **Architecture** : Auto-attention sur l'historique des événements
- **Avantages** : Flexibilité dans la capture des dépendances
- **Cas d'usage** : Données avec structures d'attention complexes

## Interface commune

Tous les modèles héritent de `Model` et implémentent :

```python
class Model(nn.Module):
    def forward(self, batch):
        """Forward pass du modèle"""
        pass
    
    def compute_loss(self, batch, outputs):
        """Calcul de la fonction de perte"""
        pass
    
    def predict(self, batch):
        """Prédiction des événements futurs"""
        pass
    
    def simulate(self, history, max_time, num_samples):
        """Simulation de nouveaux événements"""
        pass
```

## Configuration des modèles

### Configuration typique pour NHP

```python
from easy_tpp.config_factory import ModelConfig

nhp_config = ModelConfig(
    model_id='NHP',
    num_event_types=5,
    specs={
        'hidden_size': 128,       # Taille de la couche cachée
        'dropout': 0.1,           # Taux de dropout
        'use_intensity': True     # Utiliser la fonction d'intensité
    }
)
```

### Configuration typique pour THP

```python
from easy_tpp.config_factory import ModelConfig

thp_config = ModelConfig(
    model_id='THP',
    num_event_types=5,
    specs={
        'hidden_size': 128,       # Dimension du modèle
        'num_heads': 8,           # Nombre de têtes d'attention
        'num_layers': 4,          # Nombre de couches Transformer
        'dropout': 0.1,           # Taux de dropout
        'max_seq_len': 200        # Longueur maximale de séquence
    }
)
```

### Hyperparamètres recommandés

```python
from easy_tpp.config_factory import ModelConfig

# Configuration conservative (bon point de départ)
conservative_config = ModelConfig(
    model_id='NHP',  # ou 'THP', 'RMTPP', etc.
    num_event_types=5,
    specs={
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.1
    },
    base_config={
        'lr': 1e-3
    }
)

# Configuration performante (plus de ressources)
performance_config = ModelConfig(
    model_id='THP',
    num_event_types=5,
    specs={
        'hidden_size': 256,
        'num_layers': 4,
        'dropout': 0.2
    },
    base_config={
        'lr': 5e-4
    }
)
```
