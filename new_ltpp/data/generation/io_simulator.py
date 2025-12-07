import json
import os
from typing import Dict, List, Optional

from datasets import Dataset, DatasetDict


class IOSimulator:
    """
    Classe responsable de la gestion des entrées/sorties pour les simulateurs.
    Gère l'enregistrement local et le push vers Hugging Face Hub.
    """

    def __init__(self):
        """Initialise le gestionnaire d'I/O."""
        pass

    def save_to_json(
        self,
        formatted_data: List[Dict],
        output_dir: str,
        splits: Dict[str, float] = {"train": 0.6, "test": 0.2, "dev": 0.2},
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Sauvegarde les données formatées en JSON localement.

        Args:
            formatted_data (List[Dict]): Données formatées
            output_dir (str): Dossier de sortie
            splits (dict): Ratios pour la division train/test/dev
            metadata (dict, optional): Métadonnées à sauvegarder
        """
        # Vérification que les ratios somment à 1
        assert abs(sum(splits.values()) - 1.0) < 1e-10, "Les ratios doivent sommer à 1"

        # Création du dossier de sortie
        os.makedirs(output_dir, exist_ok=True)

        # Division des données
        print("Division des données en ensembles train/test/dev...")
        n = len(formatted_data)

        data_splits = {}
        start_idx = 0
        for split_name, ratio in splits.items():
            split_size = int(n * ratio)
            data_splits[split_name] = formatted_data[start_idx : start_idx + split_size]
            start_idx += split_size

        # Sauvegarde des données
        print("Sauvegarde des données...")
        for split_name, data in data_splits.items():
            filepath = os.path.join(output_dir, f"{split_name}.json")
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

        # Sauvegarde des métadonnées
        if metadata:
            # Enrichir les métadonnées avec les infos des splits
            metadata["split_info"] = {
                **{
                    f"{split_name}_size": len(data)
                    for split_name, data in data_splits.items()
                },
                **{
                    f"{split_name}_ratio": ratio for split_name, ratio in splits.items()
                },
            }
            metadata["total_events"] = sum(item["seq_len"] for item in formatted_data)

            metadata_path = os.path.join(output_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

        print(f"✓ Toutes les données ont été sauvegardées dans {output_dir}")

    def push_to_hub(
        self,
        formatted_data: List[Dict],
        repo_id: str,
        splits: Dict[str, float] = {"train": 0.6, "test": 0.2, "dev": 0.2},
        metadata: Optional[Dict] = None,
        private: bool = False,
        token: Optional[str] = None,
    ) -> None:
        """
        Pousse les données formatées vers Hugging Face Hub.

        Args:
            formatted_data (List[Dict]): Données formatées
            repo_id (str): Identifiant du dépôt (format: "username/dataset-name")
            splits (dict): Ratios pour la division train/test/dev
            metadata (dict, optional): Métadonnées du dataset
            private (bool): Si True, crée un dépôt privé
            token (str, optional): Token d'authentification Hugging Face
        """
        # Vérification que les ratios somment à 1
        assert abs(sum(splits.values()) - 1.0) < 1e-10, "Les ratios doivent sommer à 1"

        # Division des données
        print("Division des données en ensembles train/test/dev...")
        n = len(formatted_data)

        data_splits = {}
        start_idx = 0
        for split_name, ratio in splits.items():
            split_size = int(n * ratio)
            data_splits[split_name] = formatted_data[start_idx : start_idx + split_size]
            start_idx += split_size

        # Création du DatasetDict pour Hugging Face
        print("Création du DatasetDict...")
        dataset_dict = DatasetDict(
            {
                split_name: Dataset.from_list(data)
                for split_name, data in data_splits.items()
            }
        )

        # Push vers Hugging Face Hub
        print(f"Push du dataset vers {repo_id}...")
        dataset_dict.push_to_hub(
            repo_id=repo_id,
            private=private,
            token=token,
        )

        # Créer et pousser le README avec les métadonnées
        if metadata:
            # Enrichir les métadonnées avec les infos des splits
            metadata["split_info"] = {
                **{
                    f"{split_name}_size": len(data)
                    for split_name, data in data_splits.items()
                },
                **{
                    f"{split_name}_ratio": ratio for split_name, ratio in splits.items()
                },
            }
            metadata["total_events"] = sum(item["seq_len"] for item in formatted_data)

            self._create_and_push_readme(repo_id, metadata, splits)

        print(f"✓ Dataset poussé avec succès vers {repo_id}")
        print(f"  URL: https://huggingface.co/datasets/{repo_id}")

    def _create_and_push_readme(
        self, repo_id: str, metadata: Dict, splits: Dict[str, float]
    ) -> None:
        """
        Crée un README pour le dataset sur Hugging Face.

        Args:
            repo_id (str): Identifiant du dépôt
            metadata (dict): Métadonnées du dataset
            splits (dict): Ratios des splits
        """
        readme_content = f"""# {repo_id.split('/')[-1]}

## Dataset Description

This dataset contains temporal point process simulations generated using the {metadata.get('simulation_info', {}).get('simulator_type', 'Unknown')} simulator.

## Metadata

```json
{json.dumps(metadata, indent=2)}
```

## Dataset Structure

Each sequence in the dataset has the following fields:
- `dim_process`: Number of event types/dimensions
- `seq_len`: Length of the sequence
- `seq_idx`: Index of the sequence
- `time_since_start`: Time elapsed since the first event
- `time_since_last_event`: Time elapsed since the last event
- `type_event`: Type/dimension of each event

## Splits

The dataset is split into:
- Train: {splits.get('train', 0) * 100}%
- Test: {splits.get('test', 0) * 100}%
- Dev: {splits.get('dev', 0) * 100}%

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{repo_id}")

# Access train split
train_data = dataset["train"]

# Example: iterate through sequences
for sequence in train_data:
    print(f"Sequence length: {{sequence['seq_len']}}")
    print(f"Event types: {{sequence['type_event']}}")
    print(f"Time deltas: {{sequence['time_since_last_event']}}")
```

## Citation

If you use this dataset, please cite the simulator used to generate it.
"""

        # Note: Pour l'instant, on affiche juste le README
        # Dans une future version, on pourrait utiliser l'API HF pour le pousser
        print("\n--- README.md preview ---")
        print(readme_content[:500] + "...")
