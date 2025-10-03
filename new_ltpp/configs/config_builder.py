from abc import ABC, abstractmethod
from typing import Dict, Union, Any, List, Optional
from pathlib import Path
import yaml

from .config_factory import config_factory, ConfigType
from .data_config import DataConfig


class ConfigBuilder(ABC):
    """Interface pour un builder de config spécifique."""

    def __init__(self, config_type: ConfigType, config_dict: Dict[str, Any] = None):
        self.config_type = config_type
        self.config_dict = config_dict or {}

    def set_field(self, field: str, value: Any):
        self.config_dict[field] = value
        return self.get_missing_fields()

    def get_missing_fields(self) -> List[str]:
        # Par défaut aucune contrainte; les sous-classes doivent surcharger
        return []

    def get_config_dict(self) -> Dict[str, Any]:
        return self.config_dict

    def build(self, **kwargs) :
        """
        Construire une instance de Config à partir du dict courant via la factory.
        Args:
            **kwargs: passed to factory.create_config/create_config_by_name
        """
        
        return config_factory.create_config(self.config_type, self.get_config_dict(), **kwargs)

    def _load_yaml(self, yaml_path: Union[str, Path]) -> Dict[str, Any]:
        """Charge un fichier YAML avec fallback d'encodage."""
        path = Path(yaml_path)
        if not path.is_file():
            raise FileNotFoundError(f"Fichier YAML non trouvé: {yaml_path}")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except UnicodeDecodeError:
            with open(path, 'r', encoding='latin-1') as f:
                return yaml.safe_load(f)

    @abstractmethod
    def load_from_yaml(self, yaml_path: Union[str, Path]) -> Dict[str, Any]:
        """Charge un fichier YAML avec fallback d'encodage."""
        pass
    
    @abstractmethod
    def from_dict(self, data: Dict[str, Any], *args, **kwargs):
        pass

    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Récupère une valeur via un chemin avec points (ex: 'section.key')."""
        keys = path.split('.')
        current = data
        
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                raise KeyError(f"Chemin '{path}' non trouvé dans le YAML")
            current = current[key]
        
        return current


class RunnerConfigBuilder(ConfigBuilder):
    
    def __init__(self):
        super().__init__(ConfigType.RUNNER)
        self.model_builder = ModelConfigBuilder()
        self.data_builder = DataConfigBuilder()
        self.training_builder = TrainingConfigBuilder()

    def from_dict(self, data: Dict[str, Any],
                  training_config_path: str,
                  model_config_path: str,
                  data_config_path: str,
                  data_loading_config_path: str = None,
                  data_specs_path: str = None,
                  simulation_config_path: str = None,
                  thinning_config_path: str = None,
                  logger_config_path: str = None) -> List[str]:
        
        # Utiliser TrainingConfigBuilder pour charger la config training
        self.training_builder.from_dict(data, training_config_path)
        training_cfg = self.training_builder.get_config_dict()

        self.model_builder.from_dict(
            data,
            model_config_path,
            simulation_config_path,
            thinning_config_path
        )

        model_cfg = self.model_builder.get_config_dict()
        if "max_epochs" not in model_cfg["scheduler_config"]:
            model_cfg["scheduler_config"] = {
                "lr_scheduler": training_cfg.get("lr_scheduler"),
                "lr": training_cfg.get("lr"),
                "max_epochs": training_cfg.get("max_epochs")
            }

        self.data_builder.from_dict(
            data,
            data_config_path,
            data_loading_config_path,
            data_specs_path
        )

        data_cfg = self.data_builder.get_config_dict()
        self.config_dict["training_config"] = training_cfg
        self.config_dict["model_config"] = model_cfg
        self.config_dict["data_config"] = data_cfg
        
        if logger_config_path:
            logger_cfg = self._get_nested_value(data, logger_config_path)
            self.config_dict["logger_config"] = logger_cfg

        return self.get_missing_fields()
    
    def load_from_yaml(self, 
                      yaml_file_path: Union[str, Path], 
                      training_config_path: str,
                      model_config_path: str, 
                      data_config_path: str,
                      data_loading_config_path: str = None,
                      data_specs_path: str = None,
                      simulation_config_path: str = None,
                      thinning_config_path: str = None,
                      logger_config_path: str = None,
                      ) -> List[str]:
        """
        Charge la config complète depuis un YAML en utilisant les autres builders.
        
        Args:
            yaml_path: Chemin vers le fichier YAML
            training_config_path: Chemin vers la config training (ex: 'trainer_configs.quick_test')
            model_config_path: Chemin vers la config model (ex: 'model_configs.neural_small')
            data_config_path: Chemin vers la config data (ex: 'data_configs.test')
            data_loading_config_path: Chemin vers data_loading_config (ex: 'data_loading_configs.default')
            data_specs_path: Chemin vers tokenizer_specs (ex: 'tokenizer_specs.standard')
            simulation_config_path: Chemin vers la config simulation (ex: 'simulation_configs.simulation_fast')
            thinning_config_path: Chemin vers la config thinning (ex: 'thinning_configs.thinning_fast')

        Returns: 
            Liste des champs manquants après chargement
        """
        data = self._load_yaml(yaml_file_path)
        return self.from_dict(
            data,
            training_config_path,
            model_config_path,
            data_config_path,
            data_loading_config_path,
            data_specs_path,
            simulation_config_path,
            thinning_config_path,
            logger_config_path
        )

    def set_trainer_config(self, trainer_cfg: Union[Dict[str, Any], Any]):
        self.config_dict["training_config"] = trainer_cfg
        return self.get_missing_fields()

    def set_model_config(self, model_cfg: Union[Dict[str, Any], Any]):
        self.config_dict["model_config"] = model_cfg
        return self.get_missing_fields()

    def set_data_config(self, data_cfg: Union[Dict[str, Any], Any]):
        self.config_dict["data_config"] = data_cfg
        return self.get_missing_fields()
    
    # Méthodes utilitaires pour modifier les paramètres de training
    def override_max_epochs(self, epochs: int):
        """Override le nombre d'époques dans la configuration de training."""
        if "training_config" not in self.config_dict:
            self.config_dict["training_config"] = {}
        self.config_dict["training_config"]["max_epochs"] = epochs
        return self.get_missing_fields()
    
    def override_devices(self, devices: int):
        """Override le nombre de devices dans la configuration de training."""
        if "training_config" not in self.config_dict:
            self.config_dict["training_config"] = {}
        self.config_dict["training_config"]["devices"] = devices
        return self.get_missing_fields()
    
    def override_batch_size(self, batch_size: int):
        """Override la taille du batch dans la configuration de training."""
        if "training_config" not in self.config_dict:
            self.config_dict["training_config"] = {}
        self.config_dict["training_config"]["batch_size"] = batch_size
        return self.get_missing_fields()

    def get_missing_fields(self) -> List[str]:
        required = ["training_config", "model_config", "data_config"]
        return [f for f in required if f not in self.config_dict]


class ModelConfigBuilder(ConfigBuilder):
    
    def __init__(self):
        super().__init__(ConfigType.MODEL)

    def from_dict(self, data: Dict[str, Any],
                  model_config_path: str,
                  simulation_config_path: Optional[str] = None,
                  thinning_config_path: Optional[str] = None,
                  scheduler_config_path: Optional[str] = None):
        model_cfg = self._get_nested_value(data, model_config_path)
        simulation_cfg = self._get_nested_value(data, simulation_config_path) if simulation_config_path else {}
        scheduler_cfg = self._get_nested_value(data, scheduler_config_path) if scheduler_config_path else {}
        thinning_cfg = self._get_nested_value(data, thinning_config_path) if thinning_config_path else {}
        self.config_dict = model_cfg
        self.config_dict["simulation_config"] = simulation_cfg
        self.config_dict["thinning_config"] = thinning_cfg
        self.config_dict["scheduler_config"] = scheduler_cfg

        return self.get_missing_fields()

    def load_from_yaml(
            self, 
            yaml_path: Union[str, Path], 
            model_config_path: str, 
            simulation_config_path: Optional[str] = None,
            thinning_config_path: Optional[str] = None,
            scheduler_config_path: Optional[str] = None):
        """
        Charge la config model depuis un YAML.
        
        Args:
            yaml_path: Chemin vers le fichier YAML
            model_config_path: Chemin vers la config (ex: 'model_configs.neural_small')
        """
        data = self._load_yaml(yaml_path)
        return self.from_dict(
            data,
            model_config_path,
            simulation_config_path,
            thinning_config_path,
            scheduler_config_path
        )

    def set_num_event_types(self, n: int):
        return self.set_field("num_event_types", n)

    def set_specs(self, specs: Union[Dict[str, Any], Any]):
        return self.set_field("specs", specs)

    def get_missing_fields(self) -> List[str]:
        required = ["model_id", "num_event_types"]
        return [f for f in required if f not in self.config_dict]


class TrainingConfigBuilder(ConfigBuilder):
    
    def __init__(self):
        super().__init__(ConfigType.TRAINING)
    
    def from_dict(self, data: Dict[str, Any], training_config_path: str):
        """Charge la configuration d'entraînement depuis un dictionnaire."""
        training_cfg = self._get_nested_value(data, training_config_path)
        self.config_dict = training_cfg
        return self.get_missing_fields()
    
    def load_from_yaml(self, yaml_path: Union[str, Path], training_config_path: str):
        """Charge la configuration d'entraînement depuis un YAML."""
        data = self._load_yaml(yaml_path)
        return self.from_dict(data, training_config_path)
    
    def set_max_epochs(self, epochs: int):
        """Définit le nombre maximum d'époques."""
        return self.set_field("max_epochs", epochs)
    
    def set_batch_size(self, batch_size: int):
        """Définit la taille du batch."""
        return self.set_field("batch_size", batch_size)
    
    def set_learning_rate(self, lr: float):
        """Définit le taux d'apprentissage."""
        return self.set_field("lr", lr)
    
    def set_devices(self, devices: int):
        """Définit le nombre de devices/GPU."""
        return self.set_field("devices", devices)
    
    def set_patience(self, patience: int):
        """Définit la patience pour l'early stopping."""
        return self.set_field("patience", patience)
    
    def set_dropout(self, dropout: float):
        """Définit le taux de dropout."""
        return self.set_field("dropout", dropout)
    
    def get_missing_fields(self) -> List[str]:
        required = ["max_epochs"]
        return [f for f in required if f not in self.config_dict]


class DataLoadingSpecsBuilder(ConfigBuilder):
    
    def __init__(self):
        super().__init__(ConfigType.DATA_LOADING_SPECS)
    
    def from_dict(self, data: Dict[str, Any], data_loading_config_path: str):
        """Charge les spécifications de chargement depuis un dictionnaire."""
        loading_cfg = self._get_nested_value(data, data_loading_config_path)
        self.config_dict = loading_cfg
        return self.get_missing_fields()
    
    def load_from_yaml(self, yaml_path: Union[str, Path], data_loading_config_path: str):
        """Charge les spécifications de chargement depuis un YAML."""
        data = self._load_yaml(yaml_path)
        return self.from_dict(data, data_loading_config_path)
    
    def set_batch_size(self, batch_size: int):
        """Définit la taille du batch."""
        return self.set_field("batch_size", batch_size)
    
    def set_num_workers(self, num_workers: int):
        """Définit le nombre de workers pour le chargement des données."""
        return self.set_field("num_workers", num_workers)
    
    def set_shuffle(self, shuffle: bool):
        """Définit si les données doivent être mélangées."""
        return self.set_field("shuffle", shuffle)
    
    def set_pin_memory(self, pin_memory: bool):
        """Définit si la mémoire doit être épinglée (GPU)."""
        return self.set_field("pin_memory", pin_memory)
    
    def set_drop_last(self, drop_last: bool):
        """Définit si le dernier batch incomplet doit être supprimé."""
        return self.set_field("drop_last", drop_last)
    
    def set_persistent_workers(self, persistent: bool):
        """Définit si les workers doivent être persistants."""
        return self.set_field("persistent_workers", persistent)
    
    def get_missing_fields(self) -> List[str]:
        required = ["batch_size"]
        return [f for f in required if f not in self.config_dict]


class DataConfigBuilder(ConfigBuilder):
    
    def __init__(self):
        super().__init__(ConfigType.DATA)
        self.data_loading_builder = DataLoadingSpecsBuilder()
    
    def from_dict(self, data: Dict[str, Any],
                  data_config_path: str,
                  data_loading_config_path: str = None,
                  tokenizer_specs_path: str = None):
        data_cfg = self._get_nested_value(data, data_config_path)
        
        # Assurer que dataset_id existe
        if isinstance(data_cfg, dict) and "dataset_id" not in data_cfg:
            dataset_id = data_config_path.split('.')[-1]
            data_cfg["dataset_id"] = dataset_id

        # Utiliser DataLoadingSpecsBuilder si demandé
        if data_loading_config_path:
            self.data_loading_builder.from_dict(data, data_loading_config_path)
            dl_cfg = self.data_loading_builder.get_config_dict()
            data_cfg.setdefault("data_loading_specs", dl_cfg)

        # Merge tokenizer_specs si demandé
        if tokenizer_specs_path:
            specs_cfg = self._get_nested_value(data, tokenizer_specs_path)
            data_cfg.setdefault("tokenizer_specs", specs_cfg)
        # Edge case: if src_dir is present and any required dir is missing, set all to src_dir
        required_dirs = ["train_dir", "valid_dir", "test_dir"]

        if "src_dir" in data_cfg:
            for d in required_dirs:
                if d not in data_cfg:
                    data_cfg[d] = data_cfg["src_dir"]
            data_cfg.pop("src_dir", None)
        
        self.config_dict = data_cfg
        return self.get_missing_fields()

    def load_from_yaml(self, yaml_path: Union[str, Path], 
                      data_config_path: str,
                      data_loading_config_path: str = None,
                      data_specs_path: str = None):
        """
        Charge la config data depuis un YAML.
        
        Args:
            yaml_path: Chemin vers le fichier YAML
            data_config_path: Chemin vers la config data (ex: 'data_configs.test')
            data_loading_config_path: Chemin optionnel vers data_loading_config
            data_specs_path: Chemin optionnel vers tokenizer_specs
        """
        data = self._load_yaml(yaml_path)
        return self.from_dict(
            data,
            data_config_path,
            data_loading_config_path,
            data_specs_path
        )

    def set_train_dir(self, path: str):
        return self.set_field("train_dir", path)

    def set_valid_dir(self, path: str):
        return self.set_field("valid_dir", path)

    def set_test_dir(self, path: str):
        return self.set_field("test_dir", path)

    def set_src_dir(self, path: str):
        """
        Helper to set train/valid/test directories at once from a single source directory.

        This mirrors the YAML fallback behavior where a `src_dir` may be provided.
        It sets `train_dir`, `valid_dir` and `test_dir` to `path` and returns the
        remaining missing fields (usually none after this call).
        """
        self.config_dict["train_dir"] = path
        self.config_dict["valid_dir"] = path
        self.config_dict["test_dir"] = path
        return self.get_missing_fields()

    def set_data_loading_specs(self, specs: Union[Dict[str, Any], Any]):
        """Définit les spécifications de chargement des données."""
        return self.set_field("data_loading_specs", specs)

    def set_tokenizer_specs(self, specs: Union[Dict[str, Any], Any]):
        """Définit les spécifications des données (tokenizer, etc.)."""
        return self.set_field("tokenizer_specs", specs)

    # Méthodes de commodité pour les spécifications de chargement courantes
    def set_batch_size(self, batch_size: int):
        """Définit la taille du batch dans data_loading_specs."""
        if "data_loading_specs" not in self.config_dict:
            self.config_dict["data_loading_specs"] = {}
        self.config_dict["data_loading_specs"]["batch_size"] = batch_size
        return self.get_missing_fields()
    
    def set_num_workers(self, num_workers: int):
        """Définit le nombre de workers dans data_loading_specs."""
        if "data_loading_specs" not in self.config_dict:
            self.config_dict["data_loading_specs"] = {}
        self.config_dict["data_loading_specs"]["num_workers"] = num_workers
        return self.get_missing_fields()
    
    def set_shuffle(self, shuffle: bool):
        """Définit le mélange des données dans data_loading_specs."""
        if "data_loading_specs" not in self.config_dict:
            self.config_dict["data_loading_specs"] = {}
        self.config_dict["data_loading_specs"]["shuffle"] = shuffle
        return self.get_missing_fields()
    
    # Méthodes de commodité pour les spécifications de données courantes
    def set_num_event_types(self, num_types: int):
        """Définit le nombre de types d'événements."""
        self.config_dict["num_event_types"] = num_types
        return self.get_missing_fields()
    
    def set_max_len(self, max_len: int):
        """Définit la longueur maximale des séquences dans tokenizer_specs."""
        if "tokenizer_specs" not in self.config_dict:
            self.config_dict["tokenizer_specs"] = {}
        self.config_dict["tokenizer_specs"]["max_len"] = max_len
        return self.get_missing_fields()
    
    def set_padding_side(self, side: str):
        """Définit le côté de padding dans tokenizer_specs (left/right)."""
        if "tokenizer_specs" not in self.config_dict:
            self.config_dict["tokenizer_specs"] = {}
        self.config_dict["tokenizer_specs"]["padding_side"] = side
        return self.get_missing_fields()

    def set_dataset_id(self, dataset_id: str):
        return self.set_field("dataset_id", dataset_id)
    
    def set_data_format(self, data_format: str):
        return self.set_field("data_format", data_format)

    def get_missing_fields(self) -> List[str]:
        required = ["train_dir", "valid_dir", "test_dir", "dataset_id"]
        # If src_dir is present, treat all required as present
        if "src_dir" in self.config_dict:
            return []
        return [f for f in required if f not in self.config_dict]