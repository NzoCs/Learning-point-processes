# Configuration Refactoring

Ce document explique la refactorisation des fonctions `from_dict` dans le système de configuration.

## Objectifs de la refactorisation

La refactorisation a été conçue pour séparer clairement les responsabilités :

1. **Validation** : S'assurer que le dictionnaire peut être traduit en instance de classe
2. **Création d'instance** : Créer une instance de la classe à partir du dictionnaire validé
3. **Transformation** : Gérer les spécificités liées aux différents formats d'entrée (déplacé vers des utilitaires séparés)

## Structure après refactorisation

### 1. Fonctions `from_dict` simplifiées

Chaque classe de configuration a maintenant une fonction `from_dict` épurée qui suit ce pattern :

```python
@classmethod
def from_dict(cls, config_dict: Dict[str, Any]) -> "ConfigClass":
    from easy_tpp.config_factory.config_utils import ConfigValidator
    
    # 1. Validation du dictionnaire
    ConfigValidator.validate_required_fields(
        config_dict, cls._get_required_fields_list(), "ConfigClass"
    )
    filtered_dict = ConfigValidator.filter_invalid_fields(config_dict, cls)
    
    # 2. Création des sous-configurations si nécessaire
    if "sub_config" in filtered_dict and isinstance(filtered_dict["sub_config"], dict):
        filtered_dict["sub_config"] = SubConfig.from_dict(filtered_dict["sub_config"])
    
    # 3. Création de l'instance
    return cls(**filtered_dict)
```

### 2. Utilitaires de transformation (`config_utils.py`)

Les transformations spécifiques aux formats d'entrée sont maintenant dans des utilitaires séparés :

- **`ConfigTransformer`** : Gère les transformations de format (aliases, conversions de type, etc.)
- **`ConfigValidator`** : Gère la validation des dictionnaires

### 3. Utilisation

#### Utilisation simple (dictionnaire déjà propre)
```python
config_dict = {
    "model_id": "NHP",
    "num_event_types": 5,
    "device": "cuda"
}
model_config = ModelConfig.from_dict(config_dict)
```

#### Utilisation avec transformation (format legacy/YAML)
```python
yaml_config = {
    "model_id": "NHP", 
    "num_event_types": 5,
    "base_config": {
        "dropout_rate": 0.1,  # Alias pour "dropout"
        "backend": "torch"    # String au lieu d'enum
    }
}

# 1. Transformer le format
transformed = ConfigTransformer.transform_model_config(yaml_config)

# 2. Créer l'instance
model_config = ModelConfig.from_dict(transformed)
```

## Avantages de cette approche

1. **Séparation des responsabilités** : 
   - `from_dict` = validation + création d'instance
   - `ConfigTransformer` = gestion des formats d'entrée

2. **Flexibilité** : 
   - Possibilité d'avoir plusieurs transformateurs pour différents formats
   - `from_dict` reste simple et prévisible

3. **Maintenabilité** :
   - Code plus lisible et plus facile à déboguer
   - Transformations centralisées et réutilisables

4. **Extensibilité** :
   - Facile d'ajouter de nouveaux transformateurs
   - `from_dict` n'a pas besoin d'être modifié pour de nouveaux formats

## Classes modifiées

- `TrainerConfig` (runner_config.py)
- `RunnerConfig` (runner_config.py)
- `ModelConfig` (model_config.py)
- `TrainingConfig` (model_config.py)
- `ModelSpecsConfig` (model_config.py)
- `ThinningConfig` (model_config.py)
- `SimulationConfig` (model_config.py)
- `DataConfig` (data_config.py)
- `DataLoadingSpecsConfig` (data_config.py)
- `TokenizerConfig` (data_config.py)
- `LoggerConfig` (logger_config.py)

## Fichiers ajoutés

- `config_utils.py` : Utilitaires de transformation et validation
- `config_examples.py` : Exemples d'utilisation des nouvelles fonctionnalités

## Migration

### Avant
```python
# Tout était mélangé dans from_dict
config = SomeConfig.from_dict(messy_yaml_dict)
```

### Après
```python
# Pour un dictionnaire propre
config = SomeConfig.from_dict(clean_dict)

# Pour un format legacy/YAML
transformed = ConfigTransformer.transform_some_config(messy_yaml_dict)
config = SomeConfig.from_dict(transformed)
```
