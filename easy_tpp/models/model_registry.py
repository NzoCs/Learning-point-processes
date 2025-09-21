"""
Registry pour les modèles TPP avec validation automatique

Ce module fournit un registry qui vérifie automatiquement que l'enum Models
contient bien tous les modèles disponibles dans le projet.

Utilisation:
    from easy_tpp.models.model_registry import ModelRegistry
    
    registry = ModelRegistry()
    registry.validate_models()  # Vérifie que l'enum est à jour
    missing = registry.get_missing_models()  # Modèles manquants
"""

import os
import inspect
import importlib
from enum import Enum
from typing import Type, List, Dict, Set
from pathlib import Path

from easy_tpp.models.basemodel import BaseModel
from easy_tpp.utils import logger

# Import des classes de modèles
from easy_tpp.models.nhp import NHP
from easy_tpp.models.sahp import SAHP
from easy_tpp.models.thp import THP
from easy_tpp.models.rmtpp import RMTPP
from easy_tpp.models.attnhp import AttNHP
from easy_tpp.models.anhn import ANHN
from easy_tpp.models.hawkes import Hawkes
from easy_tpp.models.self_correcting import SelfCorrecting
from easy_tpp.models.intensity_free import IntensityFree
from easy_tpp.models.ode_tpp import ODETPP
from easy_tpp.models.fullynn import FullyNN


class Models(Enum):
    """Enum des modèles disponibles avec leurs classes. Aide pour les autocomplétion et pour
    éviter les erreurs."""
    
    # Modèles basés sur l'intensité
    NHP = ("NHP", NHP)
    SAHP = ("SAHP", SAHP)
    THP = ("THP", THP)
    RMTPP = ("RMTPP", RMTPP)
    AttNHP = ("AttNHP", AttNHP)
    ANHN = ("ANHN", ANHN)
    HAWKES = ("Hawkes", Hawkes)
    SELF_CORRECTING = ("SelfCorrecting", SelfCorrecting)
    INTENSITY_FREE = ("IntensityFree", IntensityFree)
    ODE_TPP = ("ODE_TPP", ODETPP)
    FULLYNN = ("FullyNN", FullyNN)

    @classmethod
    def get_registry(cls) -> Dict[str, Type[BaseModel]]:
        """Retourne un dictionnaire model_name : class"""
        registry = {}
        for model in cls:
            registry[model.get_class_name()] = model.get_class()
        return registry

    def __init__(self, model_name: str, model_class: Type[BaseModel]):
        self.model_name = model_name
        self.model_class = model_class

    def get_class_name(self) -> str:
        """Obtenir le nom de la classe du modèle."""
        return self.model_name
    
    def get_class(self) -> Type[BaseModel]:
        """Obtenir la classe du modèle."""
        return self.model_class


class ModelRegistry:
    """Registry pour valider et gérer les modèles TPP."""
    
    def __init__(self):
        self.models_dir = Path(__file__).parent
        self._discovered_models: Dict[str, Type[BaseModel]] = {}
        self._enum_models: Dict[str, Type[BaseModel]] = {}
        self._discover_all_models()
        self._extract_enum_models()
    
    def _discover_all_models(self):
        """Découvrir tous les modèles disponibles dans le dossier models."""
        logger.info("Découverte de tous les modèles disponibles...")
        
        # Lister tous les fichiers Python dans le dossier models
        python_files = [f for f in self.models_dir.glob("*.py") 
                       if f.name not in ["__init__.py", "basemodel.py", "model_factory.py", "model_registry.py"]]
        
        for py_file in python_files:
            module_name = f"easy_tpp.models.{py_file.stem}"
            
            try:
                module = importlib.import_module(module_name)
                
                # Chercher les classes qui héritent de BaseModel
                for name in dir(module):
                    obj = getattr(module, name)
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseModel) and 
                        obj != BaseModel and
                        obj.__module__ == module_name):  # Éviter les imports
                        
                        self._discovered_models[obj.__name__] = obj
                        logger.debug(f"Modèle découvert: {obj.__name__} dans {module_name}")
                        
            except ImportError as e:
                logger.debug(f"Impossible d'importer {module_name}: {e}")
            except Exception as e:
                logger.warning(f"Erreur lors de l'analyse de {module_name}: {e}")
        
        logger.info(f"Découverte terminée: {len(self._discovered_models)} modèles trouvés")
    
    def _extract_enum_models(self):
        """Extraire les modèles de l'enum Models."""
        for model_enum in Models:
            model_class = model_enum.get_class()
            self._enum_models[model_class.__name__] = model_class
    
    def validate_models(self) -> bool:
        """
        Valider que l'enum Models contient tous les modèles disponibles.
        
        Returns:
            True si tous les modèles sont présents, False sinon
        """
        logger.info("Validation de l'enum Models...")
        
        missing = self.get_missing_models()
        extra = self.get_extra_models()
        
        if not missing and not extra:
            logger.info("✅ L'enum Models est à jour!")
            return True
        
        if missing:
            logger.warning(f"❌ Modèles manquants dans l'enum: {missing}")
        
        if extra:
            logger.warning(f"⚠️ Modèles dans l'enum mais pas trouvés: {extra}")
        
        return False
    
    def get_missing_models(self) -> List[str]:
        """Obtenir la liste des modèles manquants dans l'enum."""
        discovered_names = set(self._discovered_models.keys())
        enum_names = set(self._enum_models.keys())
        
        missing = discovered_names - enum_names
        return sorted(list(missing))
    
    def get_extra_models(self) -> List[str]:
        """Obtenir la liste des modèles dans l'enum mais pas découverts."""
        discovered_names = set(self._discovered_models.keys())
        enum_names = set(self._enum_models.keys())
        
        extra = enum_names - discovered_names
        return sorted(list(extra))
    
    def get_all_discovered_models(self) -> Dict[str, Type[BaseModel]]:
        """Obtenir tous les modèles découverts."""
        return self._discovered_models.copy()
    
    def get_enum_models(self) -> Dict[str, Type[BaseModel]]:
        """Obtenir tous les modèles de l'enum."""
        return self._enum_models.copy()
    
    def generate_enum_code(self) -> str:
        """
        Générer le code pour un enum Models complet avec tous les modèles découverts.
        
        Returns:
            Code Python pour l'enum Models
        """
        logger.info("Génération du code pour l'enum Models...")
        
        imports = []
        enum_entries = []
        
        # Générer les imports et les entrées de l'enum
        for class_name, model_class in sorted(self._discovered_models.items()):
            module_name = model_class.__module__.split('.')[-1]  # Dernier partie du module
            imports.append(f"from easy_tpp.models.{module_name} import {class_name}")
            
            # Créer une entrée d'enum
            enum_name = class_name.upper().replace('TPP', '_TPP')  # Normaliser le nom
            enum_entries.append(f'    {enum_name} = ("{class_name}", {class_name})')
        
        # Générer le code complet
        code = f'''class Models(Enum):
    """Enum des modèles disponibles avec leurs classes."""
    
{chr(10).join(enum_entries)}

    def __init__(self, model_name: str, model_class: Type[BaseModel]):
        self.model_name = model_name
        self.model_class = model_class

    def get_class_name(self) -> str:
        """Obtenir le nom de la classe du modèle."""
        return self.model_name
    
    def get_class(self) -> Type[BaseModel]:
        """Obtenir la classe du modèle."""
        return self.model_class'''
        
        return code
    
    def print_validation_report(self):
        """Imprimer un rapport de validation détaillé."""
        print("\n" + "="*50)
        print("📋 RAPPORT DE VALIDATION DES MODÈLES")
        print("="*50)
        
        print(f"\n🔍 Modèles découverts: {len(self._discovered_models)}")
        for name in sorted(self._discovered_models.keys()):
            print(f"  ✓ {name}")
        
        print(f"\n📝 Modèles dans l'enum: {len(self._enum_models)}")
        for name in sorted(self._enum_models.keys()):
            print(f"  ✓ {name}")
        
        missing = self.get_missing_models()
        if missing:
            print(f"\n❌ Modèles manquants ({len(missing)}):")
            for name in missing:
                print(f"  • {name}")
        
        extra = self.get_extra_models()
        if extra:
            print(f"\n⚠️ Modèles en trop ({len(extra)}):")
            for name in extra:
                print(f"  • {name}")
        
        if not missing and not extra:
            print(f"\n✅ L'enum Models est parfaitement à jour!")
        else:
            print(f"\n❌ L'enum Models nécessite une mise à jour")
        
        print("="*50)
    
    def print_suggested_enum(self):
        """Imprimer le code suggéré pour l'enum complet."""
        print("\n" + "="*50)
        print("💡 CODE SUGGÉRÉ POUR L'ENUM MODELS")
        print("="*50)
        
        code = self.generate_enum_code()
        print(code)
        
        print("="*50)


class RegistryMeta(type):
    """Metaclasse pour enregistrer automatiquement les nouveaux modèles."""
    
    def __new__(mcls, name, bases, namespace):
        cls = super().__new__(mcls, name, bases, namespace)
        
        # Enregistrer automatiquement les nouvelles classes de modèles
        if bases and any(issubclass(base, BaseModel) for base in bases):
            # Vérifier que ce n'est pas BaseModel lui-même
            if cls != BaseModel:
                # Ajouter automatiquement à l'enum Models si pas déjà présent
                try:
                    # Vérifier si le modèle existe déjà dans l'enum
                    model_exists = any(model.get_class_name() == name for model in Models)
                    
                    if not model_exists:
                        logger.info(f"🔄 Nouveau modèle détecté: {name}")
                        logger.warning(f"⚠️ Le modèle '{name}' n'est pas dans l'enum Models")
                        logger.info(f"💡 Ajoutez cette ligne à l'enum Models:")
                        logger.info(f"    {name.upper()} = (\"{name}\", {name})")
                        
                except Exception as e:
                    logger.debug(f"Erreur lors de la vérification du modèle {name}: {e}")
        
        return cls