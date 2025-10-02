# 🤝 Guide de Contribution

Merci de votre intérêt pour contribuer au projet Learning Point Processes ! Ce guide vous explique comment contribuer efficacement.

## 🚀 Démarrage rapide

1. **Forker** le repository
2. **Cloner** votre fork
3. **Configurer** l'upstream
4. **Créer** une branche de feature
5. **Développer** et tester
6. **Soumettre** une Pull Request

## 📋 Prérequis

- Python 3.8+
- Git configuré avec les bonnes pratiques (voir `.github/README.md`)
- UV installé pour la gestion des dépendances

### Installation

```bash
# Cloner le repository
git clone https://github.com/NzoCs/Learning-point-processes.git
cd Learning-point-processes

# Configurer l'upstream
git remote add upstream https://github.com/NzoCs/Learning-point-processes.git

# Installer les dépendances
uv sync

# Activer l'environnement virtuel
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows
```

## 🌿 Workflow des branches

### Structure des branches
- **`main`** : Version stable de production
- **`dev`** : Branche de développement (intégration)
- **`feature/nom-feature`** : Nouvelles fonctionnalités
- **`hotfix/nom-fix`** : Corrections urgentes

### Workflow standard

```bash
# 1. Se synchroniser avec upstream
git checkout dev
git pull upstream dev

# 2. Créer une branche de feature
git checkout -b feature/nouvelle-fonctionnalite

# 3. Développer avec des commits atomiques
git add .
git commit  # Utilise automatiquement le template

# 4. Pousser vers votre fork
git push origin feature/nouvelle-fonctionnalite

# 5. Créer une Pull Request vers dev
```

## 📝 Conventions de commit

Nous utilisons les **Conventional Commits** :

### Format
```
<type>(<scope>): <description>

[body optionnel]

[footer optionnel]
```

### Types disponibles
- **feat**: Nouvelle fonctionnalité
- **fix**: Correction de bug  
- **docs**: Documentation
- **style**: Formatage (sans changement de logique)
- **refactor**: Refactoring
- **test**: Tests
- **chore**: Maintenance (dépendances, config)

### Exemples
```bash
feat(models): ajouter nouveau modèle SAHP avec support GPU
fix(config): corriger validation des paramètres HPO
docs(readme): mettre à jour guide d'installation
test(evaluation): ajouter tests pour métriques de simulation
refactor(factory): simplifier création des modèles avec builder pattern
```

## ✅ Standards de code

### Style Python
- Suivre **PEP 8**
- Utiliser **Black** pour le formatage
- **Isort** pour l'organisation des imports
- **Flake8** pour le linting

### Commandes de vérification
```bash
# Formatage automatique
make format

# Vérification du style
make lint

# Tests
make test

# Couverture
make coverage
```

### Docstrings
Utiliser le format **Google Style** :

```python
def compute_likelihood(events: torch.Tensor, model_params: Dict) -> float:
    """Calcule la log-vraisemblance des événements.
    
    Args:
        events: Tensor des événements [batch_size, seq_len, 2]
        model_params: Paramètres du modèle
        
    Returns:
        Log-vraisemblance moyenne
        
    Raises:
        ValueError: Si les events sont vides
    """
```

## 🧪 Tests

### Structure des tests
```
tests/
├── unit/           # Tests unitaires
├── integration/    # Tests d'intégration  
├── functional/     # Tests fonctionnels
└── conftest.py     # Configuration pytest
```

### Écriture des tests
```python
def test_nhp_model_training():
    """Test l'entraînement du modèle NHP."""
    # Arrange
    config = create_test_config()
    model = NHPModel(config)
    
    # Act
    loss = model.fit(test_data)
    
    # Assert
    assert loss < 1000
    assert model.is_fitted
```

### Coverage minimal
- **90%** pour le code nouveau
- **80%** pour le code modifié
- Tests obligatoires pour les nouvelles features

## 📄 Documentation

### README
- Mettre à jour si changement d'API
- Ajouter des exemples pour nouvelles features
- Maintenir les badges à jour

### CHANGELOG
- Ajouter entrée pour chaque changement notable
- Suivre le format [Keep a Changelog](https://keepachangelog.com/)
- Catégoriser : Added, Changed, Deprecated, Removed, Fixed, Security

### Docstrings
- Toutes les fonctions publiques
- Classes et méthodes importantes  
- Modules avec description du but

## 🔍 Pull Request

### Checklist avant soumission
- [ ] Tests passent (`make test`)
- [ ] Couverture suffisante (`make coverage`)
- [ ] Code formaté (`make format`) 
- [ ] Pas de warnings lint (`make lint`)
- [ ] Documentation mise à jour
- [ ] CHANGELOG.md mis à jour
- [ ] Commits squashés si nécessaire

### Template PR
Le template sera automatiquement appliqué. Remplissez toutes les sections :

- **Description** claire du changement
- **Type de changement** coché
- **Tests effectués** documentés
- **Checklist** complétée

### Review process
1. **Automated checks** doivent passer (CI)
2. **Code review** par au moins 1 mainteneur
3. **Tests manuels** si nécessaire
4. **Approbation** avant merge
5. **Squash and merge** vers dev

## 🚨 Hotfixes

Pour les corrections urgentes sur production :

```bash
# Partir de main
git checkout main
git pull upstream main
git checkout -b hotfix/nom-correction

# Développer la correction
# ... commits ...

# Créer PR vers main ET dev
```

## 💡 Bonnes pratiques

### Commits
- **Atomiques** : Un changement logique par commit
- **Descriptifs** : Message clair de ce qui change
- **Testés** : Chaque commit compile et les tests passent

### Branches
- **Courtes** : Quelques jours maximum
- **Focalisées** : Une seule fonctionnalité par branche
- **Synchronisées** : Rebase fréquent sur dev

### Code
- **SOLID principles**
- **DRY** : Don't Repeat Yourself
- **KISS** : Keep It Simple, Stupid
- **Tests first** quand possible

## 🆘 Support

- **Issues GitHub** : Pour bugs et feature requests
- **Discussions** : Pour questions générales
- **Email** : enzo.regna@example.com pour questions urgentes

## 📞 Contact

- **Mainteneur principal** : @NzoCs
- **Repository** : https://github.com/NzoCs/Learning-point-processes

---

Merci de contribuer à améliorer ce projet ! 🙏