# Changelog

Toutes les modifications importantes de ce projet seront documentées dans ce fichier.

Le format est basé sur [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
et ce projet adhère au [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Configuration des bonnes pratiques Git
- Templates de commit standardisés
- Templates de Pull Request
- Aliases Git pour améliorer la productivité
- Documentation des workflows dans `.github/README.md`

### Changed
- Migration de `enzo-feature` vers `dev` comme branche de développement principal
- Synchronisation de `main` et `master` avec le code de `dev`

### Infrastructure
- Configuration Git avec rebase par défaut
- Mise en place de templates pour standardiser les contributions

## [1.0.0] - 2025-09-21

### Added
- Structure complète du projet EasyTPP
- Modèles de processus ponctuels (NHP, SAHP, THP, etc.)
- Système d'évaluation et de benchmarking
- Configuration avec Factory Pattern
- Tests unitaires et d'intégration
- Documentation complète
- Workflows GitHub Actions (CI, lint, deploy)
- Support de génération de données synthétiques

### Infrastructure
- Configuration avec pyproject.toml
- Support UV pour la gestion des dépendances
- Makefile avec commandes utiles
- Scripts CLI pour faciliter l'utilisation