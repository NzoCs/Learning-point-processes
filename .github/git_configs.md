# 🚀 Configuration Git & Bonnes Pratiques

Ce document décrit toutes les bonnes pratiques Git mises en place sur ce repository pour maintenir un code de qualité et un workflow efficace.

## ✅ Configuration Git Appliquée

### Configuration globale Git
Les configurations suivantes ont été appliquées globalement :

```bash
# Historique propre avec rebase lors des pulls
git config --global pull.rebase true

# Push seulement la branche courante
git config --global push.default current

# Branche par défaut = main pour nouveaux repos
git config --global init.defaultBranch main

# Gestion correcte des fins de ligne sur Windows
git config --global core.autocrlf input
```

### Aliases Git configurés
Pour améliorer la productivité :

```bash
git co         # = git checkout
git br         # = git branch
git ci         # = git commit
git st         # = git status
git unstage    # = git reset HEAD --
git last       # = git log -1 HEAD
git graph      # = git log --oneline --graph --decorate --all
```

### Template de commit
Un template de commit a été créé (`.gitmessage`) pour standardiser les messages suivant les **Conventional Commits** :

```
<type>(<scope>): <description>

Types disponibles:
- feat     : Nouvelle fonctionnalité
- fix      : Correction de bug
- docs     : Documentation
- style    : Formatage (pas de changement de code)
- refactor : Refactoring
- test     : Tests
- chore    : Maintenance (dépendances, config)
```

## 🌿 Structure des branches recommandée

```
main (production - toujours stable)
├── dev (développement - intégration)
│   ├── feature/model-improvements
│   ├── feature/new-evaluation-metrics
│   └── feature/better-config-system
└── hotfix/critical-bug-fix (si urgence)
```

## 🔄 Workflow recommandé

### Pour une nouvelle fonctionnalité :
```bash
# 1. Partir de dev
git checkout dev
git pull origin dev
git checkout -b feature/nom-fonctionnalite

# 2. Développer avec commits conventionnels
git add .
git commit  # Utilise le template automatiquement

# 3. Push et Pull Request
git push origin feature/nom-fonctionnalite
# Créer PR vers dev sur GitHub

# 4. Après merge, nettoyer
git checkout dev
git pull origin dev
git branch -d feature/nom-fonctionnalite
```

## 📋 Actions à faire

### ⏳ En cours
- [x] Configuration Git de base
- [x] Aliases Git  
- [x] Template de commit
- [x] Templates Pull Request
- [x] CHANGELOG.md
- [x] Guide de contribution complet
- [ ] Protection des branches sur GitHub
- [ ] Amélioration des workflows CI/CD

### 🎯 Prochaines étapes
1. **Protection des branches** : Configurer la protection sur GitHub pour `main` et `dev`
2. **Templates PR** : Créer des templates de Pull Request
3. **CHANGELOG** : Mettre en place un système de changelog automatique
4. **CI/CD** : Améliorer les workflows existants
5. **Documentation** : Guide complet de contribution

## 🛠️ Workflows GitHub Actions existants

Le projet dispose déjà de workflows dans `.github/workflows/` :
- `ci.yml` : Tests d'intégration continue
- `deploy.yml` : Déploiement
- `lint.yml` : Vérification du code
- `depedandabot.yml` : Mise à jour automatique des dépendances

---

**Dernière mise à jour :** 21 septembre 2025  
**Status :** Documentation complète, protection des branches à configurer

## 📄 Fichiers créés

- `.gitmessage` : Template de commit avec conventions
- `.github/pull_request_template.md` : Template standardisé pour les PR
- `CHANGELOG.md` : Suivi des modifications du projet
- `CONTRIBUTING.md` : Guide complet de contribution (180+ lignes)