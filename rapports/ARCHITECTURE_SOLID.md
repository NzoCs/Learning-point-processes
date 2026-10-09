# Rapport et consignes — Architecture, SOLID et conception objet

Date : 9 octobre 2026. Dépôt : `NzoCs/Learning-point-processes`.
Référence examinée : `fdb6c9fb7d1116dc1f4311259c349fa383480612`.
Statut : propositions et plan de travail, pas un refactoring déjà réalisé.
Rapport complémentaire : [Reproductibilité et Ruche](REPRODUCTIBILITE_RUCHE.md).

Mise à jour de consigne : le passage à pySigLib est désormais demandé dans le rapport de reproductibilité, lot R6. Les mentions ci-dessous de `sigkernel` obligatoire s'appliquent à la référence pendant la transition : la fonctionnalité de noyau de signature reste obligatoire, mais pySigLib avec support CUDA peut remplacer la dépendance de production après validation. Ce remplacement relève de R6, pas des lots de refactoring A0–A5.

## 1. Objectif et contraintes non négociables

Rendre l'entraînement, la simulation et l'évaluation exécutables indépendamment, avec des contrats explicites et des résultats traçables. Conserver les modèles et les calculs scientifiques existants tant qu'une modification scientifique n'a pas été validée séparément.

- Le calcul GPU sur Ruche est un usage principal, pas un ajout secondaire.
- `sigkernel` reste une dépendance obligatoire. Une abstraction de backend ne signifie ni installation facultative ni remplacement automatique.
- Conserver les callbacks Lightning utiles : checkpoints, early stopping, progression, journalisation. Extraire leur orchestration scientifique, pas supprimer tous les callbacks.
- Préférer une migration incrémentale aux réécritures de modèles ou au remplacement de Lightning.
- Respecter SOLID de façon pragmatique : pas une interface et une factory pour chaque classe, pas de conteneur d'injection de dépendances sans besoin démontré.
- Aucun changement de formule de loss, d'estimateur MMD, de normalisation ou de protocole statistique dans une PR présentée comme simple refactoring.

L'exécution des tests a précédemment été bloquée sur ce poste Windows par la compilation native de `sigkernel`. Les constats ci-dessous proviennent de la lecture du code ; ils ne constituent pas une validation d'exécution sur Ruche.

## 2. Constats dans l'architecture actuelle

Les chemins de code ci-dessous sont relatifs à la racine du dépôt pour rendre ce document portable.

| Zone | Constat | Conséquence et priorité |
| --- | --- | --- |
| `new_ltpp/configs/runner_config.py` | Le validateur construit les dossiers et réécrit les chemins du logger. | Valider une configuration modifie le filesystem ; tests et réutilisation difficiles. P1. |
| `new_ltpp/configs/base_config.py` | `frozen=True`, mais `update()` utilise `setattr()` ; champs inconnus ignorés. | Contrat de mutabilité contradictoire et erreurs de configuration masquées. P1. |
| `new_ltpp/runners/model_runner.py` | Le checkpoint est choisi à la construction ; `train()` ne retourne pas le checkpoint sélectionné après `fit()`. | Une phase suivante peut consulter un état antérieur à l'entraînement. P0. |
| Même fichier | La propriété `trainer` fabrique un Trainer à chaque accès et ajoute les callbacks d'évaluation. | Cycle de vie et responsabilités de phase peu explicites. P1. |
| `new_ltpp/runners/runner_manager.py` | Relais supplémentaire ; certains paramètres stockés, dont le checkpoint et le dossier de sortie, ne sont pas transmis au Runner. | Plusieurs points de vérité et options potentiellement sans effet. P1. |
| `new_ltpp/runners/callbacks.py` | Injection de `_simulator` et `_io_manager` dans le modèle ; simulation, statistiques, écritures et figures dans les hooks. | La prédiction dépend implicitement d'un callback et de son ordre d'exécution. P0. |
| `new_ltpp/models/base/training.py` | `predict_step()` simule et persiste ; validation/test modifient les champs du batch. | Modèle couplé aux sorties ; mutation des entrées à caractériser. P1. |
| `new_ltpp/models/model_protocol.py` | Protocoles larges, incluant état privé et méthodes non supportées par tous les modèles. | Substitution fragile ; couplage inutile. P1. |
| `new_ltpp/models/implementations/intensity_free.py` | `compute_intensities()` lève `NotImplementedError`, alors que le simulateur générique attend une fonction d'intensité. | Capacité « intensité » non universelle ; besoin d'un dispatch explicite. P0. |
| `new_ltpp/evaluation/results_aggregator.py` | Append d'un CSV partagé avec colonnes variables et identifiant dataset. | Collisions, schéma ambigu, concurrence non protégée. P0. |
| `new_ltpp/models/simulation/tpp_io.py` | L'offset de séquences est incrémenté du nombre de lignes d'événements. | Les identifiants ne représentent pas correctement les séquences ; données vides non représentées. P0. |

Ces priorités visent la fiabilité des expériences, pas l'élégance abstraite. Les détails statistiques sont traités dans le second rapport.

## 3. Orchestration : avant et cible

### Avant, schéma simplifié du code actuel

```text
CLI → ExperimentRunner → RunnerConfig (création de dossiers)
    → RunnerManager → Runner (données, modèle, checkpoint initial)
    → nouveau Trainer à chaque accès
        ├─ fit → callbacks Lightning
        └─ test / predict → callbacks scientifiques
             → injection d'état privé dans le modèle
             → simulateur → collecteur → fichiers → CSV global → figures
```

### Après, cible incrémentale

```text
CLI → résolution et validation de configuration, sans I/O de sortie
    → ExperimentRunner : point de composition et orchestration des phases
    → RunContext : identité, chemins, provenance, statut
        ├─ TrainingService + TrainerFactory
        │    → fit → TrainingResult(checkpoint sélectionné)
        ├─ chargement explicite du checkpoint sélectionné
        ├─ EvaluationService
        │    → prédiction / SimulationEngine → métriques et tests statistiques
        │    → ResultWriter : streaming des simulations et résultats
        └─ PlotService : post-traitement des résultats persistés
```

Les noms proposés sont des contrats de responsabilité, pas une obligation de créer exactement ces classes. Réutiliser `Runner`, `Simulator` et les collecteurs lorsque leur séparation suffit. La CLI ne doit pas importer une chaîne de sous-systèmes pour une simple validation.

### Contrats minimaux à arrêter avant de coder

- `ResolvedExperimentConfig` : configuration finale validée, immuable, sérialisable ; ni création de dossiers ni découverte automatique d'un ancien run.
- `RunContext` : `run_id`, `experiment_label`, `config_fingerprint`, racine de sortie, identité de phase et références de provenance. Pas de modèle, de tenseurs ou de Trainer à l'intérieur.
- `TrainingResult` : checkpoint retenu, règle de sélection, métrique surveillée et valeur si disponible, statut de fin. Le nom `best-vN` n'est pas une preuve de meilleure métrique.
- `EvaluationRequest` : checkpoint et split explicites, configuration de simulation/statistique, graines dédiées, référence du run parent.
- `EvaluationResult` : métriques typées, définition des agrégations, tailles effectives, statut et références des artefacts. Ne pas garder toutes les simulations GPU en mémoire dans ce résultat.
- `SimulationEngine` : produit des lots de `SimulationResult` sans écrire ni tracer de figures. La stratégie peut utiliser l'intensité ou un échantillonnage direct selon le modèle.
- `ResultWriter` : reçoit ces lots et résultats ; gère schéma, fermeture, chemins et écriture atomique des petits fichiers de contrôle.

Les erreurs doivent identifier la phase et la cause. « Pas de données », « checkpoint absent », « capacité non supportée » ne sont pas des succès avec résultat vide ou p-value égale à 1.

## 4. SOLID appliqué au projet

### S — Une raison principale de changer par composant

Séparer apprentissage, simulation, métriques, stockage et visualisation. `ExperimentRunner` choisit l'ordre et transmet les résultats ; il ne calcule pas de MMD et ne formate pas du Parquet. Un modèle définit ses calculs, pas l'emplacement du CSV global.

Une classe qui « lance une expérience » reste légitime : elle orchestre des services cohérents. SRP ne signifie pas une méthode par classe.

### O — Extension aux frontières réellement variables

Ajouter un modèle, une stratégie de simulation, une métrique ou un writer par registre explicite ou petite interface adaptée. Le point de composition choisit les implémentations. Éviter les branches sur les noms de modèles dispersées dans les callbacks.

Conserver les factories utiles ; vérifier que seuls les modèles concrets sont enregistrés. L'enregistrement ne doit pas dépendre d'importer fortuitement tous les modules, ni inclure une classe abstraite. Une alternative à `sigkernel` n'est acceptée qu'après validation numérique et GPU séparée.

### L — Substitution fondée sur les capacités réelles

Un modèle sans intensité ne doit pas prétendre être interchangeable avec un modèle à intensité dans une opération qui l'exige. Définir les capacités consommées : vraisemblance, prédiction, intensité, échantillonnage direct. `IntensityFree` utilise son mécanisme direct ; ne pas inventer une intensité pour satisfaire une interface.

Tests de contrat par capacité : formes, masques, unités, device, dtype, séquences vides et erreurs. Faire échouer la demande incompatible avant de lancer un entraînement coûteux.

### I — Protocoles petits et orientés consommateurs

Retirer `_simulator`, `_io_manager`, journalisation et détails Lightning du contrat minimal d'un modèle simulable. Un writer n'a pas besoin de connaître la loss ; un noyau n'a pas besoin de connaître le Trainer.

Aligner les signatures avec les implémentations, notamment les masques d'événements et les types `Batch` / `SimulationResult`. Réutiliser les types existants avant d'introduire des DTO redondants.

### D — Injection explicite des dépendances

Construire simulateur, évaluateur et writer dans le point de composition, puis les transmettre par constructeur ou appel. Les services ne vont pas chercher leurs dépendances dans des attributs privés du modèle ou des chemins globaux.

Utiliser `typing.Protocol` pour les frontières testables quand c'est suffisant. Une ABC se justifie pour une réelle implémentation partagée ou une contrainte runtime. Les fonctions pures sont préférables aux objets sans état pour conversions et calculs simples.

### Principes OOP et design complémentaires

- Encapsulation : chaque ressource a un propriétaire ; fermeture garantie par context manager / `try/finally`.
- Composition plutôt qu'héritage pour les stratégies de simulation, backends et writers. Ne pas démanteler immédiatement l'héritage des modèles Lightning.
- Faible couplage et cohésion : dépendances métier vers contrats, implémentations infrastructure choisies au bord du système.
- Invariants : config finale valide, entrée non mutée, checkpoint explicite, sorties uniques ; documenter plutôt que supposer.
- Éviter le « service locator », la classe orchestratrice géante, le framework interne générique et les abstractions spéculatives.

## 5. Callbacks et cycle de vie Lightning

Conserver `ModelCheckpoint`, `EarlyStopping`, progression et logging. Un Trainer d'entraînement est construit explicitement pour une exécution ; un Trainer d'évaluation distinct est possible, mais intentionnel et sans callbacks d'entraînement inutiles.

L'évaluation finale devient appelable sans `fit()` et sans injection privée. Une évaluation périodique facultative peut rester un callback mince qui délègue au même `EvaluationService` : pas de deuxième implémentation des statistiques.

Ce callback doit restaurer mode train/eval et états RNG, éviter les changements de device, être borné en fréquence et en volume et ne pas utiliser le jeu de test pour choisir le checkpoint. Définir le comportement DDP : exécuté une fois sur un modèle/checkpoint exploitable, ou distribué avec réductions correctes ; éviter un appel rank-zero qui déclencherait des collectives sur les autres ranks absents.

Ne pas appliquer aveuglément `torch.inference_mode()` / `no_grad()` : certains calculs d'intensité ont besoin de dérivées même en évaluation. Tester et exprimer cette exigence dans l'adaptateur de capacité.

## 6. Plan d'implémentation et critères d'acceptation

### A0 — Tests de caractérisation et contrats, P0

Avant modification, documenter le parcours CLI, les sorties et les formes. Préparer de petits fixtures synthétiques et enregistrer les résultats de référence dans un environnement où `sigkernel` fonctionne. Identifier les défauts connus sans les figer comme comportements souhaités.

**Livrable :** tests couvrant `all`, entraînement seul, évaluation seule, checkpoint absent, modèle sans intensité, sortie concurrente. Tests unitaires avec doubles possibles ; un test d'intégration doit utiliser réellement le backend obligatoire.

**Acceptation :** tolérances annoncées, graines et config enregistrées ; tests séparant contrat existant, bug connu et comportement corrigé. Aucun résultat de benchmark inventé.

### A1 — Config pure et RunContext, P0/P1

Fichiers principaux : `configs/base_config.py`, `configs/runner_config.py`, `scripts/cli_runners/experiment_runner.py`.

1. Sortir `mkdir` et les décisions de reprise des validateurs.
2. Résoudre tous les overrides avant de construire les configurations dépendantes, dont le scheduler.
3. Remplacer la mutation d'une config gelée par une nouvelle instance revalidée. Attention : une copie avec `update` n'est pas nécessairement une revalidation Pydantic.
4. Rejeter les paramètres inconnus au bon niveau ; préserver les paramètres spécifiques légitimes des modèles via schémas explicites.
5. Sérialiser toute la configuration effective, pas seulement quatre sous-configurations.
6. Créer un dossier unique, sous la racine demandée, avec identifiant indépendant du libellé et du hash de config.

**Acceptation :** validation sans filesystem, round-trip complet, override du nombre d'epochs cohérent avec le scheduler, faute de frappe signalée, deux configs identiques produisant deux runs distincts, `--save-dir` respecté partout.

### A2 — Checkpoints et orchestration, P0

Fichiers principaux : `runners/model_runner.py`, `runners/runner_manager.py`, `TrainerFactory` et `ExperimentRunner`.

1. Faire retourner le résultat d'entraînement et la référence `ModelCheckpoint.best_model_path` après `fit()`.
2. Définir le cas « aucun best » : erreur ou fallback explicite vers last, enregistré. Vérifier la cohérence des fréquences de validation et de checkpoint.
3. Distinguer reprise d'entraînement et chargement pour évaluation ; ne pas reprendre silencieusement depuis un dossier existant.
4. En phase `all`, charger le checkpoint retenu puis évaluer ce checkpoint, non une référence choisie avant `fit()`.
5. Réduire progressivement `RunnerManager` à un adaptateur de compatibilité ou le retirer une fois ses consommateurs migrés.

**Acceptation :** un ancien checkpoint leurre ne peut pas être choisi par accident ; test de reprise avec état optimiseur/scheduler ; test d'évaluation autonome ; chemin et hash du checkpoint réellement évalué présents dans les résultats.

### A3 — Évaluation et simulation indépendantes, P0/P1

Fichiers principaux : `runners/callbacks.py`, `models/base/training.py`, `models/model_protocol.py`, `models/simulation/simulator.py`, collecteurs existants.

1. Extraire les opérations scientifiques des callbacks dans un service réutilisable.
2. Rendre les prédictions/simulations consommables sans writer injecté dans le modèle.
3. Injecter réellement `SimulationConfig` et tracer ses paramètres effectifs ; comparer durée, nombre d'événements et arrêt. Ne pas modifier silencieusement la règle d'arrêt.
4. Sélectionner stratégie par capacité pour les modèles à intensité et sans intensité.
5. Éviter la mutation du batch ; préserver et tester les alignements temporels, les masques et les dénominateurs.
6. Auditer les réductions de métriques : loss pondérée par événements, métriques non linéaires calculées depuis des statistiques suffisantes plutôt que moyenne naïve par batch.

**Acceptation :** même entrée non modifiée ; évaluation sans callback privé ; mêmes calculs à tolérance fixée pour les parcours préservés ; stratégie IntensityFree testée ; séquences vides et batch incomplet pris en charge ; pas de hausse non bornée de mémoire GPU.

### A4 — Persistence et agrégation, P0

Fichiers principaux : `models/simulation/tpp_io.py`, `evaluation/results_aggregator.py`, nouveaux contrats de résultats.

1. Conserver l'écriture incrémentale ; transférer vers CPU seulement à la frontière de persistence.
2. Corriger le comptage des séquences indépendamment du nombre d'événements ; stocker les séquences vides et leur horizon dans une table de métadonnées.
3. Écrire les résultats canoniques par run/évaluation. Le CSV global devient une vue reconstruite hors du chemin critique, avec union déterministe des colonnes.
4. Choisir single-writer ou shards par rank suivis d'une fusion validée ; aucun append concurrent non protégé.
5. Fermer les writers et marquer les artefacts incomplets lors d'une exception.

**Acceptation :** IDs uniques/stables, nombres de séquences/événements exacts après reload ; test avec lots de tailles variables et séquences vides ; deux runs concurrents sans écrasement ; aucun fichier partiel présenté comme final.

### A5 — Stabilisation et documentation, P1

Migrer les appels et documenter le parcours réel avant de supprimer les anciens chemins. Ajouter tests de contrats, smoke test GPU, exemple d'entraînement puis d'évaluation séparée. Les options CLI nouvelles doivent être documentées comme nouvelles, pas supposées déjà disponibles.

**Acceptation globale :** chaque phase autonome, `all` cohérent, callbacks conservés et minces, dépendances explicites, rapports de résultats reloadables ; suites exécutées dans l'environnement cible avec compte rendu des échecs et limitations.

## 7. Consigne prête à transmettre à un agent

> Lire ce rapport et le rapport de reproductibilité, vérifier l'état actuel du dépôt et les consignes locales. Implémenter uniquement le lot A[n] choisi, après ses dépendances. Préserver les modifications existantes et les calculs scientifiques ; ne pas remplacer sigkernel ni rendre son installation facultative. Proposer les contrats publics nécessaires avant de modifier plusieurs sous-systèmes. Ajouter les tests d'acceptation du lot, puis fournir fichiers modifiés, compatibilité CLI, commandes exécutées et résultats réels. Signaler toute décision scientifique ou ressource Ruche manquante au lieu de la supposer. Ne pas pousser, soumettre de jobs ou publier d'artefacts externes sans demande explicite.

Ordre recommandé : A0 → contrat A1 → A2 → A3 → A4 → A5. Un correctif P0 d'export peut être isolé plus tôt, avec tests et sans changer le futur schéma à l'insu du responsable de contrats.

## 8. Coordination avec les agents de reproductibilité

Un seul propriétaire définit `RunContext`, schémas de manifeste/résultat et configuration résolue : lot A1, en accord avec R1. R1 enrichit ensuite la provenance sans inventer un second run manager. A4 possède les writers ; R3 précise les champs de traçabilité des données et vérifie leur conservation. A3 ne modifie pas les mathématiques de R4.

Ne pas faire modifier simultanément `runner_config.py`, `model_runner.py` ou les callbacks par plusieurs agents. Découper les lots en PR petites et relisibles ; fournir un tableau des dépendances et un résumé des écarts de comportement pour chaque livraison.

## 9. Hors périmètre sans décision complémentaire

Migration de framework, réécriture des modèles, migration systématique vers Hydra, nouveau backend par défaut, refonte mathématique des tests statistiques, conversion de tout le code à une hiérarchie abstraite, changement des jeux de données. Les corrections scientifiques et les améliorations de performances doivent être identifiables séparément des refactorings.
