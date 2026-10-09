# Rapport et consignes — Reproductibilité scientifique, GPU et Ruche

Date : 9 octobre 2026. Dépôt : `NzoCs/Learning-point-processes`.
Référence examinée : `fdb6c9fb7d1116dc1f4311259c349fa383480612`.
Le diagnostic initial ci-dessous est historique. Le bilan le plus récent figure
dans la section suivante ; les validations CPU ne certifient pas Ruche/CUDA/DDP.
Rapport complémentaire : [Architecture et SOLID](ARCHITECTURE_SOLID.md).

### Bilan actuel — tests, calibration et notebooks (9 octobre 2026)

Branche : `codex/pysiglib-migration`. pySigLib reste le seul moteur de signature.
Les anciens builders étaient déjà supprimés de `main` avant cette migration ;
les exemples maintenus utilisent les schémas Pydantic actuels.

La CI Linux du commit `f58d440` a exécuté **155 tests réussis, 2 ignorés,
2 désélectionnés**. Packaging et migration pySigLib ont réussi. Le seul échec
du job général était le seuil de couverture : **63,57 % / 80 %**.
Le présent lot fixe `uv==0.11.29` dans tous les jobs de `ci.yml`, utilise les
actions checkout v4 / setup-python v5 et conserve les diagnostics de couverture.

**CI Linux validee sur le code `596b169` : 197 tests reussis, 3 ignores,
2 deselectionnes ; couverture 80,79 %.** Les jobs packaging/lanceurs,
migration pySigLib et tests ont tous reussi dans
[Test pipeline](https://github.com/NzoCs/Learning-point-processes/actions/runs/37953907105).
Le workflow
[Code Quality & Linting](https://github.com/NzoCs/Learning-point-processes/actions/runs/37953907160)
a aussi reussi (formatage, imports, noms indefinis et structure des notebooks ;
mypy et Bandit restent des diagnostics consultatifs).

**Validation locale finale : 190 tests reussis, 10 ignores, 2 deselectionnes ; couverture 80,79 %.**
Les omissions correspondent a Make indisponible (8), CUDA absent (1) et a
l'absence d'API intensite sur IntensityFree (1).

Les nouveaux tests couvrent les onze modèles : perte finie, gradients utiles
et finis, mise à jour Adam, intensités avec axes corrects, prédictions temporelles
et marques valides, répétabilité avec graine fixe. La reprise d'entraînement
reste contrôlée par comparaison à un entraînement continu. Le seuil de 80 %
et les exclusions de couverture sont conservés.

Deux problèmes concrets ont été corrigés :

- Trois benchmarks apprenaient leurs statistiques sur le jeu de test malgré
  leur contrat d'apprentissage sur l'entraînement. Des jeux disjoints aux valeurs
  contrastées contrôlent maintenant l'absence de cette fuite. Leurs anciens
  scores peuvent changer du fait de cette correction de calcul.
- Le graphique de distribution des longueurs plantait quand toutes les
  séquences avaient la même longueur ; sa plage de classes inclut ce cas.

La calibration scientifique dispose d'un oracle scalaire à tailles inégales,
d'une énumération exacte des partitions avec ex aequo et d'un contrôle du calcul
accéléré contre l'API réelle de pySigLib avec les mêmes permutations.
Sur 200 répétitions et 99 permutations, le protocole indépendant rejette H0
dans **6 % des cas Poisson et 4 % des cas Hawkes**, compatibles avec 5 % dans
les intervalles mesurés. Sa puissance contre les intensités 2 et 8 vaut 97,5 %
et 96,5 % respectivement. La troncature historique par paires atteint **9,5 %
sur Hawkes** (IC 95 % : 6,17–14,36 %) : **ce protocole n'est pas validé**.

L'API `compute_statistics(..., paired_truncation=False)` permet le protocole
indépendant. Le défaut historique est conservé pour préserver explicitement les
références existantes ; leur stabilité n'est pas une preuve de validité statistique.
La bascule des pipelines de simulation requiert encore de définir le protocole
d'observation voulu et de revoir les références scientifiques correspondantes.
Voir [la calibration, ses sources et ses limites](../docs/SCIENTIFIC_CALIBRATION.md)
et [les mesures versionnées](../docs/validation/scientific-calibration-2026-10-09.json).

Les deux notebooks expérimentaux ont été réécrits avec les API actuelles et des
données locales graînées. Ils conservent les expériences H0/H1, les sweeps et
la comparaison des noyaux actuels, sans recommandations scientifiques non
vérifiées. `scripts.validate_notebooks` exécute toutes leurs cellules et celles
du guide Getting Started dans trois processus séparés ; les journaux et
empreintes sont sauvegardés dans `artifacts/notebook-validation`.
**Les trois notebooks ont termine toutes leurs cellules de code sur CPU.**
Le nouveau workflow `scientific.yml` permet cette validation Linux à la demande.
La pipeline complete des onze modeles a aussi ete reexecutee sur CPU :
les onze parcours et leurs comparaisons numeriques historiques passent sans
modifier les references. Elle reste declenchee a la demande.

Restent hors validation : Ruche, CUDA, DDP, p-values agrégées entre batches,
modèles ajustés et comparaisons de simulations conditionnelles. Les résultats
CPU et la couverture ne lèvent pas ces limites.

### Historique — corrections et seconde validation du 9 octobre 2026

Cette section décrit les corrections ultérieures au diagnostic ci-dessous.
La branche reste `codex/pysiglib-migration`, avec pySigLib comme seule bibliothèque
pour les noyaux de signature.

**Bilan local : 110 tests réussis, 9 ignorés, 2 tests lourds désélectionnés.**
Les 9 omissions correspondent à CUDA absent (1) et à `make` absent sur Windows (8).
Les deux tests lourds Makefile dépendent du parcours distant et des benchmarks ;
la CI générale les exclut explicitement avec `-m "not slow"`.
Aucune désactivation globale de `torch.compile` n'est nécessaire : les deux
compilations automatiques du dispatcher MMD et du moteur de simulation ont été
retirées. Le calcul natif pySigLib reste réellement exécuté.

La commande avec couverture termine néanmoins en échec : **62,86 %**, sous le
seuil maintenu de **80 %**. Les tests fonctionnels ne présentent plus les erreurs
identifiées lors du premier diagnostic ; la CI globale n'est pas déclarée verte.
Le wheel est reconstruit et contrôlé ; le lock et l'environnement sont cohérents.
Les scripts Bash passent `bash -n` et les dry-runs testés.

Corrections vérifiées :

- Import d'accumulateur corrigé ; génération CLI compatible avec `--method`
  et `--model`, et tests alignés sur le véritable `metadata.json`.
- Gram des marques : les axes séquence/événement sont conservés. Des contrôles
  scalaires indépendants vérifient chaque entrée, y compris des tailles inégales.
- MKernel : même noyau temps×marques dans XX/XY/YY ; normalisation et bandwidth
  calculées sur le pool commun lors de la MMD ; transformation finie même si
  toutes les distances sont nulles. La MMD générique exclut les diagonales,
  utilise les deux tailles séparément et rejette les batches de moins de deux.
- Les tests ne supposent plus qu'une MMD² non biaisée est non négative ou nulle
  sur un même échantillon fini. Les anciens sweeps qui répétaient le même RBF
  sous plusieurs noms de noyaux inexistants ont été supprimés.
- Le dénominateur des p-values utilise le nombre de tirages effectivement
  calculés, aussi pour l'agrégation. Les sorties précisent leur nombre et
  distinguent comparaison de simulations et permutation. KSD, non implémenté,
  est rejeté dès la configuration.
- Seed d'entraînement appliquée avant modèle/loaders ; validation isolée des
  streams RNG d'entraînement ; RNG sauvegardés/restaurés avec le checkpoint
  pour la reprise en entraînement. Le mode déterministe est configurable.
  Un choix CPU explicite est respecté et une demande GPU sans CUDA échoue.
- Deux entraînements NHP courts sur CPU produisent exactement les mêmes poids.
  Avec zéro worker et un learning rate fixe, entraînement continu de deux
  époques et reprise depuis `last.ckpt` après une époque produisent aussi
  exactement les mêmes poids. Ce résultat ne certifie ni une reprise en milieu
  de minibatch, ni la reprise avec workers persistants, ni CUDA/DDP.
- Configuration finale complète sauvegardée, incluant simulation, logger,
  identité et seed ; relecture sans dépendre des presets courants. Le logger
  ne modifie plus la configuration et le chemin de sortie ne s'imbrique plus
  à chaque round-trip. Le budget du scheduler suit l'override des époques.
- Répertoires distincts par `run_id`, commun aux processus d'un lancement
  Slurm ; résultats CSV locaux à chaque run. Le manifeste enregistre config,
  hash de config/lock/données locales/checkpoints, versions, commit/dirty,
  numérics et statuts `running/completed/failed`. La reprise est explicite via
  `--checkpoint`, avec source et hash consignés.
- Les 12 presets Hugging Face sont figés sur des SHA de dataset obtenus auprès
  de l'API publique du Hub ; le loader transmet la révision y compris pour le
  fallback `dev`→`validation`.
- Génération : seed consignée, empreintes des splits sauvegardées, arrondis de
  split sans perte de séquences, conservation des séquences vides et des IDs,
  ordre stable des marques à temps identiques.
- Petit parcours **CLI → entraînement → test → simulation → statistiques,
  fichiers Parquet et graphiques** exécuté entièrement sur fixture locale avec
  pySigLib réel. L'évaluation charge le checkpoint produit par ce fit.
  Le graphique d'intensité utilise la simulation enregistrée, au lieu de
  relancer une simulation supplémentaire. La simulation NHP non conditionnée
  est aussi testée : intensité initiale issue de l'état caché nul, sans faux
  événement, puis génération de trois événements valides.

**Compatibilité scientifique :** les corrections MKernel/MMD générique et du
comptage des p-values changent certains résultats historiques. Les nouveaux
manifests portent `unbiased_off_diagonal_v2`. Ne pas fusionner ces résultats
avec l'ancienne implémentation sans identifier le protocole. Les conventions
numériques de l'adaptateur pySigLib lui-même restent inchangées. La calibration
scientifique des p-values, les horizons/normalisations et le cas de simulations
conditionnelles restent à valider ; le succès du pipeline ne les certifie pas.

Commandes finales exécutées :

```bash
uv build --wheel --no-sources
OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUTF8=1 uv run --frozen --no-sync python -m pytest tests \
  -m "not slow" -o addopts= -q --tb=short --cov=new_ltpp \
  --cov-report=term:skip-covered --cov-report=json:artifacts/validation-coverage.json \
  --cov-fail-under=80
```

Restent ouverts : couverture globale 80 %, workflow de lint historique
Poetry/Python 3.12, validation Linux/Ruche après ces nouveaux changements,
CUDA/DDP et reprise avec workers, précision/mémoire/performance représentatives,
calibration du test statistique et prise en compte effective de tous les paramètres
de fenêtre de simulation.
Le test complet validé porte sur NHP CPU et une fixture bornée, pas tous les modèles.

### Validation réelle de la branche pySigLib — 9 octobre 2026

Code vérifié : `98be5371cd14f9b891a4fac6cee64e60b237ac0a`.
**Conclusion : migration CPU validée dans le périmètre testé ; reproductibilité
complète et fonctionnement de bout en bout non validés.**

| Contrôle | Résultat observé |
| --- | --- |
| pySigLib CPU, récurrence indépendante, MMD et gradients | Tests ciblés réussis avec `TORCH_COMPILE_DISABLE=1`. Cela ne valide pas `torch.compile`. |
| Distribution et plans Slurm | Tests ciblés réussis ; les plans ne prouvent pas un job réel. |
| CI Linux du commit vérifié | Jobs `signature-migration` et `packaging-and-launchers` réussis ; job général `test` en échec à l'étape `Run tests`. |
| Suite complète locale avec dépendances dev | Collecte interrompue : `tests/test_accumulators.py` importe `mean_len_accumulator`, absent ; l'implémentation actuelle se trouve dans `len_accumulator.py`. |
| Tests restants, hors accumulateurs et Makefile | **83 réussis, 5 échoués, 1 ignoré**. Les tests Makefile sont exclus de ce diagnostic car `make` est absent du poste Windows. |
| Trois échecs CLI | Les tests de génération demandent `--method`, option non reconnue par la CLI actuelle ; le test des métadonnées échoue également faute de sortie. Cela révèle un contrat test/CLI désaligné, pas à lui seul l'échec de la commande actuelle avec ses options correctes. |
| Deux échecs MKernel | Test de symétrie du Gram et assertion de MMD sur le même échantillon en échec. Le contrat numérique et les attentes des tests doivent être examinés avant correction. Ce noyau est distinct de l'adaptateur pySigLib. |
| CUDA, clone neuf Ruche, entraînement/test/simulation et reprise | Non exécutés sur le cluster ; accès SSH différé pour le pare-feu. |

[Exécution CI du commit vérifié](https://github.com/NzoCs/Learning-point-processes/actions/runs/37915233675).

Commandes locales de validation après installation du groupe `dev` :

```bash
uv sync --frozen --no-default-groups --group dev --no-build-package pysiglib
TORCH_COMPILE_DISABLE=1 PYTHONUTF8=1 uv run --frozen --no-sync python -m pytest tests \
  -o addopts= --tb=short -q --cov=new_ltpp --cov-report=term:skip-covered \
  --cov-config=.coveragerc --cov-fail-under=80
# La commande ci-dessus s'arrête pendant la collecte ; couverture non certifiée.
TORCH_COMPILE_DISABLE=1 PYTHONUTF8=1 uv run --frozen --no-sync python -m pytest tests \
  --ignore=tests/test_accumulators.py --ignore=tests/scripts/test_makefile.py \
  -o addopts= --tb=short -q
```

Les lots suivants restent ouverts :

- **R1 :** manifeste d'expérience complet (commit, environnement, config finale,
  données, checkpoint et hashes), round-trip de configuration et isolation des
  tentatives. `RunnerConfig.get_yaml_config()` n'inclut pas la configuration de
  simulation ; les répertoires sont dérivés du dataset/modèle, sans identité
  unique de tentative dans le runner général.
- **R2 :** contrôle des RNG d'entraînement et des workers, double exécution avec
  tolérances définies et comparaison entraînement continu/repris. Aucune seed
  d'entraînement ne figure dans `TrainingConfig`.
- **R3 :** révision/hash des données et provenance des simulations. Les appels
  Hugging Face de `data_loader.py` ne fournissent pas de `revision`.
- **R4 :** spécification et calibration du protocole statistique ; corriger ou
  justifier les deux tests MKernel après examen du contrat scientifique.
- **R0/R5/R6 :** installation propre Linux sur Ruche, plugin CUDA, jobs Slurm et
  mini parcours complet, puis mémoire/performance sur les charges utilisées.
- **R7 :** suite générale verte et couverture mesurée ; aligner tests et CLI.
  Le workflow de lint historique utilise encore Poetry/Python 3.12, alors que
  le projet courant utilise uv/Python 3.11.

### Migration implémentée dans une branche dérivée

La branche `codex/pysiglib-migration`, issue du commit `19aa258`, utilise
exclusivement pySigLib 4.0.0 avec plugin CUDA 4.0.0 exigé par l'extra Ruche.
La dépendance sigkernel, sa sélection et les outils qui l'importent ont été
retirés de cette branche ; la référence reste dans `codex/reproductibilite-ruche`.
L'adaptateur préserve le solveur, les
embeddings, les noyaux spatiaux/scaling et la MMD² non biaisée.
Voir le [compte rendu de migration](../docs/PYSIGLIB_MIGRATION.md) pour les
commandes et tolérances. Les contrôles mathématiques et gradients CPU passent ;
CUDA et les jobs Ruche restent à exécuter. Aucune comparaison native historique
n’est revendiquée ; les tests utilisent une récurrence mathématique indépendante.
La branche de migration est prête à ces validations, pas certifiée en production.

La procédure Ruche distingue désormais `$HOME` (code/environnement) et
`$WORKDIR` (caches, données et résultats), et les launchers acceptent un
environnement extérieur au clone via `UV_PROJECT_ENVIRONMENT`.

Les sections d'audit et le plan initial ci-dessous décrivent la référence avant
migration. Leurs propositions de conserver ou installer l'ancien backend ne
s'appliquent plus à `codex/pysiglib-migration` ; utiliser la procédure actuelle
dans `docs/RUCHE.md` et `docs/PYSIGLIB_MIGRATION.md`.

### Premier lot implémenté dans le worktree

Branche : `codex/reproductibilite-ruche`, créée depuis la référence examinée.
L'utilisateur a identifié le blocage SSH comme un problème de pare-feu et demandé
de poursuivre les modifications puis de tester Ruche ultérieurement. L'alias
local `ruche` est configuré ; aucune nouvelle authentification n'a été effectuée.

Corrections locales réalisées :

- Installation : dépendances runtime directes `pydantic`, `pyyaml` et `rich` ;
  lock régénéré par `uv`, sans changer les 340 versions/sources de packages.
- Distribution : inclusion des sous-packages et des presets YAML dans le wheel ;
  accès aux presets depuis un autre répertoire, sorties dans le répertoire de travail.
- Launchers : options CLI corrigées, chemins personnels supprimés, environnement
  du clone utilisé avec `--frozen --no-sync`, indices contrôlés, grilles de 3 tâches
  GPU et 28 CPU, sorties par job/tâche, profil CPU masquant les GPU.
- Vérification : dry-run sans calcul ; petit job NHP d'une epoch avec contrôle
  préalable du Gram réel de signature sur CUDA et refus d'un résultat sur CPU.
- Documentation : Python 3.11 exact, commandes `uv` verrouillées, démo NHP,
  invocation du module CLI, procédure [Ruche](../docs/RUCHE.md).
- Tests : invocation pytest explicite dans la CI, cible de couverture corrigée
  de `easy_tpp` vers `new_ltpp`, job CI distinct pour wheel et launchers sans
  dépendances natives. Le seuil historique de couverture n'a pas été abaissé.

Validation locale : wheel construit avec `uv build --wheel --no-sources` ;
**10 tests ciblés réussis**, sans les options globales de couverture ; syntaxe
Bash vérifiée pour les quatre fichiers ; `uv lock --check --offline` réussi.
Les tests du wheel vérifient son contenu et exécutent l'accès aux presets dans
un processus isolé du checkout. Ces contrôles ne valident pas les imports de
tous les packages natifs. La suite complète, sa couverture, la CI distante,
le Gram CPU/GPU et le job d'entraînement Ruche restent non exécutés.

Les sections suivantes conservent les constats de baseline et le plan scientifique.
Les corrections ci-dessus ne terminent pas R0/R5/R7 : restent notamment le build
natif Linux, la validation réelle Slurm/GPU et le parcours complet. R1–R4/R6
(identité complète, checkpoints/reprise, données et protocole MMD, pySigLib)
restent à réaliser et à valider séparément.

## 0. Reprise du rapport : parcours d'un nouvel utilisateur

Complément du 9 octobre 2026, vérifié sur le checkout local. La référence locale et le HEAD distant annoncés par `git ls-remote origin HEAD` sont tous deux `fdb6c9fb7d1116dc1f4311259c349fa383480612`. Les fichiers `rapports/` sont actuellement non suivis par Git : ils ne sont donc pas livrés à une personne qui clone cette référence. Les consignes ci-dessous sont une préparation de validation, pas un compte rendu de jobs réussis.

Le premier jalon est désormais **un clone neuf directement sur Ruche, sans réutiliser l'environnement LTPP personnel**. Les améliorations se discutent à partir de ce parcours et de ses premiers échecs. La migration pySigLib reste un lot séparé : établir la baseline du dépôt actuel avant de modifier le backend.

### Constats supplémentaires vérifiés

| Constat | Conséquence pour un nouvel utilisateur | Première action |
| --- | --- | --- |
| README : Python 3.11+ ; `pyproject.toml` : `==3.11.*`. | Le lecteur peut choisir un Python incompatible. | Annoncer Python 3.11 exactement dans le parcours d'installation. |
| README : `uv sync` ; certaines dépendances utilisées par la CLI sont déclarées dans le groupe `cli`, sans contrat runtime complet vérifié. | Une installation par défaut peut échouer dès l'import de la CLI. | Tester d'abord la commande du README, puis comparer avec `uv sync --frozen --group cli --group dev` ; corriger les dépendances nécessaires au runtime. |
| Le preset `test` utilise `NzoCs/test_dataset` sur Hugging Face. | Le dépôt seul ne suffit pas à garantir un test hors ligne ; accès et révision des données restent à vérifier. | Précharger les données avant calcul, enregistrer leur révision et prévoir une petite fixture locale. |
| `make run-demo` utilise `--model Hawkes` ; le README décrit une démo NHP. | L'exemple annoncé et la commande exécutée diffèrent. | Choisir et vérifier une seule démo canonique ; commencer le diagnostic avec NHP explicitement. |
| La CI contient `uv run tests`. | Cette commande ne constitue pas une invocation explicite de pytest. | Exécuter `uv run --frozen --no-sync python -m pytest` après installation du profil de test. |
| `packages.find.include` contient seulement `new_ltpp` et `scripts` ; les YAML sont hors de ces packages. | Le paquet construit doit être vérifié séparément de l'installation editable. | Inspecter le wheel, ses sous-packages et l'accès aux presets depuis un autre répertoire. |
| Port TCP 22 inaccessible depuis le poste courant, malgré résolution DNS et ping réussi. | Aucun test SSH, Slurm ou GPU distant n'a été exécuté. | Rétablir l'accès réseau puis vérifier l'authentification ; ne pas conclure sur la cause à partir du ping. |

### Trois validations à ne pas confondre

1. **Installation et utilisation :** clone, environnement neuf, imports, CLI, données, petit entraînement, évaluation et artefacts lisibles.
2. **Reproductibilité d'exécution :** même code, données, configuration et environnement ; deux runs isolés et une reprise comparés selon des tolérances définies.
3. **Validité scientifique :** protocole MMD spécifié et calibré, puis équivalence du backend de signature. Un pipeline qui termine ne prouve pas cette validité.

Ordre proposé pour la discussion : lever les obstacles du parcours utilisateur (R0/R5/R7), rendre les sorties traçables (R1/R3), vérifier répétition et reprise (R2), puis valider les choix scientifiques et la migration (R4/R6). Pour chaque correction, consigner le symptôme avant modification, le changement et le résultat de la même commande après modification.

## 1. Objectif, contraintes et limites de l'audit

Pouvoir retrouver ce qui a réellement été exécuté, relancer une expérience dans un environnement identifié et comparer ses résultats selon des tolérances documentées. Une seed seule, un fichier YAML de preset ou un checkpoint seul ne suffisent pas.

Le projet cible notamment le GPU du mésocentre Ruche. La fonctionnalité de noyau de signature reste obligatoire. À la demande de l'utilisateur, le passage de la bibliothèque `sigkernel` à **pySigLib** (paquet/import `pysiglib`, désigné ici par « pysig ») est maintenant une cible explicite du plan, et non une simple alternative à évoquer. Cela ne signifie pas rendre le calcul optionnel : après validation, pySigLib et son support CUDA doivent être installés dans le profil Ruche. `sigkernel` reste la référence de comparaison pendant la transition, puis peut quitter les dépendances de production lorsque les critères du lot R6 sont satisfaits.

Cette demande porte sur les consignes de migration : aucune dépendance ni aucun calcul du projet n'ont été changés à ce stade. La bascule doit démontrer fidélité numérique, compatibilité et comportement GPU sur le matériel cible.

Les constats proviennent du dépôt et des scripts locaux, pas de jobs actifs, de logs Slurm ou d'une connexion SSH à Ruche. La précédente tentative `uv run --frozen pytest` sur Windows a échoué pendant la compilation Cython de `sigkernel`, faute de compilateur MSVC compatible ; elle n'a pas exécuté la suite. Le fichier `coverage.xml` versionné est historique et ne mesure pas la couverture actuelle. Ne pas annoncer de tests GPU réussis ni de performances mesurées.

## 2. Risques observés et ordre des priorités

| Priorité | Observation | Action attendue |
| --- | --- | --- |
| P0 | Checkpoint choisi avant entraînement et reprise implicite possible dans un dossier partagé. | Identité unique de run et checkpoint évalué explicite ; voir A1/A2. |
| P0 | Snapshot `RunnerConfig.get_yaml_config()` incomplet. | Sauver la configuration réellement résolue, y compris simulation, logger et paramètres runtime. |
| P0 | Sorties globales partagées et identifiants de simulation fragiles. | Artefacts canoniques par run ; schémas et comptages validés ; voir A4. |
| P0 | Scripts Ruche avec options CLI incohérentes et grille d'array incorrecte. | Validation des commandes et grilles avant soumission. |
| P0 scientifique | Plusieurs branches du test MMD ne décrivent pas le même protocole de génération de la loi nulle. | Caractériser, nommer et versionner chaque protocole ; corriger les décomptes validés. |
| P1 | Dépendances natives, versions déclarées larges et environnement HPC non décrit entièrement. | Build automatisé, lock et environnement GPU testés et archivés. |
| P1 | Migration prévue vers pySigLib, mais absence de contrat d'équivalence et de validation Ruche. | Lot R6 : adaptateur, comparaison des Gram/MMD, installation CUDA, puis bascule contrôlée. |
| P1 | Seeds, données, splits, précision et normalisation pas réunis dans une provenance complète. | Manifeste et fixtures ; RNG séparés ; données et paramètres effectifs identifiés. |

Les corrections d'installation, d'identité et de commande sont prioritaires avant une grande campagne. Les p-values demandent une validation scientifique avant d'être interprétées, même si le pipeline s'exécute correctement.

## 3. R0 — Installation reproductible sans compilation manuelle

Sources de code : `pyproject.toml`, `uv.lock`, CI et scripts de lancement. Le projet exige actuellement Python `3.11.*` et référence `sigkernel` depuis Git ; le lock de référence doit fixer le commit effectivement utilisé. Préparer ensuite un lock cible avec les versions exactes de `pysiglib` et du plugin `pysiglib-cuda`, sans changer en même temps la version Python ou le protocole MMD.

### Cible d'installation après migration vers pysig

La documentation pySigLib décrit des wheels précompilés et l'installation `pip install "pysiglib[cuda]"`, qui ajoute le plugin `pysiglib-cuda`. Le paquet de base seul est CPU ; `pysiglib.BUILT_WITH_CUDA` indique si le backend CUDA est chargé. [Installation officielle pySigLib](https://pysiglib.readthedocs.io/en/latest/pages/installation.html), consultée le 9 octobre 2026.

La commande ci-dessus illustre le mécanisme, pas une installation reproductible finale : fixer les versions et hashes dans le lock. Vérifier les wheels disponibles pour Python 3.11, le système Ruche et le pilote hôte. Exiger une installation binaire dans le profil « sans compilation manuelle » et signaler toute incompatibilité plutôt que lancer silencieusement un build source ou revenir au CPU. Si un build spécifique est nécessaire, l'automatiser dans une recette séparée et versionnée.

L'extra CUDA proposé par la bibliothèque devient une exigence du profil de production GPU du projet. Au démarrage d'un job Ruche, contrôler le plugin puis exécuter un petit Gram sur des tenseurs CUDA ; conserver ce test après la migration. L'environnement historique `sigkernel` peut rester distinct pour comparer les résultats sans réintroduire son problème de compilation dans l'installation de production pySigLib.

### Solution recommandée pour Ruche

1. Construire automatiquement un environnement Linux GPU contenant le projet et le backend obligatoire du profil : `sigkernel` pour la référence historique, `pysiglib` avec plugin CUDA pour la cible validée. Utiliser des recettes identifiables.
2. Utiliser une image compatible Apptainer sur Ruche, identifiée par digest et checksum du fichier SIF, plutôt qu'une activation d'environnement personnel non documenté.
3. Effectuer la compilation dans le build automatisé, pas au début de chaque job de calcul. « Sans compilation manuelle » ne signifie pas « aucune compilation n'existe ».
4. Fixer Python, base système, outils de build, dépendances résolues et version CUDA utilisateur compatibles avec le pilote hôte. Un conteneur ne fournit pas le pilote GPU du nœud.
5. Tester import, petit Gram de signature sur GPU, gradients si utilisés, puis mini entraînement et simulation avec le backend réel. Une disponibilité CUDA de PyTorch seule ne valide ni `sigkernel` ni pySigLib.
6. Préparer caches et données avant le calcul ; documenter les chemins bind, cache, scratch et sortie. Ne pas installer depuis Internet dans chaque job ni incorporer de jetons dans l'image.

Ruche recommande Singularity/Apptainer et ne permet pas l'installation de Docker sur ses nœuds ; sa documentation insiste sur la compatibilité des pilotes. [Documentation logicielle Ruche](https://mesocentre.pages.centralesupelec.fr/user_doc/ruche/09_softwares/), consultée le 9 octobre 2026. Vérifier les modules réellement disponibles avant de figer la recette ; ne pas supposer que le build d'image est autorisé sur un nœud de login.

### Solution secondaire : wheels préconstruits

Si l'usage hors conteneur est requis, automatiser la production et les tests de wheels `sigkernel` pour les plateformes/Python pris en charge, avec provenance du commit et checksum. Prévoir toolchain Linux et MSVC Windows, compatibilité ABI et ressources CI. Ne pas redistribuer sans vérifier la licence. Les dépendances CUDA/JIT éventuelles doivent toujours être testées à l'exécution.

Nettoyer ensuite les groupes dev/docs/logging et dépendances dupliquées sans retirer une dépendance nécessaire au runtime. Vérifier également que le wheel du projet contient bien les sous-packages, YAML et point d'entrée ; un fonctionnement en editable ne garantit pas celui d'un paquet installé.

**Acceptation R0 :** installation dans un environnement Linux propre par procédure unique documentée, lock du profil respecté, import et CLI disponibles, test réel du backend CPU/GPU selon le profil, smoke test sur nœud GPU. Le changement de lock lié à R6 doit être intentionnel et comparé à la baseline. Windows est un profil distinct ; ne pas prétendre qu'un conteneur Linux supprime tous les prérequis Windows.

## 4. R1 — Configuration finale, identité et manifeste

S'appuyer sur A1 : un seul contrat de run, pas un deuxième orchestrateur.

Structure cible, les noms sont proposés :

```text
artifacts/<dataset>/<model>/<run_id>/
  config.resolved.yaml
  manifest.json
  checkpoints/
  evaluations/<evaluation_id>/
    request.json
    metrics.json
    simulations/
    plots/
  logs/
```

Une nouvelle exécution a un nouveau `run_id`, même si config et seed sont identiques. Un hash de configuration identifie une recette, pas une tentative. Une évaluation indépendante référence son entraînement parent et le hash du checkpoint ; elle n'écrase pas ses métriques précédentes.

### Champs obligatoires du manifeste versionné

- `schema_version`, run/évaluation/parent IDs, libellé, phase, horodatages UTC, statut `created/running/completed/failed/interrupted` et raison d'erreur.
- Git SHA, indicateur dirty, référence et checksum d'un patch/snapshot des changements locaux nécessaires à reconstruire le code ; prendre aussi en compte les fichiers non suivis pertinents. Un SHA seul est insuffisant si le dépôt est dirty.
- Commande et arguments effectifs, config finale avec overrides, presets d'origine, hash du lock, recette/digest d'image, checksum SIF et versions réellement installées.
- Python, système, PyTorch/Lightning, CUDA utilisateur, pilote, backend signature et commit/version ; options de compilation/JIT pertinentes.
- Pour la migration pysig : versions de `pysiglib` et `pysiglib-cuda`, méthode du solveur, paramètres explicitement traduits, version de l'adaptateur, profil historique/cible et référence du rapport d'équivalence.
- GPU alloué visible par le job, mémoire, précision, TF32, déterminisme, stratégie distribuée, ranks/world size ; jobs et arrays Slurm, ressources demandées et constatées.
- Identité/révision/hash des données et splits, preprocessing, unités et schéma, seed principale et sous-seeds par usage.
- Checkpoints source et retenu : hash, règle best/last, métrique/valeur, epoch/step ; distinguer reprise et évaluation.
- Protocole d'évaluation, horizons et règles d'arrêt effectifs, estimateur/kernel/normalisation, nombres de séquences/événements et tirages nuls réellement utilisés.
- Chemins relatifs et checksums des artefacts finaux ; fichiers incomplets clairement distingués.

Sauver le manifeste avant calcul puis le mettre à jour atomiquement. En DDP, un propriétaire unique écrit l'état global. Ne pas sérialiser les secrets d'environnement, tokens ni payloads sensibles dans les manifests. Définir ce qui est accessible et portable dans une publication.

**Acceptation R1 :** une expérience sauvegardée est reconstruisible sans consulter les presets actuels ; round-trip de config complet ; tous les fichiers appartiennent au bon run ; un échec laisse un état exploitable, pas `completed` ; changement de checkpoint ou protocole traçable.

## 5. R2 — RNG, déterminisme et reprise

Définir deux profils explicitement : reproductibilité stricte autant que supportée, et performance. Enregistrer leurs différences. Ne pas promettre une identité bit à bit entre GPU, versions CUDA ou world sizes différents.

1. Initialiser Python, NumPy, PyTorch CPU/CUDA et les générateurs réellement utilisés par simulation/noyaux. Utiliser des générateurs locaux lorsque l'API le permet.
2. Affecter des streams distincts à initialisation/entraînement, données, simulation et permutations. Les dériver de façon stable, pas avec le `hash()` Python dépendant du processus.
3. Semer les workers DataLoader et enregistrer sampler, shuffle, `drop_last`, batch size, num_workers et ordre des données.
4. Enregistrer et contrôler les options de déterminisme, cuDNN et TF32. Une option non supportée doit produire un diagnostic, pas être ignorée.
5. Restaurer les RNG d'entraînement après une évaluation périodique pour éviter qu'en changer la fréquence change les minibatches ou le dropout.
6. Reprendre depuis un checkpoint qui restaure modèle, optimiseur, scheduler, précision et RNG nécessaires. Définir les limites de reprise mid-epoch et distribué ; distinguer reprise exacte d'un simple fine-tuning des poids.

**Acceptation R2 :** deux runs courts dans le même profil produisent des résultats dans les tolérances déclarées ; ajout d'une évaluation périodique n'altère pas la trajectoire d'entraînement attendue ; comparaison continu/repris au point supporté ; différence de matériel documentée, pas cachée.

## 6. R3 — Données et simulations traçables

Fixer la révision des datasets distants, les hashes des fichiers locaux, le schéma et le preprocessing. Enregistrer train/validation/test, nombre de séquences, ordre, unité temporelle, vocabulaire des marques et conventions de padding. Estimer les transformations sur train uniquement lorsqu'elles sont apprises.

Pour les données synthétiques, sauver paramètres, seed, générateur/version et horizon d'observation, y compris les séquences sans événement. L'instant du dernier événement n'est pas automatiquement la fin de la fenêtre observée.

Pour les simulations, faire correspondre configuration demandée et configuration appliquée : le simulateur actuel utilise notamment la largeur de séquence dans son chemin par défaut ; il faut tracer et tester la relation avec `time_window`, nombre d'événements, thinning et padding avant toute correction.

Travailler avec A4 sur le format : une table de séquences avec IDs, horizon et comptage ; événements rattachés à ces IDs. Corriger `_total_sequences += len(formatted)` dans `tpp_io.py` avec un test montrant la différence entre événements et séquences. Garantir fermeture et état des exports en cas d'interruption.

**Acceptation R3 :** fixture avec longueurs variables, séquences vides et padding ; round-trip sans perte ; pas d'IDs dupliqués ; split stable et sans fuite ; provenance identique après reload ; données distantes indisponibles signalées explicitement.

## 7. R4 — Protocole MMD et signature : validation scientifique séparée

Sources principales : `evaluation/statistical_testing/statistical_tests/mmd_test.py`, `point_process_kernels/sig_kernel.py`, `point_process_metric/` et tests associés, sous `new_ltpp/`.

Ce lot n'est pas un simple refactoring. Conserver la possibilité d'identifier les résultats de l'ancien protocole et faire approuver les choix scientifiques non déductibles du code.

### 7.1 Définir le test effectivement exécuté

Le chemin `_permutation_test` permute les échantillons poolés. Une autre branche compare données à une simulation de référence puis cette référence aux autres simulations : ce n'est pas le même mécanisme nul. Donner des noms distincts et enregistrer le nombre effectif de tirages.

Dans la branche multi-simulations, le nombre de comparaisons est lié aux simulations après la première, alors que la p-value utilise `self.n_samples + 1`. Vérifier la convention puis utiliser le nombre effectif de statistiques nulles : pour la correction Monte Carlo prévue, `(1 + nombre de valeurs nulles >= observée) / (1 + B_effectif)`.

Réinitialiser les accumulateurs à chaque test public, tester deux appels successifs, traiter les tailles inégales et détecter `zip` qui tronque silencieusement les loaders. Pas de p-value de succès sans données. Documenter l'agrégation entre batches : une somme de MMD par batch ne devient pas automatiquement la MMD sur toutes les séquences.

### 7.2 Horizons et normalisation

`_truncate_to_min_time` s'appuie sur les derniers événements de paires de séquences. Il faut distinguer cette règle d'un horizon de censure observé et indépendant du test. Choisir le protocole adapté avec le chercheur ; ne pas remplacer silencieusement la règle actuelle.

`SIGKernel` normalise actuellement les temps à partir d'un maximum des entrées du calcul. Vérifier si `Kxx`, `Kxy`, `Kyy` partagent la même transformation : la cohérence d'un unique noyau MMD ne doit pas être supposée. Tester une transformation fixée par le protocole et sa sensibilité au découpage en batches. Sauver discretisation, dyadic order, embedding, paramètres du noyau spatial, dtype et normalisation.

### 7.3 Tests requis

- Référence petite et transparente de l'estimateur réellement retenu, Gram symétrique et valeurs finies ; contrôle PSD à tolérance sur cas appropriés.
- Cas n ≠ m, padding, séquences vides et batches partiels ; nombres de tirages et p-values sur statistiques nulles connues.
- Identifier MMD ou MMD², biaisée ou non biaisée. Une estimation non biaisée peut être négative ; ne pas l'interchanger avec une implémentation dont les tests attendent la non-négativité.
- Calibration empirique sous H0 sur processus synthétiques contrôlés, avec intervalle d'incertitude sur taux de rejet ; puissance sous alternatives annoncées.
- Distinguer modèle fixé d'un modèle estimé sur données utilisées dans le test ; décrire split et procédure de calibration appropriée.
- Équivalence CPU/GPU et tolérances fp32/fp64 ; ne pas confondre variabilité Monte Carlo et erreur numérique.

**Acceptation R4 :** spécification écrite du protocole, version dans les résultats, tests de référence et calibration exécutés ; décompte exact des tirages ; migration explicitant quels résultats historiques ne sont plus directement comparables.

Le chemin KSD comporte des éléments non implémentés : rejeter une configuration non supportée avant le calcul, plutôt que promettre un test disponible. Ne pas développer KSD dans ce lot sans demande supplémentaire.

## 8. R5 — Scripts Slurm et profils Ruche

### Défauts locaux à corriger

- `scripts/bash/run_all_pipeline.sh` : array `0-4%5` pour une grille observée de trois couples modèle/dataset ; deux indices hors grille. Calculer la grille et vérifier chaque index.
- Les scripts GPU/CPU utilisent `--data-config` et `--gpu`, alors que la CLI examinée utilise `--dataset-id` et ne fournit pas cette option GPU. Tester contre `new-ltpp run --help` réel avant remplacement ; ajouter un sélecteur explicite CPU/GPU si nécessaire au contrat.
- `train_ruche_cpu.sh` contient une grille et des IDs à vérifier contre les presets, notamment `self_correcting`. Ne pas lancer une campagne dont les commandes sont invalides.
- Chemins `/gpfs/...` personnels et activations multiples ailleurs dans le dépôt : remplacer par paramètres contrôlés ; distinguer Ruche d'un éventuel profil HubIA/MIG au lieu de mélanger partitions et chemins.
- `--mem=150G` désigne la RAM hôte demandée, pas la VRAM du GPU.

La documentation Ruche consultée le 9 octobre 2026 indique pour `gpua100` une durée maximale de 24 h, quatre GPU par utilisateur et au plus huit CPU par GPU réservé. Une concurrence `%5` de jobs à un GPU dépasse cette limite de GPU simultanés ; tenir compte aussi des autres jobs de l'utilisateur. Ces paramètres doivent être revérifiés avant soumission. [Partitions Slurm Ruche](https://mesocentre.pages.centralesupelec.fr/user_doc/ruche/07_slurm_partitions_description/).

### Contrat du launcher

Valider grille, presets, fichiers et commande avant soumission. Fournir un dry-run qui imprime toutes les commandes, sans réserver de GPU. Utiliser un shell strict, un répertoire explicite, des variables citées, logs distincts `%A/%a`, `srun` adapté au profil et codes d'erreur non masqués.

Respecter l'allocation GPU et `CUDA_VISIBLE_DEVICES` ; ne pas reconfigurer les GPU du nœud. Paramétrer CPUs/workers/threads pour éviter la sursouscription. Une tâche mono-GPU, un entraînement DDP et une évaluation répartie sont trois stratégies à documenter distinctement.

Prévoir interruption et checkpoint au point supporté, fermeture des outputs, puis statut `interrupted`. Tout signal/requeue dépend de la politique du cluster : valider plutôt que supposer une reprise automatique. Séparer run neuf, reprise et évaluation d'un checkpoint.

**Acceptation R5 :** `bash -n`, dry-run complet sans indices invalides, options CLI testées, petit job réel sur allocation autorisée, manifeste/logs/checkpoint associés. Aucun agent ne soumet de jobs payants ou réservant des ressources sans autorisation explicite.

## 9. R6 — Passage de sigkernel à pySigLib et validation GPU

### 9.1 Intention et périmètre

Faire de pySigLib le backend principal du noyau de signature après validation. Le but est de simplifier l'installation grâce à une distribution binaire et de disposer d'un parcours GPU adapté à Ruche ; ce n'est pas une promesse de vitesse supérieure avant mesure. Conserver la définition du noyau, les embeddings et l'estimateur MMD pendant la première migration.

Le changement se concentre d'abord dans `new_ltpp/evaluation/statistical_testing/point_process_kernels/sig_kernel.py` et un petit adaptateur de backend. Les modèles, la simulation et l'orchestration ne doivent pas importer directement pySigLib. Réutiliser les contrats de noyau existants, conformément à DIP/OCP dans le rapport d'architecture.

### 9.2 Correspondance des calculs et pièges à éviter

Le code actuel prépare les embeddings via `_prepare_kernel()` / `_get_embedding()`, puis appelle `SigKernel.compute_Gram()` et `compute_mmd()`. Garder initialement exactement les mêmes tenseurs d'entrée et la même préparation dans les comparaisons.

L'API cible pour le Gram est `pysiglib.sig_kernel_gram`. Sa documentation distingue les méthodes `finite_difference` et `polynomial`, le raffinement `dyadic_order`, les transformations de chemins et la normalisation du Gram. [API officielle du Gram](https://pysiglib.readthedocs.io/en/latest/pages/signature_kernels/sig_kernel_gram.html).

| Élément | Consigne de migration |
| --- | --- |
| Solveur | Commencer par `method="finite_difference"`, à comparer à la référence. Une méthode polynomial est un essai ultérieur, pas une substitution supposée équivalente. |
| Raffinement | Traduire explicitement `dyadic_order`, y compris le cas 0 présent dans les presets ; tester les valeurs limites contre la version épinglée. |
| Noyau spatial | Adaptateurs explicites pour linear/RBF ; vérifier la formule, sigma et scaling, pas seulement l'égalité des noms de paramètres. |
| Embeddings | Conserver interpolation, comptage des marques, grille et masques. Ne pas passer les événements bruts à une API qui attend les chemins construits. |
| Transformations | Ne pas activer time augmentation ou lead-lag par défaut ; vérifier si l'embedding contient déjà l'information temporelle. |
| Normalisation | Préserver la règle actuelle pour la comparaison initiale ; ne pas confondre normalisation des temps et normalisation diagonale du Gram. La correction scientifique de R4 reste séparée. |
| Formes/device/dtype | Contrat de Gram `(n, m)`, y compris n ≠ m ; pas de conversion NumPy/CPU dans le chemin GPU ; tester contiguïté et dtype effectifs. |
| Gradients | Vérifier les consommateurs et la bonne API PyTorch de la version retenue ; si gradients utilisés, tester leur fidélité et ne pas détacher les tenseurs. |

**Ne pas remplacer directement `compute_mmd()` par `pysiglib.sig_mmd()` sans vérifier l'estimateur.** La documentation décrit cette dernière comme une MMD² non biaisée. [API officielle MMD](https://pysiglib.readthedocs.io/en/latest/pages/signature_kernels/sig_mmd.html). Caractériser l'estimateur de la version historique `sigkernel` effectivement installée, puis appliquer la même formule à `Kxx`, `Kxy` et `Kyy` calculés par le backend cible. Les tests actuels ou leurs noms ne suffisent pas à déterminer la formule. Si un changement d'estimateur est souhaité, l'isoler dans R4 avec nouvelle version de protocole et calibration.

### 9.3 Étapes de migration, livrables et dépendances

1. **M0 — Baseline figée.** Sauver config, versions, embeddings et résultats Gram/MMD sur petits fixtures déterministes, avec backend historique. Si `sigkernel` est bloqué sur Windows, produire la référence dans un environnement Linux fonctionnel ; ne pas comparer à des valeurs supposées.
2. **M1 — Environnement cible.** Installer les wheels épinglés pySigLib/CUDA en environnement isolé ; valider disponibilité du plugin et calcul réel sur GPU. Conserver le lock historique et les hashes des environnements.
3. **M2 — Adaptateur.** Implémenter le Gram cible et le calcul de l'estimateur caractérisé. Pendant la transition, rendre le choix de backend explicite et enregistré, sans fallback silencieux. Garder le backend requis pour toute évaluation signature.
4. **M3 — Équivalence.** Comparer chaque entrée du Gram puis MMD et statistiques nulles avec les mêmes embeddings, permutations préenregistrées et échantillons simulés. Les tolérances absolues/relatives sont fixées et justifiées avant la comparaison ; analyser les écarts plutôt qu'élargir les seuils pour faire passer les tests.
5. **M4 — Ruche.** Exécuter benchmark et smoke test intégrés sur allocation autorisée, à précision et protocole identiques. Vérifier temps, VRAM, répétabilité et erreurs sur grandes entrées.
6. **M5 — Bascule.** Une fois M0–M4 acceptés, changer le backend par défaut, les dépendances obligatoires du profil cible, le lock, la CI et la documentation. Retirer `sigkernel` de la production seulement après avoir archivé un environnement historique reproductible et une procédure de retour à la version précédente. Pas de suppression des résultats historiques.

Si les Gram divergent au-delà des tolérances, ne pas annoncer une migration équivalente : rechercher différences de solveur, kernel spatial, dtype et transformations. Si elles sont intentionnelles, les traiter comme un protocole nouveau avec accord scientifique. Le changement de bibliothèque ne doit pas cacher une correction simultanée des horizons ou des p-values.

### 9.4 Performance GPU : protocole de mesure

Commencer par `sigkernel` actuel. Mesurer les charges représentatives : batch size, longueur de séquence, nombre de marques, discretisation, nombre de tirages, dtype. Séparer coût de JIT/compilation, warm-up, transferts, Gram, permutations et calcul complet.

Synchroniser CUDA pour mesurer le temps ; répéter et rapporter dispersion, throughput, VRAM et RAM hôte. Les compteurs PyTorch ne capturent pas nécessairement les allocations Numba ou d'autres runtimes ; compléter par une mesure GPU externe. Tester sous la mémoire réellement disponible, pas la RAM demandée à Slurm.

Pré-calculer un Gram poolé et réindexer pour les permutations peut réduire les recomputations si le noyau est fixe. Démontrer l'équivalence et chiffrer le coût mémoire O(N²) ; sinon utiliser blocs/caches bornés. Ne pas cumuler toutes les simulations et matrices sur GPU. Une approximation ou un changement de normalisation doit recevoir une version de protocole différente.

Comparer `sigkernel` et pySigLib sur les mêmes charges et fixtures, en séparant premier appel et régime établi. Régler une limite de batch/blocs si nécessaire et vérifier que ce découpage ne change pas le protocole. Un gain de performance peut justifier la migration ; une installation plus simple peut aussi être un bénéfice, à condition de ne pas dégrader la fidélité ou de dépasser la mémoire disponible.

### 9.5 Critères de bascule et retour arrière

- Installation propre du profil cible sans compilateur manuel ; dépendances et plugin CUDA verrouillés et traçables.
- Tests linéaires/RBF, longueurs variables, masques, séquences vides, n ≠ m et raffinements utilisés ; erreurs explicites pour paramètres non supportés.
- Gram et estimateur MMD équivalents à tolérances annoncées ; gradients vérifiés si requis ; p-values comparées avec tirages identiques, sans garantie universelle d'égalité près du seuil de rejet.
- Smoke test GPU et mesures mémoire/temps réelles sur Ruche ; aucun retour CPU silencieux.
- Manifeste identifiant backend, versions, solveur et protocole ; exemples CLI et documentation mis à jour.
- Environnement et résultats de référence conservés. Si la bascule échoue, revenir à un profil/version historique explicite et enregistrer ce choix, pas changer de backend au milieu d'un run.

**Acceptation R6 :** livrables M0–M5, rapport d'équivalence et benchmark versionnés, installation cible reproductible et test GPU réel ; pySigLib devient principal uniquement après satisfaction des critères. Aucun gain de vitesse annoncé sans mesure. Documentation vérifiée le 9 octobre 2026 ; `latest` est une source de consultation, jamais une version à utiliser comme pin de production.

## 10. R7 — Tests continus et documentation exécutable

Construire trois niveaux : unitaires rapides, intégration avec dépendances natives réelles, smoke/benchmark GPU sur runner autorisé. Ne pas prétendre tester CUDA sur une CI sans GPU ; annoncer les tests ignorés et prévoir un contrôle GPU dédié.

Générer la couverture depuis le run de test actuel, pas depuis l'ancien `coverage.xml`. Tester également le paquet construit, la CLI, ses exemples README et les commandes de dry-run Slurm. Les mocks ne remplacent pas le test réel du backend : historique `sigkernel` pour l'équivalence, puis pySigLib avec plugin CUDA pour le profil cible.

Documenter une procédure courte : environnement → données → entraînement → évaluation d'un checkpoint → inspection des résultats → reprise. Distinguer recette, run et réplication sur plusieurs seeds. Éviter l'expression « reproductible » sans préciser environnement et tolérance.

**Acceptation R7 :** compte rendu de commandes et résultats, fixtures versionnées, exemples exécutables, limitations GPU visibles, diagnostic utile lors d'un échec natif, aucun secret dans les artefacts CI.

## 11. Handoff aux autres agents et dépendances

Ordre recommandé : R0 et tests A0 → contrat commun A1/R1 → A2/A4 et R5 → R2/R3 → R4 → validation finale R6 → R7. M0/M1/M2 de la migration pysig peuvent commencer après la baseline et les contrats, avant la fin des autres chantiers ; ne pas mélanger leur comparaison avec une modification scientifique de R4. Les nouveaux protocoles sont validés séparément avant la campagne finale.

| Lot | Périmètre propriétaire | Dépendances / coordination |
| --- | --- | --- |
| R0 | Environnement, build, lock, smoke backend | Ne pas modifier simultanément le lock avec d'autres agents. |
| R1 | Provenance du manifeste | Contrat unique créé avec A1 ; pas de nouveau runner concurrent. |
| R2 | Seeds et reprise | A2 et cycle de vie A3 ; distinguer reprise exacte et fine-tuning. |
| R3 | Identité données et simulations | A4 possède les writers ; accord sur schéma avant changements. |
| R4 | Protocole et tests scientifiques | Isoler des PR de refactoring A3 ; choix scientifiques à approuver. |
| R5 | Profils cluster et grilles | Options stabilisées avec A2 ; accès GPU autorisé requis pour test réel. |
| R6 | Adaptateur pySigLib, équivalence, benchmark et bascule | R0 et baseline ; coordonner `sig_kernel.py` avec R4, lock avec R0 ; pas de changement scientifique implicite. |
| R7 | CI et documentation | Consolider les livraisons, sans inventer des résultats non exécutés. |

### Consigne prête à transmettre

> Lire les deux rapports et vérifier le dépôt actuel. Prendre uniquement le lot R[n] choisi ; fournir un plan court, les contrats affectés et ses dépendances. Préserver la fonctionnalité obligatoire de noyau de signature et l'usage GPU. Pour R6, préparer le passage à pySigLib avec référence sigkernel conservée, équivalence explicite et plugin CUDA exigé sur Ruche ; ne retirer la dépendance historique de production qu'après les critères de bascule. Implémenter les changements et tests du lot, sans refactorer les zones possédées par un autre lot. Pour les choix scientifiques, produire la spécification et demander validation avant modification non équivalente. Rapporter les commandes exécutées, résultats, tolérances, tests non exécutés et raisons. Ne pas accéder à des secrets, pousser, déployer, publier une image ou soumettre des jobs sans demande explicite. Livrer une procédure de reproduction et les limites connues.

## 12. Critère final de campagne exploitable

- [ ] Chaque expérience a un environnement identifiable, des données identifiables et une configuration effective complète.
- [ ] Les sorties de deux tentatives ne se mélangent jamais ; les séquences vides restent représentées.
- [ ] Le checkpoint évalué est choisi explicitement après entraînement et vérifiable par hash.
- [ ] Le protocole MMD, ses tirages effectifs et ses limites sont documentés et testés.
- [ ] Un petit run GPU et une reprise supportée ont été réellement validés sur le profil cible.
- [ ] Le passage à pySigLib dispose d'un rapport d'équivalence, d'un plugin CUDA verrouillé et d'une baseline sigkernel archivée ; aucun changement d'estimateur n'est masqué.
- [ ] Les commandes Slurm ont passé le dry-run et respectent les ressources autorisées.
- [ ] Une autre personne peut reproduire le parcours sans connaissance d'un environnement personnel ni compilation manuelle.
- [ ] Les différences de matériel, précision ou protocole sont visibles ; aucune garantie bit à bit universelle n'est annoncée.

## 13. Protocole préparé : clone neuf sur Ruche

### 13.1 État de la connexion et configuration SSH

Hôte confirmé par la [documentation officielle de connexion Ruche](https://mesocentre.pages.centralesupelec.fr/user_doc/ruche/03_connection_and_file_transfer/), consultée le 9 octobre 2026 : `ruche.mesocentre.universite-paris-saclay.fr`.

Depuis le poste Windows courant, `Test-NetConnection ruche.mesocentre.universite-paris-saclay.fr -Port 22 -InformationLevel Detailed` a résolu `129.175.109.7` et `129.175.109.6` ; les tentatives TCP sur les deux adresses ont échoué. Le ping a réussi. Le client OpenSSH Windows est installé. Aucun fichier `%USERPROFILE%\.ssh\config` n'a été trouvé pendant cette vérification. L'identifiant et le mode d'authentification restent à préciser ; aucune connexion authentifiée ni configuration SSH active n'a été créée.

Mise à jour après échange : l'utilisateur indique `regnaguen` comme identifiant probable et une authentification habituelle par mot de passe. L'alias local `ruche` a été créé dans `%USERPROFILE%\.ssh\config` avec cet utilisateur, l'hôte officiel, un délai de connexion de 10 secondes et le transfert d'agent désactivé. `ssh -G ruche` confirme la résolution de la configuration. Une tentative directe avec `regnaguen` échoue encore par timeout sur le port 22, avant authentification : l'identifiant reste donc non vérifié par le serveur. La commande interactive à utiliser dans le terminal utilisateur est désormais `ssh ruche`.

Après rétablissement de l'accès réseau et confirmation de l'identifiant, ajouter un alias sans écraser les autres entrées du fichier de configuration :

```sshconfig
Host ruche
    HostName ruche.mesocentre.universite-paris-saclay.fr
    User IDENTIFIANT_RUCHE
    ServerAliveInterval 30
    ServerAliveCountMax 3
    ForwardAgent no
```

Si une clé est déjà autorisée, ajouter son chemin local avec `IdentityFile` et tester avec `ssh -o BatchMode=yes -o ConnectTimeout=10 ruche hostname`. Sinon, effectuer l'authentification interactive dans le terminal local. Ne pas transmettre de mot de passe ou de clé privée dans le rapport ou le chat. Conserver la vérification de clé d'hôte ; ne pas utiliser `StrictHostKeyChecking=no`. Une nouvelle paire de clés et son installation sur le compte distant ne sont utiles qu'après confirmation du mode d'accès existant.

### 13.2 Baseline du dépôt livré

Créer un répertoire neuf dans un espace autorisé du compte Ruche. Ne pas y copier `.venv`, caches, données ou checkpoints du poste local. Les commandes suivantes sont **préparées, non exécutées sur Ruche** ; le chemin de travail et les modules seront choisis après inventaire du cluster.

```bash
# Depuis le répertoire de travail autorisé, pour une tentative neuve.
git clone https://github.com/NzoCs/Learning-point-processes.git ltpp-baseline
cd ltpp-baseline
git checkout --detach fdb6c9fb7d1116dc1f4311259c349fa383480612
git rev-parse HEAD
git status --porcelain
sha256sum pyproject.toml uv.lock
command -v git python python3 uv sbatch srun
module list
```

Enregistrer versions et modules réellement disponibles. Si `uv` ou Python 3.11 manquent, documenter leur installation dans l'espace utilisateur ou le module exact utilisé. Ne pas supposer un accès Internet sur les nœuds de calcul. Les compilations/installations doivent respecter les règles et ressources autorisées du cluster.

Tester d'abord **le parcours publié**, sans corriger silencieusement le dépôt : commande `uv sync` du README, puis aide de la CLI. Si ce parcours échoue, conserver son diagnostic et classer l'échec. Pour le profil de diagnostic verrouillé, repartir d'un second clone/environnement neuf et utiliser :

```bash
uv --version
uv sync --frozen --python 3.11 --group cli --group dev
uv run --frozen --no-sync python --version
uv pip check
uv run --frozen --no-sync python -c 'import new_ltpp, torch, sigkernel; print(new_ltpp.__file__); print(torch.__version__, torch.version.cuda); print(sigkernel.__file__)'
uv run --frozen --no-sync new-ltpp --help
uv run --frozen --no-sync new-ltpp run --help
```

L'option `--no-sync` empêche les étapes de validation de modifier implicitement l'environnement après installation. Conserver le hash du lock avant/après ; un lock modifié invalide le statut « référence inchangée ». Vérifier aussi le paquet construit dans un environnement distinct ; la réussite en editable n'est pas une preuve de distribution complète.

### 13.3 Étapes et preuves attendues

| Étape | Exécution | Preuve de réussite | État au 9 octobre 2026 |
| --- | --- | --- | --- |
| Réseau et SSH | Accès port 22, authentification, `hostname`. | Nœud frontal identifié, connexion authentifiée. | TCP en échec ; authentification non testée. |
| Clone public | Clone neuf et checkout de la référence. | SHA exact, arbre propre, URL distante. | SHA distant vérifié depuis Windows ; clone Ruche non exécuté. |
| Installation | Parcours README puis diagnostic verrouillé dans un environnement distinct. | Code retour, versions, hashes, imports et CLI. | Non exécutée sur Ruche. |
| Données | Accès au preset `test`, préchargement dans le cache du nouveau parcours. | Révision, splits, schéma et comptages ; accès depuis le job. | Non vérifiés sur Ruche. |
| Backend GPU | Sur allocation Slurm : PyTorch CUDA puis petit Gram avec le backend réel. | GPU/pilote, dtype, forme et valeurs finies ; CPU/GPU comparés à tolérance. | Non exécuté ; PyTorch CUDA seul serait insuffisant. |
| Petit entraînement | NHP, une epoch, preset test, logger local, dossier neuf. | Logs, métriques finies, checkpoint non vide, GPU effectivement utilisé. | Non exécuté. |
| Pipeline complet | Train/test/predict bornés, puis évaluation statistique explicitement vérifiée. | Checkpoint évalué identifié, simulations et résultats ; erreurs propagées. | Non exécuté ; `--phase all` annonce train/test/predict, pas une preuve de MMD exécutée. |
| Répétition et reprise | Deux dossiers neufs ; reprise depuis checkpoint explicite après stabilisation du contrat. | Résultats comparés, tolérances et limites documentées. | Non exécutées. |

Commande de diagnostic de petit entraînement, à exécuter **sur un nœud de calcul alloué**, après les étapes précédentes :

```bash
uv run --frozen --no-sync new-ltpp run \
  --config yaml_configs/configs.yaml \
  --model NHP --dataset-id test \
  --training-config quick_test --general-specs-config quick_test \
  --data-loading-config quick_test --logger-config csv \
  --phase train --epochs 1 --save-dir artifacts/ruche-baseline-train-001
```

La CLI actuelle ne permet pas de fixer explicitement le device par `--gpu`. Vérifier le device effectivement choisi et visible dans l'allocation ; échouer le contrôle GPU s'il utilise le CPU. Un autre test dans un dossier neuf pourra utiliser `--phase all --epochs 1`, mais il devra inspecter chaque phase et ses artefacts. La simulation et l'évaluation doivent être bornées séparément ; une epoch ne borne pas à elle seule leur coût. Ne pas lancer les arrays historiques pour ce premier test.

### 13.4 Livrable et décisions à prendre ensemble

Conserver pour chaque tentative : référence Git, hashes de configuration/lock, version de `uv`, commande exacte, environnement installé, modules, identité des données, allocation/job Slurm, sorties standard et erreurs, code retour, chemins et checksums des artefacts. Utiliser des logs ciblés plutôt qu'un dump de toutes les variables d'environnement. Après un job, examiner aussi son état Slurm et son code de sortie ; une ligne de succès de la CLI ne suffit pas.

Le premier bilan attendu est une liste courte de défauts **observés à l'exécution**, avec statut corrigé/non corrigé. Nous pourrons ensuite choisir les tolérances numériques, le niveau de reprise garanti et le protocole scientifique. Tant que l'accès SSH est indisponible, les lignes distantes restent « non exécutées » et aucune reproductibilité Ruche n'est certifiée.
