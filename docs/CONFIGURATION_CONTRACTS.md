# Audit des paramètres et comportements conservés

L'absence d'effet d'un paramètre ne signifie pas que son ancien comportement
doit être rétabli. Cet audit utilise les appels actuels, les docstrings, les
tests et l'historique Git pour distinguer compatibilité et défauts.

## Simulation : les anciens contrôles sont obsolètes

Le commit `fdb6c9f` remplace explicitement la simulation par horizon/multiplicateur
par `simulate(batch, num_events_to_simulate=None)`. Sans argument explicite, le
nombre d'événements générés est la largeur du lot d'entrée. Les buffers sont
alloués exactement pour l'historique et les événements à produire. Le nombre de
séquences du lot provient du data loader.

Les champs `SimulationConfig.time_window`, `.initial_buffer_size` et
`.batch_size` sont supprimés. SimulationConfig ne contient que la graine ;
les champs inconnus sont rejetés. Les presets trompeurs `tw...` sont également
supprimés ; `fixed_events`, `quick_test` et `debug` ne contiennent que `seed: 42`.

```yaml
simulation_config:
  seed: 42
```

Le manifeste ajoute `simulation_contract`, avec le mode réellement exécuté,
les sources du nombre d'événements/de la taille du lot et la graine. Aucun
horizon temporel n'est réintroduit. La limite de génération dépend toujours
de la largeur paddée du lot, pas du nombre valide de chaque séquence : c'est
le contrat conservé, à distinguer de l'invariance au padding des chemins de
signature. Un autre protocole de génération doit faire l'objet d'un changement
scientifique explicite.

Cette suppression rompt la lecture des anciennes configurations contenant ces
champs. Pour les utiliser dans un nouveau run, retirer les trois champs et
sélectionner un preset actuel. Conserver les configurations et manifests
historiques dans leurs archives ; le rejeu strict d'un ancien run nécessite
son ancienne version du code. Ne pas réécrire un manifeste pour contourner
le contrôle d'identité de configuration.

Les graines hors de `[0, 2**32 - 1]` étaient acceptées par SimulationConfig mais
rejetées par Lightning lors de `predict`. Elles sont désormais rejetées dès le
chargement, comme les graines d'entraînement. Les graines 0 et `2**32 - 1`
restent valides.

## Signature : une seule représentation

Depuis `6e5e75e`, `_get_embedding` documentait que `embedding_type` était
conservé pour compatibilité. L'implémentation construit une grille régulière
contenant le temps, le comptage total normalisé et les comptages par marque.
Elle ne choisissait pas une interpolation différente selon `linear`/`constant`.

`counting_grid` est désormais la seule valeur acceptée, dans la configuration,
le constructeur SIGKernel et la préparation. Pour les configurations anciennes,
remplacer `embedding_type: linear` ou `constant` par `counting_grid`.
Le noyau spatial `space_kernel_type: linear` reste une option distincte et valide.
Le manifeste indique `signature_path_representation: counting_grid`. Aucune
formule de noyau, MMD ou p-value n'est changée par cette suppression des alias.

## CI : incompatibilités confirmées

Le projet exige Python 3.11 et utilise `uv.lock`. Le workflow de qualité utilisait
Python 3.12 et Poetry, puis ajoutait des dépendances pendant la vérification.
Il est remplacé par une installation `uv sync --frozen` sous Python 3.11, avec
les outils déjà verrouillés. Black/isort et les erreurs Ruff restent bloquants ;
les diagnostics mypy/Bandit restent consultatifs comme précédemment. Les huit
notebooks sont validés avec nbformat disponible dans l'environnement verrouillé.
Le faux build Sphinx sans `docs/conf.py` et les commandes d'outils non verrouillés
ne sont pas conservés comme des validations prétendument opérationnelles.

Le formatage et le tri des imports sont appliqués pour faire passer les contrôles
existants. La syntaxe des calculs a été comparée avant/après : aucune modification
hors imports et mise en page de docstrings. Les noms indéfinis de Ruff ont été
résolus sans changer les statistiques (le nom de noyau de PlotData est rempli
plus bas depuis les métadonnées, comme auparavant).

La couverture globale reste une vérification séparée avec son seuil de 80 %.
La clarification des paramètres ne justifie pas d'abaisser ce seuil ni de
remplacer les références numériques de la suite d'intégration.

## Validation locale du 9 octobre 2026

Sous Windows, Python 3.11.15, Torch CPU et pySigLib 4.0, les 11 modèles
passent la pipeline complète et la comparaison aux références existantes,
sans régénérer ces références. La suite ordinaire hors tests lents compte
148 tests réussis, 9 ignorés et 2 exclus. Black, isort, les contrôles critiques
Ruff, les huit documents notebook et la cohérence du verrou uv passent.

La dernière mesure de couverture, avant suppression des anciennes options,
était de 63,58 % contre 80 % requis. Le seuil reste inchangé. Cette validation locale ne prouve ni la réussite des workflows
distants ni celle des exécutions Ruche/CUDA/DDP. La calibration scientifique
et la couverture des chemins non exercés restent à compléter.

La configuration d'intégration est adaptée sans régénérer les snapshots : sa
transition exacte est enregistrée dans `tests/integration/config_migration.json`.
Les tests du comparateur vérifient que cette transition n'accepte ni une autre
configuration active ni une modification des valeurs numériques.

## Exemples et commandes adaptés

Le Makefile utilise `fixed_events` pour la simulation par défaut ; la CLI et
sa documentation affichent seulement les trois presets actuels. Le notebook
`NewLTPP_Getting_Started.ipynb` utilise les configurations Pydantic et huit
séquences locales, sans téléchargement. Toutes ses cellules de code ont été
exécutées sur CPU : chargement des presets, construction, sauvegarde/relecture
YAML, train/test/predict et vérification des trois phases dans le manifeste.
La suite ordinaire passe toujours avec 148 tests réussis, 9 ignorés et 2 exclus.

Les exemples de signature des deux notebooks expérimentaux utilisent les
constructeurs actuels et `counting_grid`. Leurs anciennes sorties ont été
retirées pour éviter de présenter des résultats d'une ancienne représentation
comme ceux de la grille actuelle. Ces constructeurs ont été vérifiés, mais
les autres API historiques et les longues analyses scientifiques de ces deux
notebooks n'ont pas été validées intégralement.

Les anciens builders se trouvent dans `origin/master`. `main` les a supprimés
au commit `f69fd98`, avant cette branche. Les exemples suivent l'API Pydantic
actuelle ; une éventuelle réintroduction des builders sera un changement séparé.
