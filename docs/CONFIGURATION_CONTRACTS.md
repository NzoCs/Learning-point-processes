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

Les anciennes valeurs `SimulationConfig.time_window`, `.initial_buffer_size`
et `.batch_size` ne sont donc pas des commandes à reconnecter au moteur actuel.
Elles restent acceptées et sérialisées pour rejouer les configurations anciennes,
mais deviennent optionnelles et dépréciées dans le schéma. Une nouvelle
configuration suffit :

```yaml
simulation_config:
  seed: 42
```

Le manifeste ajoute `simulation_contract`, avec le mode réellement exécuté,
les sources du nombre d'événements/de la taille du lot, la graine et les anciens
paramètres ignorés. Le preset `fixed_events` est le nom actuel ; `quick_test`,
`debug` et les anciens noms `tw...` restent des alias de compatibilité avec la
même graine historique 42. Aucun horizon temporel n'est réintroduit. La limite
de génération dépend toujours de la largeur paddée du lot, pas du nombre valide
de chaque séquence : c'est le contrat conservé, à distinguer de l'invariance au
padding des chemins de signature. Un autre protocole de génération doit faire
l'objet d'un changement scientifique explicite.

Les graines hors de `[0, 2**32 - 1]` étaient acceptées par SimulationConfig mais
rejetées par Lightning lors de `predict`. Elles sont désormais rejetées dès le
chargement, comme les graines d'entraînement. Les graines 0 et `2**32 - 1`
restent valides.

## Signature : une représentation commune, avec des alias historiques

Depuis `6e5e75e`, `_get_embedding` documente que `embedding_type` est conservé
pour compatibilité. L'implémentation construit une grille régulière contenant
le temps, le comptage total normalisé et les comptages par marque. Elle ne choisit
pas une interpolation différente selon les étiquettes `linear`/`constant`.

`counting_grid` devient le nom explicite et la valeur par défaut. Les anciennes
étiquettes restent acceptées et produisent exactement le même chemin. Le
manifeste indique `signature_path_representation: counting_grid`, indépendamment
de l'alias utilisé. Aucune formule de noyau, MMD ou p-value n'est changée par
cette clarification. Les étiquettes inconnues sont rejetées dans la préparation.

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
145 tests réussis, 9 ignorés et 2 exclus. Black, isort, les contrôles critiques
Ruff, les huit documents notebook et la cohérence du verrou uv passent.

La commande avec couverture échoue uniquement sur le seuil : 63,58 % contre
80 % requis. Cette validation locale ne prouve ni la réussite des workflows
distants ni celle des exécutions Ruche/CUDA/DDP. La calibration scientifique
et la couverture des chemins non exercés restent à compléter.
