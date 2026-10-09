# Régression d'intégration complète

Cette suite est volontairement séparée de `pytest tests` et de la CI exécutée
à chaque push. Le workflow `Complete integration regression` se déclenche
uniquement avec `workflow_dispatch` (Actions → Run workflow, puis sélectionner
la branche et `all`). Il vérifie CPU/Python 3.11 sur Linux et Windows avec
l'environnement verrouillé et conserve les logs et artefacts même en cas d'échec.
Le workflow doit être présent sur la branche par défaut pour apparaître dans
l'interface GitHub ; sur cette branche de travail, la commande locale est utilisable.

```bash
uv sync --frozen --no-default-groups --group dev --no-build-package pysiglib
uv run --frozen --no-sync python -m scripts.integration_suite
# Pour diagnostiquer un seul modèle :
uv run --frozen --no-sync python -m scripts.integration_suite --model FullyNN
```

La suite couvre les onze modèles publics : ANHN, ANHP, FullyNN, Hawkes,
IntensityFree, NHP, ODETPP, RMTPP, SAHP, SelfCorrecting et THP. Un changement
du catalogue public exige l'ajout de tests et de références ; aucun modèle
défaillant n'est silencieusement ignoré. Chaque modèle tourne dans un processus
isolé, avec un délai maximal de cinq minutes et un rapport commun conservé
progressivement. L'exécution continue après un échec pour diagnostiquer tous les modèles.

## Parcours et résultats contrôlés

Les huit séquences fixes de cinq événements et deux marques sont dans
`tests/integration/fixture.json`. Le même petit jeu est utilisé pour les trois
splits : cette suite vérifie le logiciel, pas la généralisation du modèle.
`config.json` fige les paramètres indépendamment des presets de production :
CPU, un thread, aucun worker, aucun dropout, graines fixes et une époque.
Aucune donnée Hugging Face n'est téléchargée.

Le parcours rejoue une configuration sauvegardée via la CLI publique en phase
`all` : chargement → entraînement → checkpoint → évaluation → simulation →
statistiques → sauvegarde Parquet/CSV et figures. Le manifeste doit confirmer
les trois phases et l'utilisation du même checkpoint. Les poids entraînés sont
rechargés strictement pour les sondes numériques suivantes.

Les références versionnées comparent les poids complets, les gradients des
paramètres sur une perte fixe, la perte et les intensités sondées, les métriques
de test, tous les événements simulés et les résultats numériques du CSV.
IntensityFree utilise sa densité conditionnelle pour tirer directement un
événement : sa perte, ses gradients et ses simulations sont contrôlés, sans
inventer une fonction d'intensité. Les figures doivent être présentes et non
vides ; leurs pixels, les timestamps, UUID et chemins ne sont pas comparés.
Les marques, tailles, noms de champs et valeurs entières doivent être identiques.
Pour les flottants : `rtol=1e-5`, `atol=1e-6`. Ce sont des tolérances de régression
CPU, pas une certification d'équivalence CUDA/DDP ou de toutes les architectures matérielles.

## Politique de référence historique

Les références commencent à l'état corrigé de cette branche. Elles ne prouvent
pas l'identité avec d'anciennes expériences qui n'ont pas de référence conservée.
Elles figent les résultats à partir de maintenant. Pour un changement de structure
sans changement de calcul, les tests doivent passer sans modifier les références.
Une divergence est un échec à analyser : on ne la masque pas en élargissant
automatiquement les tolérances.

Après un changement de calcul explicitement voulu, produire des **candidats** :

```bash
uv run --frozen --no-sync python -m scripts.integration_suite --record --output artifacts/integration-candidates-new
```

Cette commande ne remplace jamais les références. Examiner les différences des
snapshots, vérifier les formules avec des tests indépendants, puis copier seulement
les snapshots des modèles concernés dans `tests/integration/baselines/`. Conserver
la provenance d'environnement et écrire dans `BASELINE_HISTORY.md` la raison
mathématique, les modèles concernés et l'effet observé. Vérifier ensuite les
références sur un second passage et inclure les différences dans le commit/PR.
Les changements d'environnement doivent eux aussi être identifiés et examinés.
La CI ne produit ni n'approuve automatiquement une nouvelle référence.

Les artefacts d'exécution sont sous `artifacts/integration/` (ou `--output`) :
rapport, logs par modèle, snapshots, provenance, configurations, manifests,
checkpoints, métriques, événements et figures. Réutiliser un dossier contenant
déjà une exécution est refusé afin de ne pas mélanger les résultats.

## Validation initiale

Le 9 octobre 2026, sur Windows/Python 3.11 et Torch 2.8.0+cpu : les onze modèles
ont terminé un premier parcours de capture puis un second parcours complet de
comparaison, tous réussis. Une modification volontaire d'une perte capturée a
été rejetée par le comparateur. La suite habituelle a donné 136 tests réussis,
9 ignorés (Make/CUDA indisponibles) et 2 tests lourds exclus. Les tests ciblés
contrôlent aussi l'alignement CSV, les identifiants Parquet et la cohérence du
dernier pas SelfCorrecting. Le workflow Linux/Windows est configuré, mais son
exécution distante et la validation Ruche/CUDA ne sont pas certifiées par ce
passage local. Le seuil global de couverture de 80 % reste à traiter séparément.
