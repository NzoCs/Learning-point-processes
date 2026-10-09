# Migration pySigLib : implémentation et validation

Branche : `codex/pysiglib-migration`, issue de `codex/reproductibilite-ruche`
au commit `19aa258`. Cette branche utilise exclusivement pySigLib ; son acceptation
sur Ruche reste conditionnée aux contrôles ci-dessous. Elle ne modifie ni
l'embedding, ni la normalisation historique des temps, ni le protocole MMD.

Les corrections ultérieures de reproductibilité et le parcours CPU complet sont
consignés dans le [rapport actualisé](../rapports/REPRODUCTIBILITE_RUCHE.md).
La compilation automatique du dispatcher MMD et du moteur de simulation a été
retirée ; le parcours fonctionne désormais sans `TORCH_COMPILE_DISABLE`.
Les résultats ciblés ci-dessous décrivent la validation initiale de migration.

## Environnements

- Production CPU : `pysiglib==4.0.0`, obligatoire dans les dépendances runtime.
- Profil Ruche : extra `ruche`, ajoutant `pysiglib-cuda==4.0.0`.
- Baseline complète avant migration : code et lock conservés dans le commit
  `19aa258` de la branche distante `codex/reproductibilite-ruche`.

La dépendance historique, son groupe d'installation, sa sélection et les outils
qui l'importent ont été retirés de cette branche. Le champ de provenance
`signature_backend` accepte uniquement `pysiglib` ; une configuration demandant
l'ancien backend est rejetée. Aucun fallback de backend ou de device n'est effectué. L'extra CUDA doit être installé avant soumission GPU. Les wheels
pySigLib sont imposés par les commandes de la [procédure Ruche](RUCHE.md).
Une dépendance Python pure (`kauri`) peut être construite automatiquement ;
la contrainte binaire vise les deux composants natifs pySigLib.

## Contrat numérique

L'adaptateur utilise `method="finite_difference"`, conserve `dyadic_order`,
n'ajoute ni time augmentation ni lead-lag et ne normalise pas la diagonale du
Gram. Les chemins sont calculés en float64. `signature_max_batch` borne les
paires traitées à la fois ; les tests vérifient l'invariance au découpage.

Le noyau spatial linéaire conserve `scaling * <x,y>`. Le RBF conserve
`scaling * exp(-||x-y||² / sigma)`. Le facteur de scaling s'applique au noyau
spatial avant résolution, pas au Gram de signature final. Les noyaux non
supportés et paramètres invalides produisent une erreur explicite.

La MMD est la même MMD² non biaisée que la
[référence sigkernel épinglée](https://github.com/crispitagorico/sigkernel/blob/40a583155ea8d2194af0e90dddab37e2659cfcfd/sigkernel/sigkernel.py) :

```text
sum(i != j, Kxx[i,j]) / (n*(n-1))
+ sum(i != j, Kyy[i,j]) / (m*(m-1))
- 2 * mean(Kxy)
```

Elle peut être négative. Comparer le même échantillon fini aux deux entrées
ne produit pas nécessairement zéro. Les tailles n et m peuvent différer ;
chacune doit être au moins deux. Les trois Gram utilisent les mêmes embeddings
préparés ensemble, pour préserver la normalisation actuelle. Sa pertinence
scientifique reste une question séparée du remplacement de bibliothèque.

Les inputs nécessitant des gradients passent par `pysiglib.torch_api`.
Les gradchecks vérifient les dérivées du nouveau solveur par différences finies,
pour linear/RBF avec scaling. Cela ne certifie pas l'équivalence des gradients
approximatifs de la bibliothèque historique.

## Vérifications exécutées sur Windows CPU

- Installation runtime : `uv sync --frozen --no-default-groups`, réussie.
- `uv pip check` et `uv lock --check --offline`, réussis.
- Aide réelle `new-ltpp run --help`, exécutée sans import `sigkernel` requis.
- 52 tests réussis : signature, migration, launchers et contenu du wheel ;
  1 ignoré pour CUDA indisponible. Les tests ne chargent aucune bibliothèque historique.
- Comparaison de Gram à la récurrence mathématique indépendante du solveur
  historique, linear/RBF, scaling, raffinements 0/1/2 et tailles inégales :
  `rtol=atol=1e-10` sur les petites fixtures de chemins.
- Cas intégré avec padding et séquence vide masquée : `rtol=atol=1e-9`.
- MMD comparée à la formule historique et à l'API native pySigLib.
- Gradchecks : `eps=1e-6`, `atol=1e-5`, `rtol=1e-4`.
- Wheel construit et testé hors du checkout : le bilan incluant ce contrôle
  est de **52 tests réussis et 1 ignoré**. Contrôle CPU réel du backend exécuté,
  avec version, solveur, précision et état du plugin enregistrés dans sa sortie.

Commande des tests ciblés (les options globales de couverture sont écartées) :

```bash
TORCH_COMPILE_DISABLE=1 uv run --frozen --no-sync python -m pytest -o addopts= \
  tests/stat_metrics/test_signature_migration.py \
  tests/stat_metrics/test_sig_kernel.py tests/scripts/test_ruche_launcher.py \
  tests/scripts/test_wheel_contents.py -q
```

Ces contrôles ne certifient pas `torch.compile`. La récurrence Python de test
est un oracle mathématique indépendant ; aucune comparaison avec la bibliothèque
native historique n'est annoncée. Le code et le lock de cette bibliothèque
restent disponibles dans la branche de référence avant migration.

## Validation Ruche à exécuter

Le job CI `signature-migration` installe uniquement pySigLib et vérifie sur CPU
les valeurs du solveur et ses gradients. Pour Ruche, suivre l'organisation
HOME/WORKDIR décrite dans `RUCHE.md` et installer :

```bash
uv sync --frozen --no-default-groups --extra ruche \
  --no-build-package pysiglib --no-build-package pysiglib-cuda
```

Sur un nœud GPU alloué, exécuter :

```bash
uv run --frozen --no-sync python -m scripts.check_backend --device cuda
TORCH_COMPILE_DISABLE=1 uv run --frozen --no-sync python -m pytest -o addopts= tests/stat_metrics/test_signature_migration.py -q
```

Le contrôle enregistre les métadonnées du backend et vérifie qu'un véritable
Gram a été calculé sur CUDA. Le test dédié compare CPU et GPU à
`rtol=atol=1e-8`. Mesurer également les charges réelles et la mémoire GPU.

Restent à vérifier sur Ruche : compatibilité glibc/pilote, plugin CUDA réellement
chargé, gradients GPU si utilisés, mini entraînement, évaluation et simulations,
performances/mémoire représentatives. Le protocole de p-values et sa calibration
ne sont pas modifiés ni validés par ce lot.

Sources API consultées le 9 octobre 2026 :
[installation](https://pysiglib.readthedocs.io/en/latest/pages/installation.html),
[Gram](https://pysiglib.readthedocs.io/en/latest/pages/signature_kernels/sig_kernel_gram.html),
[noyaux spatiaux](https://pysiglib.readthedocs.io/en/latest/pages/signature_kernels/static_kernels.html),
[API torch](https://pysiglib.readthedocs.io/en/latest/pages/torch_api.html).
