# Migration pySigLib : implémentation et validation

Branche : `codex/pysiglib-migration`, issue de `codex/reproductibilite-ruche`
au commit `19aa258`. Cette branche utilise pySigLib par défaut ; son acceptation
sur Ruche reste conditionnée aux contrôles ci-dessous. Elle ne modifie ni
l'embedding, ni la normalisation historique des temps, ni le protocole MMD.

## Environnements

- Production CPU : `pysiglib==4.0.0`, obligatoire dans les dépendances runtime.
- Profil Ruche : extra `ruche`, ajoutant `pysiglib-cuda==4.0.0`.
- Référence native : groupe séparé `legacy-reference`, avec `sigkernel` fixé au
  commit `40a583155ea8d2194af0e90dddab37e2659cfcfd`.
- Baseline complète avant migration : code et lock conservés dans le commit
  `19aa258` de la branche distante `codex/reproductibilite-ruche`.

L'installation de la référence est explicite ; elle ne s'effectue pas lors
d'une installation de production. Aucun fallback de backend ou de device
n'est effectué. L'extra CUDA doit être installé avant soumission GPU. Les wheels
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
- 51 tests réussis : signature, migration et launchers ; 7 ignorés
  (6 pour la référence native indisponible, 1 pour CUDA indisponible).
- Comparaison de Gram à la récurrence mathématique indépendante du solveur
  historique, linear/RBF, scaling, raffinements 0/1/2 et tailles inégales :
  `rtol=atol=1e-10` sur les petites fixtures de chemins.
- Cas intégré avec padding et séquence vide masquée : `rtol=atol=1e-9`.
- MMD comparée à la formule historique et à l'API native pySigLib.
- Gradchecks : `eps=1e-6`, `atol=1e-5`, `rtol=1e-4`.
- Wheel construit et testé hors du checkout : le bilan incluant ce contrôle
  est de **52 tests réussis et 7 ignorés**. Contrôle CPU réel du backend exécuté,
  avec version, solveur, précision et état du plugin enregistrés dans sa sortie.

Commande des tests ciblés (les options globales de couverture sont écartées) :

```bash
TORCH_COMPILE_DISABLE=1 uv run --frozen --no-sync python -m pytest -o addopts= \
  tests/stat_metrics/test_signature_migration.py \
  tests/stat_metrics/test_sig_kernel.py tests/scripts/test_ruche_launcher.py -q
```

Ces contrôles ne certifient pas `torch.compile`. La tentative d'installation
du groupe `legacy-reference` sur Windows a échoué à la compilation Cython,
faute de MSVC ; aucune comparaison native historique locale n'est annoncée.
La récurrence Python de test est un oracle mathématique, pas cette bibliothèque.

## Comparaison native et Ruche à exécuter

Dans un environnement Linux de comparaison, installer explicitement les deux
backends. Le job CI `signature-migration` prépare cette comparaison sur CPU ;
il importe la référence native avant les tests pour empêcher un succès fondé
sur leur simple omission. Pour Ruche, suivre l'organisation HOME/WORKDIR
décrite dans `RUCHE.md` et installer :

```bash
uv sync --frozen --no-default-groups --extra ruche --group legacy-reference \
  --no-build-package pysiglib --no-build-package pysiglib-cuda
```

Sur un nœud GPU alloué, exécuter :

```bash
uv run --frozen --no-sync python -m scripts.check_backend --device cuda
uv run --frozen --no-sync python -m pytest -o addopts= tests/stat_metrics/test_signature_migration.py -q
uv run --frozen --no-sync python -m scripts.compare_signature_backends \
  --device cuda --output "$LTPP_OUTPUT_ROOT/comparaison-signature-001.json"
```

Le script sauvegarde les chemins exacts, seeds, permutations, statistiques
nulles, erreurs Gram/MMD, métadonnées du backend et timings après warm-up.
Il échoue en cas de désaccord numérique et refuse d'écraser un rapport existant.
Le CPU et le GPU doivent aussi être comparés à `rtol=atol=1e-8` sur la fixture
dédiée. Les timings des petites fixtures ne sont pas un benchmark représentatif
de production ; mesurer aussi les charges réelles et la mémoire GPU.

Restent à vérifier sur Ruche : compatibilité glibc/pilote, plugin CUDA réellement
chargé, comparaison native, gradients GPU si utilisés, mini entraînement,
évaluation et simulations, performances/mémoire représentatives. Le protocole
de p-values et sa calibration ne sont pas modifiés ni validés par ce lot.

Sources API consultées le 9 octobre 2026 :
[installation](https://pysiglib.readthedocs.io/en/latest/pages/installation.html),
[Gram](https://pysiglib.readthedocs.io/en/latest/pages/signature_kernels/sig_kernel_gram.html),
[noyaux spatiaux](https://pysiglib.readthedocs.io/en/latest/pages/signature_kernels/static_kernels.html),
[API torch](https://pysiglib.readthedocs.io/en/latest/pages/torch_api.html).
