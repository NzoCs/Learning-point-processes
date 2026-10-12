# Comparaison de la troncature recalculée et de l'absence de troncature

## Question et protocole

La troncature par paires avait pour objectif de comparer deux séquences sur
la même durée malgré une génération à nombre d'événements fixé. L'ancienne
implémentation figeait les masques à partir des groupes observés avant de
permuter. Elle ne recalculait donc pas la même statistique sur chaque nouvelle
affectation aux groupes.

L'expérience du 12 octobre 2026 compare :

- **Sans troncature** : MMD des séquences brutes, avec leurs masques fournis.
- **Troncature recalculée** : les deux groupes sont reconstruits à partir des
  séquences brutes à chaque permutation, puis les séquences sont appariées
  selon leur nouvel ordre et tronquées à leur horizon minimal. La normalisation
  et le noyau sont également recalculés après cette préparation.

Une préparation qui dépend des groupes peut faire partie de la statistique si
elle est réappliquée à chaque transformation. Une préparation commune figée doit
en revanche permettre l'échangeabilité des observations transformées. La condition
fondamentale reste l'invariance sous H0, pas une interdiction générale de tronquer.
Voir [Hemerik et Goeman, section 2](https://link.springer.com/article/10.1007/s11749-017-0571-1).

L'expérience conserve **les mêmes données brutes et les mêmes permutations**
pour les deux protocoles. Les empreintes des données et des permutations sont
enregistrées. Le groupe de permutations inclut aussi les réordonnancements
à l'intérieur de chaque groupe, car ils changent l'appariement des séquences.

```sh
uv run --frozen --no-sync python -m scripts.compare_truncation \
  --trials 500 --power-trials 200 --permutations 99 --seed 20261012
```

Les deux processus de référence sont Poisson et Hawkes scalaire (excitation .3,
décroissance 1). Chaque groupe contient huit séquences de cinq événements.
Sous H0, les deux intensités de base valent 2. La puissance est mesurée contre
les intensités 4 et 8, avec 200 répétitions par alternative. Le noyau est
SIGKernel/pySigLib/RBF/counting_grid, huit points, ordre dyadique 0, sur CPU.

## Contrôles numériques

La MMD calculée à partir de la matrice de Gram du pool est comparée au calcul
réel de pySigLib pour chaque permutation. La troncature est vérifiée par un
oracle scalaire indépendant. Un petit cas énumère les 24 permutations des
quatre séquences, y compris leurs différents appariements, et vérifie les rangs
exacts ainsi que l'absence de mutation des masques bruts.

Les p-values utilisent `(1 + nombre de statistiques nulles >= observation)/(B+1)`.
Le quotient est calculé à partir de comptes entiers en float64 Python pour
qu'une p-value exactement égale à .05 soit rejetée avec la règle `p <= .05`.
Les égalités numériques proches sont comptées conservativement comme
supérieures ou égales (`rtol=1e-10`, `atol=1e-12`), dans les deux protocoles.

Ce traitement évite que des arrondis float32 au seuil ou des variations de
sommation entre statistiques théoriquement égales faussent la comparaison.
Il est propre à l'expérience. Les valeurs de l'exploration du 9 octobre,
avec un autre tirage et des arrondis au seuil, ne sont pas une comparaison
appariée avec les valeurs de cette nouvelle expérience.

## Portée

Les contrôles sont dans `tests/stat_metrics/test_recomputed_truncation.py`.
L'implémentation expérimentale est isolée dans `scripts/compare_truncation.py`.
Les defaults de production et les références historiques restent inchangés.

La calibration étudiée concerne des séquences indépendantes sous H0, avec le
même mécanisme d'observation à nombre d'événements fixé. Elle ne certifie ni
les modèles ajustés, ni les simulations conditionnelles aux observations,
ni les p-values agrégées entre batches, ni les marques multiples, CUDA ou DDP.
L'horizon minimal d'une paire est aléatoire ; il n'équivaut pas à une fenêtre
d'observation commune fixée à l'avance.

## Calibration sous H0

Le niveau demandé est 5 %. Chaque ligne repose sur 500 répétitions indépendantes.

| Processus | Sans troncature | IC Wilson 95 % | Troncature recalculée | IC Wilson 95 % |
|---|---:|---|---:|---|
| Poisson | 27/500 = 5,4 % | 3,74–7,74 % | 21/500 = 4,2 % | 2,76–6,34 % |
| Hawkes | 23/500 = 4,6 % | 3,08–6,81 % | 26/500 = 5,2 % | 3,57–7,51 % |

Les quatre intervalles contiennent 5 %. Cette expérience ne montre pas
d'inflation des faux positifs avec la troncature recalculée. Cela ne prouve
pas une calibration universelle sur tous les processus et tous les paramètres.

La différence appariée entre les taux de rejet (troncature moins absence de
troncature) vaut -1,2 point pour Poisson et +0,6 point pour Hawkes, avec des
erreurs standard respectives de 1,27 et 1,31 point. Ces différences ne permettent
pas de départager les protocoles sur leur calibration dans les cas testés.

## Puissance sous H1

Chaque ligne repose sur 200 répétitions et les mêmes données/permutations pour
les deux protocoles. L'intensité de base du premier groupe vaut toujours 2.

| Processus | Intensité du second groupe | Sans troncature | Troncature recalculée |
|---|---:|---:|---:|
| Poisson | 4 | 50,5 % | 58 % |
| Poisson | 8 | 99 % | 98,5 % |
| Hawkes | 4 | 39 % | 44 % |
| Hawkes | 8 | 96 % | 94,5 % |

Les différences appariées valent respectivement +7,5, -0,5, +5 et -1,5 points.
Les intervalles normaux approximatifs à 95 % de ces différences contiennent
tous zéro. Ces mesures ne démontrent ni une supériorité générale ni une
non-infériorité formelle ; elles ne montrent pas de perte majeure dans les
alternatives et paramètres testés. Les intervalles Wilson de chaque proportion,
les désaccords appariés et toutes les p-values figurent dans
[`validation/truncation-comparison-2026-10-12.json`](validation/truncation-comparison-2026-10-12.json).

## Recommandation et état de la branche

Pour l'objectif initial de comparaison sur une même durée au sein de chaque
paire, nous recommandons **la troncature recalculée après chaque permutation**.
Sa calibration est compatible avec le niveau demandé dans ces expériences et
elle conserve l'objectif scientifique de la préparation d'origine. L'absence
de troncature demeure pertinente pour comparer directement les trajectoires
observées à nombre d'événements fixé, avec leurs durées propres.

Cette recommandation ne valide pas l'ancienne troncature figée. La future
bascule de production doit repartir des séquences brutes à chaque permutation,
réappliquer la même préparation à l'observation et au nul et contrôler les
égalités numériques ainsi que la décision au seuil. Les simulations
conditionnelles et l'agrégation de batches ont des hypothèses supplémentaires
et ne doivent pas être déclarées calibrées sur la seule base de ce test.

À ce stade, la comparaison est isolée dans un script expérimental. La pipeline
de production et ses références numériques n'ont pas encore été basculées.
Le workflow scientifique manuel rejoue aussi cette expérience et préserve son
rapport. Il doit être enregistré sur la branche par défaut pour être lancé
depuis l'interface GitHub.

La suite Windows passe avec **194 tests réussis**, 10 ignorés et 2 désélectionnés,
pour une couverture de **80,79 %**. Sur le code `475b4b1`, la
[CI Linux](https://github.com/NzoCs/Learning-point-processes/actions/runs/38226405195)
réussit avec **201 tests**, 3 ignorés, 2 désélectionnés et la même couverture.
Le [workflow qualité](https://github.com/NzoCs/Learning-point-processes/actions/runs/38226405176)
réussit également. L'expérience complète de comparaison a été exécutée sur
Windows CPU ; son exécution complète sur Linux n'est pas encore certifiée.
