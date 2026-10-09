# Calibration scientifique du test MMD

## Ce qui a ete verifie le 9 octobre 2026

Le calcul non biaise exclut les diagonales et utilise les tailles propres aux
deux groupes. Un oracle manuel sur des groupes de tailles 2 et 3 le controle.
Un second test enumere les 20 partitions d'un ensemble de six observations et
verifie le controle exact des rangs, y compris les ex aequo. Les simulations
Monte Carlo utilisent `(1 + nombre de valeurs >= observation)/(B + 1)`.

Ces controles reposent sur les hypotheses de la MMD et des permutations :
[Gretton et al. (2012)](https://jmlr.org/beta/papers/v13/gretton12a.html) et
[Hemerik et Goeman (2018)](https://doi.org/10.1007/s11749-017-0571-1).
Les observations doivent etre echangeables sous H0 et la preparation doit etre
independante des etiquettes de groupe. La correction Monte Carlo ne remplace
pas cette condition.

## Experience reproductible sur des processus connus

Commande depuis la racine, dans l'environnement verrouille :

```sh
uv run --frozen --no-sync python -m scripts.scientific_calibration --trials 200 --permutations 99
```

Graine 20261009, huit sequences independantes par groupe, cinq evenements
par sequence, CPU, pySigLib, RBF, counting_grid de huit points, ordre dyadique 0.
Le Poisson utilise directement des intervalles exponentiels. Le Hawkes scalaire
de reference utilise Ogata avec excitation .3 et decroissance 1, sans burn-in.
Sous H0 les deux intensites de base sont 2; sous H1 elles sont 2 et 8.
Les rejets utilisent `p <= .05`, avec resolution minimale .01.

| Processus | Preparation | Rejets H0 / 200 | IC Wilson 95 % | Puissance H1 |
|---|---|---:|---|---:|
| Poisson | Echantillons independants | 12 (6 %) | 3,47–10,19 % | 97,5 % |
| Hawkes | Echantillons independants | 8 (4 %) | 2,04–7,69 % | 96,5 % |
| Poisson | Troncature historique par paires | 12 (6 %) | 3,47–10,19 % | 98 % |
| Hawkes | Troncature historique par paires | 19 (9,5 %) | 6,17–14,36 % | 96,5 % |

La mesure du protocole independant est compatible avec 5 % dans ces experiences.
Elle ne constitue pas une preuve universelle de calibration ou de puissance.
Le protocole historique montre une inflation des faux positifs sur ce Hawkes.
Un test demontre aussi que ses masques changent selon les etiquettes initiales.
Il ne doit donc pas etre presente comme un test de permutation calibre.

Le calcul accelere utilise une matrice de Gram preparee sur le pool complet.
Sa statistique et ses p-values sont comparees a l'API reelle de pySigLib,
avec les memes permutations, avec et sans troncature et avec tailles inegales.
Cette equivalence est testee pour SIGKernel; elle n'est pas generalisee a
n'importe quel noyau ou a une normalisation dependante des groupes.

## API et compatibilite des resultats

Pour comparer des echantillons independants soumis a la meme regle d'observation :

```python
result = test.compute_statistics(x, y, accumulate=False, paired_truncation=False)
```

Le defaut `paired_truncation=True` est conserve pour expliciter la compatibilite
avec les references historiques. Les pipelines existantes ne sont pas declarees
scientifiquement calibrees par cette validation. Leur bascule exige un choix
documente du protocole d'observation et une revision explicite des references.
Une meme fenetre fixe pour toutes les observations peut aussi etre preparee
avant d'appeler l'API sans troncature par paires.

Les comparaisons de simulations conditionnelles, les modeles ajustes sur les
donnees de test, les p-values agregees entre batches, les marques multiples,
les autres noyaux, CUDA et DDP restent hors de cette certification.

Le resume mesure et ses empreintes sont versionnes dans
`docs/validation/scientific-calibration-2026-10-09.json`.
Le rapport complet avec toutes les p-values est produit dans
`artifacts/scientific-calibration/report.json`.

## Notebooks et execution optionnelle

`Hawkes_MMD_Test.ipynb` execute les experiences H0/H1 et les sweeps de largeur
RBF, intensite et taille des groupes. `Test_MMD_Metric.ipynb` compare les noyaux
actuels, controle la symetrie et la formule sur Gram a tailles inegales.
Les exemples sont courts et ne contiennent aucun telechargement.

```sh
uv run --frozen --no-sync python -m scripts.validate_notebooks
```

Chaque notebook est execute integralement dans un nouveau processus Python.
Les journaux, nombres de cellules et empreintes vont dans
`artifacts/notebook-validation`. Le workflow manuel `scientific.yml` execute
les notebooks et l'experience complete sur Linux; il n'est pas lance a chaque push.

## Correction des benchmarks

Les baselines moyenne des intervalles, distribution des intervalles et
distribution des marques apprenaient sur le jeu de test malgre leur contrat
explicite d'apprentissage sur le jeu d'entrainement. Elles apprennent maintenant
sur l'entrainement et evaluent toujours sur le test. Des jeux disjoints aux
valeurs contrastees controlent ce comportement. Leurs scores historiques
peuvent changer : c'est une correction de fuite de donnees, pas un simple
changement d'architecture. Les references des onze modeles restent distinctes.
