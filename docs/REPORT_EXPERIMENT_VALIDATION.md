# Validation scientifique des expériences du rapport

Ce protocole prolonge l'audit de `SignatureMMDTesting` du 15–16 octobre
2026. Il teste les scénarios synthétiques du rapport avec des données fraîches,
une comparaison équitable des kernels et un contrôle de l'information apportée
par les temps au-delà du compte final. Il ne modifie aucun modèle de production.

## Contrat principal, fixé avant les calculs

- Fenêtre observée entière `[0,1]`, 128 trajectoires par groupe, grille de 100
  points, dyadique 0, float64, amplitude statique 1, RBF `sigma=1`.
- Trois méthodes sur **les mêmes observations et les mêmes échanges** :
  signature/linéaire, signature/RBF et RBF utilisant seulement `N(T)`.
- Les deux canaux des chemins sont `N(t)/std_pilote` et `t/T` : cela conserve
  la représentation du rapport et isole la comparaison des méthodes. Ce n'est
  pas l'embedding multimarque de la pipeline conditionnelle de production.
- Un pilote indépendant de 512 trajectoires par loi fixe l'échelle des comptes.
  Il est partagé entre les cohortes et entre les kernels, puis reste inchangé.
  Les données évaluées ne choisissent ni cette échelle ni les hyperparamètres.
- 499 échanges indépendants au sein des paires ; statistique U appariée et
  p-value `(1 + nombre de statistiques nulles >= observation)/(499+1)` via les
  fonctions de production. Les égalités numériques suivent leur règle actuelle.
- Deux cohortes indépendantes : **1 000 essais H0 par cohorte et cas**, soit
  2 000 par cas ; **250 essais H1 par cohorte et cas**, soit 500 par cas.
- Quatre cas H0 et six H1, donc **11 000 essais** et 33 000 p-values au total.
  Les effectifs et paramètres ne sont pas prolongés ou ajustés pour obtenir
  un résultat favorable.

## Cas et interprétation

| Cas | Comparaison | Ce qu'il contrôle |
|---|---|---|
| `poisson_H0` | Poisson 100 / Poisson 100 indépendants | Calibration élémentaire |
| `hawkes_exp_H0` | Hawkes exponentiel / même loi | Calibration avec clustering |
| `hawkes_power_H0` | Hawkes puissance p=2 / même loi | Calibration avec mémoire longue |
| `poisson_timing_H0` | Poisson / temps uniformes indépendants conditionnellement aux mêmes comptes | Calibration du contrôle temporel apparié |
| `poisson_rate_110` | Taux 100 / 110 | Sensibilité au changement de taux |
| `hawkes_alpha_8` | alpha=5,mu=50 / alpha=8,mu=20, beta=10 | Dynamique à moyenne stationnaire commune |
| `exp_vs_power_p2` | Exponentiel / puissance p=2, n=0.5,mu=50,beta=10 | Scénario du rapport, avec réserve de burn-in |
| `exp_vs_power_p3` | Même comparaison, puissance p=3 | Formes différentes, délai moyen d'un descendant direct identique, effet de burn-in réduit |
| `hawkes_exp_timing` | Hawkes exponentiel / temps uniformes aux comptes exactement conservés | Information temporelle au-delà de N(T) |
| `hawkes_power_timing` | Hawkes puissance / temps uniformes aux comptes exactement conservés | Même contrôle pour la mémoire longue |

Dans les contrôles temporels, chaque trajectoire conserve exactement son
nombre d'événements. Les temps du second bras sont de nouveaux tirages
uniformes triés sur `[0,1]`. Le test sur `N(T)` doit alors rendre `p=1` : sa
cécité est un contrôle attendu, pas un défaut. Sous Poisson homogène, les deux
bras sont échangeables conditionnellement aux comptes : le cas est un vrai H0.
Pour Hawkes, c'est une alternative scientifique distincte, pas une modification
du simulateur ou du modèle utilisé dans la bibliothèque.

## Références indépendantes et limites

Poisson utilise un nombre d'événements Poisson puis des positions uniformes.
Hawkes utilise la représentation en grappes de Poisson : immigrants puis
descendants avec moyenne `n<1`, délais exponentiels ou Lomax. Il s'agit d'une
implémentation indépendante du thinning des modèles de production.

Les Hawkes commencent avec un passé vide à `-10`. Le code ne les déclare pas
exactement stationnaires. La moyenne stationnaire cible est 100 ; l'espérance
du compte réellement observé peut différer, surtout pour une puissance longue.
Pour p=2, elle est environ 98,99 sur `[0,1]` ; pour p=3, environ 99,991.
Les contrôles temporels à comptes conservés éliminent ce facteur de comparaison.

Les oracles ont un budget de calcul explicite. Son dépassement produit un
échec enregistré de l'essai, jamais un événement déplacé ou une trajectoire
tronquée. Tous les échecs empêchent la certification de la cellule et de la
campagne complète ; le sous-ensemble de succès ne remplace pas l'effectif prévu.

Ce protocole ne valide pas les modèles appris, les marques multiples, les
quotas ou les arrêts. La campagne `pipeline_protocol_calibration` sur Ruche
teste séparément ce contrat exact de la pipeline. Ses contrôles H0 internes
sur certains modèles neuronaux ne certifient pas à eux seuls leur simulateur
non constant ; les références indépendantes ont leur périmètre propre.

Les courbes RMSE Hawkes du rapport demandent encore une référence commune
entre prédicteurs. Elles ne sont pas remplacées par cette campagne. Une
comparaison formelle de puissance MMD/RMSE nécessitera un test RMSE de même
niveau, avec le même contrat d'observation et de prédiction.

## Critères de lecture

L'agrégateur exige les 20 cellules complètes et cohérentes, sur un commit propre,
avec le même environnement et le même pilote entre cohortes. Il recalcule
chaque p-value depuis les statistiques archivées.

- Excès de faux positifs : test binomial unilatéral contre 5 %, seuil
  `0.01/36` pour les 4 H0 x 3 méthodes x (2 cohortes + agrégation).
- Précision : bornes supérieures simultanées à 99 % sur les 12 H0 agrégés,
  avec critère annoncé d'exclusion d'un taux >=7,5 %. Une borne imprécise
  est **inconclusive**, pas la preuve d'un défaut. Ne pas confondre ce critère
  de précision avec le niveau nominal 5 %.
- Puissance : taux et intervalles binomiaux exacts. Comparaisons appariées
  entre méthodes, avec correction de Holm sur les 18 comparaisons prévues.
- Aucune absence de rejet H0 ne prouve l'égalité exacte au niveau nominal.
  Une faible puissance n'est pas automatiquement une erreur de code.

## Exécution à la demande

Depuis un environnement installé avec le `uv.lock` du dépôt :

```bash
python -m scripts.report_experiment_validation \
  --case hawkes_exp_timing --cohort 0 --trials 250 \
  --n 128 --permutations 499 --grid 100 --device cuda \
  --output /workdir/scientific-report/hawkes_exp_timing/cohort-0

python -m scripts.report_experiment_summary /workdir/scientific-report
```

Un pilote technique peut utiliser `--trials 2 --n 8 --permutations 19`.
L'agrégateur principal refuse ces petits effectifs : un pilote ne devient
jamais une preuve de calibration.

`python -m scripts.check_report_experiment_backend --device cuda --output ...`
compare les Grams/statistiques CPU et CUDA puis mesure trois petits pilotes à
l'effectif principal. Ces pilotes techniques utilisent un domaine de graines
distinct, exclu par l'agrégateur principal.

Avant une campagne GPU, vérifier la concordance CPU/CUDA des Grams et le
temps d'un petit pilote. Soumettre ensuite des cellules bornées à Slurm,
avec un seul GPU simultané. La campagne longue n'est pas exécutée par la CI.
Les tests ordinaires vérifient uniquement les invariants et de petits cas.

Chaque cellule conserve son plan, ses graines, les empreintes des observations,
les statistiques observées/nulles et les p-values dans `trials.jsonl`. Une reprise
doit garder le même plan, source et environnement ; elle reprend un préfixe
continu sans remplacer les essais échoués. Un append final déchiré est conservé
à part et l'essai interrompu est reconstruit à l'identique depuis sa graine.
