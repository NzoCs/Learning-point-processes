# Validation des marques et des modèles appris

Le script `python -m scripts.extended_scientific_validation` réalise neuf expériences
avec la troncature recalculée à chaque permutation et les p-values float64
conservatrices de production. Deux marques, six événements par séquence, huit
séquences par groupe, 99 permutations : 300 répétitions par hypothèse nulle,
100 par diagnostic ou alternative. Les expériences sont optionnelles ; le workflow
`Extended scientific validation` est déclenchable manuellement, ou sur la branche
de développement par un commit contenant `[scientific-extended]`.

Les oracles NumPy sont indépendants des simulateurs de la bibliothèque : Poisson
marqué (taux 2, probabilités 0.65/0.35), et Hawkes multivarié exponentiel. Pour
Hawkes, les probabilités des marques sont calculées après décroissance de
l'excitation au temps candidat, conformément à l'intensité instantanée.
Voir [Ogata, 1981](https://bemlar.ism.ac.jp/zhuang/Refs/Refs/ogata1981ieee.pdf).
Il s'agit de processus démarrant avec un historique vide, observés pendant un
nombre fixé d'événements ; aucune stationnarité n'est revendiquée.

Les deux premières hypothèses nulles comparent des tirages indépendants d'un même
processus marqué. Les alternatives échangent les labels 0 et 1 du deuxième groupe,
sans changer ses temps : elles évaluent la sensibilité aux marques.

Hawkes et NHP sont effectivement entraînés (256 séquences indépendantes, 200
pas Adam, CPU), puis leurs paramètres sont figés. Les deux contrôles internes H0
comparent des simulations indépendantes issues du même modèle figé. Le contrôle
Hawkes supplémentaire compare son simulateur à l'oracle NumPy avec les paramètres
effectivement appris. Pour NHP, le contrôle interne ne démontre pas à lui seul
l'exactitude de l'algorithme de simulation.

Les deux comparaisons des modèles avec un échantillon indépendant du processus
cible sont des diagnostics d'ajustement. Leur taux de rejet ne constitue pas un
taux de faux positifs : un modèle estimé sur un petit jeu de données peut différer
de la véritable loi. Cette distinction est nécessaire pour les tests composites,
voir [Key et al., JMLR 2025](https://jmlr.org/beta/papers/v26/24-0276.html).
Nous n'appliquons pas ici leur théorème ni un bootstrap composite.

Chaque rapport contient toutes les p-values, les intervalles de Wilson à 95 %, les
graines, les empreintes des données et permutations, les configurations et les
empreintes des checkpoints. Les poids appris sont conservés dans les artefacts.
Les jeux d'entraînement et d'évaluation utilisent des flux NumPy distincts.
Un intervalle contenant 5 % indique une compatibilité empirique ; ce n'est pas une
preuve universelle de validité. Les historiques hétérogènes conditionnant chaque
ligne et les neuf autres modèles ne sont pas couverts par cette expérience.

## Contrôle analytique du premier événement NHP

Le NHP part d'un état CT-LSTM nul. Son intensité avant le premier événement
est constante : le premier délai doit suivre une exponentielle de taux égal
à la somme des intensités, et la marque une loi catégorielle proportionnelle
à ces intensités. Ce contrôle ne repose pas sur deux sorties du même simulateur.

```bash
uv run --frozen --no-sync python -m scripts.extended_scientific_validation \
  --first-arrival-only --seed 20261014 --steps 200 \
  --output artifacts/nhp-first-arrival
```

Deux passages locaux Windows CPU ont reproduit exactement les mêmes 4096
tirages après entraînement : moyenne du délai attendue 0.58424, observée
0.57841 ; variance attendue 0.34134, observée 0.34096 ; test KS exponentiel
p=0.722. Probabilité de la marque 0 attendue 0.55807, observée 0.55298,
IC Wilson 95 % [0.53771, 0.56815], test binomial p=0.519.
Ces contrôles sont compatibles avec les lois attendues. Ils ne valident pas
à eux seuls la simulation NHP après un historique non vide.

## Résultats Linux complets du 12 octobre 2026

Code `6388706`, [run 38237687152](https://github.com/NzoCs/Learning-point-processes/actions/runs/38237687152),
Python 3.11.17, Torch 2.8.0+cpu, environnement verrouillé, graine 20261013.
Les [résultats bruts et la provenance](../validation/extended-scientific-2026-10-12.json)
contiennent les 1900 p-values et les empreintes des checkpoints ; l'artefact
GitHub conserve les deux checkpoints. Les neuf expériences ont terminé.

| Contrôle H0 | Rejets / 300 | Taux | IC Wilson 95 % |
|---|---:|---:|---:|
| Poisson, deux marques | 19 | 6.33 % | 4.09–9.68 % |
| Hawkes, deux marques | 10 | 3.33 % | 1.82–6.03 % |
| Hawkes appris, contrôle interne | 16 | 5.33 % | 3.31–8.49 % |
| Hawkes appris, oracle NumPy indépendant | 19 | 6.33 % | 4.09–9.68 % |
| NHP appris, contrôle interne | 17 | 5.67 % | 3.57–8.89 % |

Les cinq intervalles contiennent 5 % : la calibration est empiriquement
compatible avec le niveau nominal pour ces lois, ce checkpoint par modèle
et ce protocole. La répétition de plusieurs entraînements indépendants reste
à étudier pour couvrir la variabilité de l'estimation.

Les alternatives de changement de marques sont détectées dans **32/100**
cas Poisson (IC 23.67–41.66 %) et **28/100** cas Hawkes (20.14–37.49 %).
La sensibilité est mesurée, mais sa puissance est limitée avec huit séquences
de six événements par groupe. Une absence de rejet ne démontre donc pas
l'égalité des lois.

Contre le vrai processus cible sur données indépendantes, les rejets sont
**4/100** pour Hawkes appris (IC 1.57–9.84 %) et **2/100** pour NHP appris
(0.55–7.00 %). Ce sont des diagnostics d'ajustement ; ni des mesures de faux
positifs ni une certification de bonne spécification.

L'apprentissage a modifié les paramètres et diminué la NLL moyenne du même
jeu d'entraînement avec les mêmes tirages d'intégration : Hawkes **1.0273 →
0.8332**, NHP **0.9304 → 0.7706**. Le rayon spectral de la matrice d'excitation
Hawkes apprise vaut 0.3057. La baisse de NLL sur entraînement ne mesure pas
la généralisation. Le pilote local ne contribue pas à ces conclusions.

Le [rapport analytique NHP local](../validation/nhp-first-arrival-2026-10-12.json)
est distinct de l'expérience Linux. Les futures exécutions du workflow incluent
également ce contrôle, et la commande complète calcule une vérification du
premier événement à partir du pool NHP déjà généré.
