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

Résultats complets : à compléter après l'exécution Linux indépendante. Le pilote
local vérifie l'exécution et ne sert pas à conclure sur la calibration.
