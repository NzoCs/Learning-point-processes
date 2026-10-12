# Historiques hétérogènes et agrégation des batches

Deux continuations peuvent être échangées sous H0 si elles proviennent du même
historique et sont indépendantes conditionnellement à cet historique. Lorsque
les historiques diffèrent entre lignes, une permutation globale ne bénéficie
pas de cette invariance. Le groupe valide échange chaque paire indépendamment,
sans déplacer les lignes entre historiques. La troncature à la durée commune
commute avec ces échanges, puisque min(T_xi,T_yi) ne dépend pas du sens de la paire.
Voir [Hemerik et Goeman, 2018](https://link.springer.com/article/10.1007/s11749-017-0571-1).

La statistique doit également inclure le contexte. Une MMD des seuls futurs
peut manquer des erreurs conditionnelles qui se compensent entre historiques.
L'API utilise donc le noyau produit histoire/futur et la statistique appariée
hors diagonale :

\[
T=\frac{1}{n(n-1)}\sum_{i\ne j} w_H(h_i,h_j)
[k_F(x_i,x_j)+k_F(y_i,y_j)-k_F(x_i,y_j)-k_F(y_i,x_j)].
\]

Les quatre termes excluent les contributions de la même paire. Le poids
d'historique est une gaussienne de bande unité de la distance entre les
représentations RKHS fournies par le noyau d'historiques. Aucun seuil ni bande
n'est ajusté aux taux de rejet. Historiques, troncature par paire, normalisation
du pool et matrices de noyau sont fixes sur l'orbite des échanges autorisés.
Le protocole est nommé `within_history_pair_swaps_joint_kernel_v1`.

```python
from new_ltpp.evaluation.statistical_testing.statistical_tests import (
    ConditionalMMDTwoSampleTest, split_history_future,
)

test = ConditionalMMDTwoSampleTest(kernel, n_samples=99)
history, heldout = split_history_future(batch, history_events=3, future_events=6)
simulated = model.simulate(history, num_events_to_simulate=6)
stats = test.compute_statistics(
    heldout, simulated, histories=history, pair_ids=independent_sequence_ids,
)
# Répéter sur des batches disjoints, puis :
p_value = test.get_final_p_value()
```

Les nombres d'événements sont fixés avant d'observer le futur. Les temps des
deux continuations sont relatifs au dernier événement du préfixe. Le découpage
compacte les événements valides, préserve leurs marques et rejette les séquences
trop courtes ou non chronologiques ; il ne modifie pas le batch original.
Le modèle doit être figé et ajusté sur des données indépendantes de l'évaluation.
Un modèle estimé n'est pas nécessairement égal à la véritable loi : un rejet
contre la cible reste un diagnostic d'ajustement, même avec un test valide.

`test_model(..., history_events=3, future_events=6)` utilise ce découpage et le
simulateur natif du modèle. L'ancienne comparaison de l'historique avec son
propre futur n'est plus acceptée par cette API : il ne s'agit pas de deux
échantillons conditionnellement échangeables. Pour des simulations préexistantes,
il faut fournir explicitement leurs historiques communs à `compute_statistics`.

L'accumulateur prend également `conditional=True`, puis
`update(heldout, simulated, histories=history, pair_ids=ids)`. Son flux persistant
de rééchantillonnage ne modifie pas le flux des trajectoires.

## Agrégation

La statistique globale reste la **somme des statistiques par batch**. Chaque
colonne j du nul est la somme des statistiques obtenues par des randomisations
indépendantes dans chaque batch. Il s'agit du groupe produit des groupes de
chaque batch ; ce n'est ni une moyenne des p-values ni la MMD recalculée sur
toutes les données concaténées. Des tailles et lois différentes entre batches
sont autorisées, sous leurs hypothèses d'échangeabilité respectives.

Les paires doivent être disjointes entre batches. Des trajectoires réutilisées,
fenêtres glissantes partageant des événements et individus communs ne sont pas
couverts. Les identifiants `pair_ids` permettent de rejeter explicitement les
recouvrements, les doublons et un mélange de batches identifiés et non identifiés.
Sans identifiants, l'indépendance reste une condition à assurer par l'appelant ;
on ne peut pas l'inférer des temps ou des historiques identiques.

Les appels sur des loaders redémarrent leur agrégation. Des loaders de longueurs
différentes sont rejetés, ainsi que des vecteurs nuls de tailles différentes et
des statistiques non finies. Aucun seuil de tolérance historique n'est élargi.

## Validation

Les tests rapides comparent les huit échanges possibles de trois paires à l'API
réelle, puis énumèrent les 64 orientations du produit de deux batches de trois
paires. Les p-values exactes contrôlent le niveau pour toutes les orientations.
Ils vérifient aussi l'origine des temps, l'absence de fuite du futur, la remise
à zéro, le flux aléatoire, les données manquantes et les recouvrements explicites.

Le script de validation couvre 500 répétitions
H0 et 200 H1, avec 99 permutations : lois Poisson dépendant du contexte,
Hawkes appris contre un oracle NumPy conditionnel, NHP appris contre ses propres
tirages indépendants, et batches Poisson indépendants de tailles 8/5/3 et de
taux différents. Une permutation globale des contextes et des batches réutilisés
servent de contrôles volontairement hors hypothèses. Le contrôle interne NHP
ne certifie pas l'exactitude de son simulateur pour chaque historique.

Le workflow optionnel `Conditional and batch calibration` rejoue les contrôles
conditionnels et le contrôle de padding. Le script sans `--conditional-only`
exécute aussi les contrôles indépendants et de réutilisation de trajectoires.

```bash
uv run --frozen --no-sync python -m scripts.conditional_calibration
```

## Résultats CPU Linux du 12 octobre 2026

Code `83e2f76`, [run 38246520797](https://github.com/NzoCs/Learning-point-processes/actions/runs/38246520797),
500 essais H0, 200 H1, 99 randomisations, deux marques, modèles entraînés puis
figés. La [mesure brute](validation/conditional-joint-2026-10-12.json) conserve
toutes les p-values, les checkpoints identifiés par hash et les graines.

| H0 | Premier batch | Somme des batches 8/5/3 | IC 95 % de la somme |
|---|---:|---:|---:|
| Poisson dépendant de l'historique | 19/500 (3.8 %) | 14/500 (2.8 %) | 1.68–4.64 % |
| Hawkes appris, oracle NumPy conditionnel | 23/500 (4.6 %) | 26/500 (5.2 %) | 3.57–7.51 % |
| NHP appris, tirages conditionnels indépendants | 22/500 (4.4 %) | 30/500 (6.0 %) | 4.23–8.44 % |

Hawkes et NHP sont empiriquement compatibles avec 5 %. Le Poisson est
conservateur : son intervalle de la somme n'inclut pas 5 %, mais il n'y a pas
d'inflation observée. Les tests discrets avec égalités ne sont pas nécessairement
uniformes sous H0 ; le critère théorique est le contrôle du risque de rejet.
Ces mesures ne démontrent pas une validité universelle pour tous les modèles,
les ajustements et les types de dépendance.

Pour le changement de marques conditionnelles, la puissance passe de **6/200
(3 %) à 188/200 (94 %)** sur le premier batch, et de **1/200 (0.5 %) à 199/200
(99.5 %)** sur la somme (IC nouveau 97.22–99.91 %). Les empreintes de toutes les
données brutes et de tous les échanges sont identiques entre les deux expériences
comparées, y compris pour les modèles appris : seule la statistique change.
La bande d'historique est restée fixée à un ; aucun ajustement aux résultats
n'a été effectué après cette mesure.

Le [contrôle supplémentaire Hawkes](validation/conditional-ragged-hawkes-2026-10-12.json)
avec historiques de 2 ou 6 événements et padding mixte gauche/droite donne
**29/500, soit 5.8 %**, IC 4.07–8.21 %. Il a également tourné sur le runner Linux
du même run. Les historiques NHP à longueurs variables ne sont pas certifiés
par cette expérience.

La CI du même code passe avec **225 tests**, **82.40 %** de couverture et les
**onze modèles d'intégration**. Aucune référence historique, aucun seuil
numérique de régression et aucun calcul de modèle n'a été modifié.

La première expérience, avec une MMD portant uniquement sur les futurs, est
[archivée avec ses p-values](validation/conditional-future-only-2026-10-12.json).
Elle donne seulement **6/200** détections du changement conditionnel de marques
sur un batch et **1/200** après agrégation. Les échanges sont valides sous H0,
mais la statistique ne détecte pas correctement cette alternative. La correction
ci-dessus change explicitement la statistique conditionnelle ; la MMD
inconditionnelle historique demeure inchangée.

Cette même expérience valide l'agrégation de batches Poisson indépendants de
tailles 8/5/3 et de taux différents : **18/500**, soit 3.6 % (IC 2.29–5.62 %).
En revanche, la réutilisation des mêmes trajectoires dans trois batches donne
**72/500**, soit 14.4 % (IC 11.59–17.75 %), contre 4.2 % sur le premier batch
seul. Ce contrôle confirme le défaut d'une agrégation indépendante de données
réutilisées et justifie le rejet des identifiants en recouvrement.

Les taux à 2.8 % des premières agrégations conditionnelles sont conservateurs,
avec des intervalles n'incluant pas 5 %. Ils ne démontrent pas une inflation du
risque sous H0. Il ne faut pas exiger l'uniformité exacte des p-values lorsqu'il
y a des égalités et des randomisations discrètes.

Le nouveau noyau est évalué avec les mêmes données, graines et échanges : aucune
référence numérique n'est remplacée pour faire passer une régression.
Le workflow rejoue les seuls contrôles conditionnels modifiés et conserve les
mesures indépendantes/recouvrement ci-dessus. Une seconde étape vérifie Hawkes
contre l'oracle avec 2 ou 6 événements historiques et du padding mixte gauche/droite.

La pipeline historique `predict` continue de simuler après le batch entier et
de produire ses comparaisons descriptives avec ce batch. Elle n'utilise pas le
holdout ci-dessus et ses p-values ne sont **pas certifiées comme test conditionnel
de bonne spécification**. L'évaluation conditionnelle est désormais accessible
dans cette même pipeline par activation explicite. Les références historiques
continuent de tester le mode historique.

## Évaluation conditionnelle dans la CLI

Dans une configuration effective utilisée par `ltpp run --phase all` :

```yaml
simulation_config:
  seed: 321
  conditional_evaluation:
    history_events: 2
    future_events: 3
```

Pour chaque séquence, seuls les deux premiers événements valides sont transmis
au modèle figé. Les trois événements suivants constituent le futur réservé.
Le modèle simule trois événements depuis le même historique. Les deux futurs
ont pour origine le dernier événement historique. La MMD utilise le noyau
joint historique/futur et les échanges au sein des paires ; les statistiques
et vecteurs de randomisation sont ensuite sommés entre batches indépendants.
Les autres métriques portent également sur les futurs, dans ce mode.

Le loader transporte les identifiants jusqu'au test et aux exports Parquet.
Le champ source `sequence_id` est considéré comme un identifiant global : le
même identifiant dans plusieurs batches est rejeté avant simulation. À défaut,
`seq_idx` ou l'indice original de ligne est associé au SHA256 du contenu du
snapshot, indépendamment du chemin local et de l'ordre du DataLoader.
Les identifiants ne changent ni les tenseurs ni les générateurs aléatoires.
Le Parquet conditionnel conserve le premier délai après l'historique, au lieu
de ramener le premier événement simulé à zéro comme dans le mode historique.

Les contrôles exigent au moins deux séquences par batch, chacune avec au moins
`history_events + future_events` événements valides. Ils rejettent les identifiants
manquants, répétés dans un batch ou réutilisés après un batch précédent.
Choisir une taille de batch qui ne laisse pas une seule séquence dans le dernier
batch. L'exécution conditionnelle est actuellement limitée à un seul processus.

Un identifiant de ligne ne prouve pas l'indépendance des observations : pour
des fenêtres qui se recouvrent ou plusieurs lignes issues du même individu,
fournir un identifiant global commun pour détecter leur réutilisation.
Le checkpoint doit être ajusté indépendamment des futurs évalués ; cette
condition est explicitement enregistrée comme non vérifiée par la pipeline.
Le test logiciel de bout en bout peut réutiliser une petite fixture pour toutes
les phases : il vérifie le câblage, pas la calibration scientifique.

Sans `conditional_evaluation`, la configuration sérialisée, le mode de simulation
et les calculs historiques restent identiques. Les références numériques des
onze modèles ne sont pas réenregistrées pour ce changement.

### Vérification du branchement

Code `a5cab14`, [CI Linux 38250465654](https://github.com/NzoCs/Learning-point-processes/actions/runs/38250465654) :
236 tests réussis, 3 ignorés, 2 désélectionnés, couverture 82.76 %.
Les onze modèles historiques passent leur pipeline et les références inchangées.
Le nouveau parcours CLI complet est testé sur NHP et Hawkes : préfixe seul fourni
à la simulation, futur réservé transmis au test, identifiants contrôlés avant
génération puis exportés, conservation du premier délai et vérification de la
p-value agrégée sur deux batches. Les tests refusent également les entrées trop
courtes, les batches d'une seule séquence et l'exécution conditionnelle distribuée.
Ce contrôle du branchement complète les expériences scientifiques ci-dessus ;
il ne constitue pas une nouvelle mesure de calibration.
