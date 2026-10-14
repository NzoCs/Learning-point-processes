# Historiques hétérogènes et agrégation des batches

## Répétabilité, reprise et modèles appris — 14 octobre 2026

Code `de93f98`, [validation Linux des onze modèles](https://github.com/NzoCs/Learning-point-processes/actions/runs/38538882377).
Les [preuves numériques archivées](validation/conditional-repeat-calibration-2026-10-14.json)
contiennent les rapports de répétabilité, les paramètres ajustés, graines,
hashes de checkpoints, environnements, toutes les p-values et les empreintes
des données et randomisations. Les onze jobs ont terminé avec succès.

### Répétabilité et reprise

Pour chacun des onze modèles, la suite exécute deux pipelines train/test/predict
dans des dossiers neufs avec les mêmes données et graines. Les poids entraînés,
24 événements exportés, statistiques observées, 99 tirages nuls par batch et
p-values individuelles et agrégées concordent **exactement**, sans tolérance.
Une exécution séparée `predict --checkpoint` retrouve les mêmes événements et
statistiques, sans réentraîner le modèle.

La suite compare ensuite un entraînement continu de deux epochs avec une epoch
suivie d'une reprise depuis `last.ckpt`. Les poids finaux, états Adam, étapes,
poids du checkpoint sélectionné et résultats conditionnels concordent exactement
sur les onze modèles. Ces petites configurations utilisent un taux d'apprentissage
constant ; les états de scheduler y sont vides, explicitement indiqués dans
l'archive. La garantie porte sur CPU, un processus, zéro worker et une reprise
à la frontière d'epoch. Elle ne porte pas sur une reprise au milieu des batches
de prédiction ni sur CUDA/DDP.

Un contrôle supplémentaire NHP active le scheduler cosinus avec un budget total
de deux epochs identique pour les parcours continu et repris. Les poids, états
Adam/scheduler, événements et statistiques conditionnelles concordent exactement.
Le taux final atteint le plancher existant de 1 % du taux initial ; l'assertion
initiale qui attendait zéro a été corrigée, sans modification du scheduler.
Ce test passe dans la [CI du code `94230c6`](https://github.com/NzoCs/Learning-point-processes/actions/runs/38541503999),
avec le lint et le packaging. Il vérifie un scheduler actif sur NHP, sans étendre
cette garantie aux onze modèles ou aux changements de budget total en cours de reprise.

### Calibration et puissance

Chaque modèle est ajusté par 200 mises à jour sur 256 séquences Hawkes à deux
marques, puis figé. Les futurs évalués ne servent pas à cet ajustement.
Le protocole fixe trois événements historiques hétérogènes, six événements par
continuation et trois batches indépendants de tailles 8/5/3. Il utilise 99
randomisations par batch, 500 essais H0 et 200 essais pour chacune des deux
alternatives H1 définies avant les mesures : temps multipliés par quatre, ou
marques futures imposées par le contexte historique. Les poids sont contrôlés
inchangés après tous les essais. Le cache numérique est comparé à l'API publique,
sur tous les tirages du premier essai de chaque cas et taille de batch.

Pour **Hawkes**, les observations proviennent d'un oracle NumPy indépendant,
initialisé avec les paramètres ajustés et l'historique de chaque ligne.
Pour **les dix autres modèles**, H0 compare deux tirages indépendants du même
simulateur figé : cela vérifie la calibration du test sous sa loi interne,
sans valider le simulateur contre une loi externe ni l'incertitude d'un modèle
réestimé à chaque essai.

| Modèle | Rejets H0 agrégés | IC 95 % ponctuel | Puissance H1 temps | Puissance H1 marques |
|---|---:|---:|---:|---:|
| ANHN | 4.8 % | 3.25–7.04 % | 97.0 % | 91.0 % |
| ANHP | 5.6 % | 3.90–7.97 % | 89.5 % | 64.0 % |
| FullyNN | 6.4 % | 4.57–8.90 % | 100.0 % | 100.0 % |
| Hawkes, oracle indépendant | 5.2 % | 3.57–7.51 % | 89.5 % | 70.0 % |
| IntensityFree | 5.8 % | 4.07–8.21 % | 82.5 % | 72.0 % |
| NHP | 5.4 % | 3.74–7.74 % | 85.5 % | 57.0 % |
| ODETPP | 5.6 % | 3.90–7.97 % | 89.0 % | 50.0 % |
| RMTPP | 6.8 % | 4.91–9.35 % | 87.5 % | 69.0 % |
| SAHP | 5.0 % | 3.41–7.28 % | 92.0 % | 75.0 % |
| SelfCorrecting | 3.2 % | 1.98–5.13 % | 96.5 % | 35.5 % |
| THP | 6.0 % | 4.23–8.44 % | 86.0 % | 75.0 % |

Tous les intervalles H0, sur le premier batch comme après agrégation, contiennent
5 %. Ces mesures sont compatibles avec le niveau annoncé dans ce protocole ;
elles ne prouvent pas une validité universelle ou l'uniformité exacte des p-values.
Les intervalles sont ponctuels et ne constituent pas une certification simultanée
avec correction pour onze modèles. La puissance sur les marques est limitée pour
SelfCorrecting et ODETPP dans cette alternative : aucun paramètre du noyau,
seuil de rejet ou calcul n'a été retouché après observation pour l'augmenter.

Les calculs historiques et les modèles sont inchangés. Les modifications de ce
lot ajoutent les suites optionnelles, leur CI et les tests de leurs contrôles.

Rejouer un modèle dans un dossier neuf :

```bash
uv run --frozen --no-sync python -m scripts.conditional_reproducibility \
  --model NHP --output artifacts/conditional-repeat-new
uv run --frozen --no-sync python -m scripts.learned_conditional_calibration \
  --model NHP --output artifacts/conditional-science-new \
  --trials 500 --power-trials 200 --permutations 99 --steps 200
```

Le workflow manuel **Conditional repeat resume and learned calibration** exécute
ces deux suites sur les onze modèles. Il reste optionnel et peut être déclenché
sur la branche de développement par `[conditional-validation]`. Les logs,
checkpoints et rapports complets sont conservés dans ses artefacts pendant 30 jours.

## Protocole conditionnel

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

### Pipeline conditionnelle sur les onze modèles, avec entraînement indépendant

Code `1fa68c0`, [run Linux 38251661306](https://github.com/NzoCs/Learning-point-processes/actions/runs/38251661306) :
**les onze modèles passent** train/test/predict conditionnels. La CI ordinaire
passe également : 237 tests, 3 ignorés, 2 désélectionnés, couverture 82.76 %.
Le lint et le packaging passent. Aucune correction des modèles ni modification
des références historiques n'a été nécessaire.

Trois jeux Poisson marqués sont générés par des flux aléatoires distincts
(graines 1701/1702/1703) pour train/dev/test : chacun contient huit séquences,
cinq événements et deux marques. Les identifiants et les trajectoires sont
disjoints ; les trois hashes de fichiers diffèrent. La validation utilisée pour
choisir le checkpoint est également séparée du jeu évalué. Chaque modèle est
entraîné une epoch, puis évalué avec deux événements historiques et trois futurs,
sur deux batches de quatre séquences, avec 99 randomisations par batch.

La suite vérifie le checkpoint utilisé dans les trois phases, le mode eval et
l'absence de modification de ses poids pendant predict, le préfixe effectivement
fourni à `simulate`, le futur réservé reçu par la MMD, huit identifiants uniques,
24 événements simulés valides, leurs marques, temps et ordre, la correspondance
des exports aux simulations et le recalcul de la p-value agrégée.
Les [mesures brutes archivées](validation/conditional-eleven-2026-10-12.json)
conservent les données, graines, environnement, hashes des checkpoints et fichiers,
ainsi que les statistiques et vecteurs de randomisation de chaque modèle.

| Modèle | Pipeline | P-value agrégée de ce petit essai |
|---|---|---:|
| ANHN | passe | 0.70 |
| ANHP | passe | 0.63 |
| FullyNN | passe | 0.05 |
| Hawkes | passe | 0.05 |
| IntensityFree | passe | 0.45 |
| NHP | passe | 0.63 |
| ODETPP | passe | 0.10 |
| RMTPP | passe | 0.05 |
| SAHP | passe | 0.40 |
| SelfCorrecting | passe | 0.66 |
| THP | passe | 0.86 |

Ces p-values ne sont pas un critère de réussite de la suite : après une seule
epoch, le modèle appris peut différer de la loi des données. Ce contrôle confirme
le fonctionnement logiciel et la séparation des données ; il ne mesure pas la
calibration H0 ou la convergence scientifique de chacun des onze modèles.

Rejouer localement depuis le dépôt, dans un dossier de sortie neuf :

```bash
uv run --frozen --no-sync python -m scripts.conditional_integration_suite \
  --model all --output artifacts/conditional-integration-new
```

Dans le workflow manuel **Complete integration regression**, sélectionner
`conditional: true` et `model: all`. La suite complète reste optionnelle ; elle
peut également être déclenchée sur la branche de développement par le marqueur
de commit `[conditional-integration]`. Les logs, checkpoints, exports, figures
et contrôles sont conservés dans l'artefact du run GitHub pendant 14 jours.
