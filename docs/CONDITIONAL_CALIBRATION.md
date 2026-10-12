# Historiques hétérogènes et agrégation des batches

Deux continuations peuvent être échangées sous H0 si elles proviennent du même
historique et sont indépendantes conditionnellement à cet historique. Lorsque
les historiques diffèrent entre lignes, une permutation globale ne bénéficie
pas de cette invariance. Le groupe valide échange chaque paire indépendamment,
sans déplacer les lignes entre historiques. La troncature à la durée commune
commute avec ces échanges, puisque min(T_xi,T_yi) ne dépend pas du sens de la paire.
Voir [Hemerik et Goeman, 2018](https://link.springer.com/article/10.1007/s11749-017-0571-1).

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

Le workflow optionnel `Conditional and batch calibration` exécute 500 répétitions
H0 et 200 H1, avec 99 permutations : lois Poisson dépendant du contexte,
Hawkes appris contre un oracle NumPy conditionnel, NHP appris contre ses propres
tirages indépendants, et batches Poisson indépendants de tailles 8/5/3 et de
taux différents. Une permutation globale des contextes et des batches réutilisés
servent de contrôles volontairement hors hypothèses. Le contrôle interne NHP
ne certifie pas l'exactitude de son simulateur pour chaque historique.

```bash
uv run --frozen --no-sync python -m scripts.conditional_calibration
```

Résultats Linux : en attente de l'expérience complète.

La pipeline historique `predict` continue de simuler après le batch entier et
de produire ses comparaisons descriptives avec ce batch. Elle n'utilise pas le
holdout ci-dessus et ses p-values ne sont **pas certifiées comme test conditionnel
de bonne spécification**. Ce lot fournit l'API correcte et sa validation sans
modifier silencieusement les trajectoires ou les références historiques.
