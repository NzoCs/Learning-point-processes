# Campagne scientifique du protocole courant — 15 octobre 2026

Cette campagne teste le protocole conditionnel de la pipeline après le
raccordement du MMD global. Elle n'utilise ni Ruche ni GPU. Le modèle, ses poids,
ses paramètres de thinning et la borne du sampler restent ceux du parcours
d'intégration archivé ; aucun réglage n'est choisi à partir des résultats.

## Plan fixé avant les résultats

Chaque modèle reprend son checkpoint conditionnel effectivement entraîné,
archivé dans `docs/validation/global-pipeline-20261014`. Les données utilisées
pour cet entraînement proviennent des graines 1701/1702 ; les futurs évalués
utilisent de nouvelles graines, sans réentraînement ni sélection de checkpoint.
Les configurations et statistiques de durée de l'entraînement sont rechargées
par le même DataModule que le runner. La configuration du sampler n'est pas
remplacée par celle d'anciens outils de diagnostic.

Deux scénarios sont déclarés pour chacun des douze modèles :

| Scénario | Historique | Quota futur | Paires | Batches |
|---|---:|---:|---:|---|
| quota3 | 2 événements | 3 événements | 32 | 16 / 11 / 5 |
| quota6 | 2 événements | 6 événements | 128 | 64 / 63 / 1 |

Les historiques varient en temps, densité et marques, avec deux marques possibles.
Les futurs observés passent par `split_history_future`, puis les simulations
sont produites par `model.simulate` depuis le seul historique. Le véritable
`StatisticalTestAccumulator` applique le MMD global de production : kernel de
signature `counting_grid` à huit points, RBF sigma=1, pySigLib, normalisation
commune, 99 swaps par identifiant stable, correction plus-un et égalités
conservatrices. Les blocs de Gram restent bornés à 64.

FullyNN/RMTPP conservent le quota avec arrêt permanent explicite, y compris les
futurs vides. Les autres modèles conservent la troncature symétrique par paire
de la pipeline actuelle. Aucun quota n'est présenté comme une fenêtre temporelle.

- FullyNN et RMTPP : **1 000 essais H0 par scénario** ; **200 essais** pour
  chacune des quatre alternatives.
- Autres modèles : **500 essais H0 par scénario** ; **200 essais** pour
  chacune des deux alternatives communes.
- ANHN, ANHP, NHP, ODETPP, SAHP et THP : **500 essais H0 supplémentaires** par
  scénario sur un contrôle Poisson indépendant à intensité constante.

La graine principale est `2026101523`, distincte des petits essais techniques
de préparation (`2026101507`). Le seuil de rejet est 5 %. Les 36 cellules H0 (24 principales et 12 contrôles
Poisson) partagent un critère d'excès binomial unilatéral `0,01 / 36`.
Les intervalles de Wilson à 95 %, les taux de rejet, le caractère conservateur,
les tirages nuls, les p-values et les erreurs sont conservés. L'absence d'excès
détecté n'est pas une preuve universelle de calibration ; la puissance est
rapportée sans seuil de réussite choisi après observation.

## Références et portée

| Modèles | Référence H0 |
|---|---|
| FullyNN, RMTPP | Décodeur, intégrale, limite à l'infini, RNG et inversion NumPy/SciPy indépendants ; encodeur Torch figé partagé |
| IntensityFree | Décodeur NumPy du mélange log-normal, tirages du mélange et des marques indépendants ; encodeur Torch figé partagé |
| Poisson | Exponentielles et marques NumPy aux taux effectivement figés |
| Hawkes | Oracle NumPy d'Ogata initialisé avec chaque historique et les paramètres effectifs figés |
| SelfCorrecting | Intégrale NumPy et inversion SciPy du risque total, puis tirage indépendant des marques |
| Six autres neuronaux | Deux tirages indépendants du simulateur figé pour contrôler le test statistique ; contrôle séparé contre un Poisson connu |

Le contrôle Poisson utilise une **copie séparée** du modèle neuronal, avec
projection d'intensité nulle et Softplus de pente unité, donnant deux taux
connus `log(2)`. Il ne modifie pas le modèle appris et conserve la configuration
de son sampler. Le test contrôle explicitement la valeur des intensités : une
Softplus apprise ne doit pas être supposée de pente unitaire sans ce réglage
de la fixture séparée.

Le H0 interne des six neuronaux vérifie les permutations, le conditionnement,
l'observation et l'agrégation aux paramètres appris. Il ne démontre pas que
leur simulateur non constant suit exactement leur intensité théorique. Le
contrôle Poisson ne généralise pas cette conclusion aux paramètres arbitraires.
Les références neuronales partageant l'encodeur ne valident pas cet encodeur
contre une réimplémentation indépendante. L'incertitude liée à l'estimation des
poids et CUDA sont exclus du périmètre.

## Alternatives et contrôles

Les alternatives communes transforment la loi du futur observé : tous ses
temps relatifs sont divisés par deux, ou ses marques sont forcées à la dernière
marque de l'historique. La première est une transformation de trajectoire,
pas une réestimation du modèle ni l'inversion de `H(2t)` avec réencodage.
FullyNN/RMTPP ont aussi deux alternatives de terminaison : suppression des
arrêts naturels en conditionnant chaque événement sur son existence, et arrêt
immédiat de tous les futurs. Une mauvaise détection est conservée comme résultat.

Sur le premier essai de chaque cas, toutes les statistiques nulles et observées
du calcul par blocs sont comparées au Gram dense public. L'essai est ensuite
rejoué à l'identique. Le RNG des trajectoires est vérifié avant/après chaque
mise à jour statistique. Les empreintes des poids avant/après doivent coïncider.

Les horloges et gaps restent ceux du protocole float32 de production. La
référence conserve les gaps tirés séparément des temps absolus arrondis. Aucun
délai n'est déplacé pour éviter un arrondi, aucun arrêt n'est déduit d'un silence.
Un essai en erreur garde son identifiant, sa graine et son exception. La campagne
continue pour documenter les autres cas, mais une cellule incomplète est marquée
non évaluable pour la calibration ; le job échoue et ses preuves sont conservées.

## Exécution

Le workflow optionnel `pipeline-calibration.yml` commence par les quatre jobs
FullyNN/RMTPP (deux scénarios chacun). Les vingt jobs des autres modèles suivent,
y compris si un premier résultat signale un problème, sans effacer ce problème.
Le parallélisme de l'extension est limité à dix jobs CPU.

```bash
uv run --frozen --no-sync python -m scripts.pipeline_protocol_calibration \
  --model FullyNN --scenario quota3 --output artifacts/pipeline-calibration-FullyNN-quota3
```

Le même programme accepte tous les modèles et les deux scénarios. Les dossiers
doivent être neufs ; `plan.json` est écrit avant les tirages d'évaluation. Le
marqueur explicite `[pipeline-calibration]` ou un lancement manuel déclenche la
campagne Linux. Les rapports, plans, premiers essais et statistiques nulles
complètes sont déposés en artefacts pendant trente jours.

Statut initial : protocole et contrôles préparés ; les résultats scientifiques
seront rapportés à partir des exécutions, sans les confondre avec les petits
essais techniques de préparation.
