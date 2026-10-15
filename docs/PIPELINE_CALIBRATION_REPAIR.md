# Rejeu Hawkes / IntensityFree — 15 octobre 2026

## Résultat final

Le [rejeu Linux `38759901384`](https://github.com/NzoCs/Learning-point-processes/actions/runs/38759901384)
réussit les trois jobs, au commit `3e52a9e`. Les **2 700 essais** sont complets,
les **50 anciens rejets de validation** sont résolus, sans réentraînement ni
modification des modèles, paramètres, simulateurs ou réglages statistiques.

| Job | Essais complets | Anciens échecs résolus | Rejets H0 / 500 | Accélération / 200 | Marques / 200 |
|---|---:|---:|---:|---:|---:|
| Hawkes quota 6 | 900 | 25 | 29 (5,8 %) | 200 (100 %) | 200 (100 %) |
| IntensityFree quota 3 | 900 | 3 | 20 (4 %) | 67 (33,5 %) | 184 (92 %) |
| IntensityFree quota 6 | 900 | 22 | 26 (5,2 %) | 200 (100 %) | 200 (100 %) |

Aucun excès H0 n’est détecté au critère initial `0.01 / 36`. Ces résultats
concernent les checkpoints, observations et alternatives déclarés ; ils ne
valident pas tous les paramètres, GPU ou modèles appris possibles.

Les **2 650 p-values historiques** et graines nulles restent strictement
identiques. Les 1 775 anciens essais IntensityFree gardent toutes leurs
statistiques bit à bit ; les 875 anciens essais Hawkes restent équivalents
numériquement avec un écart maximal de `4,996003610813204e-16`, selon le contrat
public de précision décrit plus bas. Ce n’est pas une modification des valeurs
enregistrées pour forcer une égalité.

Sur le même runner Linux, le validateur original et le nouveau retrouvent
exactement les mêmes simulations, statistiques observées et 99 statistiques
nulles pour les premiers essais H0 et des deux alternatives Hawkes. Les AST
des autres méthodes statistiques et les sources des modèles, simulateurs,
kernel et MMD global sont vérifiés inchangés.

La [CI](https://github.com/NzoCs/Learning-point-processes/actions/runs/38759901384)
passe **391 tests**, 7 ignorés, 2 désélectionnés, couverture **85,16 %** ; le
[lint](https://github.com/NzoCs/Learning-point-processes/actions/runs/38759901387)
réussit. Les rapports avant/après, 2 700 tirages avec leurs 99 statistiques
nulles, références historiques, audits et preuve du rejeu sur un même runner
sont conservés dans `docs/validation/pipeline-repair-20261015`, avec manifeste
SHA-256. Les anciens échecs et le premier contrôle historique trop strict
restent visibles dans les preuves et dans l’explication ci-dessous.

## Problème confirmé

La campagne Linux `38714040860` a terminé avec trois jobs en échec :

| Job | Essais prévus | Échecs de validation | Essais déjà réussis |
|---|---:|---:|---:|
| Hawkes quota 6 | 900 | 25 | 875 |
| IntensityFree quota 3 | 900 | 3 | 897 |
| IntensityFree quota 6 | 900 | 22 | 878 |

Les 50 erreurs ont toutes le même message :
`Continuation gaps must use the shared history endpoint as origin`.
Les résultats calculés sur les sous-ensembles restants ne constituent pas une
campagne complète. Les rapports originaux sont conservés, sans remplacer les
échecs par des graines plus favorables.

Le rejeu local des trois premières graines en échec reproduit le problème sur
les simulations natives. Exemple Hawkes : un délai conservé de
`0.0200958251953125`, à un temps relatif d’environ 176, donne
`0.020111083984375` lorsqu’on soustrait les deux temps relatifs float32.
La différence `0.0000152587890625` est un pas de précision de l’horloge à cet
instant. IntensityFree produit aussi légitimement de longs délais suivis de
délais courts : vers 1006, une différence de `0.000029221177101135254`
dépassait la tolérance fixe du contrôle. Il ne s’agit pas d’une modification
de l’origine ni d’une erreur confirmée de la loi des simulateurs.

## Correction limitée au contrôle numérique

`validate_pairs` compare désormais les écarts en float64 et conserve la
tolérance initiale `1e-5 + 1e-5 * abs(délai attendu)`. Elle l’étend uniquement
si le budget d’arrondi de l’horloge source est plus grand : addition en temps
absolu, soustractions de l’origine, stockage du délai. Pour une précision de
machine eps, le budget utilise `eps / (2 - eps) * abs(valeur arrondie)`, borne
standard d’une opération arrondie au plus proche pour les nombres normaux.
La tolérance initiale couvre les nombres proches de zéro et sous-normaux.

La borne dépend du dtype, des temps et du dernier événement valide de
l’historique, jamais des rejets H0 ou de la puissance. Elle n’est pas un
nouveau seuil absolu fixé pour faire passer la campagne. Les délais négatifs
ou non finis restent explicitement rejetés ; les erreurs d’origine dépassant
la précision de représentation restent rejetées. La précision du format reste
une limite : le contrôle ne peut certifier des écarts sous sa résolution.

Aucune modification des paramètres, architectures, simulateurs, tirages,
horloges retournées, observations, kernel, formule MMD, permutations ou
p-values. Aucun clipping de délai, filtrage d’essai ou déplacement d’événement.
Les petits écarts d’arrondi ne sont pas corrigés dans les données : seul le
contrôle de leur cohérence est adapté à leur précision réelle.

## Plan initial du rejeu, fixé avant les nouveaux résultats

Le workflow optionnel `pipeline-calibration-repair.yml` relance les trois jobs
sur Linux CPU, avec les dépendances verrouillées : 500 H0 et 200 essais de
chacune des deux alternatives, mêmes checkpoints, mêmes quotas, mêmes batches,
même graine `2026101523`, mêmes 99 permutations. Le critère familial H0 de la
campagne originale reste `0.01 / 36`, sans le redéfinir sur ces seuls jobs.

Les baselines de `docs/validation/pipeline-repair-20261015/*_before.json`
conservent les rapports originaux et, pour chaque essai réussi, une empreinte
SHA-256 canonique du tuple comprenant la graine, la statistique observée,
les 99 statistiques nulles et la graine du générateur nul.
`scripts/pipeline_calibration_regression.py` exige :

- les 2 700 essais complets, sans aucun échec ;
- les mêmes paramètres scientifiques, empreintes de checkpoints et poids ;
- les 2 650 anciens résultats numériques strictement identiques, sans tolérance ;
- les 50 graines auparavant en échec présentes et évaluées ;
- aucun excès H0 selon le critère initial.

Les tests couvrent les trois lignes réellement reproduites, le dtype float64,
un grand point d’origine avec padding, les mauvais décalages d’origine, les
délais négatifs/non finis, l’absence de mutation et la conservation du RNG.
Le contrôle historique est lui-même testé contre une permutation numérique
modifiée, un RNG modifié, un checkpoint modifié et des essais manquants.

Les autres campagnes et les réglages de production du kernel restent
inchangés. Ce rejeu ne constitue pas une validation CUDA/Ruche/DDP.

## Complément après le premier rejeu : précision inter-exécutions

Le premier rejeu `f8d16e1`, workflow `38757445347`, termine les 2 700 essais
scientifiques sans erreur : les 50 rejets de validation disparaissent.
Les deux jobs IntensityFree passent aussi leur contrôle bit à bit, soit
897 et 878 anciens essais strictement identiques. Le job Hawkes échoue
uniquement sur ce nouveau contrôle historique strict : les 875 anciens essais
diffèrent en derniers bits, au maximum `4,996003610813204e-16` sur les
statistiques observées et nulles, avec toutes les p-values et graines identiques.
Les versions déclarées, dépendances verrouillées et checkpoints coïncident.
La cause matérielle précise de cette variation n’est pas établie.

Avant de modifier l’audit, un rejeu local dans le même processus compare le
validateur original du commit `3cb3e9` au nouveau sur le premier essai de H0 et
des deux alternatives : statistiques et simulations sont strictement identiques.
Le calcul statistique, le kernel et les simulateurs n’ont pas changé.

Le rejeu suivant conserve donc deux niveaux de contrôle explicites :

- SHA-256 exact lorsqu’il coïncide ; sinon comparaison de toutes les valeurs
  avec `rtol=1e-10, atol=1e-12`, **limites déjà présentes dans le contrôle public
  Gram dense / calcul par blocs**, sans seuil ajusté à l’écart observé ;
- p-values, graines, RNG nul, cardinalités, protocole et checkpoint toujours
  strictement identiques. Les références numériques sont vérifiées contre
  les SHA-256 originaux avant toute comparaison.

Le mode strict sans référence numérique reste disponible pour l’audit ; le
workflow active explicitement la comparaison inter-exécutions. Une p-value
modifiée, une référence altérée ou une différence hors du contrat numérique
fait échouer le job. Les résultats enregistrés ne sont ni arrondis ni remplacés.

En complément, `continuation_precision_parent_check.py` vérifie sur le même
runner Linux que les méthodes statistiques hors validation sont identiques au
parent, que modèles/kernel/MMD global n’ont pas changé, puis compare les deux
validateurs sur les mêmes futurs et les mêmes trois premiers essais. Ce contrôle
exige encore l’égalité bit à bit des statistiques et trajectoires. Il utilise
le validateur original lu dans Git, sans modifier le modèle.

Les rapports initiaux et le premier rejeu Hawkes restent conservés. Le résultat
du contrôle historique est séparé de celui des 900 essais scientifiques ; le
premier échec d’audit n’est pas présenté comme une campagne GitHub réussie.
