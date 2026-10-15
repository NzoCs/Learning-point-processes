# Rejeu Hawkes / IntensityFree — 15 octobre 2026

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

## Plan du rejeu, fixé avant les nouveaux résultats

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
