# Taille des pools : MMD figé contre RMSE à référence apprise

Cette campagne volontaire compare 128, 512 et 1 024 trajectoires par groupe.
Elle isole l'effet de taille sur la calibration et la puissance, sans modifier
les modèles, les simulateurs, le kernel ou la campagne grille/kernel en cours.

Le MMD utilise le RBF statique sigma=0.3 choisi auparavant sur des données
séparées, sur la grille uniforme historique de 100 points, ordre dyadique zéro.
Le temps est t/T et le comptage N(t)/écart-type du pilote historique indépendant.
Ces échelles sont communes aux groupes et à toutes les tailles, sans recalibrage.
Le kernel reste une approximation de signature par différences finies ; cette
campagne ne modifie pas son intégration numérique ni sa représentation.

La référence RMSE est l'une des six références paramétriques figées du commit
6ea57564a6a13efceaef02499febef7360accb70 : famille Poisson/Hawkes connue, paramètres
appris sur 1 024 trajectoires H0 indépendantes par famille/cohorte. Aucun nouvel
apprentissage. Le même prédicteur sert aux deux groupes et aux trois tailles.
Le score inclut l'origine zéro, chaque événement, la censure terminale et les
trajectoires vides ; sa cible est l'attente limitée au temps restant avant T=1.
Normalisation par la moyenne restreinte prédite et poids égaux entre trajectoires.
La RMSE prédictive reste distincte du test bilatéral par permutation des pertes.

## Données et comparaison appariée

Graine d'évaluation nouvelle : 2026102051. Entropie
`(graine, étape, scénario, cohorte, essai, flux)` ; étape 0 pour le pilote technique,
1 pour l'évaluation. Les flux 1/2 génèrent X/Y et le flux 3 les permutations.
Cette graine est distincte des banques d'entraînement, de la sélection du kernel
et des évaluations historiques. Aucun résultat de l'autre campagne ne choisit
un paramètre de celle-ci : seul son choix indépendant déjà figé est consommé.

Chaque essai génère deux pools de 1 024 trajectoires. Leurs préfixes donnent
les pools 128/512 : les méthodes utilisent exactement les mêmes événements,
identifiants et échanges. Une matrice de 499 échanges Rademacher sur 1 024 paires
est générée et ses colonnes préfixes sont partagées par les deux tests et les
trois tailles. L'identité observée et la correction plus-un sont conservées.
La statistique RMSE est la différence moyenne des pertes au carré ; le MMD est
le même contraste apparié hors diagonale que dans les campagnes précédentes.
Les intervalles d'une même trajectoire ne deviennent jamais des observations
indépendantes. Les trois tailles sont corrélées par construction, ce qui permet
de comparer leurs décisions par essai apparié.

On conserve les dix scénarios, deux cohortes, 1 000 essais H0 par scénario/cohorte
et 250 H1. Au total : 8 000 essais H0, 3 000 H1, chacun aux trois tailles, donc
33 000 comparaisons de pools et 66 000 p-values. Les mêmes paramètres et burn-in
des générateurs indépendants précédents sont conservés. Les contrôles temporels
gardent exactement les comptes par paire. Le calcul du Gram complet est fait
une seule fois par essai, avec les blocs pySigLib max_batch=16 ; ses sous-matrices
incluent les comparaisons entre tous les blocs et ne sont pas des MMD moyennés.
Les pertes par trajectoire sont calculées une fois, puis sous-échantillonnées.

## Contrôles et interprétation

Avant calcul : source propre et commit gelé, banques NPZ/JSON et fichier de
sélection hachés. Un pilote vérifie les six références sur CPU/CUDA, les deux
statistiques et les mêmes p-values sur de petits pools imbriqués. Il mesure
ensuite le coût réel des trois tailles sur chacune des trois familles à 1 024.
Les petits pools du pilote ne mesurent aucune puissance. Aucun fallback CPU
silencieux, aucune exclusion, aucune adaptation de paramètre après les résultats.

Cent fragments indépendants couvrent les vingt cellules : cinq fragments par
scénario/cohorte, 200 essais H0 ou 50 H1 par fragment. Cela limite la durée d'un
job sans changer les effectifs. L'agrégation exige les cent fragments, les mêmes
banques et choix figés, tous les vecteurs de pertes, graines, origines et p-values
cohérents. Elle recalcule les pertes/permutations depuis les scores archivés et
rejoue les hashes des pools aux deux extrémités de chaque fragment. Une panne
bloque la chaîne : ne pas remplacer un fragment par des essais différents.

Calibration : 72 contrôles d'excès (4 H0 × 3 tailles × 2 méthodes × cohortes 0/1
et réunion), seuil 0.01/72 ; bornes supérieures simultanées 99 % sur 24 résultats
réunis. Puissance : famille unique de 54 McNemar exacts bilatéraux, Holm54 :
18 comparaisons MMD/RMSE et 36 comparaisons de tailles (3 paires de tailles,
2 méthodes, 6 alternatives). Intervalles binomiaux exacts 95 % par cohorte ;
présenter aussi les taux réunis et des intervalles stratifiés si nécessaire.
L'absence d'excès significatif n'est pas une preuve d'égalité exacte à 5 %.
Une sortie d'agrégation 2 indique un excès scientifique avec données complètes,
pas une autorisation de relance ou de changement des réglages.

Portée : données synthétiques Poisson/Hawkes non marquées ; références apprises
paramétriques et familles connues. Deux banques par famille ne mesurent pas
l'incertitude générale de réentraînement. Les deux exposants Hawkes puissance
atteignent leur borne préfixée ; le passé du burn-in reste caché aux prévisions.
La famille d'alternatives est issue d'un audit exploratoire antérieur ; les
nouvelles trajectoires permettent une confirmation indépendante dans ce périmètre,
sans validation universelle ni conclusion sur les modèles neuronaux.

Le lanceur volontaire est `python -m scripts.report_pool_sizes --help`.
Les scripts Slurm archivés attendent la fin de l'autre campagne sans modifier
ses dépendances, priorités ou fichiers. Un seul GPU simultané est demandé.
Les preuves sont écrites dans une nouvelle racine ; aucune archive précédente
n'est réécrite. Aucun modèle de production, défaut, dépendance ou workflow CI
modifié ; aucune table créée, modifiée ou supprimée, aucune migration.
