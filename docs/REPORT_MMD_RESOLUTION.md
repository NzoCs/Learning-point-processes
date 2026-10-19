# Diagnostic indépendant de la grille et du kernel MMD

Cette expérience explique la sensibilité du MMD de signature dans les scénarios
synthétiques Poisson/Hawkes du rapport. Elle ne modifie ni les modèles, ni les
simulations sources, ni les campagnes RMSE, ni les paramètres de production.
Elle ne vise pas à démontrer une supériorité universelle du MMD ou de la RMSE.

## Contrat fixé avant calcul

Le plan factoriel de sélection croise trois représentations (grille uniforme
100 points, grille 400 points, chemin aux événements) et quatre kernels statiques
(linéaire, RBF sigma=0.3, 1 et 3). Le sigma suit la convention pySigLib existante.
La normalisation du comptage et du temps reste celle du pilote indépendant
historique : N(t)/écart-type du compte pilote et t/T. Aucune normalisation propre
à un groupe, aucune modification du nombre d'événements ou du modèle.

Le chemin aux événements contient l'origine, le déplacement temporel avant
chaque saut, le saut de comptage au même instant puis l'extrémité à T=1.
Le padding répète cette extrémité. Les trajectoires vides sont conservées.
Les sommets sont conservés exactement ; le calcul du kernel reste une
approximation par différences finies pySigLib, ordre dyadique zéro comme la
référence. Un pilote documente la sensibilité aux ordres 1 et 2 sur de petits
échantillons ; une différence de puissance entre représentations ne prouve
donc pas à elle seule une erreur numérique de la bibliothèque.

La graine indépendante est 2026102037. Les espaces de graines sélection (1) et
confirmation H0 (2) sont distincts des campagnes historiques et de leurs pilotes.
Tous les essais utilisent n=128 par groupe, T=1 et 499 échanges de trajectoires
entières, avec la correction plus-un et les égalités conservatrices existantes.

1. Sélection : 32 nouveaux essais pour chacun des dix scénarios. Pour chaque
   représentation, choisir le kernel maximisant la moyenne non pondérée des
   puissances sur les six alternatives H1. Les égalités suivent l'ordre déclaré
   linéaire, RBF 0.3, 1, 3. Les résultats H0 ne filtrent pas les candidats.
   Ce faible effectif constitue un réglage exploratoire, pas un optimum garanti.
2. Figer `selection.json`, ses données et leurs empreintes avant confirmation.
3. Calibration indépendante : 1 000 nouveaux essais pour chacun des quatre H0,
   soit 4 000 essais, sur les trois choix figés et les deux références historiques.
   Au plus vingt contrôles effectifs ; seuil conservateur préfixé 0.01/48.
   Signaler tout excès, ne pas changer le choix pour obtenir une calibration favorable.
4. Rejouer les vingt cellules historiques, 8 000 H0 et 3 000 H1 : mêmes temps,
   nombres d'événements et permutations vérifiés par hachage. Évaluer linéaire et
   RBF sigma=1 pour les trois représentations, ainsi que les choix indépendants
   (au plus neuf méthodes distinctes). Les deux références grille 100 doivent
   retrouver les statistiques dans les tolérances et les p-values exactes.

Les intervalles de proportions sont binomiaux exacts à 95 %. Les 72 comparaisons
de puissance McNemar exactes bilatérales sont corrigées ensemble par Holm :
pour chaque alternative, quatre comparaisons de résolution à kernel fixe,
six comparaisons du choix indépendant aux deux kernels fixes de sa représentation,
et deux comparaisons entre représentations avec choix indépendant. Les doublons
issus de choix identiques sont conservés, avec p=1 pour une méthode comparée à elle-même.
Les taux H0 historiques sont descriptifs ; la confirmation formelle utilise les
nouveaux essais indépendants. Absence d'excès détecté n'est pas preuve d'égalité
exacte à 5 %. Ne pas conclure à partir de cellules incomplètes.

## Sources, séparation et intégrité

La source MMD est le commit 9398381cb6d2b71b618b37d78299c7501d121f80.
Ses 61 fichiers plan/report/trials/summary sont hachés avant exécution et vérifiés
à l'agrégation. Les sources et les environnements du diagnostic sont figés.
Chaque p-value est recalculée depuis sa statistique et ses 499 valeurs nulles.
Les erreurs numériques, échecs CPU/CUDA, archives incomplètes, modifications
de source ou divergences des références bloquent le diagnostic : aucun essai
n'est exclu silencieusement. Une sortie 2 de l'agrégation indique un excès H0
scientifique avec résultats complets, pas une panne à relancer automatiquement.

Les résultats de puissance historiques ayant déjà été examinés, leur réutilisation
reste exploratoire, même si les nouveaux réglages utilisent des données séparées.
Les trajectoires sont synthétiques et non marquées ; la comparaison ne valide
pas les modèles neuronaux. Aucune table créée, modifiée ou supprimée, aucune
migration, aucun workflow CI ni défaut de production modifié.

## Exécution volontaire

`python -m scripts.mmd_resolution_diagnostic --help` expose les phases initialize,
pilot, selection, select, calibration, replay et aggregate. La sélection doit
précéder calibration/replay. Le pilote CUDA vérifie les douze candidats contre
CPU, les références historiques et les temps d'une paire de groupes n=128.
Les scripts Slurm archivés avec le reçu d'exécution attendent la fin de la
campagne en cours ; ils ne modifient aucun job déjà soumis.
