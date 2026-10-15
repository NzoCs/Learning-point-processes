# Diagnostic contrôlé de la puissance contre l’accélération

## Plan fixé avant les résultats

`scripts/acceleration_diagnosis.py` utilise les checkpoints FullyNN et RMTPP de
la pipeline actuelle, leurs configurations effectives et leurs simulateurs
inchangés. Une référence NumPy/SciPy indépendante génère X (encodeur Torch
partagé) et le simulateur de production génère Y. Les poids sont vérifiés par
empreinte avant/après. Aucun entraînement sur les futurs évalués.

Pour chaque modèle : 300 tirages H0, puis évaluation de l’alternative sur les
200 premiers de ces mêmes tirages en divisant les temps de X par deux. Il
s’agit de transformer le chemin complet, pas de recalculer le réseau après
chaque événement accéléré. Graine indépendante 2026101561, 128 paires,
99 échanges au sein des paires, seuil 5 %. Les historiques hétérogènes ont deux
événements et deux marques. Toutes les variantes partagent données, signes
d’échange, poids du kernel des historiques et kernel de l’indicateur d’arrêt.

Les quotas 3 et 6 sont les préfixes des mêmes trajectoires, au même effectif
128. Un arrêt après le troisième événement n’est pas révélé au quota 3.
Une trajectoire incomplète sans arrêt réel produit une erreur. Le quota
supérieur n’est pas traité comme une fenêtre temporelle complètement observée.

Variantes déclarées, sans sélection sur leurs résultats :

| Variante | Représentation | Échelle du canal temps |
|---|---|---|
| grid8_pool | Grille actuelle de 8 points | Maximum commun du pool |
| grid32_pool | Grille de 32 points | Maximum commun du pool |
| grid8_physical | Mêmes cases que grid8_pool | dtime_max des données d’entraînement |
| grid32_physical | Mêmes cases que grid32_pool | dtime_max des données d’entraînement |
| events_pool | Tous les sauts conservés | Maximum commun du pool |
| events_physical | Tous les sauts conservés | dtime_max des données d’entraînement |
| first_time_physical | log(1 + premier temps / dtime_max), présence | RBF de largeur fixée 1 |
| counts | Comptages par marque | RBF de largeur fixée 1 |

Les signatures gardent le backend pySigLib, kernel statique RBF sigma 1 et
dyadic_order 0. Les chemins événementiels utilisent un point avant/après
chaque saut, une origine et une extrémité commune ; leur padding répète
l’extrémité. La normalisation du canal comptage est celle de production.

Les variantes de grille « physical » changent uniquement le canal temps après
construction des cases. Fixer le diviseur des événements tout en laissant la
grille sur [0,1] perdrait les événements tardifs : ce serait une censure cachée.
Le croisement avec les chemins événementiels sépare cet effet des pertes de
résolution dues aux cases du maximum commun. Les contrôles premier temps et
comptage servent au diagnostic ; ils ne remplacent pas le test de production.

Chaque cellule H0 donne son taux, intervalle de Wilson 95 % et test binomial
unilatéral d’excès, avec seuil familial 0,01/32 (2 modèles × 2 quotas ×
8 variantes). Tous les résultats H1 sont publiés, pas seulement le meilleur.
Les essais appariés ne sont pas des confirmations indépendantes. Une erreur
interrompt le diagnostic et est conservée, sans filtrage des échecs.

Mesures descriptives : fraction de futurs vides, événements dans la première
case, chemins distincts, proportion de chemins non vides dont l’encodage
reste exactement identique après accélération à une échelle commune figée.
Cette dernière échelle sert seulement à décrire la perte d’information, pas
à calculer une p-value.

Contrôles : statistique et toutes les permutations de la référence grid8
comparées au Gram public, compression des doublons exacte, invariance aux
échanges, conservation des événements tardifs et absence de modification
des modèles. Les résultats ne garantissent pas la calibration universelle,
la stabilité après réentraînement ni le fonctionnement CUDA/Ruche.

Le workflow `acceleration-diagnosis.yml` est facultatif (déclenchement manuel
ou marqueur `[acceleration-diagnosis]` sur cette branche) et utilise les
dépendances verrouillées sur un runner Linux CPU standard.
