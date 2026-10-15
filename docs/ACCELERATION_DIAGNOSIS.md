# Diagnostic contrôlé de la puissance contre l’accélération

## Résultats du plan initial — 15 octobre 2026

Les **deux plans (quatre jobs scientifiques) réussissent**, sans erreur ni modification des
poids. Le manque de puissance a plusieurs causes : la résolution de la grille,
l’échelle du temps face au kernel statique RBF et, pour FullyNN, le bruit des
futurs vides/comptages. Le quota supplémentaire apporte beaucoup d’information
sur RMTPP, très peu sur ce checkpoint FullyNN. Les contrôles ne révèlent pas
d’erreur de formule MMD ni de mauvaise calibration du test actuel sur ces cas.

Puissance contre les temps divisés par deux, **200 essais par cellule**,
**128 paires partout**, mêmes futurs et mêmes échanges au sein d’un plan :

| Variante | FullyNN quota 3 | FullyNN quota 6 | RMTPP quota 3 | RMTPP quota 6 |
|---|---:|---:|---:|---:|
| grid8_pool (production) | 3,5 % | 3,5 % | 25 % | 45 % |
| grid32_pool | 3,5 % | 3,5 % | 38,5 % | 83 % |
| grid8_physical | 3,5 % | 3,5 % | 30 % | 55 % |
| grid32_physical | 4 % | 3,5 % | 92,5 % | 100 % |
| events_pool | 2,5 % | 2,5 % | 10 % | 14 % |
| events_physical | 2 % | 1,5 % | 95 % | 99 % |
| first_time_physical | 5,5 % | 5 % | 87 % | 83,5 % |
| counts | 1,5 % | 1,5 % | 4 % | 5,5 % |

Sous H0, **300 essais par cellule** : FullyNN 2,67–4 %, RMTPP 3,67–6 %.
Le plus petit test binomial d’excès vaut 0,2467, au-dessus du seuil familial
0,01/32. Pas d’excès détecté, ce qui ne certifie pas tous les paramètres.
À quota 6, les intervalles Wilson 95 % RMTPP sont 38,26–51,92 % pour grid8,
77,18–87,57 % pour grid32_pool et 98,12–100 % pour grid32_physical. Le 100 %
est le résultat de 200 essais, pas une garantie de détection parfaite.

### Ce que ces comparaisons isolent

- **Grille et échelle interagissent.** Sur RMTPP quota 6, affiner seul donne
  45 → 83 %, changer le canal temps seul 45 → 55 %, combiner les deux
  45 → 100 %. Les chemins événementiels, qui ne perdent aucun saut,
  passent de 14 à 99 % quand seule leur échelle du temps change. Conserver
  les événements ne suffit donc pas si leurs différences temporelles ont un
  poids trop faible dans le kernel RBF de largeur fixée.
- **Le maximum commun comprime les premiers événements.** Sur RMTPP, il
  place 79,5 % des événements dans la première case au quota 3, 71,7 % au
  quota 6. À échelle commune figée pour décrire le phénomène, 64,9 % et
  41,2 % des chemins X non vides ont exactement le même encodage grid8
  après division des temps par deux. Le maximum reste symétrique et valide
  pour les échanges ; sa faiblesse ici concerne la puissance.
- **Le quota a un effet propre à effectif constant.** Les 300 tirages RMTPP
  contiennent 223 751 événements X+Y au quota 3 contre 435 280 au quota 6 ;
  grid8 passe de 25 à 45 % de puissance. FullyNN passe seulement de 19 713
  à 19 860 événements (+147, moins de 1 %), et la puissance reste à 3,5 %.
- **Ce checkpoint FullyNN produit 79,26 % de futurs X vides.** Le nombre de
  futurs X non vides est 7 965 sur 38 400. Environ 50–51 % d’entre eux
  perdent le changement temporel dans grid8, mais les chemins événementiels
  seuls ne restaurent pas la puissance. Le contrôle centré ci-dessous montre
  qu’une partie du signal est masquée par les caractéristiques de présence
  et de comptage ; ce n’est pas simplement une absence d’information.
- **L’alternative conserve les marques, le nombre d’événements et les arrêts.**
  Le contrôle « counts » ne doit donc pas gagner de puissance : il retrouve
  exactement les mêmes statistiques avant/après accélération sur les mêmes
  tirages. Une observation par fenêtre de temps complètement observée serait
  un autre protocole, où une accélération peut augmenter le comptage.

Ces chiffres concernent deux checkpoints figés, deux quotas et cette
alternative de transformation des chemins. Aucun défaut de la pipeline n’a
été changé. Une évolution éventuelle doit conserver le test global des
distributions et valider séparément sa sensibilité temporelle : le contrôle
centré n’est pas un remplacement universel. Les signatures à grille fine et
échelle fixe restent des variantes expérimentales, pas de nouveaux défauts.

## Résultats du contrôle temporel indépendant — 15 octobre 2026

Les modèles, leurs poids et leurs simulateurs restent inchangés. Le contrôle
temporel identifie un effet de choix des caractéristiques, au-delà du seul
nombre de futurs non vides : centrer les caractéristiques du temps parmi les
futurs non vides réduit le bruit de présence/comptage, sans retirer les vides.

| Modèle | Quota | Premier temps initial : H1 / 200 | Premier temps centré : H1 / 200 | H0 initial / 300 | H0 centré / 300 |
|---|---:|---:|---:|---:|---:|
| FullyNN | 3 | 20 (10 %) | 70 (35 %) | 13 (4,33 %) | 14 (4,67 %) |
| FullyNN | 6 | 19 (9,5 %) | 73 (36,5 %) | 13 (4,33 %) | 13 (4,33 %) |
| RMTPP | 3 | 175 (87,5 %) | 177 (88,5 %) | 10 (3,33 %) | 12 (4 %) |
| RMTPP | 6 | 164 (82 %) | 178 (89 %) | 19 (6,33 %) | 17 (5,67 %) |

À quota 6, les intervalles Wilson 95 % du contrôle centré sont **30,14–43,37 %**
pour FullyNN et **83,91–92,62 %** pour RMTPP. Sur les mêmes 200 essais FullyNN,
le centrage ajoute 64 détections et en perd 10, soit +27 points nets. Ce sont
des comparaisons appariées, sans recherche de largeur du kernel. Aucun excès
H0 n’est détecté au critère familial 0,01/8 ; 300 essais ne démontrent pas une
calibration universelle.

La première campagne FullyNN avait donné 5 % pour le contrôle premier temps
à quota 6, contre 9,5 % sur les nouveaux tirages. Les deux résultats restent
archivés : le gain du centrage se mesure contre le **9,5 % des mêmes tirages**,
pas contre la valeur de la campagne précédente.

Ce contrôle vise les temps du premier événement. Il n’est pas un remplacement
du test global de distribution : ses caractéristiques ignorent certaines
différences d’arrêt, de marques ou d’événements ultérieurs. Il montre que la
faible puissance actuelle de FullyNN ne prouve pas une absence de signal.

Campagne initiale : commit `9c448ae`, [Linux](https://github.com/NzoCs/Learning-point-processes/actions/runs/38718120167).
Contrôle indépendant : commit `244b804`, [Linux réussi](https://github.com/NzoCs/Learning-point-processes/actions/runs/38718888764).
Suite logicielle : **366 tests réussis**, 7 ignorés, 2 désélectionnés,
**85,13 %** de couverture. Les rapports complets et leur provenance sont
conservés dans `docs/validation/acceleration-diagnosis-20261015`.

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
Les simulations natives sont faites par blocs de 64 (64/64 à 128 paires),
mais la statistique est globale. La référence grid8 et ses tirages nuls sont
comparés au Gram public sur ces mêmes futurs : il ne s’agit pas de réutiliser
les p-values des essais précédents avec d’autres graines et tailles de batch.

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

## Contrôle complémentaire fixé après le premier résultat FullyNN

Après les résultats FullyNN du premier plan (puissance 1,5–5,5 %), un second
plan, distinct et conservé séparément, compare uniquement le RBF premier
temps initial à sa version centrée sur les futurs non vides du pool. Les
futurs vides ont le vecteur nul dans cet espace de caractéristiques ; ils ne
sont pas exclus et aucun temps fictif ne leur est attribué. Le Gram est
positif (translation puis multiplication des caractéristiques par l’indicateur
de présence), et le centrage commun est invariant aux échanges. Les kernels
des historiques et de l’arrêt sont conservés.

Ce contrôle cherche à séparer manque d’information temporelle et bruit de
présence/comptage dans la statistique. Nouvelle graine 2026101583, mêmes
effectifs 300 H0 / 200 H1, 128 paires et quotas 3/6 ; famille H0 de 8 cellules,
critère 0,01/8. Largeur RBF toujours 1, échelle d’entraînement inchangée, aucun
ajustement selon les résultats du second plan. Il ne remplace pas le plan
initial et ne constitue pas une validation d’un nouveau test de production.
