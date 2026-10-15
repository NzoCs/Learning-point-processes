# Diagnostic de puissance pour FullyNN — 14 octobre 2026

La suite de ce diagnostic est désormais réalisée sur les futurs complets avec
le kernel de signature : voir [la calibration globale](GLOBAL_SIGNATURE_CALIBRATION.md).
Elle retrouve un gain de puissance, avec 42,5–45,5 % pour FullyNN à 2 048 paires,
et 100 % des essais d'accélération RMTPP dès 128 paires. Les taux H0 indépendants
FullyNN sont de 4,4 % et 4,6 % sur 1 000 essais chacun. Le défaut de la pipeline
reste inchangé ; les résultats de ce premier diagnostic sont conservés ci-dessous.

Sur les deux petits checkpoints FullyNN utilisés pour les validations, le signal
d'accélération est faible parce que les futurs contiennent peu d'événements.
L'addition de MMD calculés dans des batches de 16, 11 et 5 paires réduit ensuite
fortement la puissance. Affiner la grille de signature ne suffit pas à résoudre
ce cas. Un contrôle sur le premier événement isole cet effet : avec les mêmes
données, le même kernel RBF et les mêmes permutations, réunir les 2 048 paires
donne **87–95 %** de détection, contre **5–9 %** avec les petits batches.

**Aucun modèle, poids, sampler ou kernel de production n'a été modifié.**
Ce lot ajoute des diagnostics optionnels, leurs contrôles et leurs preuves.
Il ne remplace pas automatiquement le test de la pipeline.

## Expériences et contrôles

Les paramètres archivés de FullyNN et RMTPP après une et deux époques sont
rechargés strictement, puis vérifiés inchangés après chaque expérience. Les
données de diagnostic sont indépendantes de l'entraînement. Deux historiques
fixes alternent : trois marques 0 aux temps `[0.1, 0.2, 0.3]` et trois marques 1
aux temps `[0.2, 2.0, 5.0]`. Fenêtre observée : `T=2`, deux marques possibles.
Les fenêtres vides sont conservées ; aucune terminaison permanente n'est
déduite du silence dans la fenêtre.

Chaque comparaison comporte **100 essais sous H0 et 100 essais d'accélération**,
**99 permutations**, et un seuil de rejet inchangé de 5 %. Les comparaisons
sont déclarées dans `plan.json` avant les tirages de chaque étape. Les étapes
suivantes répondent aux observations précédentes ; elles ne choisissent pas
des paramètres pour obtenir un résultat favorable.

1. Futurs complets : production contre décodeur/inversion NumPy/SciPy, avec
   encodeur Torch figé partagé. Comparaisons sur exactement les mêmes données
   et swaps : signature à 8/64 points, RBF sur les comptes par marque, RBF sur
   le premier temps censuré, 32/128 paires. Le kernel d'historique reste à 8
   points pour isoler la représentation du futur. Bande passante fixée à un
   événement pour les comptes et `T/10` pour le premier temps.
2. Premier événement seulement : contrôle ciblé de la taille d'échantillon et
   test bilatéral de sa moyenne censurée, sur sa loi analytique indépendante.
3. Même loi du premier événement, mêmes données et mêmes swaps concaténés :
   comparaison du même RBF et des mêmes poids d'historique avec MMD par petits
   batches ou MMD sur toutes les paires. Tailles : 32, 128, 512 et 2 048.
   Les p-values finales utilisent aussi la règle conservatrice de la production
   pour les égalités numériques. Cette étape remplace le contrôle préliminaire.

Pour les contrôles du premier événement, H0 compare deux échantillons
indépendants de la même loi NumPy/SciPy. **Ce n'est pas une nouvelle calibration
du simulateur de production ou de tous les futurs.** L'inversion est contrôlée
contre les délais censurés exacts d'un Poisson. La statistique vectorisée et
la version globale sont comparées aux swaps explicites du même Gram.

## Peu d'événements et faible changement des comptes

Mesures des futurs complets, 12 800 séquences par colonne et checkpoint :

| Checkpoint | Événements/fenêtre, loi originale | Accélération ×2 | Fenêtres vides originales |
|---|---:|---:|---:|
| FullyNN, 1 époque | 0,196 | 0,225 | 83,18 % |
| FullyNN, 2 époques | 0,208 | 0,239 | 82,65 % |
| RMTPP, 1 époque | 2,983 | 5,535 | 12,05 % |
| RMTPP, 2 époques | 3,000 | 5,586 | 11,63 % |

Pour les historiques FullyNN, `P(pas de prochain événement)=exp(-H∞)` vaut
environ **79–84 %**. Accélérer les temps conserve cette masse de terminaison.
La probabilité d'au moins un événement dans la fenêtre passe de **14,4–19,4 %**
à **15,8–21,1 %** : seulement environ 1,4–1,7 point de pourcentage en plus.
L'allongement de la fenêtre de 2 à 8 augmente cette probabilité à environ
16,0–21,4 %, sans faire disparaître la masse de terminaison.

Une petite expérience à 32 séquences FullyNN ne contient donc typiquement
que cinq à six fenêtres non vides. Le déplacement des temps existe bien :
la moyenne analytique du premier temps censuré diminue d'environ 0,05–0,07
selon l'historique. Ces valeurs décrivent **ces checkpoints et ces historiques**,
pas une limite de performance universelle de FullyNN.

## Grille et kernel sur les futurs complets

Puissance mesurée à 128 paires :

| Kernel du futur | FullyNN 1 | FullyNN 2 | RMTPP 1 | RMTPP 2 |
|---|---:|---:|---:|---:|
| Signature, 8 points | 4 % | 5 % | 31 % | 43 % |
| Signature, 64 points | 4 % | 6 % | 70 % | 68 % |
| RBF des comptes par marque | 3 % | 5 % | 57 % | 59 % |
| RBF du premier temps censuré | 6 % | 5 % | 30 % | 19 % |

Le contrôle couplé divise uniquement les temps d'une copie de chaque futur
par deux, sans changer marques ou comptes. Parmi les fenêtres **non vides**,
environ 12,7–12,8 % de ces copies ont exactement le même embedding à 8 points,
contre 1,4–1,5 % à 64 points. La grille grossière perd donc réellement de
l'information, mais restaurer cette information ne suffit pas ici pour FullyNN.
Les fenêtres vides ne sont pas comptées comme une erreur de discrétisation.

Les contrôles H0 des futurs complets donnent 2–10 % de rejets sur 100 essais
par cellule. Aucun excès n'est détecté au critère familial préannoncé `0.01/32`.
**Cela ne certifie pas un taux exact de 5 %.** RMTPP, deuxième checkpoint,
signature à 64 points et 128 paires donne notamment 10 % de rejets sous H0 :
une validation plus longue est nécessaire avant d'adopter cette variante.
Les intervalles de Wilson et toutes les p-values sont conservés dans les preuves.

## Le découpage du MMD est un facteur important

Contrôle du premier événement, même RBF de bande passante `T/10`, mêmes poids
d'historique, mêmes données, mêmes swaps à l'intérieur de chaque paire :

| Paires indépendantes | Petits batches, FullyNN 1 / 2 | Toutes les paires, FullyNN 1 / 2 | Moyenne censurée bilatérale, FullyNN 1 / 2 |
|---:|---:|---:|---:|
| 32 | 7 % / 7 % | 5 % / 4 % | 4 % / 6 % |
| 128 | 5 % / 6 % | 9 % / 12 % | 10 % / 14 % |
| 512 | 8 % / 6 % | 33 % / 28 % | 36 % / 36 % |
| 2 048 | 5 % / 9 % | **87 % / 95 %** | 86 % / 88 % |

À 2 048 paires, les intervalles de Wilson à 95 % sont **79,0–92,2 %** et
**88,8–97,8 %** pour le MMD global. Sous H0, ce MMD donne respectivement
**3 % et 6 %** de rejets à cette taille ; aucun excès n'est détecté parmi les
24 contrôles finaux au critère familial préannoncé `0.01/24`.

La somme des MMD des petits batches conserve la validité du schéma de
permutation, mais elle n'est pas le MMD de toutes les données. Elle retire les
comparaisons entre batches et donne le même poids aux statistiques de batches
de tailles différentes. À 2 048 paires, le découpage `16/11/5` utilise 23 680
positions hors diagonale, contre 4 192 256 pour le Gram global. Le faible nombre
d'événements rend cette perte de comparaisons particulièrement coûteuse.

Il faut donc distinguer **calibration des p-values** et **puissance** : une
statistique peut rester correctement randomisée tout en détectant mal une
alternative. Changer arbitrairement les paramètres FullyNN ne résoudrait pas
ce défaut d'efficacité du protocole.

## Portée et suite justifiée

Le diagnostic identifie deux facteurs : le peu d'événements disponibles et
l'agrégation de MMD sur de petits batches. Le kernel RBF du premier temps peut
détecter l'accélération avec assez de données et un calcul global ; la grille
de signature à huit points n'explique pas, à elle seule, le mauvais résultat.

Le prochain changement à évaluer serait un MMD conditionnel global, calculé
éventuellement par blocs **en conservant les termes entre blocs**, avec une
normalisation commune et les identifiants des séquences. Il faudrait comparer
sa calibration et sa puissance sur les **futurs complets avec le kernel de
signature** avant de changer le défaut de la pipeline. Les 87–95 % mesurés ici
ne valident pas encore ce nouveau parcours complet. Le test de moyenne est
un contrôle ciblé ; il ne remplace pas un test général de la distribution.

Pas de nouvelle validation Ruche, CUDA, DDP ou de calibration scientifique
des onze modèles dans ce lot. La borne du sampler générique reste inchangée.

## Rejouer et vérifier

```powershell
.venv\Scripts\python.exe -X utf8 -m scripts.window_power_diagnosis --output artifacts/window-power-replay
.venv\Scripts\python.exe -X utf8 -m scripts.window_first_event_power --epoch-number 1 --output artifacts/window-first-event-replay-1
.venv\Scripts\python.exe -X utf8 -m scripts.window_first_event_power --epoch-number 2 --output artifacts/window-first-event-replay-2
```

Workflows Linux optionnels : `window-power-diagnosis.yml` et
`window-first-event-power.yml`, appelés uniquement sur demande manuelle ou
avec les marqueurs explicites de commit `[window-power]` / `[window-first-event]`.

Preuves : [rapports, plans, p-values et provenance](validation/window-power-20261014/).
Futurs complets, code `9882678` : [exécution Linux réussie](https://github.com/NzoCs/Learning-point-processes/actions/runs/38622375150).
Comparaison globale finale, code `cb16a69` : [exécution Linux réussie](https://github.com/NzoCs/Learning-point-processes/actions/runs/38624318432),
[lint réussi](https://github.com/NzoCs/Learning-point-processes/actions/runs/38624318596).
La CI finale donne **332 tests réussis**, 7 ignorés, 2 désélectionnés et
**84,35 % de couverture**. Les paramètres figés sont déjà versionnés dans les
preuves de vraisemblance ; les ZIP CI et leurs sommes SHA-256 sont référencés
dans `provenance.json`. Les rapports préliminaires restent identifiés comme
tels et sont remplacés, pour les conclusions, par la comparaison globale finale.
