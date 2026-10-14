# MMD global sur les futurs complets — 14 octobre 2026

Le MMD conditionnel global avec le kernel de signature à huit points améliore
la détection d'une accélération ×2 sans changer les modèles. À 2 048 paires,
FullyNN atteint **42,5–45,5 %**, contre **7–8,5 %** avec la somme historique des
petits batches. RMTPP atteint **100 % des 200 essais dès 128 paires**, contre
48–49 % avec cette somme. FullyNN reste donc peu puissant à petite taille et
modérément puissant à 2 048 paires sur cette alternative.

## Protocole

Quatre checkpoints déjà archivés sont rechargés : FullyNN et RMTPP, chacun
après une et deux époques. Tous les poids sont vérifiés strictement inchangés
après les expériences ; aucune nouvelle estimation de paramètres n'intervient.
Les futurs évalués sont indépendants des données d'entraînement. Les deux
historiques hétérogènes alternés sont ceux du diagnostic précédent, à trois
événements, avec deux marques possibles.

Les futurs sont observés **en entier dans une fenêtre fixe `T=2`**, avec tous
leurs événements et leurs marques, y compris les fenêtres vides. Aucune
troncature au minimum d'une paire, quota d'événements ou sélection des seules
séquences non vides. Une fenêtre vide ne déclare pas une terminaison permanente.

L'expérience principale prévoit **300 essais H0 et 200 essais H1 par checkpoint**,
99 swaps indépendants à l'intérieur des paires d'historique, et les tailles
32, 128, 512 et 2 048 paires. Le même jeu de données et les mêmes swaps servent
aux trois statistiques :

- **Global** : U-statistique appariée sur toutes les paires et normalisation
  commune du pool complet, avec tous les termes entre batches.
- **Batches, normalisation globale** : somme sur les batches `16/11/5` extraite
  du même Gram global ; permet d'isoler le retrait des termes entre batches
  et la pondération de la somme, sans modifier les embeddings.
- **Batches, normalisation locale** : règle historique, avec la préparation du
  kernel séparément dans chaque petit batch, appliquée aux mêmes nouvelles données.

Kernel inchangé : pySigLib 4, `counting_grid`, huit points, ordre dyadique zéro,
kernel statique RBF de `sigma=1`, deux marques. Poids d'historique : gaussienne
unitaire de la distance RKHS de ce kernel, comme dans le test conditionnel public.
Les chemins exactement identiques sont regroupés **après** la préparation du
pool. Les indices inverses restituent toutes les lignes et leurs multiplicités.
Il ne s'agit pas d'une approximation ou d'une suppression d'observations.

H0 compare le simulateur de production, appelé une étape à la fois, au décodeur,
à l'inversion et aux tirages NumPy/SciPy indépendants. L'encodeur Torch figé est
partagé : cette expérience n'est pas une réimplémentation indépendante de
l'encodeur. H1 inverse `H(2τ)` à chaque étape, évalue les marques au temps `2τ`,
puis met à jour l'historique avec le futur effectivement accéléré. Les poids
du modèle restent les mêmes ; seule la loi alternative simulée diffère.

Les plans sont écrits avant les tirages. Le seuil de rejet est toujours 5 %,
avec la correction Monte Carlo `+1` et la gestion conservatrice des égalités
numériques de la production. Le critère d'excès H0 principal est un test
binomial unilatéral à `0.01/48` pour les 48 cellules de calibration prévues.

## Puissance observée

Chaque pourcentage repose sur 200 essais d'accélération :

| Paires | Global FullyNN 1 / 2 | Batches locaux FullyNN 1 / 2 | Global RMTPP 1 / 2 | Batches locaux RMTPP 1 / 2 |
|---:|---:|---:|---:|---:|
| 32 | 5,5 % / 7 % | 2,5 % / 6,5 % | 54,5 % / 53 % | 30 % / 29 % |
| 128 | 8 % / 5 % | 5 % / 6 % | **100 % / 100 %** | 49 % / 48 % |
| 512 | 11,5 % / 10,5 % | 7,5 % / 8,5 % | 100 % / 100 % | 84 % / 89,5 % |
| 2 048 | **45,5 % / 42,5 %** | 7 % / 8,5 % | 100 % / 100 % | 100 % / 100 % |

À 2 048 paires, les batches avec normalisation globale donnent seulement
**6 % / 8 %** pour FullyNN. La préparation commune des temps ne suffit donc
pas : conserver les termes entre batches et changer la pondération de la
statistique apporte l'essentiel du gain observé ici.

Les intervalles de Wilson à 95 % de la puissance FullyNN globale à cette taille
sont **38,7–52,4 %** et **35,9–49,4 %**. Pour RMTPP, les 200 rejets sur 200
correspondent à un intervalle **98,1–100 %**, pas à une garantie de puissance
parfaite sur toute future expérience.

## Calibration H0 et contrôle indépendant

Sur l'expérience principale, le MMD global à 2 048 paires rejette :

| Checkpoint | Rejets H0 sur 300 | Taux | Wilson 95 % |
|---|---:|---:|---:|
| FullyNN 1 | 15 | 5 % | 3,1–8,1 % |
| FullyNN 2 | **24** | **8 %** | **5,4–11,6 %** |
| RMTPP 1 | 17 | 5,7 % | 3,6–8,9 % |
| RMTPP 2 | 15 | 5 % | 3,1–8,1 % |

Aucun excès n'est détecté au critère familial principal. Cependant, les 8 %
de FullyNN 2 ne sont pas dissimulés : l'intervalle individuel exclut 5 %.
Un contrôle indépendant supplémentaire a donc été déclaré pour **les deux**
checkpoints FullyNN : 1 000 nouveaux essais H0, 2 048 paires, mêmes kernel,
statistique et 99 permutations, nouvelle graine `2026101441`. Aucun réglage
du modèle, du kernel ou du seuil, aucune répétition jusqu'à un résultat favorable.
Le critère d'excès de cette confirmation est fixé à `0.01/2`.

| Checkpoint | Rejets H0 indépendants sur 1 000 | Taux | Wilson 95 % |
|---|---:|---:|---:|
| FullyNN 1 | 44 | **4,4 %** | 3,3–5,9 % |
| FullyNN 2 | 46 | **4,6 %** | 3,5–6,1 % |

Les deux confirmations ne détectent aucun excès ; les p-values binomiales
unilatérales sont 0,827 et 0,739. L'excès initial de FullyNN 2 ne se confirme
donc pas sur la cohorte indépendante. Le résultat initial à 8 % reste conservé
et les cohortes ne sont pas fusionnées pour masquer ce signal.

L'absence d'excès détecté n'est pas une preuve de calibration universelle à
exactement 5 %. Ces essais portent sur les checkpoints, historiques, fenêtre,
marques et alternative précisés ; ils ne mesurent pas l'incertitude d'estimation
des paramètres ni la puissance contre toutes les erreurs possibles.

## Précision de l'horloge et contrôles numériques

La première exécution RMTPP a échoué dans l'outil d'observation : un délai
positif plus court que l'espacement float32 autour du temps absolu disparaissait
lors du stockage. **Tous les essais principaux ont été relancés avec les mêmes
graines**, sans écarter l'essai en échec. L'horloge d'observation est désormais
float64 dans cette expérience. Les entrées et paramètres neuronaux restent
float32 ; les gaps sont calculés avant la conversion. Aucun délai n'est déplacé
à un epsilon et aucun échec n'est assimilé à une censure.

L'option historique de l'outil reste disponible par défaut pour rejouer les
anciennes validations ; le nouveau diagnostic demande explicitement l'horloge
float64. La pipeline de production et la borne du sampler ne sont pas modifiées.

Contrôles : Gram direct contre compression exacte, toutes les permutations
contre la matrice publique sur la première petite cohorte de chaque expérience,
préservation des fenêtres vides et du padding, respect du contrat de fenêtre,
et conservation d'un petit délai positif ainsi que de son gap neuronal.

## Rejouer

```powershell
.venv\Scripts\python.exe -X utf8 -m scripts.global_signature_calibration --model-name FullyNN --epoch-number 1 --output artifacts/global-signature-replay-FullyNN-1
.venv\Scripts\python.exe -X utf8 -m scripts.global_signature_calibration --model-name RMTPP --epoch-number 1 --output artifacts/global-signature-replay-RMTPP-1
.venv\Scripts\python.exe -X utf8 -m scripts.global_signature_calibration --model-name FullyNN --epoch-number 2 --hypothesis H0 --trials 1000 --seed 2026101441 --confirmation --output artifacts/global-signature-confirmation-replay-2
```

Rejouer aussi les autres époques pour retrouver les quatre checkpoints et les
deux confirmations. Un dossier neuf est exigé pour préserver les preuves existantes.
Les workflows Linux sont optionnels : `global-signature-calibration.yml` et
`global-signature-confirmation.yml`, appelés manuellement ou par les marqueurs
explicites `[global-signature]` / `[global-signature-confirm]`.

Preuves : [plans, p-values, rapports et provenance](validation/global-signature-20261014/).
L'expérience principale, code `47d5655`, passe sur [Linux](https://github.com/NzoCs/Learning-point-processes/actions/runs/38628475313)
et le [lint](https://github.com/NzoCs/Learning-point-processes/actions/runs/38628475108).
La confirmation indépendante, code `2311860`, passe dans son
[exécution Linux](https://github.com/NzoCs/Learning-point-processes/actions/runs/38630581439)
et son [lint](https://github.com/NzoCs/Learning-point-processes/actions/runs/38630581694).
La suite CPU donne **336 tests réussis**, 7 ignorés, 2 désélectionnés,
**84,35 % de couverture**. Pas de nouvelle validation Ruche, CUDA, DDP ou
scientifique des onze modèles dans ce lot. Le défaut de la pipeline reste inchangé.
