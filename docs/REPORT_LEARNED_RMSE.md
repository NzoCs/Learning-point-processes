# Référence apprise : RMSE sur tous les événements et MMD

Le protocole connu reste inchangé et ses archives restent immuables. Cette
extension isole l'effet du remplacement de ses paramètres théoriques par des
paramètres estimés sur des données séparées. Les modèles de production ne sont
pas modifiés. Il s'agit de références **paramétriques apprises**, pas neuronales.

## Apprendre puis figer avant de lire les résultats évalués

Trois familles sont fixées : Poisson, Hawkes exponentiel et Hawkes puissance.
La famille est connue, ses paramètres ne le sont pas. Pour chacune, deux banques
indépendantes de 1 024 trajectoires H0 sont générées, une par cohorte. Leur durée
observée est 1, avec le même burn-in génératif de 10 pour Hawkes. Le passé du
burn-in n'est transmis ni à l'estimateur ni à la prévision. La vraisemblance
conditionne sur un passé visible vide à zéro : elle peut donc être mal spécifiée
au bord de la fenêtre. Aucune prétention d'estimation exacte des paramètres
générateurs n'est faite dans ce cadre.

La graine d'entraînement est `2026101931`, avec entropie
`(2026101931, 731, famille, cohorte)`. Cet espace est distinct de tous les
tirages évalués, du pilote de normalisation MMD et du pilote local Poisson.
Les identifiants des banques, leurs événements et leurs SHA256 sont conservés.
Les trajectoires évaluées ne servent jamais à l'estimation, à l'initialisation,
au choix d'une famille, d'un seuil ou d'un hyperparamètre.

Poisson utilise le maximum de vraisemblance exact : nombre total d'événements
divisé par l'exposition totale, en comptant les trajectoires vides. Hawkes
optimise `∫λ(t)dt − Σ log λ(t_i−)` sur les trajectoires d'entraînement. Toutes
les composantes libres sont estimées : fond, branchement, décroissance et,
pour la famille puissance, exposant. Le terme intégré est exact. Les
gradients analytiques sont vérifiés par différences finies et par une
intégration indépendante de l'intensité sur de petites trajectoires.
Référence : [Laub, Taimre et Pollett, Hawkes Processes](https://arxiv.org/abs/1507.02822).

Les trois points de départ de l'optimiseur sont fixés dans le code. Le meilleur
optimum convergé est choisi uniquement par la vraisemblance d'entraînement.
Les bornes numériques, points de départ, sources et diagnostics sont archivés.
Un ajustement sans convergence bloque la campagne ; une solution au bord est
signalée, sans changer les bornes après examen de la puissance. Une optimisation
multi-départ ne constitue pas une garantie d'optimum global.

## Comparaison appariée fixée avant exécution

Six références sont figées, puis vérifiées sur CPU/CUDA avant le tableau Slurm.
Une même référence est appliquée aux deux groupes de chaque essai. Elle est
réutilisée dans tous les scénarios de sa famille/cohorte, sans réentraînement
par alternative ni permutation. La normalisation des pertes utilise sa propre
prévision restreinte, commune aux deux groupes.

Les 20 cellules reprennent exactement les 11 000 essais archivés : 8 000 H0,
3 000 H1, 128 trajectoires par groupe, T=1 et 499 échanges appariés. Les temps,
comptages, signatures, normalisation MMD et permutations restent identiques.
La RMSE garde les origines zéro et chaque événement, le dernier intervalle
censuré, les trajectoires vides et les poids égaux par trajectoire. La cible
reste `min(prochaine attente, horizon restant)`. Les unités échangées sont les
trajectoires entières, jamais leurs intervalles dépendants.

Cinq méthodes sont comparées : RMSE avec référence apprise, RMSE avec référence
connue, signatures MMD linéaire/RBF et RBF du compte. L'excès H0 est contrôlé
sur 60 tests (quatre scénarios, cinq méthodes, deux cohortes et leur réunion),
au seuil `0.01/60`. Les bornes simultanées à 99 % portent sur 20 cellules réunies.
Les 24 McNemar exacts bilatéraux (six alternatives, quatre comparateurs de la
nouvelle RMSE) sont corrigés par Holm. Les intervalles binomiaux ponctuels
restent à 95 %. Aucune erreur ou cellule incomplète n'est exclue.

Les conclusions de calibration et puissance sont **conditionnelles aux six
références figées**, avec seulement deux répliques d'entraînement par famille.
Ce n'est pas une estimation générale de la variabilité d'un apprentissage
répété, ni une validation des modèles neuronaux. La famille correcte reste une
information préalable. Les données restent synthétiques et non marquées ;
le passé Hawkes caché et le choix exploratoire des métriques subsistent.

## Exécution manuelle

```bash
python -m scripts.report_learned_rmse train /workdir/learned/training
# Sur allocation GPU, après réussite de l'apprentissage :
python -m scripts.report_learned_rmse backend /workdir/learned/training /workdir/learned/pilot
# Le lanceur exige les trois variables LTPP_KNOWN_RMSE_SOURCE,
# LTPP_LEARNED_TRAINING et LTPP_LEARNED_OUTPUT ; tableau 0-19%1.
bash scripts/bash/ruche_learned_rmse.sh --dry-run
python -m scripts.report_learned_rmse aggregate /workdir/learned/cells
```

Le code et les données d'entraînement sont figés avant le rejeu. Les SHA256,
versions et contrôles de reprise empêchent un changement silencieux de la
référence. Les archives d'entrée sont lues sans troncature ni écriture.
Aucun entraînement ou calcul scientifique long n'est ajouté à la CI.
