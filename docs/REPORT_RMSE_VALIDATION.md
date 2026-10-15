# RMSE à référence commune et comparaison au MMD

Ce complément scientifique ne modifie ni les modèles, ni les simulateurs,
ni les observations du MMD. Les résultats historiques restent conservés.
Il corrige la métrique d'évaluation et définit explicitement ce qui est comparé.

## Référence commune

Pour chaque historique H et chaque candidat q, l'erreur normalisée devient
`((attente_observée - prédiction_q(H)) / prédiction_référence(H))²`.
La référence est toujours la loi H0 du scénario, y compris lorsque q change.
Elle n'est jamais estimée sur les futurs évalués. Les trajectoires ont le même
poids ; la racine est prise après la moyenne des erreurs quadratiques.

L'ancien code Hawkes pouvait utiliser la prédiction du candidat au dénominateur.
Cela comparait des pertes différentes. Sur Poisson de taux 100, prédire une
attente 0,02 au lieu de 0,01 peut alors sembler meilleur, simplement parce que
le diviseur double. Le test de régression vérifie que la référence commune
rétablit le classement attendu sur des données indépendantes.

La formule exponentielle de l'ancienne métrique initialisait aussi l'excitation
à zéro au premier événement, au lieu de son saut d'intensité. Sa contribution
manquait ensuite dans la récurrence. Le nouveau calcul intègre
l'intensité conditionnelle sur l'historique disponible, événement courant
inclus. Les intégrales exponentielles et puissance sont vérifiées par
intégration numérique indépendante de l'intensité dans les tests.

Les trajectoires de la campagne ne contiennent pas le passé du burn-in.
Le prédicteur reçoit exclusivement les événements visibles depuis zéro.
La référence est donc une règle de prévision fixée, **pas un oracle optimal**
pour le Hawkes marginalisé sur son passé caché. Cette limite est commune aux
candidats et n'invalide pas l'échangeabilité du test sous H0.

## Courbes descriptives et bords de fenêtre

La commande `curves` reprend les quatre balayages historiques :

- Poisson : ratios de taux 0,5 à 2.
- Hawkes optimisé : référence puissance p=2 ; alpha exponentiel de 2 à 8,
  beta=2*alpha, mu=50, moyenne stationnaire 100.
- Forme du noyau : référence exponentielle ; puissance p=1,5 ; 2 ; 3 ; 5 ; 8.
- Hawkes à beta=10 : alpha de 2 à 8, mu=100-10*alpha.

Chaque balayage utilise le premier banc X de la cohorte 0 du scénario
correspondant, fixé par ses graines et son empreinte MMD. Tous les candidats
scorent exactement ce même banc. Le contrat est harmonisé à **T=1 et n=128**,
donc ces courbes ne reproduisent pas à l'identique les figures anciennes T=10.
Une prévision constante de taux 100 remplace la moyenne calculée sur les
trajectoires évaluées ; cette dernière utilisait les cibles à prédire.

Deux sorties sont distinguées :

1. RMSE descriptive sur les intervalles entre événements effectivement vus,
   avec référence commune (`--completed-gaps`). Aucun minimum arbitraire de
   cinq événements ; les trajectoires de moins de deux événements sont
   comptabilisées comme non scorables, jamais comme erreur zéro. L'intervalle
   terminal censuré n'est pas connu : cette métrique reste conditionnée à
   l'observation des événements et ne mesure pas l'erreur sur toutes les
   attentes futures. C'est un diagnostic, pas le test de détection principal.
2. RMSE sur quatre origines fixes `0, 0.25, 0.5, 0.75`, avec cible
   `min(prochaine_attente, 0.25)`. Une absence d'événement jusqu'à la limite
   fournit exactement cette cible, sans inventer de prochain événement.
   Chaque modèle prédit la moyenne restreinte `∫₀^0.25 S(u|H) du`, au lieu de
   comparer une attente non bornée à une cible censurée. Toutes les trajectoires,
   même vides, restent présentes. La limite appartient à l'observation et à la
   cible évaluée : elle ne plafonne pas le simulateur ni les événements du MMD.

Les deux RMSE sont des quantités différentes et leurs valeurs ne doivent pas
être confondues. Les courbes seules ne sont pas des mesures de puissance.

## Test de détection au même niveau que le MMD

La prévision H0, fixée avant évaluation, transforme chaque trajectoire en sa
MSE moyenne aux quatre origines. La statistique est le carré de la différence
des MSE moyennes entre les deux groupes. C'est un test bilatéral : il détecte
une hausse ou une baisse de l'erreur. Sous échange, la MSE totale reste fixe ;
ordonner les écarts absolus de MSE revient à ordonner les écarts de RMSE.

Chaque essai utilise les **mêmes X/Y et les mêmes 499 échanges appariés** que
le MMD, avec la même règle de p-value plus-un et le même niveau de 5 %.
Les observations sont régénérées depuis les graines archivées ; leurs deux
SHA256 et la graine des échanges doivent correspondre exactement, sinon
l'essai échoue. Aucun nouveau calcul de signature n'est nécessaire.

Le test RMSE ne teste pas toute la loi du processus : deux lois différentes
peuvent donner la même MSE pour cette prévision. Une puissance faible ne suffit
donc pas à conclure à une erreur, ni à une supériorité universelle du MMD.
La précision prédictive des candidats est aussi évaluée sur les mêmes X,
avec le même dénominateur H0, séparément de ce test à deux échantillons.

## Effectifs, intégrité et lecture

Les 20 cellules et 11 000 essais du MMD sont repris sans sélection. Deux
cohortes, 2 000 essais par H0 et 500 par alternative. Une cellule MMD incomplète
ou avec échecs est refusée. Une erreur RMSE est conservée et empêche la
certification ; aucun filtrage des seuls essais réussis. Les reprises exigent
les mêmes sources, environnement, archives MMD et protocole.

L'agrégateur relit les archives sources, vérifie leurs empreintes et recalcule
les p-values MMD et RMSE. Il exige les paramètres principaux, la source propre
et versionnée, puis produit :

- Calibration des quatre méthodes, cohortes séparées et réunies ; intervalles
  binomiaux et test d'excès contre 5 %, seuil simultané `0.01/48` pour
  4 cas H0 × 4 méthodes × 3 regroupements. Les anciennes p-values MMD restent
  inchangées ; la correction de lecture englobe désormais la méthode ajoutée.
- Bornes supérieures simultanées à 99 % sur les 16 cellules H0 réunies.
- Puissance et comparaison appariée RMSE/MMD par test exact de McNemar,
  avec Holm sur les 18 comparaisons prévues (six alternatives × trois MMD).

Ce complément a été défini après le début de la campagne MMD et son audit
exploratoire. Il ne constitue pas une confirmation indépendante des choix
du MMD. Les critères RMSE sont fixés avant sa campagne ; aucune sélection de
prévision, de cap ou d'origine n'est faite d'après les puissances obtenues.

## Exécuter à la demande

```bash
python -m scripts.report_rmse_validation replay \
  /workdir/mmd/cells/poisson_H0/cohort-0 \
  /workdir/rmse/cells/poisson_H0/cohort-0 --descriptive
python -m scripts.report_rmse_validation curves \
  /workdir/mmd/cells /workdir/rmse/curves.json --completed-gaps
python -m scripts.report_rmse_validation aggregate /workdir/rmse/cells
```

`scripts/bash/ruche_report_rmse.sh` prépare un tableau Slurm 0–19. Le lancer
avec une dépendance `aftercorr` sur le tableau MMD permet de démarrer chaque
cellule uniquement après la réussite de sa source correspondante. Les calculs
RMSE utilisent le CPU et n'ajoutent aucune campagne longue à la CI.

Références : la moyenne restreinte est l'intégrale de la survie jusqu'à la
limite ([Zhang, Wu et Yin](https://wuyuanshan.github.io/Pubs/20_ZhangWuYin_SIM.pdf)).
Le MMD et une différence de risque prédictif ont des objets différents
([Gretton et al., 2012](https://www.jmlr.org/papers/v13/gretton12a.html)).
