# Comparaison RMSE/MMD : chaque événement et censure administrative

Les quatre dates de la première campagne étaient des origines fixes pour
éviter de sélectionner uniquement des attentes dont l'événement final est
observé. Ce choix était une référence simple, pas une nécessité statistique.
Il utilisait moins d'information que le MMD. Les conclusions de cette première
campagne restent celles de ce test précis ; elles ne prouvent pas que tout
test RMSE est moins puissant qu'un MMD.

## Protocole fixé avant la nouvelle campagne complète

Les mêmes 11 000 essais, 128 trajectoires par groupe, fenêtre `[0,1]` et
499 échanges appariés sont repris, sans nouveau tirage des données MMD.
La référence de prévision reste H0. Aucun modèle, simulateur, entraînement ou
paramètre n'est changé ; la représentation et les statistiques MMD sont
conservées exactement. Les archives originales restent immuables.

Chaque trajectoire est évaluée au temps zéro et après **chaque événement**
avant la fin de la fenêtre. À une origine `t`, on dispose de `c=1-t` unités
d'observation. La cible est `min(prochaine attente, c)`, et la prévision est
la moyenne restreinte correspondante `∫₀^c S(u|H_t) du`.

Cela inclut l'intervalle initial et le dernier intervalle censuré. L'absence
de prochain événement avant la fin de la fenêtre produit une cible connue
`c` ; elle ne permet pas de connaître l'attente non bornée. Aucune censure
n'est imputée comme un événement survenu à la fin : c'est la cible restreinte
qui est évaluée. Il n'y a **pas de plafond arbitraire de 0,25** dans ce test.
Une trajectoire vide garde l'origine zéro et une cible égale à 1. Un événement
exactement à la frontière est une cible mais ne démarre pas une prévision
avec exposition nulle. Les historiques de prévision n'incluent jamais le futur.

Chaque erreur est divisée par la même prévision H0, et les erreurs sont
moyennées dans chaque trajectoire. Les trajectoires ont ensuite le même poids.
Les intervalles d'une trajectoire étant dépendants, **seules les trajectoires
entières sont échangées**, exactement comme dans le MMD. Ce score est une
fonction de la trajectoire observée ; son origine et son effectif dépendent du
processus. Ce n'est pas une affirmation de risque conditionnel optimal ni un
test omnibus de toute différence de loi.

## Séparer l'effet du nombre de prévisions de celui de l'horizon

Un deuxième contrôle utilise toujours les quatre dates `0, 0.25, 0.5, 0.75`,
mais avec le même horizon restant `1-t` que le test sur tous les événements.
Il permet de comparer le changement d'origines sans le confondre avec le
retrait du plafond 0,25. La première RMSE à quatre dates/plafond 0,25 est relue
dans ses archives, sans la recalculer ni modifier ses p-values.

La nouvelle campagne contient donc six méthodes : RMSE à tous les événements,
les deux RMSE à quatre dates, les signatures linéaire/RBF et le RBF du compte.
Les tests d'excès sous H0 utilisent une famille de 72 contrôles
(4 scénarios × 6 méthodes × cohortes séparées et réunies), au seuil `0.01/72`.
Les bornes supérieures simultanées à 99 % portent sur les 24 cellules réunies.
Les comparaisons de puissance prévues opposent la nouvelle RMSE aux cinq
autres méthodes sur six alternatives : McNemar exact et Holm sur **30** tests.

Chaque essai vérifie les empreintes des données et la graine des échanges.
L'agrégateur vérifie aussi les archives MMD et quatre dates, leurs plans,
les pertes, les p-values et les nombres d'origines : chaque événement doit
avoir sa prévision, plus le dernier intervalle censuré. Les erreurs restent
enregistrées ; aucune campagne partielle ou filtrée n'est déclarée complète.

## Calcul numérique et validation

Le calcul float64 peut utiliser CPU ou CUDA, sans fallback de périphérique.
Les origines et nœuds de quadrature sont traités par blocs pour borner la
mémoire. Les ordres 32 et 64 sont comparés ; seuls les cas non résolus passent
aux ordres 128 et 256, puis à une vérification adaptative indépendante. Un
désaccord numérique produit un échec explicite, jamais un événement tronqué.

Les tests comparent les prévisions à une intégration indépendante de
l'intensité puis de sa survie, vérifient les séquences vides, les frontières,
la censure finale, l'absence de fuite du futur, le découpage en blocs et la
reprise exacte des expériences. Les contrôles CUDA doivent être exécutés
sur la Ruche avant la campagne complète ; un test CPU ne les remplace pas.

```bash
# Pilote numérique CPU/CUDA, sur allocation Slurm GPU :
python -m scripts.check_all_event_rmse --device cuda --output /workdir/all-event/pilot

# Reprise d'une cellule dans un répertoire neuf :
python -m scripts.report_rmse_validation replay \
  /workdir/mmd/cells/hawkes_exp_H0/cohort-0 \
  /workdir/all-event/cells/hawkes_exp_H0/cohort-0 \
  --mode all_events --device cuda \
  --landmark-source /workdir/four-date/cells/hawkes_exp_H0/cohort-0

# Vérification stricte des vingt cellules :
python -m scripts.report_rmse_validation aggregate /workdir/all-event/cells --mode all_events
```

Le lanceur `scripts/bash/ruche_all_event_rmse.sh --dry-run` affiche les vingt
commandes sans créer de résultats. L'exécution réelle exige Slurm ; soumettre
avec un seul GPU simultané après réussite du pilote, depuis un commit propre
et gelé. Aucune campagne longue n'est ajoutée à la CI.

Un contrôle Poisson local distinct (`check_all_event_rmse --poisson-pilot`)
comporte 1 000 essais H0 et 500 alternatives de taux 100/110, aux mêmes
effectifs et permutations. Ses graines sont séparées de la campagne MMD.
Il vérifie la nouvelle métrique et ses deux contrôles RMSE ; il ne remplace
pas la comparaison appariée complète aux MMD archivés.

Le pilote local indépendant du 16 octobre donne **36/1 000 rejets H0 (3,6 %,
intervalle à 95 % 2,53–4,95 %)** pour tous les événements, contre 44/1 000
(4,4 %) pour chacun des contrôles à quatre dates. Sur le changement de taux,
la détection est **500/500 (100 %, intervalle 99,26–100 %)** contre 92/500
(18,4 %, intervalle 15,10–22,08 %) pour les deux contrôles. Aucune décision
de rejet ne diffère entre les deux versions à quatre dates sur ces 1 500 essais.
Le gain observé vient ici des origines supplémentaires, sans inflation des
faux positifs ; la variante complète apparaît plutôt conservatrice sur ce
tirage. McNemar/Holm sur les deux comparaisons donne p=6,05e-123.

Ces données sont nouvelles et séparées du MMD archivé. Elles ne permettent pas
de déclarer une nouvelle puissance MMD appariée, ni de conclure pour Hawkes.
Le choix de la nouvelle métrique suit l'audit de la première campagne : la
réutilisation ultérieure de ses 11 000 essais sera une extension exploratoire,
pas une confirmation indépendante du choix de métrique.

La limite du passé Hawkes caché subsiste : les prévisions reçoivent uniquement
les événements visibles, pas le passé du burn-in. Les processus sont
synthétiques non marqués. Les modèles appris, quotas et arrêts sont évalués
dans une campagne distincte. Les résultats anciens à T=10 ne sont pas
remplacés silencieusement par ces expériences à T=1.
