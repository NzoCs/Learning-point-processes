# Historique des références numériques

## Version 1 — 9 octobre 2026

Première référence pour les onze modèles publics, sur huit séquences locales
de cinq événements avec deux marques. Elle capture l'état corrigé de la branche
`codex/pysiglib-migration`, après le commit de préparation des signatures
`4e7469e` et les corrections d'intégration de ce lot. La provenance des paquets,
de Python, du verrou et l'empreinte des sources sont dans `provenance.json`.

Corrections nécessaires avant de figer les références :

- Import de `Simulator` différé dans le test MMD, pour supprimer un cycle
  rencontré lors de l'import direct du catalogue de modèles.
- Branchement du simulateur sur le tirage de densité existant d'IntensityFree :
  un seul tirage d'intertemps et une marque, sans thinning par intensité.
  Les calculs existants d'entraînement et de prédiction un pas sont conservés.
  La figure d'intensité, indisponible pour ce modèle, est omise ; les figures
  statistiques et les sorties numériques restent exigées.
- Dans SelfCorrecting, sélection également du dernier pas de `sample_dtimes`
  lors du calcul `compute_last_step_only`, pour supprimer un mauvais broadcasting.
- CSV de test/simulation aligné sur l'union des colonnes, au lieu d'ajouter
  des lignes de longueurs différentes sous le premier en-tête.
- Identifiants de séquences Parquet incrémentés par le nombre de séquences,
  et non par le nombre d'événements.

Les anciennes simulations SelfCorrecting/IntensityFree ne terminaient pas ce
parcours commun. Les identifiants et la forme du CSV changent intentionnellement.
Cette référence ne prétend pas reproduire des résultats antérieurs absents du dépôt.
Les futures restructurations du code doivent conserver les snapshots sans les
mettre à jour ; un changement de calcul doit être expliqué ici et évalué séparément.

## Nettoyage des configurations — 9 octobre 2026

La fixture active utilise désormais uniquement `simulation_config.seed` et
`embedding_type: counting_grid`. Les trois paramètres de simulation ignorés
et les deux alias de représentation sont supprimés du code public. Les
snapshots numériques et leur provenance d'origine restent inchangés : cette
adaptation de configuration ne modifie aucun calcul.

L'empreinte de configuration passe de
`7adc6e2718b6ffd58edd572d728283c4b756c4812427fe9ad433124f67aa9162`
à `58bc3241a9813dab282f9abf1936774819dc436c551260be2bee9f1051b49ce9`.
`config_migration.json` enregistre exclusivement cette transition et les champs
modifiés. Le comparateur accepte cette paire précise, contrôle l'empreinte de
la fixture active, puis compare tous les autres champs sans exception. Toute
modification future de graine/configuration ou de résultat continue à échouer.
