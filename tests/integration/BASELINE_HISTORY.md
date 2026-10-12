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

## Référence v2 — troncature recalculée, 12 octobre 2026

À la demande de l'utilisateur après comparaison de calibration, le seul
protocole de production est désormais la permutation des séquences brutes,
avec troncature par paires et préparation du noyau recalculées pour chaque
tirage. Le nul construit par simulations supplémentaires a été supprimé.
`n_samples` désigne exactement le nombre de permutations. Les p-values sont
en float64, avec correction +1 et traitement conservateur des égalités proches.

Le rééchantillonnage dispose de flux persistants CPU/CUDA séparés, dérivés de
la graine initiale par addition de `0x4D4D44` modulo `2**63`. L'état des
générateurs de trajectoires est restauré après chaque calcul statistique.
Cette correction supprime le couplage historique entre le nombre de tirages
du nul et les trajectoires du batch suivant. Les anciennes simulations
suivantes et leurs histogrammes/statistiques changent donc intentionnellement.

Les onze parcours train/test/predict ont été exécutés. Avant toute mise à jour,
un contrôle complet a vérifié l'égalité exacte des poids entraînés, pertes,
gradients, intensités, métriques de test et trajectoires du premier batch.
Les changements des autres colonnes ont été limités aux statistiques dérivées
des simulations et aux résultats MMD. L'audit et les empreintes des anciennes
références figurent dans
`docs/validation/truncation-reference-migration-2026-10-12.json`.

La version précédente reste accessible dans Git au commit `5156aa2` ; elle
n'est pas présentée comme une référence valable pour ce nouveau calcul.
Les snapshots v2 ajoutent le protocole statistique et enregistrent l'empreinte
de la configuration active. Le comparateur conserve ses tolérances d'origine
et compare tous les champs ; aucune exception sur les résultats n'est ajoutée.
Une nouvelle restructuration doit désormais conserver ces références v2.

La calibration sur Poisson/Hawkes indépendants ne certifie pas à elle seule
les p-values de modèles ajustés ou de simulations conditionnelles, ni leur
agrégation entre batches. Ruche/CUDA/DDP restent à valider séparément.

## 12 octobre 2026 — profils CPU Linux confirmés

La première exécution complète Linux du code `883ae826` a terminé les onze
pipelines. Huit snapshots passent les références Windows v2 ; pour ANHN, ANHP
et THP, certains poids entraînés dépassent les tolérances initiales. Tous les
autres champs des onze modèles passent les références historiques, notamment
les pertes, gradients sondés, intensités, métriques et simulations.

Les profils `linux_trained_states` ne contiennent que les poids capturés des
trois modèles et leurs empreintes d'entrées. Le comparateur continue de
vérifier tous les autres champs contre Windows, sans élargir les tolérances.
Aucun calcul de modèle et aucune référence Windows n'est modifié. L'explication
précise des écarts interplateformes n'est pas établie.

La seconde exécution Linux indépendante du code `6388706`, run `38237687152`,
confirme les onze modèles et la concordance de tous les champs des snapshots
avec le premier passage Linux. L'archive a été téléchargée et son SHA256
vérifié ; l'audit et la provenance sont dans
`tests/integration/linux_trained_states/provenance.json`.
