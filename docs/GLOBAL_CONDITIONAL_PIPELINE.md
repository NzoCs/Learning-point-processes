# MMD conditionnel global dans la pipeline

Le mode `simulation_config.conditional_evaluation` utilise désormais une seule
statistique sur toutes les paires de futurs, avec le kernel de signature.
L'agrégation historique par somme des MMD de batches reste un diagnostic local,
mais ne détermine plus `pooled_mmd_p_value`.

## Calcul et identité

Pour chaque séquence `i`, `X_i` est le futur tenu à l'écart et `Y_i` est une
simulation depuis le même historique. Le modèle reste figé. On prépare les
chemins de tous les historiques ensemble, puis les chemins de tous les futurs
`X` et `Y` ensemble. La normalisation ne dépend pas des blocs de calcul.

Avec le kernel produit histoire/futur, la statistique est

```text
D_ij = K(X_i,X_j) + K(Y_i,Y_j) - K(X_i,Y_j) - K(Y_i,X_j)
T(s) = sum_{i != j} s_i D_ij s_j / [N(N-1)]
```

L'observation utilise `s_i=1`. Chaque tirage nul échange indépendamment `X_i` et
`Y_i`, soit un signe par identifiant stable, sur toutes les séquences.
La p-value conserve la correction plus-un et les égalités numériques conservatrices.
Tous les termes entre batches sont inclus. Les identifiants sont triés avant les
tirages globaux ; leur absence, leur réutilisation ou un mauvais appariement sont
rejetés. Une dernière batch de taille un est acceptée ; il faut au moins deux
paires dans l'évaluation entière. Aucun résultat local fictif n'est créé pour
cette batch.

`signature_max_batch` borne aussi les dimensions des blocs de Gram globaux et
des diagnostics locaux, à 64 par défaut. Les observations sont copiées sans
gradient sur CPU pour éviter de conserver les graphes d'entraînement. Les chemins
préparés et les contractions peuvent être évalués sur le device de la simulation.
La mémoire ne contient pas de matrice complète `N × N` ; elle contient néanmoins
les événements et chemins de toutes les séquences et `n_samples × N` signes.
Le temps reste quadratique en nombre de paires. CUDA n'est pas validé dans ce lot.

## Comportement conservé et changements attendus

- Les paramètres, pertes, prédictions et algorithmes de simulation ne changent pas.
- FullyNN et RMTPP gardent tous leurs futurs jusqu'au quota, y compris les futurs
  vides ou courts avec un arrêt permanent explicite. Le kernel sur le statut
  terminal est conservé.
- Les autres modèles gardent leur observation conditionnelle à troncature
  symétrique par paire. Ce lot ne remplace pas les quotas par une fenêtre fixe.
- Les diagnostics ordinaires sans `conditional_evaluation` restent inchangés.
- La p-value globale et sa statistique changent volontairement : davantage de
  termes et une normalisation commune remplacent la somme des statistiques locales.
- Les diagnostics locaux restent dans `p_values`, `observed_statistic` et
  `permuted_statistic`. Leur calcul par blocs peut seulement changer l'arrondi.

Les graines des permutations sont isolées de celles des trajectoires. L'ajout
d'observations invalide le cache global ; relire un résultat final ne refait pas
de tirages. Un reset efface aussi les observations et les identifiants. Les
erreurs de finalisation sont propagées, sans remplacement silencieux par une
p-value absente.

`simulation_results/mmd_global_statistics.json` conserve les statistiques locales
et globales, les tirages nuls, les identifiants, la graine et la taille des blocs.
Le CSV expose `global_mmd_value` et `pooled_mmd_p_value`. Le manifeste donne le
protocole `global_paired_u_mmd_pooled_signature_blocks_v1`.
Les graphes et les moyennes de batches restent des diagnostics locaux.

## Validation

La référence dense publique contrôle chaque permutation pour trois observations :
troncature par paire, quota avec arrêt et fenêtre fixe. Les tests changent l'ordre
et le découpage des batches, contrôlent un singleton, les masques, les deux marques,
le statut terminal, la borne des appels de Gram, le RNG, les identifiants et le reset.

Le workflow optionnel `.github/workflows/global-pipeline.yml` compare les onze
modèles historiques et Poisson au parent revu
`5b41f13fb4eee9abba4754df45959cb5a1b0c46a`, sur le même runner Linux et le même
environnement verrouillé. Les splits d'entraînement, validation et évaluation
utilisent des graines indépendantes et des identifiants disjoints. Il contrôle
aussi deux exécutions identiques, le replay du checkpoint, la reprise à une
frontière d'époque et le redémarrage complet après interruption de prédiction.
FullyNN et RMTPP ont un cas supplémentaire avec des arrêts immédiats connus.

```bash
python scripts/global_pipeline_validation.py --capture \
  --code-root artifacts/global-pipeline-before-code --output artifacts/global-pipeline-before
python scripts/global_pipeline_validation.py \
  --reference-dir artifacts/global-pipeline-before --output artifacts/global-pipeline-after
```

Le premier checkout doit être celui du parent revu. Utiliser des dossiers neufs.
La campagne Linux se déclenche manuellement ou avec le marqueur `[global-pipeline]`.

Cette intégration logicielle n'est pas une nouvelle calibration scientifique des
onze modèles. La [validation scientifique précédente](GLOBAL_SIGNATURE_CALIBRATION.md)
porte sur des fenêtres complètes `T=2`, quatre checkpoints figés de FullyNN/RMTPP
et deux marques. Ses taux de rejet et de puissance ne sont pas automatiquement
transférables à tous les quotas, historiques ou paramètres. L'indépendance de
l'entraînement reste une condition du protocole, pas une preuve que la pipeline
peut inférer d'un checkpoint quelconque. Ruche, CUDA et DDP restent à valider.
