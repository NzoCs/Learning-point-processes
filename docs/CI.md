# Contrôles GitHub Actions

Les workflows `ci.yml` et `lint.yml` contrôlent les pushes sur `main`, `master`,
`fix/**`, `features/**` et `migration/**`, ainsi que les PR vers `main` ou `master`.
Une branche de correction est donc contrôlée avant même l'ouverture de sa PR.
Les deux workflows peuvent aussi être lancés manuellement depuis Actions.

La CI ordinaire exécute les tests CPU avec le seuil de couverture de 80 %, les
contrôles pySigLib, le packaging et les plans de lancement Ruche. Elle ne lance
aucun calcul sur Ruche. Le lint vérifie le formatage, les imports, la syntaxe et
les documents notebooks ; mypy et Bandit restent des diagnostics non bloquants
explicitement nommés dans le workflow.

Les intégrations complètes et les campagnes scientifiques restent optionnelles :
lancer leur workflow dédié manuellement ou utiliser le marqueur de commit prévu
dans `ci.yml`. Un push ordinaire ou une PR ne relance pas ces campagnes. Un push
sur une branche avec une PR ouverte produit deux exécutions ordinaires : celle
du push et celle de la PR, qui vérifie aussi l'intégration avec sa branche cible.

## État de départ du correctif de déclenchement

Le commit fusionné `927b402` passe la
[CI sur main](https://github.com/NzoCs/Learning-point-processes/actions/runs/38764275570)
et le [lint](https://github.com/NzoCs/Learning-point-processes/actions/runs/38764275400) :
391 tests réussis, 7 ignorés, 2 désélectionnés, couverture 85,16 %.

Les anciennes notifications d'échec restent dans l'historique GitHub. Les trois
cellules Hawkes/IntensityFree et l'audit numérique Hawkes ont été corrigés avant
cette branche ; leur [rejeu réussi](https://github.com/NzoCs/Learning-point-processes/actions/runs/38759901384)
est documenté dans [le rapport de réparation](PIPELINE_CALIBRATION_REPAIR.md).
Le présent correctif remplace uniquement la liste de trois anciennes branches
`codex/` par les préfixes de travail actuels et ajoute le lancement manuel de la
CI ordinaire. Il ne modifie aucun modèle, référence numérique, seuil de test ou
calcul statistique.
