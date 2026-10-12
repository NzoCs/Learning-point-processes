# Profils de poids Linux

La première exécution Linux complète, commit `883ae826`, run GitHub `38236030823`,
a exécuté les onze pipelines. Huit références Windows passent entièrement.
ANHN, ANHP et THP diffèrent uniquement dans les poids entraînés au-delà des
tolérances historiques `rtol=1e-5`, `atol=1e-6`.

Tous les autres champs des onze snapshots passent ces mêmes tolérances contre
les références Windows v2 : pertes, gradients de contrôle, intensités, métriques,
trajectoires simulées, statistiques et empreintes des entrées/protocoles.
Les écarts maximaux de poids pour ANHN/ANHP/THP sont respectivement
6.68e-6, 1.43e-5 et 6.83e-6 ; les sorties contrôlées diffèrent de moins de 5e-7.
Ces écarts sont concentrés notamment dans les biais de projection des clés de
l'attention. L'explication précise de chaque écart n'est pas établie.

PyTorch [ne garantit pas l'identité des résultats entre plateformes](https://docs.pytorch.org/docs/2.8/notes/randomness.html).
Les trois fichiers capturent donc seulement les poids du premier run Linux ;
le reste du contrat observable continue de comparer les résultats aux références
Windows historiques. Aucun seuil n'est élargi, aucun champ n'est exclu, aucun
calcul de modèle n'est changé. Les profils sont liés au modèle et aux empreintes
de configuration et de données. Les références Windows demeurent intactes.

La promotion doit être confirmée par une seconde exécution Linux indépendante.
Sa provenance et son résultat sont enregistrés dans `provenance.json`.
Un futur changement de poids Linux sera toujours détecté aux tolérances initiales.
