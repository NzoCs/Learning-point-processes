# Installation neuve et première exécution sur Ruche

Le 15 octobre 2026, un clone propre de `main`, commit
`63053beb55cc6d4a687b5d5f75746ae3d4f6d6ed`, a été installé sans réutiliser
l'environnement personnel existant. Code sous `$HOME/src`, environnement,
caches, données et sorties sous `$WORKDIR/ltpp-validation/20261015-main-63053be`.

## Installation et données

- Python 3.11.15 ; uv 0.11.29 ; installation `uv sync --frozen --no-default-groups
  --extra ruche --no-build-package pysiglib --no-build-package pysiglib-cuda`.
- PyTorch 2.8.0+cu128 ; pySigLib et plugin CUDA 4.0.0 ; glibc 2.28.
- Installation, `uv pip check`, CLI et Gram CPU réussis.
- Données `NzoCs/test_dataset`, révision
  `1ca2a24f16f377555175c99e5cc1704dc44b92db` : 6/2/2 séquences train/validation/test.
  L'entraînement GPU a utilisé le cache en mode hors ligne.

## GPU et problèmes observés

Le job direct `2074168`, sur `ruche-gpu15`, termine avec `COMPLETED`, code `0:0` :
Gram de signature sur une NVIDIA A100-SXM4-40GB, puis une époque NHP réellement
sur CUDA. Un checkpoint `last.ckpt` et le manifeste d'entraînement sont produits.
Le pilote est 580.167.08. Le Gram CPU/CUDA diffère au maximum de `2.84e-14`.
Cela valide ce petit parcours, pas les douze modèles ou DDP sur GPU.

Deux incidents sont conservés dans les logs :

1. Sur `ruche-gpu16`, job `2074161`, l'initialisation CUDA échoue. Un diagnostic
   séparé (`2074163`) retrouve `cuInit(0) == 999`, sans importer le dépôt.
   Le diagnostic attrape les exceptions : son exit code nul ne signifie pas un
   succès CUDA. Le même environnement calcule ensuite le Gram sur `ruche-gpu19`.
   Cette preuve oriente vers l'allocation ou le nœud ; elle ne certifie pas sa
   cause matérielle et ne justifie aucune modification des modèles.
2. Le launcher initial reste en attente dans `srun uv run` après l'impression
   du Gram, avec un enfant Python défunt (job `2074164`). Les jobs en échec ou
   bloqués ont été annulés. L'appel direct à l'interpréteur installé permet au
   contrôle CUDA et à l'entraînement d'aboutir (`2074168`). Des sondes limitées
   à 60 secondes ont expiré pour les deux modes : elles ne prouvent pas à elles
   seules un défaut de uv. La cause interne du blocage reste non isolée.

Le correctif du launcher appelle donc l'interpréteur de l'environnement
explicitement choisi, après contrôle de sa présence. Le lock reste utilisé à
l'installation. Aucun paquet ni paramètre de modèle n'est changé dans le job.
Les tests du launcher vérifient l'environnement dont le chemin contient des
espaces, l'absence de dépendance à uv à l'exécution et la propagation d'un échec
du backend avant entraînement.

## Campagne scientifique distincte

Le pilote technique `2074170` (FullyNN, 256 historiques) termine en code `0:0`.
La campagne `2074172` couvre les onze modèles historiques et Poisson, deux
quotas (3/6 événements) et deux cohortes indépendantes par modèle/quota :
48 tâches CPU, au maximum quatre en parallèle, 1 CPU/4 Go/4 heures par tâche.
L'agrégation `2074177` dépend de leur terminaison, y compris en cas d'échec.

Chaque cohorte utilise 256 historiques, deux marques, 1 000 essais H0 et 250
essais par alternative H1 ; deux cohortes donnent 2 000 H0 et 500 H1 par
alternative. Avec les contrôles Poisson et alternatives d'arrêt, cela représente
72 000 essais H0 et 28 000 H1. Les paramètres restent figés aux checkpoints
archivés et le test utilise 99 tirages nuls, au seuil 5 %. Les 72 contrôles H0 par
cohorte et 36 contrôles groupés ont un seuil familial d'excès `0.01/108` déclaré
avant exécution. Les graines de cohortes sont 2026101601 et 441726.

Cette campagne est en cours ; aucun résultat de calibration n'est revendiqué.
Elle utilise le module existant `scripts.pipeline_protocol_calibration`, avec
les tailles de batches `(64, 63, 64, 64, 1)` pour les deux quotas. Le plan, les
drivers hachés, les tirages et rapports sont conservés dans
`$WORKDIR/ltpp-validation/20261015-main-63053be/campaign-12models`.

Limites : un état appris figé et deux marques par modèle ; pas d'incertitude
d'estimation couverte. Les références paramétriques/à compensateur ont des
oracles indépendants. Pour six autres modèles neuronaux, le H0 appris est
interne et complété par un contrôle Poisson indépendant : cela ne certifie pas
le sampler neuronal non constant. La calibration CPU et la validation logicielle
CUDA sont deux preuves distinctes.
