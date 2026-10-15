# Tester un clone neuf sur Ruche

Cette procédure prépare la validation d'installation et d'exécution. Elle ne
certifie pas encore la reproductibilité scientifique. La branche de migration
utilise pySigLib 4.0.0 ; le profil GPU exige son plugin CUDA 4.0.0.

## Espaces de stockage

Ruche fournit deux espaces de stockage personnels, pas deux environnements
Python prédéfinis. Selon la [documentation officielle](https://mesocentre.pages.centralesupelec.fr/user_doc/ruche/03_connection_and_file_transfer/),
consultée le 9 octobre 2026 :

| Espace | Chemins équivalents | Quota annoncé | Usage de cette procédure |
| --- | --- | --- | --- |
| `$HOME` | `/home/<login>` ou `/gpfs/users/<login>` | 50 Go | Clone Git, scripts et environnement Python. |
| `$WORKDIR` | `/workdir/<login>` ou `/gpfs/workdir/<login>` | 500 Go | Caches volumineux, données, logs, checkpoints et résultats. |

Les deux espaces sont accessibles depuis les nœuds de calcul. Vérifier les
quotas effectifs du compte avec `ruche-quota`. Les chemins ci-dessous sont une
organisation proposée ; ils ne réutilisent aucun ancien environnement personnel.

## 1. Télécharger et installer

Se connecter avec son propre compte SSH et choisir un espace de travail autorisé.
Créer un nouveau clone et un environnement identifié hors du clone. Exporter
les mêmes variables pour l'installation et la soumission :

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/envs/ltpp-pysiglib"
export UV_CACHE_DIR="$WORKDIR/cache/uv"
export HF_HOME="$WORKDIR/cache/huggingface"
export LTPP_OUTPUT_ROOT="$WORKDIR/ltpp-artifacts"
mkdir -p "$HOME/src" "$HOME/envs" "$WORKDIR/ltpp-logs"
cd "$HOME/src"
git clone --branch codex/pysiglib-migration https://github.com/NzoCs/Learning-point-processes.git Learning-point-processes-pysiglib
cd Learning-point-processes-pysiglib
git rev-parse HEAD
git status --porcelain
sha256sum pyproject.toml uv.lock
uv --version
uv sync --frozen --python 3.11 --no-default-groups --extra ruche \
  --no-build-package pysiglib --no-build-package pysiglib-cuda
uv pip check
uv run --frozen --no-sync new-ltpp run --help
```

Les modifications décrites ici doivent être disponibles dans la référence clonée.
Pour valider une branche de développement publiée, cloner cette branche avec
`git clone --branch NOM_BRANCHE URL` et enregistrer son SHA exact.

Python **3.11** est requis. `uv` doit être installé dans l'espace utilisateur ou
fourni par un module du cluster. Noter les modules chargés avec `module list`.
La procédure impose des wheels pour pySigLib et le plugin CUDA. Le wheel CPU
Linux de pySigLib 4.0.0 exige au moins glibc 2.27 ; vérifier `ldd --version`,
l'architecture x86_64 et le pilote GPU avant de déclarer le profil compatible.
En cas d'incompatibilité, préparer une recette de conteneur versionnée plutôt
qu'un retour CPU silencieux. D'autres dépendances, notamment `fastdtw`, peuvent
encore nécessiter une compilation selon la plateforme. L'installation Linux/Ruche
reste à valider. Ne pas installer des dépendances au début de chaque job.

## 2. Préparer les données

Le preset `test` utilise `NzoCs/test_dataset` sur Hugging Face. Précharger les
données dans le cache de ce nouveau parcours avant soumission, sur une machine
où l'accès réseau et ce téléchargement sont autorisés :

```bash
uv run --frozen --no-sync python -c 'from datasets import load_dataset; d = load_dataset("NzoCs/test_dataset"); print(d)'
```

Si `HF_HOME` est personnalisé, conserver le même chemin accessible aux nœuds de
calcul. Relever la révision et les comptages des splits pour le bilan. La révision
des données n'est pas encore verrouillée dans le projet : ce préchargement est
une vérification d'accès, pas une garantie d'identité scientifique des données.

## 3. Vérifier les commandes avant de soumettre

Depuis la racine du clone :

```bash
bash -n scripts/bash/ruche_common.sh
bash -n scripts/bash/smoke_ruche_gpu.sh
bash -n scripts/bash/run_all_pipeline.sh
bash -n scripts/bash/train_ruche_cpu.sh
bash scripts/bash/smoke_ruche_gpu.sh --dry-run
bash scripts/bash/run_all_pipeline.sh --dry-run
bash scripts/bash/train_ruche_cpu.sh --dry-run
```

Les dry-runs affichent respectivement 1, 3 et 28 commandes et ne réservent aucune
ressource. Les indices hors grille sont rejetés. Les scripts n'activent aucun
environnement personnel prédéfini : ils utilisent `UV_PROJECT_ENVIRONMENT`, ou
le `.venv` du clone si cette variable est absente, avec le lock figé. Ils
conservent les modules choisis pour cette installation.

## 4. Petit job GPU

Vérifier les ressources et la partition disponibles pour le compte avant de
soumettre. Créer le dossier de logs **avant** `sbatch`, car Slurm ouvre ces fichiers
avant l'exécution du script :

```bash
mkdir -p err_logs
sbatch --output="$WORKDIR/ltpp-logs/smoke_gpu_%j.out" \
  --error="$WORKDIR/ltpp-logs/smoke_gpu_%j.err" scripts/bash/smoke_ruche_gpu.sh
```

Le job demande un GPU, deux CPU, 8 Go de RAM hôte et 15 minutes au maximum.
Il exécute d'abord un petit Gram avec le backend de signature réel sur CUDA et
échoue si CUDA est indisponible, si le résultat revient sur CPU ou si le Gram
contient des valeurs non finies. Il lance ensuite NHP sur `test` pour une epoch,
en phase `train`. Il ne lance ni campagne, ni simulation complète.

Ce contrôle est aussi disponible sur un nœud de calcul alloué :

```bash
uv run --frozen --no-sync python -m scripts.check_backend --device cuda
```

Le profil CPU utilise `--device cpu` et masque les GPU aux processus du job.
Respecter `CUDA_VISIBLE_DEVICES` fourni par Slurm dans le profil GPU.

Les sorties du launcher sont isolées par profil, job et tâche sous
`artifacts/ruche/`. Pour utiliser le stockage de travail du cluster :

```bash
export LTPP_OUTPUT_ROOT="$WORKDIR/ltpp-artifacts"
```

Soumettre depuis la racine du clone. Si un autre répertoire de soumission est
nécessaire, exporter `LTPP_REPO_DIR` avec le chemin absolu de ce clone. Pour les
campagnes uniquement, `LTPP_PHASE` peut valoir `train`, `test`, `predict` ou `all`,
et `LTPP_EPOCHS` fixe un entier positif. Le smoke test impose train et une epoch.

## 5. Vérifier le résultat

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

Inspecter le log `$WORKDIR/ltpp-logs/smoke_gpu_JOB_ID.out`, son fichier d'erreurs, la matrice
Gram, le GPU identifié, les métriques et le checkpoint produit. Conserver la
commande, le SHA, les hashes du lock et des données, les versions Python/PyTorch,
les modules et les ressources demandées. Un exit code nul seul ne prouve pas
la validité du protocole MMD.

Après ce premier succès : valider test/predict, le choix du checkpoint, les
simulations, puis deux répétitions et une reprise. Ces étapes restent ouvertes
dans le [rapport de reproductibilité](../rapports/REPRODUCTIBILITE_RUCHE.md).
