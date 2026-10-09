# Tester un clone neuf sur Ruche

Cette procédure prépare la validation d'installation et d'exécution. Elle ne
certifie pas encore la reproductibilité scientifique. Le backend actuel reste
`sigkernel` ; la migration pySigLib sera validée séparément.

## 1. Télécharger et installer

Se connecter avec son propre compte SSH et choisir un espace de travail autorisé.
Créer un nouveau répertoire, sans réutiliser de `.venv` ou checkpoint existant :

```bash
git clone https://github.com/NzoCs/Learning-point-processes.git
cd Learning-point-processes
git rev-parse HEAD
git status --porcelain
sha256sum pyproject.toml uv.lock
uv --version
uv sync --frozen --python 3.11
uv pip check
uv run --frozen --no-sync new-ltpp run --help
```

Les modifications décrites ici doivent être disponibles dans la référence clonée.
Pour valider une branche de développement publiée, cloner cette branche avec
`git clone --branch NOM_BRANCHE URL` et enregistrer son SHA exact.

Python **3.11** est requis. `uv` doit être installé dans l'espace utilisateur ou
fourni par un module du cluster. Noter les modules chargés avec `module list`.
Les dépendances natives, dont `sigkernel` et `fastdtw`, peuvent nécessiter une
compilation et une toolchain compatible. L'installation Linux/Ruche reste à
valider ; le problème de compilation Windows de `sigkernel` n'est pas résolu
par ces corrections. Ne pas installer des dépendances au début de chaque job.

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
environnement personnel et utilisent exclusivement le `.venv` du clone avec le
lock figé. Ils conservent les modules choisis pour cette installation.

## 4. Petit job GPU

Vérifier les ressources et la partition disponibles pour le compte avant de
soumettre. Créer le dossier de logs **avant** `sbatch`, car Slurm ouvre ces fichiers
avant l'exécution du script :

```bash
mkdir -p err_logs
sbatch scripts/bash/smoke_ruche_gpu.sh
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

Inspecter le log `err_logs/smoke_gpu_JOB_ID.out`, son fichier d'erreurs, la matrice
Gram, le GPU identifié, les métriques et le checkpoint produit. Conserver la
commande, le SHA, les hashes du lock et des données, les versions Python/PyTorch,
les modules et les ressources demandées. Un exit code nul seul ne prouve pas
la validité du protocole MMD.

Après ce premier succès : valider test/predict, le choix du checkpoint, les
simulations, puis deux répétitions et une reprise. Ces étapes restent ouvertes
dans le [rapport de reproductibilité](../rapports/REPRODUCTIBILITE_RUCHE.md).
