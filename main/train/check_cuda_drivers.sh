#!/bin/bash
#SBATCH --output=err_logs/cuda_check_%A.out
#SBATCH --error=err_logs/cuda_check_%A.err
#SBATCH --partition=gpua100
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1

echo "=== Diagnostic complet des drivers CUDA ==="
echo "Date: $(date)"
echo "Node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"

# Informations système
echo -e "\n--- Informations système ---"
uname -a
cat /etc/os-release | head -3

# Vérifier les drivers NVIDIA
echo -e "\n--- Drivers NVIDIA ---"
if command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi version:"
    nvidia-smi --version
    echo -e "\nnvidia-smi output:"
    nvidia-smi
    echo -e "\nDétails GPU:"
    nvidia-smi --query-gpu=name,driver_version,cuda_version,memory.total,memory.free --format=csv
else
    echo "ERREUR: nvidia-smi non trouvé dans PATH"
    echo "PATH actuel: $PATH"
fi

# Vérifier les bibliothèques CUDA
echo -e "\n--- Bibliothèques CUDA système ---"
if [ -d "/usr/local/cuda" ]; then
    echo "CUDA toolkit trouvé dans /usr/local/cuda"
    ls -la /usr/local/cuda/version.txt 2>/dev/null || echo "version.txt non trouvé"
else
    echo "Pas de CUDA toolkit dans /usr/local/cuda"
fi

# Rechercher les bibliothèques libcuda
echo -e "\n--- Recherche libcuda ---"
find /usr -name "*libcuda*" 2>/dev/null | head -10 || echo "libcuda non trouvé"

# Variables d'environnement
echo -e "\n--- Variables d'environnement CUDA ---"
env | grep -i cuda | sort

# Tester Python et PyTorch
echo -e "\n--- Test Python/PyTorch ---"
module purge
source /gpfs/workdir/regnaguen/LTPP/bin/activate

python -c "
import sys
print(f'Python: {sys.version}')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'PyTorch CUDA compiled version: {torch.version.cuda}')
    print(f'PyTorch cuDNN version: {torch.backends.cudnn.version()}')
    
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA device count: {torch.cuda.device_count()}')
        try:
            print(f'Current device: {torch.cuda.current_device()}')
            print(f'Device name: {torch.cuda.get_device_name()}')
            cap = torch.cuda.get_device_capability()
            print(f'Device capability: {cap}')
            
            # Test allocation mémoire simple
            x = torch.randn(100, 100).cuda()
            print('Test allocation GPU: OK')
            
        except RuntimeError as e:
            print(f'ERREUR RuntimeError: {e}')
        except Exception as e:
            print(f'ERREUR autre: {e}')
    else:
        print('CUDA non disponible')
        
except ImportError as e:
    print(f'Erreur import PyTorch: {e}')
except Exception as e:
    print(f'Erreur inattendue: {e}')
"

echo -e "\n=== Fin diagnostic ==="
