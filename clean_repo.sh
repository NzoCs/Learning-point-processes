#!/bin/bash

set -e  # Arrêter le script si une erreur se produit

echo "🔍 Vérification de git-filter-repo..."

if ! command -v git-filter-repo &> /dev/null
then
    echo "❌ git-filter-repo n'est pas installé. Installation via pip..."
    pip install git-filter-repo
fi

echo "✅ git-filter-repo est prêt."

echo "📁 Nettoyage des fichiers volumineux de l'historique Git..."

# Liste des fichiers à supprimer
files_to_remove=(
    "*.json"
)

for file in "${files_to_remove[@]}"
do
    echo "🧹 Suppression de $file de l'historique..."
    git filter-repo --path "$file" --invert-paths --force
    echo "✅ $file supprimé de l'historique."
done

echo "🚀 Push forcé vers GitHub (attention, l'historique sera réécrit)..."
