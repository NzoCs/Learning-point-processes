#!/bin/bash
echo "Cleaning Git cache..."

# Remove files from the index
git rm -r --cached .

# Add all files back (except those in .gitignore)
git add .

# Status check
echo ""
echo "Current git status:"
git status

echo ""
echo "To complete the process, commit the changes with:"
echo "git commit -m \"Clean Git cache and update .gitignore\""
