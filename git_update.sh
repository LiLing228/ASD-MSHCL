#!/bin/bash

set -e

cd "$(dirname "$0")"

MSG=${1:-"update $(date '+%Y-%m-%d %H:%M:%S')"}

echo "Checking git status..."
git status

echo "Staging files..."
git add .

echo "Committing changes..."
git commit -m "$MSG"

echo "Pushing to GitHub..."
git push

echo "Done: $MSG"