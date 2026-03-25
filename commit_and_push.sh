#!/bin/bash
# Usage: ./commit_and_push.sh "commit message"
# Example: ./commit_and_push.sh "feat: add symptom checker screen"

if [ -z "$1" ]; then
  echo "Error: commit message required"
  echo "Usage: ./commit_and_push.sh \"commit message\""
  exit 1
fi

git add -A
git commit -m "$1"
git push origin main
echo "Pushed to GitHub: $1"
