#!/bin/bash
# Publie une nouvelle version sur GitHub Releases ; les apps installées la proposeront d'elles-mêmes.
#
#   1. augmentez VERSION dans setup.py (ex. 0.2.1 -> 0.2.2) et commitez
#   2. ./release.sh "Ce qui change dans cette version"
set -e
cd "$(dirname "$0")"

REPO="AyDevPro/MacDictee"
VERSION=$(sed -n 's/^VERSION = "\(.*\)"/\1/p' setup.py)
NOTES="$1"

[ -n "$NOTES" ] || { echo 'Usage : ./release.sh "Notes de version"'; exit 1; }
[ -z "$(git status --porcelain)" ] || { echo "Des changements ne sont pas commités : commitez d'abord."; exit 1; }
if git rev-parse -q --verify "refs/tags/v$VERSION" >/dev/null; then
  echo "La version $VERSION est déjà publiée : augmentez VERSION dans setup.py."; exit 1
fi

./build.sh
git tag "v$VERSION"
git push origin HEAD "v$VERSION"
gh release create "v$VERSION" "dist/MacDictee-$VERSION.dmg" --repo "$REPO" \
  --title "MacDictée $VERSION" --notes "$NOTES"

echo ""
echo "✅ Version $VERSION publiée : https://github.com/$REPO/releases/tag/v$VERSION"
