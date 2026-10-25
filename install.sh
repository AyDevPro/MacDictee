#!/bin/bash
# Installation de MacDictée : crée l'environnement, installe les dépendances, construit l'app.
set -e
cd "$(dirname "$0")"

command -v brew >/dev/null || { echo "Homebrew est requis : https://brew.sh"; exit 1; }
brew list portaudio >/dev/null 2>&1 || brew install portaudio
brew list ffmpeg >/dev/null 2>&1 || brew install ffmpeg
# Python 3.12 : version la mieux supportée par mlx, pyobjc et torch (le python3 système peut être trop récent)
brew list python@3.12 >/dev/null 2>&1 || brew install python@3.12
PY="$(brew --prefix python@3.12)/bin/python3.12"

[ -d .venv ] && ! .venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12)' 2>/dev/null && rm -rf .venv
"$PY" -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Option : identification des locuteurs
if [ "$1" = "--locuteurs" ]; then
  pip install pyannote.audio
fi

# Construit MacDictée.app (mode alias = rapide, utilise le venv du dossier)
python setup.py py2app -A
rm -rf "$HOME/Applications/MacDictée.app"
mkdir -p "$HOME/Applications"
cp -R dist/MacDictée.app "$HOME/Applications/"

echo ""
echo "✅ Installé dans ~/Applications/MacDictée.app"
echo "Au premier lancement, autorisez : Microphone, Accessibilité et Surveillance de l'entrée"
echo "(Réglages Système > Confidentialité et sécurité)."
