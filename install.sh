#!/bin/bash
# Installation de MacDictée : crée l'environnement, installe les dépendances, construit l'app.
set -e
cd "$(dirname "$0")"

command -v brew >/dev/null || { echo "Homebrew est requis : https://brew.sh"; exit 1; }
brew list portaudio >/dev/null 2>&1 || brew install portaudio
brew list ffmpeg >/dev/null 2>&1 || brew install ffmpeg

python3 -m venv .venv
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
