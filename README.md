# MacDictée

Dictée vocale et transcription longue durée pour Mac, 100 % locale (aucune donnée envoyée sur internet une fois le modèle téléchargé). Nécessite un Mac Apple Silicon (M1 ou plus récent).

## Installer (utilisateurs)

1. Ouvrez `MacDictee-x.y.z.dmg` et glissez **MacDictée** dans **Applications**.
2. Premier lancement : l'app n'est pas encore signée par Apple, macOS la bloque. Allez dans **Réglages Système > Confidentialité et sécurité**, tout en bas, et cliquez sur **Ouvrir quand même**.
3. La fenêtre MacDictée s'ouvre et vous guide pour les autorisations (Microphone, Accessibilité). Le modèle Whisper (~1,5 Go) se télécharge à la première utilisation ; tout reste ensuite sur le Mac.

Nécessite un Mac Apple Silicon (M1 ou plus récent), macOS 13 ou plus récent.

## Construire le DMG (développeurs)

```bash
./install.sh   # une fois : crée l'environnement Python 3.12
./build.sh     # -> dist/MacDictee-x.y.z.dmg (app autonome, Python inclus)
```

Version : `VERSION` dans `setup.py`. Sans compte développeur Apple, l'app est signée « ad hoc » : chaque nouvelle version oblige à redonner l'autorisation Accessibilité. Avec un compte (99 $/an), `SIGN_ID="Developer ID Application: …" NOTARY_PROFILE=… ./build.sh` signe et notarise (plus d'avertissement au premier lancement, autorisations conservées entre les versions). L'identification des locuteurs (pyannote/torch) n'est pas incluse dans l'app autonome.

## Installation en mode développement

```bash
cd ~/Dev/MacDictee   # sur le disque interne : l'app pointe vers ce dossier
chmod +x install.sh
./install.sh              # ou ./install.sh --locuteurs pour l'identification des personnes
```

Puis ouvrez `~/Applications/MacDictée.app`. Une icône 🎙 apparaît dans la barre de menus.
Le premier lancement télécharge le modèle Whisper (~1,5 Go).

## Autorisations (Réglages Système > Confidentialité et sécurité)

- **Microphone** : pour enregistrer
- **Accessibilité** : pour détecter la touche, repérer le champ de texte actif et coller

La fenêtre *Ouvrir MacDictée…* (menu 🎙) montre l'état des autorisations, avec un bouton pour chacune et un bouton *Relancer*. Elle regroupe aussi l'historique des dictées, les réunions, le vocabulaire, les mots appris et les réglages.

## Utilisation

**Dictée rapide (comme Wispr Flow)** : maintenez la touche **Option droite**, parlez, relâchez.
- Si le curseur est dans un champ de texte : le texte est collé.
- Si aucun champ de texte n'est actif : le texte est seulement copié, vous recevez une notification et vous collez où vous voulez avec Cmd+V.
- Après un collage, votre presse-papiers précédent est remis en place (comme Wispr Flow). Option désactivable dans le menu 🎙 (*Remettre le presse-papiers après collage*) pour garder le texte dicté.
- Le menu 🎙 permet de désactiver le collage automatique (presse-papiers seul).

**Vocabulaire personnalisé** : menu 🎙 > *Modifier le vocabulaire…*, un mot ou nom propre par ligne (prénoms, marques, jargon). Whisper les écrira correctement.

**Apprentissage automatique** : après un collage, MacDictée surveille le champ pendant 45 secondes (ou jusqu'à votre prochaine dictée). Si vous corrigez un mot, il le note. Dès que vous faites **la même correction 2 fois**, il l'apprend : le mot est ajouté au vocabulaire et remplacé automatiquement dans toutes les futures transcriptions (dictée et réunions), avec une notification « Nouveau mot appris ». Menu 🎙 > *Voir les mots appris…* pour consulter ou supprimer une règle (fichier `~/Transcriptions/mots_appris.json`).
- Seuls les remplacements de mots comptent (une reformulation complète est ignorée).
- Ça marche dans les champs dont l'app expose le texte (Notes, Mail, Messages, Slack, champs web classiques…). Dans certains éditeurs (Google Docs, par exemple), la correction n'est pas visible : pas d'apprentissage, mais le reste fonctionne.

**Compte rendu avec Claude** : à la fin d'une réunion, glissez le fichier `~/Transcriptions/...txt` (ou `..._locuteurs.txt` si activé) dans Claude et demandez un compte rendu.

**Enregistrement long (réunions, conversations)** : menu 🎙 > *Démarrer l'enregistrement long*.
- L'audio est sauvegardé en continu, donc aucune limite de durée.
- La transcription se construit au fil de l'eau dans `~/Transcriptions/AAAA-MM-JJ_HHhMM.txt`, avec des horodatages. Les blocs sont coupés sur les silences pour ne pas couper les phrases.
- À l'arrêt, si l'identification des locuteurs est activée, un second fichier `..._locuteurs.txt` est créé (SPEAKER_00, SPEAKER_01…).

Attention : seul le **micro** est enregistré. Pour capter aussi l'autre côté d'un appel (Zoom, Teams, Meet), il faut un périphérique audio virtuel type BlackHole, à choisir comme entrée.

## Identification des locuteurs (optionnel)

1. Installez avec `./install.sh --locuteurs`
2. Créez un compte gratuit sur huggingface.co et acceptez les conditions de `pyannote/speaker-diarization-3.1` et `pyannote/segmentation-3.0`
3. Créez un token d'accès et lancez l'app avec : `HF_TOKEN=hf_xxx open ~/Applications/MacDictée.app`

Les locuteurs sont séparés mais pas nommés.

## Réglages

Variables d'environnement : `WHISPER_LANG` (`fr` par défaut, vide pour détection auto), `WHISPER_MODEL` (par défaut `mlx-community/whisper-large-v3-turbo`, mettre `mlx-community/whisper-small-mlx` pour plus de rapidité sur un vieux Mac).

## Test rapide sans construire l'app

```bash
source .venv/bin/activate && python app.py
```
