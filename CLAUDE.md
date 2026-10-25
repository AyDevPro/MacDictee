# MacDictée : contexte pour Claude Code

Alternative maison, 100 % locale, à Wispr Flow (trop cher pour l'utilisateur : 15 €/mois). App Mac dans la barre de menus, Python, Apple Silicon.

## ⚠️ État : écrit sans jamais avoir été exécuté sur un Mac
Tout le code a été rédigé depuis un environnement Linux. Seuls la syntaxe et la logique pure d'apprentissage (`learn_from_edit`, `apply_rules`) ont été testées (avec des modules macOS simulés). **Premier objectif : lancer `python app.py` sur le Mac, corriger tout ce qui casse, puis vérifier chaque fonction ci-dessous à la main.**

## Ce qui existe (`app.py`)
- **Dictée** : touche Option droite maintenue -> enregistrement -> transcription (mlx-whisper, `whisper-large-v3-turbo`) -> collage via Cmd+V.
- **Champ actif** : détection via l'API Accessibilité (`AXFocusedUIElement`, rôle / `AXValue` modifiable / `AXSelectedTextRange`). Si aucun champ de texte : on copie seulement dans le presse-papiers + notification. Le texte reste toujours dans le presse-papiers. `AXManualAccessibility` est activé pour Chrome/Electron.
- **Enregistrement long** : WAV écrit en continu dans `~/Transcriptions`, transcription par blocs de 20 à 45 s coupés sur les silences (RMS), fichier `.txt` horodaté au fil de l'eau.
- **Locuteurs (optionnel)** : pyannote 3.1 après coup (`HF_TOKEN`), fichier `_locuteurs.txt`.
- **Vocabulaire** : `~/Transcriptions/vocabulaire.txt` (un mot par ligne) injecté dans `initial_prompt` (600 derniers caractères).
- **Apprentissage** : après un collage, `_watch_edits` compare le texte du champ (avant/après collage, puis jusqu'à 45 s plus tard ou prochaine dictée). Un remplacement de mot répété 2 fois devient une règle de remplacement automatique + entrée de vocabulaire (`~/Transcriptions/mots_appris.json`).
- Filtres anti-hallucinations Whisper (silence, `no_speech_prob`, phrases type « Sous-titres… »), verrou autour du modèle MLX.

## Prochaine grosse fonction : capter le son système (réunions Zoom/Meet/Teams)
Objectif : comme le Notetaker de Wispr Flow (sans bot dans l'appel). Aujourd'hui seul le micro est enregistré.
- Piste recommandée : petit binaire **Swift** (ScreenCaptureKit, audio système, macOS 13+ ; ou Core Audio process taps, macOS 14.2+) qui envoie du PCM 16 kHz mono (stdout ou fichier). `app.py` le lance en sous-processus pendant l'enregistrement long.
- Garder **deux pistes séparées** (micro = « moi », son système = « les autres »). Ça donne gratuitement la séparation vous / les autres, puis pyannote seulement sur la piste système.
- Autorisation à gérer : « Enregistrement de l'écran et audio système ». Prévoir un message clair si elle manque.
- Plan B plus simple : BlackHole + périphérique audio multiple (à documenter).

## Autres pistes (par priorité)
1. Corriger le WAV en cas de plantage (l'en-tête n'est finalisé qu'à la fermeture : réparer avec ffmpeg ou mettre à jour l'en-tête périodiquement).
2. Tests automatisés pour `learn_from_edit`, `apply_rules`, la logique de découpage sur silence.
3. Raccourci configurable (aujourd'hui Option droite codée en dur), réglages persistants (langue, modèle, collage auto) au lieu de variables d'environnement.
4. Résumé/compte rendu : l'utilisateur envoie pour l'instant le `.txt` à Claude à la main. Option future : bouton « Compte rendu » via l'API Anthropic.
5. Si le projet grossit : réécriture en app Swift native (distribution plus simple, permissions plus fiables qu'avec py2app).

## Pièges connus
- Permissions macOS liées à l'app/au binaire : Microphone, Accessibilité, Surveillance de l'entrée (pynput). En mode `py2app -A` (alias) ou en lançant `python app.py` depuis le Terminal, la permission peut s'attacher au Terminal ou à Python plutôt qu'à l'app.
- `AXValue` n'est pas exposé partout (Google Docs, certains éditeurs web) : l'apprentissage est alors impossible, le reste fonctionne.
- MLX ne doit pas être appelé en parallèle : tout passe par `_model_lock`.

## Décisions du propriétaire
100 % local (pas de cloud pour la transcription), Apple Silicon uniquement, français par défaut (`WHISPER_LANG`), fichiers dans `~/Transcriptions`, un mot est « appris » après 2 corrections identiques.
