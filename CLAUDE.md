# MacDictée : contexte pour Claude Code

Alternative maison, 100 % locale, à Wispr Flow (trop cher pour l'utilisateur : 15 €/mois). App Mac dans la barre de menus, Python, Apple Silicon.

## ⚠️ État : écrit sans jamais avoir été exécuté sur un Mac
Projet installé dans `~/Dev/MacDictee` (disque interne, venv Python 3.12) : l'app py2app en mode alias pointe vers ce dossier.
Tout le code a été rédigé depuis un environnement Linux. Seuls la syntaxe et la logique pure d'apprentissage (`learn_from_edit`, `apply_rules`) ont été testées (avec des modules macOS simulés). **Premier objectif : lancer `python app.py` sur le Mac, corriger tout ce qui casse, puis vérifier chaque fonction ci-dessous à la main.**

## Optimisation (0.6.0, mesurée)
- **Modèle** : large-v3-turbo converti en **8 bits sur le Mac** au premier lancement (`ensure_model`, dans `~/Library/Application Support/MacDictee/modeles/whisper-large-v3-turbo-8bit`), puis le modèle complet est retiré du cache HF via `scan_cache_dir().delete_revisions` (blobs partagés : un simple rmtree du dossier du dépôt laisse 1,5 Go orphelin). Texte identique au modèle complet sur 4 tests ; mémoire 874 Mo au lieu de 1 618 ; disque 820 Mo au lieu de 1,5 Go. Le 4 bits (`-q4`) a été écarté : il casse les accents (« dictée » -> « DicT ») et hallucine.
- `mx.clear_cache()` après chaque transcription (le pic de ~700 Mo n'est plus gardé en réserve) ; modèle déchargé après 10 min sans usage (`free_memory`, rechargé en ~1 s).
- **`whisper_lite.py`** : DTW et filtre médian en numpy à la place de numba/llvmlite/scipy (horodatages de mots identiques sur la vraie réunion, 152/152) -> app 572 -> 369 Mo, DMG 288 -> 171 Mo, pas de LLVM en mémoire.
- La fenêtre ne recalcule l'état que si elle est visible (`push_state`), rafraîchie à l'ouverture.

## Longues réunions (0.9.1, mesuré sur M4 16 Go)
- Audio compressé en AAC 32 kb/s (`afconvert`, fourni par macOS) une fois les intervenants séparés : 2 h = ~30 Mo par piste au lieu de 230 Mo (÷ 9,9 mesuré), WAV supprimé seulement si la durée du M4A concorde. `load_wav` relit les M4A (décodage afconvert) ; séparation des intervenants identique après compression.
- Séparation des intervenants : 30 min en 118 s (1,6 Go) -> ~8 min pour 2 h.
- Compte rendu : au-delà de ~6 000 jetons, notes partie par partie puis synthèse (`SUMMARY_CHUNK_TOKENS`). 2 h (18 000 mots, 35 000 jetons) : 217 s et 4,6 Go, contre 350 s et 12,2 Go d'un seul tenant.

## Distribution
- `./build.sh` : app autonome py2app (Python 3.12 + dépendances embarquées, ~570 Mo, DMG ~270 Mo) -> `dist/`. Signature ad hoc par défaut ; Developer ID + notarisation si `SIGN_ID` / `NOTARY_PROFILE` (chemin écrit mais jamais testé, pas de compte développeur pour l'instant).
- Pièges py2app réglés dans `setup.py` / `build.sh` : limite de récursion, `mlx` et `tiktoken_ext` sont des namespace packages copiés après coup, torch/sympy exclus (jamais chargés pour transcrire), noms de fichiers ASCII (`MacDictee.app`, codesign casse sur le « é ») et nom affiché via `CFBundleDisplayName`.
- `MACDICTEE_SELFTEST=fichier.wav dist/MacDictee.app/Contents/MacOS/MacDictee` : vérifie qu'une app construite transcrit, sans interface.
- Mises à jour : `updates.py` interroge `api.github.com/repos/AyDevPro/MacDictee/releases/latest` (dépôt public), certificats via `certifi` (le Python embarqué n'a pas ceux de Homebrew). `./release.sh "notes"` : build + tag `vX.Y.Z` + `gh release create` avec le DMG. Version = `VERSION` dans `setup.py` (`APP_VERSION` le lit aussi en mode sources).
- Icône : `tools/make_icon.py` -> `assets/MacDictee.icns`.
- `install.sh` reste le mode développement (py2app alias pointant vers le venv).

## Ce qui existe (`app.py`)
- **Fenêtre** (`ui.py` + `ui/index.html`) : WKWebView, pont JS -> `App.handle(action, args)`, événements Python -> `md.onEvent`. Réglages persistants dans `~/Transcriptions/reglages.json`, historique dans `historique.json`.
- Presse-papiers écrit via `NSPasteboard` (pas `pbcopy` : sans `LANG`, lancé depuis le Finder, il cassait les accents).
- Option « Ouvrir au démarrage » via `SMAppService`.
- **Dictée** : touche Option droite maintenue (moniteur `NSEvent` sur le thread principal ; pynput retiré car il plantait l'app sous macOS 27 en appelant TSM hors du thread principal) -> enregistrement -> transcription (mlx-whisper, `whisper-large-v3-turbo`) -> collage via Cmd+V.
- **Champ actif** : détection via l'API Accessibilité (`AXFocusedUIElement`, rôle / `AXValue` modifiable / `AXSelectedTextRange`). Si aucun champ de texte : on copie seulement dans le presse-papiers + notification. Après un collage, l'ancien presse-papiers est restauré (option de menu) ; sans collage, le texte reste dans le presse-papiers. Les mises à jour d'interface depuis les threads passent par `notify()` / `set_title()` (thread principal via `callAfter`). `AXManualAccessibility` est activé pour Chrome/Electron.
- **Enregistrement long** : WAV écrit en continu dans `~/Transcriptions`, transcription par blocs de 20 à 45 s coupés sur les silences (RMS), fichier `.txt` horodaté au fil de l'eau.
- **Intervenants** (`speakers.py`) : sherpa-onnx (segmentation pyannote 3.0 ONNX + WeSpeaker voxceleb), modèles ~35 Mo téléchargés dans `~/Library/Application Support/MacDictee/modeles`. Seuil de clustering 0,4 puis `merge_phantoms` : fusion des groupes dont les empreintes globales dépassent 0,85 (vraie réunion à deux dans une pièce : même personne coupée en deux 0,90, personnes différentes 0,79-0,81 ; les seuils calés sur des voix `say` ne transfèrent pas) + rattachement des groupes minuscules. Cas de test : vraie réunion 2 personnes, synthèse 1/2/3 voix, tous corrects. Attribution **mot par mot** via `<réunion>_mots.jsonl` (word_timestamps écrits pendant la réunion, ou retranscription à la volée pour les anciennes) + lissage des îlots de ≤ 2 mots ; repli phrase par phrase sinon. Limite connue : répliques très courtes qui se chevauchent (« Ouais, c'est ça ») parfois mal attribuées. Wispr Flow (recherche oct. 2026) : transcription et Notetaker dans le cloud, noms des intervenants tirés du calendrier, avis mitigés sur la séparation. `_intervenants.json` = source de vérité (noms renommables), `_intervenants.txt` régénéré. pyannote/torch abandonnés (600 Mo + token HF).
- **Mains libres / pastille** (`overlay.py`) : appui < 0,3 s puis second appui dans les 0,35 s = mains libres (l'enregistrement continue depuis le premier appui), un appui termine, Échap annule ; un tap seul est ignoré. Pastille NSPanel non activable (ne vole pas le focus), vu-mètre via `Dictation.level`. États : ecoute / mainslibres / transcription.
- **Touche 🌐** : si `com.apple.HIToolbox AppleFnUsageType` ≠ 0 (2 = émojis), l'appui bref qui termine le mode mains libres ouvre le panneau d'émojis, qui prend le focus -> texte seulement copié. `fn_key_conflict()` affiche un avertissement avec un bouton vers les réglages Clavier (on ne modifie pas ce réglage système nous-mêmes).
- **Mise en forme** (`formatting.py`, dictées seulement) : listes numérotées (« point numéro N », « étape N », « premièrement… », « premier point… »), puces (« tiret »), « à la ligne » / « nouveau paragraphe ». Règles, pas d'IA (Wispr Flow le fait avec un LLM cloud). Liste créée seulement si ≥ 2 marqueurs consécutifs commençant à 1. Tests : `.venv/bin/python -m unittest discover tests` (12 cas, dont les faux positifs à éviter).
- **IA locale** (`rewrite.py`, option `ai_rewrite`, 0.9.0) : Qwen3-4B-Instruct-2507 4 bits via mlx-lm. Banc d'essai sur vraies dictées : Qwen2.5 1,5B répondait aux questions dictées, Qwen2.5 3B et Gemma 3 4B perdaient des phrases. `needs_rewrite` (hésitations, corrections, ≥ 2 marqueurs d'énumération, > 45 mots) ; `guard` (≥ 80 % des mots porteurs gardés, ≥ 60 % s'il y a une correction, ≤ 20 % ajoutés) sinon transcription brute ; puis `format_dictation`. `transformers` remplacé par un module inerte + `tokenizers` + gabarit ChatML écrit à la main (vérifié identique jeton par jeton à `apply_chat_template`) : pas de transformers ni torch chargés, +18 Mo seulement dans l'app. Chargement strictement hors ligne (`local_files_only`), téléchargement explicite à l'activation. Comptes rendus : `summarize` -> `<réunion>_compte-rendu.md`. Les deux modèles (Whisper + Qwen) partagent `_model_lock` et sont libérés ensemble au repos. Testé dans l'app autonome sous `sandbox-exec (deny network*)`.
- **Raccourci** : `HOTKEYS` (touches modificatrices à maintenir, + fn) ; une autre touche tapée pendant le maintien annule la dictée.
- **Suppression** : audio seul ou réunion entière, vers la Corbeille (`NSFileManager.trashItemAtURL`).
- **Vocabulaire** : `~/Transcriptions/vocabulaire.txt` (un mot par ligne) injecté dans `initial_prompt` (600 derniers caractères).
- **Apprentissage** : après un collage, `_watch_edits` compare le texte du champ (avant/après collage, puis jusqu'à 45 s plus tard ou prochaine dictée). Un remplacement de mot répété 2 fois devient une règle de remplacement automatique + entrée de vocabulaire (`~/Transcriptions/mots_appris.json`).
- Filtres anti-hallucinations Whisper (silence, `no_speech_prob`, phrases type « Sous-titres… »), verrou autour du modèle MLX.

## Son du Mac (visios) : fait en 0.5.0, à valider en vrai
`native/audio_capture.swift` -> binaire `macdictee-audio` (macOS 14.2+ : **Core Audio process tap** + agrégat privé, autorisation « Enregistrement des sons du système uniquement » comme Wispr Flow, `NSAudioCaptureUsageDescription` ; avant, ou avec `--screen` : ScreenCaptureKit, autorisation « écran et sons du système », **validé en vrai par l'utilisateur** ; repli auto vers `--screen` si le tap échoue et que l'autorisation écran est accordée. Un tap sans autorisation renvoie du silence, pas d'erreur : à la fin d'une réunion dont la piste système est entièrement nulle, notification. PCM float32 mono 16 kHz sur stdout, `ready`/`error:` sur stderr, s'arrête quand stdin se ferme). Compilé par `build.sh` dans `Contents/MacOS/` et par `install.sh` dans `native/`. `SystemAudio` (app.py) le lance pendant une réunion, `take(n)` aligne sur les blocs micro (dette de silence rattrapée). `<réunion>.wav` = micro + son du Mac (transcrit), `<réunion>_systeme.wav` = son du Mac seul. `speakers.run` : mot pendant que la piste système est active (RMS > 0,01) = visio, sinon salle. **Hybride (0.8.0)** : salle diarisée sur micro = mélange − son du Mac (passages où la visio parle ± 0,3 s mis à zéro contre l'écho), visio sur sa piste ; noms « Moi » (une seule voix en salle) ou « Sur place n », et « Visio n ». Testé : visio simulée (Moi / Visio 1 / Visio 2) et hybride simulée 2 + 2 voix avec écho, toutes correctes ; réunion réelle sans visio inchangée. Statut de l'autorisation « sons du système uniquement » lu via `TCCAccessPreflight` (framework privé TCC, pas d'API publique ; 0 accordé / 1 refusé / 2 jamais demandé), « Autoriser… » lance une capture d'1 s pour faire apparaître la demande. Testé : visio simulée (moi + 2 voix distantes, écho léger) -> Moi / Intervenant 1 / Intervenant 2 corrects ; synchro testée avec un faux module temps réel. **Jamais testé avec la vraie capture** (le terminal n'a pas l'autorisation) : premier test réel par l'utilisateur.
Autorisation « Enregistrement de l'écran et audio système » : `CGPreflightScreenCaptureAccess` / `CGRequestScreenCaptureAccess`, facultative (la fenêtre ne s'ouvre pas toute seule pour elle). Piste alternative si l'autorisation « écran » rebute : Core Audio process taps (macOS 14.2+, autorisation audio seule).

## Notes de conception initiales (son système)
Objectif : comme le Notetaker de Wispr Flow (sans bot dans l'appel). Aujourd'hui seul le micro est enregistré.
- Piste recommandée : petit binaire **Swift** (ScreenCaptureKit, audio système, macOS 13+ ; ou Core Audio process taps, macOS 14.2+) qui envoie du PCM 16 kHz mono (stdout ou fichier). `app.py` le lance en sous-processus pendant l'enregistrement long.
- Garder **deux pistes séparées** (micro = « moi », son système = « les autres »). Ça donne gratuitement la séparation vous / les autres, puis la séparation des intervenants seulement sur la piste système.
- Autorisation à gérer : « Enregistrement de l'écran et audio système ». Prévoir un message clair si elle manque.
- Plan B plus simple : BlackHole + périphérique audio multiple (à documenter).

## Autres pistes (par priorité)
1. ~~WAV en cas de plantage~~ : faux problème (le module wave de Python met l'en-tête à jour à chaque writeframes, vérifié) ; `load_wav` lit de toute façon toutes les données après l'en-tête de 44 octets.
2. Tests automatisés pour `learn_from_edit`, `apply_rules`, la logique de découpage sur silence (ceux de `formatting.py` existent, dans `tests/`).
3. Raccourci configurable (aujourd'hui Option droite codée en dur), réglages persistants (langue, modèle, collage auto) au lieu de variables d'environnement.
4. Résumé/compte rendu : fait en local (0.9.0). Ne PAS passer par une API cloud : l'engagement auprès de l'utilisateur est qu'aucun mot ne quitte le Mac.
5. Si le projet grossit : réécriture en app Swift native (distribution plus simple, permissions plus fiables qu'avec py2app).

## Pièges connus
- Permissions macOS liées à l'app/au binaire : Microphone, Accessibilité, (Surveillance de l'entrée n'est plus nécessaire). En mode `py2app -A` (alias) ou en lançant `python app.py` depuis le Terminal, la permission peut s'attacher au Terminal ou à Python plutôt qu'à l'app.
- `AXValue` n'est pas exposé partout (Google Docs, certains éditeurs web) : l'apprentissage est alors impossible, le reste fonctionne.
- MLX ne doit pas être appelé en parallèle : tout passe par `_model_lock`.

## Décisions du propriétaire
100 % local (pas de cloud pour la transcription), Apple Silicon uniquement, français par défaut (`WHISPER_LANG`), fichiers dans `~/Transcriptions`, un mot est « appris » après 2 corrections identiques.
