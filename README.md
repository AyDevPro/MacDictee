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

Version : `VERSION` dans `setup.py`. Sans compte développeur Apple, l'app est signée « ad hoc » : chaque nouvelle version oblige à redonner l'autorisation Accessibilité. Avec un compte (99 $/an), `SIGN_ID="Developer ID Application: …" NOTARY_PROFILE=… ./build.sh` signe et notarise (plus d'avertissement au premier lancement, autorisations conservées entre les versions).

## Mises à jour

Les apps installées vérifient au lancement, puis toutes les 6 heures, s'il existe une version plus récente sur [GitHub Releases](https://github.com/AyDevPro/MacDictee/releases). Si oui : notification, entrée « ⬆️ Installer la version… » dans le menu 🎙, et bandeau « Mettre à jour » dans la fenêtre. Le DMG se télécharge dans `~/Downloads`, s'ouvre, l'app se ferme : il reste à glisser MacDictée dans Applications (Remplacer) et à la rouvrir. Sans signature Apple, l'autorisation Accessibilité est à redonner après chaque mise à jour.

Publier une version (développeurs) :

```bash
# 1. augmenter VERSION dans setup.py, puis commiter
./release.sh "Ce qui change dans cette version"
```

## Installation en mode développement

```bash
cd ~/Dev/MacDictee   # sur le disque interne : l'app pointe vers ce dossier
chmod +x install.sh
./install.sh
```

Puis ouvrez `~/Applications/MacDictée.app`. Une icône 🎙 apparaît dans la barre de menus.
Le premier lancement télécharge le modèle Whisper (~1,5 Go).

## Autorisations (Réglages Système > Confidentialité et sécurité)

- **Microphone** : pour enregistrer
- **Accessibilité** : pour détecter la touche, repérer le champ de texte actif et coller

La fenêtre *Ouvrir MacDictée…* (menu 🎙) montre l'état des autorisations, avec un bouton pour chacune et un bouton *Relancer*. Elle regroupe aussi l'historique des dictées, les réunions, le vocabulaire, les mots appris et les réglages.

## Utilisation

**Dictée rapide (comme Wispr Flow)** : maintenez la touche de dictée (**Option droite** par défaut, modifiable dans Réglages : Option gauche, Commande droite, Contrôle droite, Maj droite ou fn/🌐), parlez, relâchez. Si vous tapez une autre touche pendant qu'elle est maintenue (⌥ + e pour un caractère, par exemple), la dictée est annulée.
- Si le curseur est dans un champ de texte : le texte est collé.
- Si aucun champ de texte n'est actif : le texte est seulement copié, vous recevez une notification et vous collez où vous voulez avec Cmd+V.
- Après un collage, votre presse-papiers précédent est remis en place (comme Wispr Flow). Option désactivable dans le menu 🎙 (*Remettre le presse-papiers après collage*) pour garder le texte dicté.
- Le menu 🎙 permet de désactiver le collage automatique (presse-papiers seul).

**Vocabulaire personnalisé** : menu 🎙 > *Modifier le vocabulaire…*, un mot ou nom propre par ligne (prénoms, marques, jargon). Whisper les écrira correctement.

**Apprentissage automatique** : après un collage, MacDictée surveille le champ pendant 45 secondes (ou jusqu'à votre prochaine dictée). Si vous corrigez un mot, il le note. Dès que vous faites **la même correction 2 fois**, il l'apprend : le mot est ajouté au vocabulaire et remplacé automatiquement dans toutes les futures transcriptions (dictée et réunions), avec une notification « Nouveau mot appris ». Menu 🎙 > *Voir les mots appris…* pour consulter ou supprimer une règle (fichier `~/Transcriptions/mots_appris.json`).
- Seuls les remplacements de mots comptent (une reformulation complète est ignorée).
- Ça marche dans les champs dont l'app expose le texte (Notes, Mail, Messages, Slack, champs web classiques…). Dans certains éditeurs (Google Docs, par exemple), la correction n'est pas visible : pas d'apprentissage, mais le reste fonctionne.

**Compte rendu avec Claude** : à la fin d'une réunion, glissez le fichier `~/Transcriptions/..._intervenants.txt` (ou `...txt`) dans Claude et demandez un compte rendu. Le bouton *Copier le texte* de la fenêtre copie la version avec les intervenants.

**Enregistrement long (réunions, conversations)** : menu 🎙 > *Démarrer l'enregistrement long*.
- L'audio est sauvegardé en continu, donc aucune limite de durée.
- La transcription se construit au fil de l'eau dans `~/Transcriptions/AAAA-MM-JJ_HHhMM.txt`, avec des horodatages. Les blocs sont coupés sur les silences pour ne pas couper les phrases.
- À l'arrêt, les intervenants sont séparés automatiquement (« Intervenant 1, 2, 3… », réglage désactivable) : renommez-les dans la fenêtre, onglet Réunions. Si le nombre détecté est faux, relancez en indiquant le nombre de personnes. Fichiers : `..._intervenants.txt` (lisible) et `..._intervenants.json`. Les modèles (~35 Mo, sherpa-onnx) se téléchargent la première fois dans `~/Library/Application Support/MacDictee/modeles`.
- Onglet Réunions : *Supprimer l'audio* (garde le texte, utile une fois le résumé fait) ou *Supprimer la réunion* (audio + textes). Les fichiers vont à la Corbeille.

**Visios (Meet, WhatsApp, Zoom…)** : avec le réglage *Capter le son du Mac pendant les réunions* (activé par défaut) et l'autorisation « Enregistrement de l'écran et audio système », MacDictée enregistre votre micro **et** le son du Mac. Dans la transcription, vous apparaissez comme « Moi » et les gens de la visio sont séparés entre eux (Intervenant 1, 2…). Sans l'autorisation, seul le micro est enregistré.

**Mains libres** : appuyez deux fois rapidement sur la touche de dictée, parlez aussi longtemps que vous voulez, puis appuyez une fois pour terminer (Échap pour annuler). Une pastille en bas de l'écran indique que MacDictée écoute.

## Réglages

Variables d'environnement : `WHISPER_LANG` (`fr` par défaut, vide pour détection auto), `WHISPER_MODEL` (par défaut : whisper-large-v3-turbo converti en 8 bits au premier lancement ; mettre par exemple `mlx-community/whisper-small-mlx` pour plus de rapidité sur un vieux Mac).

Sobriété : le modèle 8 bits occupe ~870 Mo de mémoire pendant l'utilisation et ~820 Mo sur le disque ; après 10 minutes sans dictée il est libéré (réglage *Libérer la mémoire au repos*), la dictée suivante prend alors ~1 seconde de plus.

## Test rapide sans construire l'app

```bash
source .venv/bin/activate && python app.py
```
