#!/bin/bash
# Construit MacDictée.app autonome (Python et dépendances inclus) et un DMG à distribuer.
#
#   ./build.sh                                  -> signature « ad hoc » (pour soi et les amis)
#   SIGN_ID="Developer ID Application: Nom (TEAMID)" NOTARY_PROFILE=macdictee ./build.sh
#                                               -> signée et notarisée par Apple (compte développeur requis)
set -e
cd "$(dirname "$0")"

[ -x .venv/bin/python ] || { echo "Lancez d'abord ./install.sh (crée l'environnement Python)"; exit 1; }
PY=.venv/bin/python
"$PY" -m pip install -q -r requirements.txt
VERSION=$("$PY" -c 'import re; print(re.search(r"VERSION = \"(.+)\"", open("setup.py").read())[1])')
APP="dist/MacDictee.app"
DMG="dist/MacDictee-$VERSION.dmg"

[ -f assets/MacDictee.icns ] || "$PY" tools/make_icon.py
rm -rf build dist
"$PY" setup.py py2app
SITE=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
for ns in mlx tiktoken_ext; do  # namespace packages ignorés par py2app (voir setup.py)
  cp -R "$SITE/$ns" "$APP/Contents/Resources/lib/python3.12/"
done
# module Swift de capture du son du Mac (visios), lancé par l'app pendant les réunions
swiftc -O -target arm64-apple-macos13.0 -o "$APP/Contents/MacOS/macdictee-audio" native/audio_capture.swift

if [ -n "$SIGN_ID" ]; then
  # chaque binaire doit être signé, de l'intérieur vers l'extérieur, avec le hardened runtime
  find "$APP" -type f \( -name "*.so" -o -name "*.dylib" -o -perm -u+x \) -print0 |
    xargs -0 file | grep "Mach-O" | cut -d: -f1 |
    while read -r bin; do
      codesign --force --timestamp --options runtime --entitlements assets/entitlements.plist --sign "$SIGN_ID" "$bin"
    done
  codesign --force --timestamp --options runtime --entitlements assets/entitlements.plist --sign "$SIGN_ID" "$APP"
else
  codesign --force --deep --sign - "$APP"
fi
codesign --verify --deep --strict "$APP"

STAGE=$(mktemp -d)
cp -R "$APP" "$STAGE/MacDictée.app"
ln -s /Applications "$STAGE/Applications"
hdiutil create -quiet -volname "MacDictée" -srcfolder "$STAGE" -ov -format UDZO "$DMG"
rm -rf "$STAGE"

if [ -n "$SIGN_ID" ] && [ -n "$NOTARY_PROFILE" ]; then
  codesign --force --sign "$SIGN_ID" "$DMG"
  xcrun notarytool submit "$DMG" --keychain-profile "$NOTARY_PROFILE" --wait
  xcrun stapler staple "$DMG"
fi

echo ""
echo "✅ $DMG ($(du -h "$DMG" | cut -f1))"
