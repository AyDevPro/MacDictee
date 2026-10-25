from setuptools import setup

APP = ["app.py"]
OPTIONS = {
    "argv_emulation": False,
    "plist": {
        "CFBundleName": "MacDictée",
        "CFBundleDisplayName": "MacDictée",
        "CFBundleIdentifier": "fr.local.macdictee",
        "LSUIElement": True,  # pas d'icône dans le Dock, seulement la barre de menus
        "NSMicrophoneUsageDescription": "MacDictée a besoin du micro pour transcrire votre voix.",
    },
    "packages": ["mlx", "mlx_whisper", "rumps", "sounddevice", "numpy", "pynput",
                 "AppKit", "ApplicationServices"],
}

setup(app=APP, options={"py2app": OPTIONS}, setup_requires=["py2app"])
