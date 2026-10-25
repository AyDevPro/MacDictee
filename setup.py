import sys

from setuptools import setup

sys.setrecursionlimit(10000)  # l’analyse des imports de py2app est récursive (numpy, scipy, mlx…)

VERSION = "0.4.0"

APP = ["app.py"]
DATA_FILES = [("ui", ["ui/index.html"])]
OPTIONS = {
    "argv_emulation": False,
    "iconfile": "assets/MacDictee.icns",
    "plist": {
        "CFBundleName": "MacDictee",  # nom de fichier ASCII ; l’utilisateur voit CFBundleDisplayName
        "CFBundleDisplayName": "MacDictée",
        "CFBundleIdentifier": "fr.local.macdictee",
        "CFBundleShortVersionString": VERSION,
        "CFBundleVersion": VERSION,
        "LSMinimumSystemVersion": "13.0",
        "LSArchitecturePriority": ["arm64"],
        "LSUIElement": True,  # pas d'icône dans le Dock, seulement la barre de menus
        "NSMicrophoneUsageDescription": "MacDictée a besoin du micro pour transcrire votre voix.",
        "NSHumanReadableCopyright": "MacDictée : dictée vocale 100 % locale.",
    },
    "packages": ["mlx_whisper", "rumps", "sounddevice", "_sounddevice_data", "numpy", "scipy", "numba",
                 "llvmlite", "tiktoken", "sherpa_onnx", "huggingface_hub", "certifi", "tqdm",
                 "AppKit", "ApplicationServices", "Quartz", "WebKit", "AVFoundation", "ServiceManagement"],
    "includes": ["ui", "updates", "speakers"],
    # torch n'est jamais chargé pour transcrire (seulement pour convertir des modèles) : ~600 Mo économisés.
    # mlx et tiktoken_ext sont des « namespace packages » que py2app ne sait pas collecter : build.sh les copie.
    "excludes": ["mlx", "tiktoken_ext", "torch", "torchaudio", "sympy", "pynput", "tkinter", "matplotlib", "IPython", "pytest"],
}

# nom ASCII pour l’exécutable interne (codesign s’y perd avec le « é ») ; le nom affiché reste « MacDictée »
setup(name="MacDictee", app=APP, data_files=DATA_FILES, options={"py2app": OPTIONS}, setup_requires=["py2app"])
