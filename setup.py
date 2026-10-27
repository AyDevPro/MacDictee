import sys

from setuptools import setup

sys.setrecursionlimit(10000)  # l’analyse des imports de py2app est récursive (numpy, scipy, mlx…)

VERSION = "0.9.1"

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
        "NSAudioCaptureUsageDescription": "MacDictée enregistre le son du Mac pendant vos réunions en visio pour les transcrire.",
        "NSHumanReadableCopyright": "MacDictée : dictée vocale 100 % locale.",
    },
    "packages": ["mlx_whisper", "rumps", "sounddevice", "_sounddevice_data", "numpy", "tiktoken", "sherpa_onnx", "mlx_lm", "tokenizers", "huggingface_hub", "certifi", "tqdm",
                 "AppKit", "ApplicationServices", "Quartz", "WebKit", "AVFoundation", "ServiceManagement"],
    "includes": ["ui", "updates", "speakers", "overlay", "whisper_lite", "formatting", "rewrite"],
    # torch n'est jamais chargé pour transcrire (seulement pour convertir des modèles) : ~600 Mo économisés.
    # mlx et tiktoken_ext sont des « namespace packages » que py2app ne sait pas collecter : build.sh les copie.
    # numba, llvmlite et scipy : remplacés par whisper_lite (~230 Mo de moins, pas de LLVM en mémoire)
    # transformers (115 Mo) : remplacé dans rewrite.py par tokenizers, mlx-lm n'en a besoin que pour les jetons
    "excludes": ["mlx", "tiktoken_ext", "transformers", "numba", "llvmlite", "scipy", "torch", "torchaudio", "sympy", "pynput", "tkinter", "matplotlib", "IPython", "pytest"],
}

# nom ASCII pour l’exécutable interne (codesign s’y perd avec le « é ») ; le nom affiché reste « MacDictée »
setup(name="MacDictee", app=APP, data_files=DATA_FILES, options={"py2app": OPTIONS}, setup_requires=["py2app"])
