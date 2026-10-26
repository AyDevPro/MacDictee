"""MacDictée : dictée vocale + transcription longue durée, 100 % locale (Apple Silicon).

- Dictée : maintenez Option DROITE, parlez, relâchez.
    * curseur dans un champ de texte  -> le texte est collé (et reste dans le presse-papiers)
    * pas de champ de texte actif     -> le texte est seulement copié dans le presse-papiers
- Mode réunion : menu de la barre > "Démarrer l'enregistrement long".
- Fenêtre « Ouvrir MacDictée… » : état, historique, réunions, vocabulaire, mots appris, réglages.
"""
import datetime
import difflib
import json
import os
import platform
import queue
import re
import subprocess
import threading
import time
import wave
from pathlib import Path

import gc
import shutil

import whisper_lite  # noqa: F401  (avant mlx_whisper : remplace numba et scipy)
import mlx.core as mx
import mlx.nn as nn
import mlx_whisper
import objc
import numpy as np
import rumps
import sounddevice as sd
from AppKit import (NSApp, NSApplicationActivationPolicyAccessory, NSApplicationActivationPolicyRegular, NSBundle, NSEvent, NSEventMaskFlagsChanged, NSEventMaskKeyDown, NSPasteboard, NSPasteboardItem,
                    NSPasteboardTypeString, NSWorkspace)
from ApplicationServices import (AXIsProcessTrusted, AXIsProcessTrustedWithOptions, AXUIElementCopyAttributeValue,
                                 AXUIElementCreateApplication, AXUIElementIsAttributeSettable,
                                 AXUIElementSetAttributeValue)
from AVFoundation import AVCaptureDevice, AVMediaTypeAudio
from Foundation import NSURL, NSFileManager, NSUserDefaults
from PyObjCTools.AppHelper import callAfter, callLater
from ServiceManagement import SMAppService
from Quartz import CGPreflightScreenCaptureAccess, CGRequestScreenCaptureAccess, CGEventCreateKeyboardEvent, CGEventPost, CGEventSetFlags, kCGEventFlagMaskCommand, kCGHIDEventTap

import speakers
from rewrite import MODEL_SIZE as AI_MODEL_SIZE, Rewriter, needs_rewrite as needs_ai
from formatting import format_dictation
import updates
from overlay import Overlay
from ui import Window, install_edit_menu

SR = 16000
# Whisper large-v3-turbo converti en 8 bits sur le Mac : texte identique au modèle complet sur nos tests
# (accents compris, contrairement au 4 bits), 874 Mo de mémoire au lieu de 1,6 Go, 864 Mo sur le disque.
MODEL_SOURCE = "mlx-community/whisper-large-v3-turbo"
MODEL = os.environ.get("WHISPER_MODEL") or str(speakers.MODELS_DIR / "whisper-large-v3-turbo-8bit")
IDLE_UNLOAD = 10 * 60  # sans dictée ni réunion pendant ce temps, on libère la mémoire du modèle
# Touche à maintenir pour dicter : (keyCode, masque « device-dependent » de la touche, libellé).
# Écoutée via NSEvent : pynput appelait des API clavier hors du thread principal et macOS tuait l'app.
HOTKEYS = {
    "alt_r": (61, 0x40, "⌥ Option droite"),
    "alt_l": (58, 0x20, "⌥ Option gauche"),
    "cmd_r": (54, 0x10, "⌘ Commande droite"),
    "ctrl_r": (62, 0x2000, "⌃ Contrôle droite"),
    "shift_r": (60, 0x04, "⇧ Maj droite"),
    "fn": (63, 0x800000, "fn / 🌐"),
}
V_KEYCODE = 9
ESCAPE_KEYCODE = 53
TAP_MAX = 0.3  # appui plus court : c'est un « tap » (peut-être le premier d'un double appui)
DOUBLE_TAP_WINDOW = 0.35  # délai pour le second tap qui passe en mains libres
OUT_DIR = Path.home() / "Transcriptions"
OUT_DIR.mkdir(exist_ok=True)



def app_version() -> str:
    bundle = NSBundle.mainBundle()
    if bundle.bundleIdentifier() == "fr.local.macdictee":
        return str(bundle.objectForInfoDictionaryKey_("CFBundleShortVersionString"))
    setup_py = Path(__file__).resolve().parent / "setup.py"  # lancé depuis les sources
    m = re.search(r'VERSION = "(.+)"', setup_py.read_text()) if setup_py.exists() else None
    return m[1] if m else "0"


APP_VERSION = app_version()
UPDATE_EVERY = 6 * 3600

SETTINGS_FILE = OUT_DIR / "reglages.json"
DEFAULT_SETTINGS = {
    "paste": True,  # coller dans le champ de texte actif
    "restore_clipboard": True,  # remettre l'ancien presse-papiers après collage
    "language": os.environ.get("WHISPER_LANG", "fr"),  # "" = détection automatique
    "dock": True,  # icône dans le Dock en plus de la barre de menus
    "hotkey": "alt_r",  # clé de HOTKEYS
    "speakers": True,  # séparer les intervenants à la fin d'une réunion
    "overlay": True,  # pastille en bas de l'écran pendant la dictée
    "handsfree": True,  # double appui = dictée continue jusqu'au prochain appui
    "system_audio": True,  # réunions : enregistrer aussi le son du Mac (visios), sur une piste à part
    "free_memory": True,  # décharger le modèle après IDLE_UNLOAD sans utilisation (rechargé en ~1 s)
    "format": True,  # listes dictées (« point numéro 1… ») et « à la ligne » mis en forme (formatting.py)
    "ai_rewrite": False,  # réécriture IA locale des dictées (rewrite.py, ~2,3 Go téléchargés une fois)
}


def load_settings():
    try:
        saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        saved = {}
    return {k: saved.get(k, v) for k, v in DEFAULT_SETTINGS.items()}


SETTINGS = load_settings()


def save_settings():
    SETTINGS_FILE.write_text(json.dumps(SETTINGS, ensure_ascii=False, indent=2), encoding="utf-8")


HISTORY_FILE = OUT_DIR / "historique.json"
HISTORY_MAX = 200
_history_lock = threading.Lock()


def load_history():
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def add_history(text, app_name):
    with _history_lock:
        items = load_history()
        items.insert(0, {"time": datetime.datetime.now().isoformat(timespec="seconds"), "text": text, "app": app_name})
        HISTORY_FILE.write_text(json.dumps(items[:HISTORY_MAX], ensure_ascii=False, indent=2), encoding="utf-8")


def clear_history():
    with _history_lock:
        HISTORY_FILE.write_text("[]", encoding="utf-8")
VOCAB_FILE = OUT_DIR / "vocabulaire.txt"  # un mot / nom propre par ligne : aide Whisper à bien les écrire
if not VOCAB_FILE.exists():
    VOCAB_FILE.write_text("", encoding="utf-8")

# Découpage intelligent de l'enregistrement long : on coupe sur un silence, entre 20 s et 45 s
MIN_CHUNK, MAX_CHUNK = 20, 45
BLOCK = 1600  # 0,1 s par bloc audio
SILENCE_RMS = 400  # niveau int16 en dessous duquel on considère que c'est un silence

HALLUCINATIONS = ("sous-titres", "sous-titrage", "amara.org", "merci d'avoir regardé", "abonnez-vous")
_model_lock = threading.Lock()


# ----------------------------------------------------------------------------- transcription
# --- apprentissage : mots corrigés à la main -> règles de remplacement + vocabulaire
LEARN_FILE = OUT_DIR / "mots_appris.json"
LEARN_THRESHOLD = 2  # nombre de fois qu'une même correction doit être faite avant d'être apprise
TOKEN = re.compile(r"[\w'’-]+", re.UNICODE)
_learn_lock = threading.Lock()


def load_store():
    try:
        data = json.loads(LEARN_FILE.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    data.setdefault("counts", {})  # "mal écrit\tcorrect" -> nombre de corrections vues
    data.setdefault("rules", {})  # "mal écrit" -> "correct" (appliqué automatiquement)
    return data


def apply_rules(text: str) -> str:
    for wrong, right in load_store()["rules"].items():
        text = re.sub(rf"(?<!\w){re.escape(wrong)}(?!\w)", right, text, flags=re.IGNORECASE)
    return text


def register_correction(wrong: str, right: str):
    with _learn_lock:
        store = load_store()
        key = f"{wrong.lower()}\t{right}"
        store["counts"][key] = store["counts"].get(key, 0) + 1
        learned = store["counts"][key] >= LEARN_THRESHOLD and store["rules"].get(wrong.lower()) != right
        if learned:
            store["rules"][wrong.lower()] = right
            known = {w.strip().lower() for w in VOCAB_FILE.read_text(encoding="utf-8").splitlines()}
            if right.lower() not in known:
                with open(VOCAB_FILE, "a", encoding="utf-8") as f:
                    f.write(right + "\n")
        LEARN_FILE.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    if learned:
        notify("MacDictée", "Nouveau mot appris", f"{wrong} → {right}")


def learn_from_edit(original: str, edited: str):
    """Compare le texte dicté et le texte final : retient les mots remplacés (pas les ajouts/suppressions)."""
    a, b = TOKEN.findall(original), TOKEN.findall(edited)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace" or i2 - i1 > 3 or j2 - j1 > 3:
            continue
        wrong, right = " ".join(a[i1:i2]), " ".join(b[j1:j2])
        if len(right) < 3 or wrong == right:
            continue
        if difflib.SequenceMatcher(None, wrong.lower(), right.lower()).ratio() < 0.4:
            continue  # trop différent : c'est une reformulation, pas une correction d'orthographe
        register_correction(wrong, right)


def vocab_prompt():
    words = [w.strip() for w in VOCAB_FILE.read_text(encoding="utf-8").splitlines() if w.strip()]
    prompt = ", ".join(words)
    return prompt[-600:] or None  # Whisper limite la taille du prompt : on garde les plus récents


def ensure_model():
    """Prépare le modèle 8 bits une fois pour toutes : téléchargement du modèle complet, conversion, puis
    suppression du modèle complet (1,5 Go libérés). Appelé sous _model_lock."""
    global _cache_checked
    target = Path(MODEL)
    if os.environ.get("WHISPER_MODEL"):
        return
    if (target / "weights.safetensors").exists():
        if not _cache_checked:  # une ancienne version a pu retélécharger le modèle complet : on le retire
            _cache_checked = True
            drop_cached_model(MODEL_SOURCE)
        return
    from huggingface_hub import snapshot_download
    from mlx.utils import tree_flatten
    from mlx_whisper.load_models import load_model

    source = Path(snapshot_download(MODEL_SOURCE))
    model = load_model(str(source), dtype=mx.float16)
    nn.quantize(model, group_size=64, bits=8)
    config = json.loads((source / "config.json").read_text())
    config["quantization"] = {"group_size": 64, "bits": 8}
    tmp = target.with_name(target.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    (tmp / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(tmp / "weights.safetensors"), dict(tree_flatten(model.parameters())))
    tmp.rename(target)
    del model
    drop_cached_model(MODEL_SOURCE)


def drop_cached_model(repo_id):
    """Supprime un modèle du cache Hugging Face, fichiers de poids compris (ils sont dans un dossier partagé)."""
    from huggingface_hub import scan_cache_dir
    try:
        cache = scan_cache_dir()
        revisions = [r.commit_hash for repo in cache.repos if repo.repo_id == repo_id for r in repo.revisions]
        if revisions:
            cache.delete_revisions(*revisions).execute()
    except Exception:
        pass  # le cache n'est qu'une économie de place : on ne bloque jamais la dictée pour ça


_last_use = time.monotonic()
_cache_checked = False


REWRITER = Rewriter()  # IA locale : réécriture des dictées et comptes rendus (utilise aussi _model_lock)


def unload_model():
    """Libère la mémoire des modèles (rechargés automatiquement à la prochaine utilisation)."""
    with _model_lock:
        holder = mlx_whisper.transcribe.__globals__["ModelHolder"]
        if holder.model is None and REWRITER.model is None:
            return False
        holder.model = holder.model_path = None
        REWRITER.unload()
        gc.collect()
        mx.clear_cache()
        return True


def ai_call(fn, *args):
    """Appel du modèle d'IA locale sous le verrou MLX (un seul calcul à la fois sur le GPU)."""
    global _last_use
    with _model_lock:
        try:
            return fn(*args)
        finally:
            mx.clear_cache()
            _last_use = time.monotonic()


def transcribe(audio: np.ndarray, words: bool = False) -> dict:
    """words=True : horodatage de chaque mot (réunions, pour attribuer chaque mot au bon intervenant)."""
    kwargs = dict(path_or_hf_repo=MODEL, condition_on_previous_text=False, word_timestamps=words,
                  no_speech_threshold=0.6, compression_ratio_threshold=2.4)
    if SETTINGS["language"]:
        kwargs["language"] = SETTINGS["language"]
    prompt = vocab_prompt()
    if prompt:
        kwargs["initial_prompt"] = prompt
    global _last_use
    with _model_lock:
        ensure_model()
        res = mlx_whisper.transcribe(audio, **kwargs)
        mx.clear_cache()  # rend la mémoire de travail (jusqu'à ~700 Mo au pic) au lieu de la garder en réserve
        _last_use = time.monotonic()
    segs = []
    for s in res.get("segments", []):
        text = s["text"].strip()
        if not text:
            continue
        if s.get("no_speech_prob", 0) > 0.6 and s.get("avg_logprob", 0) < -1.0:
            continue  # probable silence/bruit
        if len(text) < 80 and any(h in text.lower() for h in HALLUCINATIONS):
            continue  # hallucination classique de Whisper sur du silence
        s["text"] = apply_rules(text)
        segs.append(s)
    return {"text": " ".join(s["text"].strip() for s in segs).strip(), "segments": segs}


# ----------------------------------------------------------------------------- champ de texte actif
EDITABLE_ROLES = {"AXTextField", "AXTextArea", "AXComboBox", "AXSearchField"}
_ax_enabled = set()


def front_pid():
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    return app.processIdentifier() if app else None


def front_app_name():
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    return str(app.localizedName()) if app else ""


def enable_accessibility(pid):
    """Chrome / Electron n'exposent leurs champs qu'après cette demande (une fois par app)."""
    if pid is None or pid in _ax_enabled:
        return
    try:
        app = AXUIElementCreateApplication(pid)
        AXUIElementSetAttributeValue(app, "AXManualAccessibility", True)
        AXUIElementSetAttributeValue(app, "AXEnhancedUserInterface", True)
    except Exception:
        pass
    _ax_enabled.add(pid)


def text_field_focused():
    """True = champ de texte actif ; False = aucun ; None = impossible à déterminer."""
    try:
        pid = front_pid()
        if pid is None:
            return None
        enable_accessibility(pid)
        app = AXUIElementCreateApplication(pid)
        err, el = AXUIElementCopyAttributeValue(app, "AXFocusedUIElement", None)
        if err == -25212 or (err == 0 and el is None):  # kAXErrorNoValue : rien n'a le focus
            return False
        if err != 0:
            return None
        err, role = AXUIElementCopyAttributeValue(el, "AXRole", None)
        if err == 0 and role in EDITABLE_ROLES:
            return True
        err, settable = AXUIElementIsAttributeSettable(el, "AXValue", None)
        if err == 0 and settable:
            return True
        err, _ = AXUIElementCopyAttributeValue(el, "AXSelectedTextRange", None)  # contenteditable web
        return err == 0
    except Exception:
        return None


def get_focused_element():
    try:
        pid = front_pid()
        if pid is None:
            return None
        err, el = AXUIElementCopyAttributeValue(AXUIElementCreateApplication(pid), "AXFocusedUIElement", None)
        return el if err == 0 else None
    except Exception:
        return None


def get_value(el):
    """Texte complet du champ (None si l'app ne l'expose pas)."""
    try:
        err, v = AXUIElementCopyAttributeValue(el, "AXValue", None)
        return str(v) if err == 0 and v is not None else None
    except Exception:
        return None


# ----------------------------------------------------------------------------- utilitaires
def notify(title, subtitle, message):
    """AppKit n'accepte l'interface que sur le thread principal : on y renvoie les notifications."""
    callAfter(rumps.notification, title, subtitle, message)


def play(name):
    path = f"/System/Library/Sounds/{name}.aiff"
    if os.path.exists(path):
        subprocess.Popen(["afplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def set_clipboard(text: str) -> None:
    # pas de pbcopy : lancé depuis le Finder (sans LANG), il lit l'UTF-8 comme du Mac Roman et casse les accents
    pb = NSPasteboard.generalPasteboard()
    pb.clearContents()
    pb.setString_forType_(text, NSPasteboardTypeString)


def save_clipboard():
    """Copie de tout le contenu du presse-papiers (texte, images, fichiers…) pour le remettre après collage."""
    pb = NSPasteboard.generalPasteboard()
    return [{t: item.dataForType_(t) for t in item.types()} for item in (pb.pasteboardItems() or [])]


def restore_clipboard(saved) -> None:
    pb = NSPasteboard.generalPasteboard()
    pb.clearContents()
    items = []
    for types in saved:
        item = NSPasteboardItem.alloc().init()
        for t, data in types.items():
            if data is not None:
                item.setData_forType_(data, t)
        items.append(item)
    if items:
        pb.writeObjects_(items)


def clipboard_count() -> int:
    return NSPasteboard.generalPasteboard().changeCount()


def press_paste() -> None:
    """Cmd+V envoyé directement au système (utilisable depuis n'importe quel thread)."""
    for down in (True, False):
        event = CGEventCreateKeyboardEvent(None, V_KEYCODE, down)
        CGEventSetFlags(event, kCGEventFlagMaskCommand)
        CGEventPost(kCGHIDEventTap, event)


# macOS 14.2+ : « process tap » Core Audio, autorisation « sons du système uniquement » (comme Wispr Flow) ;
# avant : ScreenCaptureKit, qui exige « Enregistrement de l'écran et des sons du système ».
SYSTEM_AUDIO_TAP = tuple(int(x) for x in (platform.mac_ver()[0] or "0").split(".")[:2]) >= (14, 2)


def fn_key_conflict() -> bool:
    """La touche 🌐 ouvre les émojis (ou autre) : en mains libres, l'appui bref final ouvre le panneau,
    qui prend le focus au moment où MacDictée cherche le champ de texte."""
    if SETTINGS["hotkey"] != "fn":
        return False
    prefs = NSUserDefaults.alloc().initWithSuiteName_("com.apple.HIToolbox")
    return int(prefs.integerForKey_("AppleFnUsageType")) != 0  # 0 = « Ne rien faire »


def _tcc_preflight(service: str):
    """Statut d'une autorisation (0 accordée, 1 refusée, 2 jamais demandée) via TCC, framework privé d'Apple :
    il n'existe pas d'API publique pour « Enregistrement des sons du système uniquement ». None si indisponible."""
    import ctypes
    try:
        cf = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        cf.CFStringCreateWithCString.restype = ctypes.c_void_p
        cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
        cf.CFRelease.argtypes = [ctypes.c_void_p]
        tcc = ctypes.cdll.LoadLibrary("/System/Library/PrivateFrameworks/TCC.framework/TCC")
        tcc.TCCAccessPreflight.restype = ctypes.c_int
        tcc.TCCAccessPreflight.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        name = cf.CFStringCreateWithCString(None, service.encode(), 0x08000100)  # kCFStringEncodingUTF8
        try:
            return tcc.TCCAccessPreflight(name, None)
        finally:
            cf.CFRelease(name)
    except Exception:
        return None


def system_audio_permission() -> str:
    if not SYSTEM_AUDIO_TAP:
        return "ok" if CGPreflightScreenCaptureAccess() else "refusé"
    return {0: "ok", 1: "refusé", 2: "inconnu"}.get(_tcc_preflight("kTCCServiceAudioCapture"), "tap")


def permissions():
    mic = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeAudio)  # 0 pas demandé, 3 accordé
    return {"micro": {0: "inconnu", 3: "ok"}.get(mic, "refusé"),
            "accessibility": "ok" if AXIsProcessTrusted() else "refusé",
            "screen": system_audio_permission()}  # son du Mac (visios)


def login_item_enabled() -> bool:
    return SMAppService.mainAppService().status() == 1  # SMAppServiceStatusEnabled


def set_login_item(enabled: bool) -> None:
    service = SMAppService.mainAppService()
    ok, err = (service.registerAndReturnError_ if enabled else service.unregisterAndReturnError_)(None)
    if not ok:
        raise RuntimeError(str(err.localizedDescription()) if err else "Impossible de modifier l'ouverture au démarrage")


PRIVACY_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_"


def fmt(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


# ----------------------------------------------------------------------------- enregistrements
class Dictation:
    """Enregistrement d'une dictée (touche maintenue, ou mains libres) ; level = niveau du micro pour la pastille."""

    def __init__(self):
        self.frames = []
        self.stream = None
        self.level = 0.0

    def _on_audio(self, data, frames, time_info, status):
        self.frames.append(data.copy())
        self.level = float(np.sqrt(np.mean(data ** 2)))

    def start(self):
        self.frames = []
        self.level = 0.0
        self.stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32", callback=self._on_audio)
        self.stream.start()

    def stop(self) -> np.ndarray:
        self.stream.stop()
        self.stream.close()
        self.stream = None
        if not self.frames:
            return np.zeros(0, dtype="float32")
        return np.concatenate(self.frames)[:, 0]


def audio_helper() -> Path:
    """Binaire Swift qui capture le son du Mac : dans l'app (Contents/MacOS) ou compilé dans native/."""
    bundle = NSBundle.mainBundle()
    if bundle.bundleIdentifier() == "fr.local.macdictee":
        bundled = Path(str(bundle.executablePath())).parent / "macdictee-audio"
        if bundled.exists():
            return bundled
    return Path(__file__).resolve().parent / "native" / "macdictee-audio"


class SystemAudio:
    """Son du Mac (visios…) lu en continu depuis macdictee-audio, distribué au rythme du micro."""

    def __init__(self, *flags):
        helper = audio_helper()
        if not helper.exists():
            raise RuntimeError("module de capture du son du Mac absent")
        self.proc = subprocess.Popen([str(helper), *flags], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
        self.buf = bytearray()
        self.lock = threading.Lock()
        self.debt = 0  # échantillons de silence ajoutés faute de données, à rattraper pour rester synchrone
        self.ready = threading.Event()
        self.error = None
        threading.Thread(target=self._read_audio, daemon=True).start()
        threading.Thread(target=self._read_log, daemon=True).start()
        self.ready.wait(5)
        if self.error or not self.ready.is_set():
            self.stop()
            raise RuntimeError(self.error or "le son du Mac ne démarre pas")

    def _read_audio(self):
        while chunk := self.proc.stdout.read1(65536):
            with self.lock:
                self.buf += chunk

    def _read_log(self):
        for line in self.proc.stderr:
            line = line.decode("utf-8", "replace").strip()
            if line == "ready":
                self.ready.set()
            elif line.startswith("error"):
                self.error = line.removeprefix("error:").strip()
                self.ready.set()

    def take(self, n) -> np.ndarray:
        """n échantillons int16, alignés sur le bloc micro du même instant."""
        deadline = time.monotonic() + 0.25
        while len(self.buf) < n * 4 and time.monotonic() < deadline and self.proc.poll() is None:
            time.sleep(0.01)
        with self.lock:
            got = min(len(self.buf) // 4, n)
            data = np.frombuffer(bytes(self.buf[:got * 4]), dtype="float32")
            del self.buf[:got * 4]
            self.debt += n - got
            spare = len(self.buf) // 4 - n  # retard rattrapé : on jette l'excédent dû aux silences ajoutés
            if self.debt and spare > 0:
                drop = min(self.debt, spare)
                del self.buf[:drop * 4]
                self.debt -= drop
        out = np.zeros(n, dtype="int16")
        out[:got] = np.clip(data * 32767, -32768, 32767).astype("int16")
        return out

    def stop(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=2)
        except Exception:
            self.proc.kill()


class LongRecording:
    """Sans limite de durée : WAV écrit en continu + transcription par blocs coupés sur les silences.
    Avec le son du Mac : <réunion>.wav = micro + son du Mac (ce qu'on transcrit et écoute),
    <réunion>_systeme.wav = son du Mac seul (sert à distinguer « Moi » des gens de la visio)."""

    def __init__(self, system=None):
        self.system = system
        self.started = datetime.datetime.now()
        stamp = self.started.strftime("%Y-%m-%d_%Hh%M")
        self.wav_path = OUT_DIR / f"{stamp}.wav"
        self.txt_path = OUT_DIR / f"{stamp}.txt"
        self.txt_path.write_text(
            f"Transcription de l'enregistrement du {datetime.datetime.now():%d/%m/%Y à %Hh%M}\n\n",
            encoding="utf-8")
        self.q = queue.Queue()
        self.segments = []  # (début, fin, texte)
        self.stream = sd.InputStream(samplerate=SR, channels=1, dtype="int16", blocksize=BLOCK,
                                     callback=lambda d, f, t, s: self.q.put(d.copy()))
        self.worker = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.stream.start()
        self.worker.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()
        self.q.put(None)
        self.worker.join()
        if self.system:
            self.system.stop()

    @staticmethod
    def _tail_is_silent(buf):
        tail = np.concatenate(buf[-5:]).astype("float32")  # 0,5 s
        return float(np.sqrt(np.mean(tail ** 2))) < SILENCE_RMS

    def _run(self):
        wf = wave.open(str(self.wav_path), "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        sys_wf = None
        if self.system:
            sys_wf = wave.open(str(speakers.system_path(self.txt_path)), "wb")
            sys_wf.setnchannels(1)
            sys_wf.setsampwidth(2)
            sys_wf.setframerate(SR)
        buf, n_buf, offset = [], 0, 0.0
        while True:
            item = self.q.get()
            if item is not None:
                if self.system:
                    other = self.system.take(len(item))
                    sys_wf.writeframes(other.tobytes())
                    item = np.clip(item[:, 0].astype("int32") + other, -32768, 32767).astype("int16")[:, None]
                wf.writeframes(item.tobytes())
                buf.append(item)
                n_buf += len(item)
            secs = n_buf / SR
            cut = item is None or secs >= MAX_CHUNK or (secs >= MIN_CHUNK and self._tail_is_silent(buf))
            if cut:
                if secs > 0.5:
                    audio = np.concatenate(buf)[:, 0].astype("float32") / 32768.0
                    try:
                        self._transcribe_chunk(audio, offset)
                    except Exception as e:  # un bloc raté ne doit pas casser l'enregistrement
                        with open(self.txt_path, "a", encoding="utf-8") as f:
                            f.write(f"[{fmt(offset)}] (erreur de transcription : {e})\n")
                    offset += secs
                buf, n_buf = [], 0
            if item is None:
                break
        wf.close()
        if sys_wf:
            sys_wf.close()

    def _transcribe_chunk(self, audio, offset):
        res = transcribe(audio, words=True)
        with open(self.txt_path, "a", encoding="utf-8") as f, \
                open(speakers.words_path(self.txt_path), "a", encoding="utf-8") as fw:
            for s in res["segments"]:
                a, b, text = s["start"] + offset, s["end"] + offset, s["text"].strip()
                self.segments.append((a, b, text))
                f.write(f"[{fmt(a)}] {text}\n")
                words = [[w["word"], round(float(w["start"]) + offset, 2), round(float(w["end"]) + offset, 2)]
                         for w in s.get("words", [])]
                fw.write(json.dumps({"start": round(a, 2), "end": round(b, 2), "words": words}, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------- réunions (pour la fenêtre)
RECORDING_STEM = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}h\d{2}$")


SUMMARIZING = set()  # réunions dont le compte rendu est en cours de rédaction


def summary_path(txt: Path) -> Path:
    return Path(f"{txt.with_suffix('')}_compte-rendu.md")


def recording_txt(stem) -> Path:
    if not RECORDING_STEM.match(stem):
        raise ValueError("Nom de fichier invalide")
    return OUT_DIR / f"{stem}.txt"


def trash(path: Path):
    """Corbeille plutôt que suppression définitive : on peut toujours récupérer un fichier."""
    if path.exists():
        ok, _, err = NSFileManager.defaultManager().trashItemAtURL_resultingItemURL_error_(
            NSURL.fileURLWithPath_(str(path)), None, None)
        if not ok:
            raise RuntimeError(str(err.localizedDescription()) if err else f"Impossible de supprimer {path.name}")


def list_recordings(diarizing=None):
    diarizing = diarizing or {}
    out = []
    for txt in sorted(OUT_DIR.glob("*.txt"), reverse=True):
        if not RECORDING_STEM.match(txt.stem):
            continue
        wav = txt.with_suffix(".wav")
        speakers_json, speakers_txt = speakers.paths(txt)
        lines = [l for l in txt.read_text(encoding="utf-8").splitlines()[2:] if l.strip()]
        out.append({
            "stem": txt.stem,
            "txt": str(txt),
            "wav": str(wav) if wav.exists() else None,
            "speakers_txt": str(speakers_txt) if speakers_json.exists() else None,
            "summary": str(summary_path(txt)) if summary_path(txt).exists() else None,
            "summarizing": txt.stem in SUMMARIZING,
            "system": speakers.system_path(txt).exists(),
            "diarizing": diarizing.get(txt.stem),
            "duration": max(0, wav.stat().st_size - 44) / (2 * SR) if wav.exists() else None,
            "lines": len(lines),
            "preview": re.sub(r"^\[[\d:]+\]\s*", "", lines[0])[:160] if lines else "",
        })
    return out


def inside_out_dir(path) -> Path:
    p = Path(path).resolve()
    if OUT_DIR.resolve() not in p.parents:
        raise ValueError("Chemin hors du dossier des transcriptions")
    return p


# ----------------------------------------------------------------------------- application
STATUS_TITLES = {"pret": "🎙", "ecoute": "🔴", "mainslibres": "🔴", "transcription": "⏳", "reecriture": "✨",
                 "reunion": "🔴 réunion"}


class App(rumps.App):
    def __init__(self):
        super().__init__("🎙", quit_button="Quitter")
        self.paste_item = rumps.MenuItem("Coller dans le champ de texte actif", callback=self.toggle_setting)
        self.restore_item = rumps.MenuItem("Remettre le presse-papiers après collage", callback=self.toggle_setting)
        self.toggle_item = rumps.MenuItem("Démarrer l'enregistrement long", callback=self.toggle_long)
        self.menu = [rumps.MenuItem("Ouvrir MacDictée…", callback=lambda _: self.window.show()), None,
                     self.toggle_item, self.paste_item, self.restore_item, None,
                     rumps.MenuItem("Ouvrir le dossier des transcriptions", callback=lambda _: subprocess.run(["open", str(OUT_DIR)]))]
        self.sync_menu()
        self.window = Window(self.handle)
        self.dictation = Dictation()
        self.dictating = False  # un enregistrement de dictée est en cours (maintenu, tap en attente ou mains libres)
        self.handsfree = False
        self.tap_pending = False  # tap bref : on attend un éventuel second tap
        self.ignore_release = False  # relâchement d'un appui qui a déjà servi (passage en / sortie de mains libres)
        self.press_time = 0.0
        self.tap_gen = 0
        self.overlay = Overlay(lambda: self.dictation.level)
        self.paste_gen = 0
        self.long = None
        self.status = "pret"
        self.model_ready = False
        self._monitors = []
        self.diarizing = {}  # réunion -> avancement (0..1) de la séparation des intervenants
        self.ai_state = "pret" if REWRITER.downloaded() else "absent"  # absent / telechargement / pret / erreur
        self.ai_error = ""
        self.update = None  # dernière version disponible (dict de updates.latest_release) si plus récente
        self.update_item = None
        rumps.events.before_start.register(self._on_start)
        threading.Thread(target=self._update_loop, daemon=True).start()
        threading.Thread(target=self._warm_up, daemon=True).start()

    def _on_start(self):
        install_edit_menu()
        self.apply_dock()
        # écoute d'Option droite sur le thread principal (nécessite l'autorisation Accessibilité)
        self._monitors = [
            NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(NSEventMaskFlagsChanged, self.on_flags),
            NSEvent.addLocalMonitorForEventsMatchingMask_handler_(NSEventMaskFlagsChanged, self._on_local_flags),
            NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(NSEventMaskKeyDown, self.on_key_down),
            NSEvent.addLocalMonitorForEventsMatchingMask_handler_(NSEventMaskKeyDown, self._on_local_key_down),
        ]
        perms = permissions()
        if perms["accessibility"] != "ok":
            AXIsProcessTrustedWithOptions({"AXTrustedCheckOptionPrompt": True})
        if perms["micro"] == "inconnu":
            AVCaptureDevice.requestAccessForMediaType_completionHandler_(AVMediaTypeAudio, lambda ok: self.push_state())
        if perms["micro"] != "ok" or perms["accessibility"] != "ok":  # le son du Mac est facultatif
            self.window.show()  # premier lancement ou autorisation manquante : on montre où ça coince

    def _warm_up(self):
        transcribe(np.zeros(SR, dtype="float32"))  # la 1re dictée est ensuite instantanée
        self.model_ready = True
        self.push_state()
        while True:  # libère le modèle quand on ne s'en sert plus (une vérification par minute, rien d'autre)
            time.sleep(60)
            idle = time.monotonic() - _last_use > IDLE_UNLOAD
            if SETTINGS["free_memory"] and idle and not self.dictating and self.long is None and not self.diarizing:
                unload_model()

    # --- état partagé avec la fenêtre
    def set_status(self, status):
        self.status = status
        callAfter(setattr, self, "title", STATUS_TITLES[status])  # appelé depuis des threads
        callAfter(self._sync_overlay, status)
        self.push_state()

    def _sync_overlay(self, status):
        if SETTINGS["overlay"] and status in Overlay.TEXTS:
            self.overlay.show(status)
        else:
            self.overlay.hide()

    def snapshot(self):
        bundle = str(NSBundle.mainBundle().bundlePath())
        return {
            "status": self.status,
            "model_ready": self.model_ready,
            "model": os.environ.get("WHISPER_MODEL") or "whisper-large-v3-turbo · 8 bits",
            "permissions": permissions(),
            "settings": SETTINGS,
            "long": {"since": self.long.started.isoformat(timespec="seconds"), "txt": str(self.long.txt_path)}
                    if self.long else None,
            "out_dir": str(OUT_DIR),
            "can_restart": bundle.endswith(".app"),
            "login": login_item_enabled(),
            "version": APP_VERSION,
            "hotkeys": {k: v[2] for k, v in HOTKEYS.items()},
            "fn_conflict": fn_key_conflict(),
            "audio_tap": SYSTEM_AUDIO_TAP,
            "ai": {"state": self.ai_state, "error": self.ai_error, "size": AI_MODEL_SIZE},
            "update": self.update,
        }

    # --- mises à jour
    def _update_loop(self):
        time.sleep(10)
        while True:
            try:
                self.check_update()
            except Exception:
                pass  # hors ligne, GitHub indisponible… on réessaiera
            time.sleep(UPDATE_EVERY)

    def check_update(self):
        release = updates.latest_release()
        if not updates.is_newer(release, APP_VERSION):
            return False
        first = self.update is None or self.update["version"] != release["version"]
        self.update = {**release, "progress": None}
        if first:
            notify("MacDictée", f"Nouvelle version {release['version']} disponible",
                   "Ouvrez MacDictée pour la télécharger.")
            callAfter(self._add_update_item)
        self.push_state()
        return True

    def _add_update_item(self):
        title = f"⬆️ Installer la version {self.update['version']}…"
        if self.update_item is None:
            self.update_item = rumps.MenuItem(title, callback=lambda _: self.window.show())
            self.menu.insert_before("Ouvrir MacDictée…", self.update_item)
        else:
            self.update_item.title = title

    def install_update(self):
        if self.update is None or self.update["progress"] is not None:
            return
        self.update["progress"] = 0.0
        self.push_state()
        threading.Thread(target=self._download_update, daemon=True).start()

    def _download_update(self):
        release = self.update
        dest = Path.home() / "Downloads" / f"MacDictee-{release['version']}.dmg"
        last = [0.0]

        def progress(fraction):
            if fraction - last[0] >= 0.02:  # pas plus d'un rafraîchissement par 2 %
                last[0] = fraction
                release["progress"] = fraction
                self.push_state()

        try:
            updates.download(release["dmg"], dest, progress)
        except Exception as e:
            release["progress"] = None
            self.push_state()
            notify("MacDictée", "Téléchargement impossible", str(e)[:200])
            return
        release["progress"] = 1.0
        self.push_state()
        subprocess.run(["open", str(dest)])  # ouvre le DMG : il n'y a plus qu'à glisser l'app dans Applications
        notify("MacDictée", "Mise à jour téléchargée",
               "Glissez MacDictée dans Applications (Remplacer), puis rouvrez-la.")
        time.sleep(2)
        callAfter(rumps.quit_application)  # l'ancienne version doit être fermée pour être remplacée

    def push_state(self):
        if self.window.visible():  # fenêtre fermée : rien à calculer (elle se met à jour en s'ouvrant)
            self.window.emit("state", self.snapshot())

    def apply_dock(self):
        # LSUIElement démarre l'app sans Dock ; on l'y ajoute si le réglage le demande
        NSApp.setActivationPolicy_(NSApplicationActivationPolicyRegular if SETTINGS["dock"]
                                   else NSApplicationActivationPolicyAccessory)

    def sync_menu(self):
        self.paste_item.state = SETTINGS["paste"]
        self.restore_item.state = SETTINGS["restore_clipboard"]

    def toggle_setting(self, sender):
        key = "paste" if sender is self.paste_item else "restore_clipboard"
        SETTINGS[key] = not SETTINGS[key]
        save_settings()
        self.sync_menu()
        self.push_state()

    # --- actions demandées par la fenêtre (thread principal)
    def handle(self, action, args):
        if action == "state":
            return self.snapshot()
        if action == "history":
            return load_history()
        if action == "history_clear":
            clear_history()
            return []
        if action == "copy":
            set_clipboard(args["text"])
            return True
        if action == "recordings":
            return list_recordings(self.diarizing)
        if action == "recording_text":  # version avec les intervenants si elle existe (pour copier / résumer)
            txt = recording_txt(args["stem"])
            named = speakers.paths(txt)[1]
            return (named if named.exists() else txt).read_text(encoding="utf-8")
        if action == "recording_delete":  # what : "audio" (garde le texte) ou "all"
            txt = recording_txt(args["stem"])
            if (self.long and self.long.txt_path == txt) or args["stem"] in self.diarizing:
                raise ValueError("Cette réunion est en cours de traitement")
            files = [txt.with_suffix(".wav"), speakers.system_path(txt)]
            if args["what"] == "all":
                files += [txt, *speakers.paths(txt), speakers.words_path(txt), summary_path(txt),
                          txt.with_name(txt.stem + "_locuteurs.txt")]
            for f in files:
                trash(f)
            return list_recordings(self.diarizing)
        if action == "speakers_get":
            return speakers.read(recording_txt(args["stem"]))
        if action == "speakers_run":
            self.run_speakers(recording_txt(args["stem"]), args.get("count") or None)
            return list_recordings(self.diarizing)
        if action == "summary_get":
            path = summary_path(recording_txt(args["stem"]))
            return path.read_text(encoding="utf-8") if path.exists() else None
        if action == "summary_run":
            self.run_summary(recording_txt(args["stem"]))
            return list_recordings(self.diarizing)
        if action == "ai_download":
            self.download_ai()
            return self.snapshot()
        if action == "speakers_rename":
            txt = recording_txt(args["stem"])
            data = speakers.read(txt)
            data["names"] = {k: (v.strip() or f"Intervenant {int(k) + 1}") for k, v in args["names"].items()}
            speakers.write(txt, data)
            return data
        if action == "open":
            subprocess.run(["open", str(inside_out_dir(args["path"]))])
            return True
        if action == "reveal":
            subprocess.run(["open", "-R", str(inside_out_dir(args["path"]))])
            return True
        if action == "open_folder":
            subprocess.run(["open", str(OUT_DIR)])
            return True
        if action == "toggle_long":
            self.toggle_long(None)
            return self.snapshot()
        if action == "vocab":
            return [w.strip() for w in VOCAB_FILE.read_text(encoding="utf-8").splitlines() if w.strip()]
        if action == "vocab_set":
            words = list(dict.fromkeys(w.strip() for w in args["words"] if w.strip()))
            VOCAB_FILE.write_text("".join(w + "\n" for w in words), encoding="utf-8")
            return words
        if action == "rules":
            return load_store()["rules"]
        if action == "rule_delete":
            with _learn_lock:
                store = load_store()
                store["rules"].pop(args["wrong"], None)
                store["counts"] = {k: v for k, v in store["counts"].items() if not k.startswith(args["wrong"] + "\t")}
                LEARN_FILE.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
            return store["rules"]
        if action == "setting":
            if args["key"] not in DEFAULT_SETTINGS:
                raise ValueError("Réglage inconnu")
            if args["key"] == "hotkey" and args["value"] not in HOTKEYS:
                raise ValueError("Raccourci inconnu")
            SETTINGS[args["key"]] = args["value"]
            save_settings()
            self.sync_menu()
            if args["key"] == "dock":
                self.apply_dock()
                self.window.show()  # repasser en mode Dock fait perdre le premier plan à la fenêtre
            if args["key"] == "ai_rewrite" and args["value"] and self.ai_state in ("absent", "erreur"):
                self.download_ai()  # seule connexion de cette option : le modèle, une fois
            return self.snapshot()
        if action == "update_check":  # réseau : en arrière-plan, la réponse arrive par l'événement "update_checked"
            def run():
                try:
                    self.window.emit("update_checked", {"found": self.check_update()})
                except Exception as e:
                    self.window.emit("update_checked", {"error": str(e)})
            threading.Thread(target=run, daemon=True).start()
            return True
        if action == "update_install":
            self.install_update()
            return self.snapshot()
        if action == "open_url":
            if not args["url"].startswith(f"https://github.com/{updates.REPO}/"):
                raise ValueError("Lien non autorisé")
            subprocess.run(["open", args["url"]])
            return True
        if action == "login":
            set_login_item(bool(args["value"]))
            return self.snapshot()
        if action == "keyboard_settings":
            subprocess.run(["open", "x-apple.systempreferences:com.apple.Keyboard-Settings.extension"])
            return True
        if action == "privacy":
            if args["what"] == "ScreenCapture" and SYSTEM_AUDIO_TAP:
                args = {"what": "AudioCapture"}
            if args["what"] == "AudioCapture" and system_audio_permission() == "inconnu":
                # jamais demandée : un court essai de capture fait apparaître la demande de macOS
                def ask():
                    try:
                        SystemAudio().stop()
                    except Exception:
                        pass
                    time.sleep(1)
                    self.push_state()
                threading.Thread(target=ask, daemon=True).start()
                return True
            elif args["what"] == "ScreenCapture" and not CGPreflightScreenCaptureAccess():
                CGRequestScreenCaptureAccess()
            if args["what"] == "Microphone" and permissions()["micro"] == "inconnu":
                AVCaptureDevice.requestAccessForMediaType_completionHandler_(AVMediaTypeAudio, lambda ok: self.push_state())
            elif args["what"] == "Accessibility":
                AXIsProcessTrustedWithOptions({"AXTrustedCheckOptionPrompt": True})
            subprocess.run(["open", PRIVACY_URL + args["what"]])
            return True
        if action == "restart":
            bundle = str(NSBundle.mainBundle().bundlePath())
            subprocess.Popen(["/bin/sh", "-c", 'sleep 1; open "$0"', bundle])
            rumps.quit_application()
            return True
        raise ValueError(f"Action inconnue : {action}")

    # --- dictée
    def on_flags(self, event):
        keycode, flag, _ = HOTKEYS.get(SETTINGS["hotkey"], HOTKEYS["alt_r"])
        if event.keyCode() != keycode:
            return
        if event.modifierFlags() & flag:
            self.on_press()
        else:
            self.on_release()

    def on_key_down(self, event):
        if not self.dictating:
            return
        if self.handsfree:
            if event.keyCode() == ESCAPE_KEYCODE:  # Échap annule le mode mains libres
                self.cancel_dictation()
            return
        # touche tapée pendant que le raccourci est maintenu (ex. ⌥ + e pour un caractère) : ce n'est pas une dictée
        self.cancel_dictation()

    def _on_local_key_down(self, event):
        self.on_key_down(event)
        return event

    def on_press(self):
        if self.handsfree:  # appui pendant le mode mains libres : on termine
            self.handsfree = False
            self.ignore_release = True
            self.finish_dictation()
            return
        if self.tap_pending:  # second tap rapide : mains libres, l'enregistrement continue
            self.tap_pending = False
            self.handsfree = True
            self.ignore_release = True
            play("Morse")
            self.set_status("mainslibres")
            return
        if self.dictating or self.long is not None:
            return
        self.dictating = True
        self.press_time = time.monotonic()
        self.paste_gen += 1  # clôt la surveillance de la dictée précédente (analyse la correction)
        self.set_status("ecoute")
        play("Tink")
        try:
            self.dictation.start()
        except Exception as e:
            self.dictating = False
            self.set_status("pret")
            notify("MacDictée", "Micro indisponible", str(e)[:200])
            return
        threading.Thread(target=lambda: enable_accessibility(front_pid()), daemon=True).start()

    def on_release(self):
        if self.ignore_release:
            self.ignore_release = False
            return
        if not self.dictating or self.handsfree or self.tap_pending:
            return
        if SETTINGS["handsfree"] and time.monotonic() - self.press_time < TAP_MAX:
            self.tap_pending = True  # on continue d'enregistrer en attendant un éventuel second tap
            self.tap_gen += 1
            callLater(DOUBLE_TAP_WINDOW, self._tap_timeout, self.tap_gen)
            return
        self.finish_dictation()

    def _tap_timeout(self, gen):
        if self.tap_pending and gen == self.tap_gen:  # un seul tap bref : rien à transcrire
            self.cancel_dictation()

    def cancel_dictation(self):
        self.dictating = self.handsfree = self.tap_pending = False
        self.dictation.stop()
        self.set_status("pret")

    def finish_dictation(self):
        self.dictating = self.handsfree = self.tap_pending = False
        audio = self.dictation.stop()
        play("Pop")
        self.set_status("transcription")
        threading.Thread(target=self._finish_dictation, args=(audio, front_app_name()), daemon=True).start()

    def _finish_dictation(self, audio, app_name):
        try:
            if len(audio) < SR // 4 or float(np.abs(audio).max()) < 0.01:
                return  # trop court ou silence
            text = transcribe(audio)["text"]
            if not text:
                return
            if SETTINGS["ai_rewrite"] and self.ai_state == "pret" and needs_ai(text):
                self.set_status("reecriture")
                try:
                    text, _ = ai_call(REWRITER.clean, text)  # repli sur la transcription si le garde-fou refuse
                except Exception as e:
                    notify("MacDictée", "Réécriture IA impossible", f"Texte collé tel quel : {e}"[:200])
            if SETTINGS["format"]:
                text = format_dictation(text)
            add_history(text, app_name)
            self.window.emit("history", load_history())
            saved = save_clipboard()
            set_clipboard(text)  # sans collage, le texte reste dans le presse-papiers
            if not SETTINGS["paste"]:
                return
            focused = text_field_focused()
            if focused is False:
                notify("MacDictée", "Copié dans le presse-papiers",
                       "Aucun champ de texte actif : faites Cmd+V où vous voulez.")
            else:  # True, ou indéterminé : on tente de coller
                el = get_focused_element()
                before = get_value(el) if el is not None else None
                count = clipboard_count()
                press_paste()
                if SETTINGS["restore_clipboard"]:
                    time.sleep(0.5)  # laisse l'app cible lire le presse-papiers avant de le remettre
                    if clipboard_count() == count:  # sauf si vous avez copié autre chose entre-temps
                        restore_clipboard(saved)
                self.paste_gen += 1
                threading.Thread(target=self._watch_edits, args=(text, el, before, self.paste_gen),
                                 daemon=True).start()
        except Exception as e:
            notify("MacDictée", "Erreur de dictée", str(e)[:200])
        finally:
            if not self.dictating:  # une nouvelle dictée a pu commencer pendant la transcription
                self.set_status("pret")

    def _watch_edits(self, text, el, before, gen):
        """Surveille le champ après le collage : si vous corrigez un mot, on l'apprend."""
        try:
            time.sleep(0.7)
            after = get_value(el) if el is not None else None
            if before is None or after is None:
                return  # l'app n'expose pas son texte : pas d'apprentissage possible ici
            m = min(len(before), len(after))
            i = 0
            while i < m and before[i] == after[i]:
                i += 1
            j = 0
            while j < m - i and before[-1 - j] == after[-1 - j]:
                j += 1
            inserted = after[i:len(after) - j]
            if difflib.SequenceMatcher(None, inserted.strip(), text.strip()).ratio() < 0.8:
                return
            pid = front_pid()
            deadline = time.time() + 45
            while time.time() < deadline and self.paste_gen == gen and front_pid() == pid:
                time.sleep(2)
            cur = get_value(el)
            if cur is None or len(cur) < i + j:
                return
            final = cur[i:len(cur) - j]
            if final.strip() != inserted.strip():
                learn_from_edit(inserted, final)
        except Exception:
            pass

    # --- enregistrement long
    def toggle_long(self, _):
        if self.long is None:
            if self.dictating:
                return
            system = None
            if SETTINGS["system_audio"]:
                if not SYSTEM_AUDIO_TAP and not CGPreflightScreenCaptureAccess():
                    CGRequestScreenCaptureAccess()  # demande l'autorisation (une fois) ; on enregistre le micro seul
                    notify("MacDictée", "Son du Mac non capté",
                           "Autorisez « Enregistrement de l'écran et audio système » pour MacDictée, puis relancez-la.")
                else:
                    try:
                        system = SystemAudio()
                    except Exception as e:
                        try:  # le tap Core Audio refuse : ScreenCaptureKit si son autorisation est déjà accordée
                            if not (SYSTEM_AUDIO_TAP and CGPreflightScreenCaptureAccess()):
                                raise
                            system = SystemAudio("--screen")
                        except Exception:
                            notify("MacDictée", "Son du Mac non capté", f"Micro seul : {e}"[:200])
            self.long = LongRecording(system)
            self.long.start()
            self.toggle_item.title = "Arrêter l'enregistrement long"
            self.set_status("reunion")
        else:
            rec, self.long = self.long, None
            self.toggle_item.title = "Démarrer l'enregistrement long"
            self.set_status("transcription")
            threading.Thread(target=self._finish_long, args=(rec,), daemon=True).start()

    def _finish_long(self, rec):
        try:
            rec.stop()
            system = speakers.system_path(rec.txt_path)
            if rec.system and system.exists() and not np.any(speakers.load_wav(system)):
                notify("MacDictée", "Aucun son du Mac capté",
                       "Si une visio était en cours : Réglages Système > Confidentialité > « Enregistrement des sons "
                       "du système uniquement » > MacDictée, puis relancez MacDictée.")
        finally:
            self.set_status("pret")
            self.window.emit("recordings", list_recordings(self.diarizing))
        if SETTINGS["speakers"] and rec.segments:
            self.run_speakers(rec.txt_path)
        else:
            notify("MacDictée", "Transcription terminée", rec.txt_path.name)

    # --- intervenants
    def run_speakers(self, txt, count=None):
        stem = txt.stem
        if stem in self.diarizing:
            return
        if not txt.with_suffix(".wav").exists():
            raise ValueError("L'audio de cette réunion a été supprimé")
        self.diarizing[stem] = 0.0
        threading.Thread(target=self._speakers_worker, args=(txt, count), daemon=True).start()

    # --- IA locale : téléchargement (une fois) et comptes rendus
    def download_ai(self):
        if self.ai_state == "telechargement":
            return
        self.ai_state, self.ai_error = "telechargement", ""
        self.push_state()

        def run():
            try:
                REWRITER.download()
                self.ai_state = "pret"
                notify("MacDictée", "IA locale prête", "Réécriture des dictées et comptes rendus disponibles.")
            except Exception as e:
                self.ai_state, self.ai_error = "erreur", str(e)[:200]
            self.push_state()
        threading.Thread(target=run, daemon=True).start()

    def run_summary(self, txt):
        if self.ai_state != "pret":
            raise ValueError("Activez d'abord l'IA locale dans Réglages")
        if txt.stem in SUMMARIZING:
            return
        SUMMARIZING.add(txt.stem)

        def run():
            named = speakers.paths(txt)[1]
            transcript = (named if named.exists() else txt).read_text(encoding="utf-8")
            try:
                summary = ai_call(REWRITER.summarize, transcript)
                summary_path(txt).write_text(
                    f"# Compte rendu : {txt.stem}\n\n{summary}\n\n_Généré sur ce Mac par l'IA locale de MacDictée, "
                    f"à relire._\n", encoding="utf-8")
                notify("MacDictée", "Compte rendu prêt", txt.stem)
            except Exception as e:
                notify("MacDictée", "Compte rendu impossible", str(e)[:200])
            finally:
                SUMMARIZING.discard(txt.stem)
                self.window.emit("recordings", list_recordings(self.diarizing))
        threading.Thread(target=run, daemon=True).start()

    def _speakers_worker(self, txt, count):
        stem, last = txt.stem, [0.0]

        def progress(fraction):
            if fraction - last[0] >= 0.02:
                last[0] = fraction
                self.diarizing[stem] = fraction
                self.window.emit("speakers_progress", {"stem": stem, "progress": fraction})

        try:
            data = speakers.run(txt, txt.with_suffix(".wav"), count, progress, fix_text=apply_rules,
                                transcribe_words=lambda audio: transcribe(audio, words=True)["segments"])
            notify("MacDictée", "Réunion prête", f"{len(data['names'])} intervenant(s) identifié(s) : renommez-les dans MacDictée.")
        except Exception as e:
            notify("MacDictée", "Intervenants non identifiés", str(e)[:200])
        finally:
            self.diarizing.pop(stem, None)
            self.window.emit("recordings", list_recordings(self.diarizing))
            self.window.emit("speakers_done", {"stem": stem})


def _reopen(delegate, app, has_visible_windows):
    """Clic sur l'icône du Dock : on rouvre la fenêtre (méthode ajoutée au délégué de rumps)."""
    _APP.window.show()
    return True


objc.classAddMethods(rumps.rumps.NSApp, [
    objc.selector(_reopen, selector=b"applicationShouldHandleReopen:hasVisibleWindows:", signature=b"Z@:@Z")])
_APP = None


def self_test(wav_path):
    """MACDICTEE_SELFTEST=fichier.wav : vérifie qu'une app construite sait transcrire, sans interface."""
    import sys
    with wave.open(wav_path) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype="int16").astype("float32") / 32768.0
    print("SELFTEST:", transcribe(audio)["text"], flush=True)
    words = [w for seg in transcribe(audio, words=True)["segments"] for w in seg.get("words", [])]
    print("SELFTEST mots horodatés :", len(words), "| premier :", words[0]["word"].strip(), round(float(words[0]["start"]), 2), flush=True)
    speakers.ensure_models()
    print("SELFTEST intervenants :", len({s for _, _, s in speakers.diarize(audio)}), flush=True)
    if REWRITER.downloaded():
        text, used = REWRITER.clean("euh on se retrouve à 14h, non pardon à 15h devant la gare")
        print("SELFTEST IA locale :", used, "->", text, "| transformers embarqué :",
              hasattr(sys.modules.get("transformers"), "__file__"), flush=True)
    try:
        SystemAudio().stop()
        print("SELFTEST son du Mac : ok", flush=True)
    except Exception as e:  # depuis un terminal sans autorisation : refus attendu, mais le module a bien tourné
        print("SELFTEST son du Mac :", audio_helper().name, "->", e, flush=True)
    try:
        print("SELFTEST mise à jour :", APP_VERSION, "->", updates.latest_release()["version"], flush=True)
    except Exception as e:  # pas encore de version publiée, hors ligne…
        print("SELFTEST mise à jour :", type(e).__name__, e, flush=True)
    sys.exit(0)


if __name__ == "__main__":
    if os.environ.get("MACDICTEE_SELFTEST"):
        self_test(os.environ["MACDICTEE_SELFTEST"])
    _APP = App()
    _APP.run()
