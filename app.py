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
import queue
import re
import subprocess
import threading
import time
import wave
from pathlib import Path

import mlx_whisper
import numpy as np
import rumps
import sounddevice as sd
from AppKit import NSBundle, NSEvent, NSEventMaskFlagsChanged, NSPasteboard, NSPasteboardItem, NSWorkspace
from ApplicationServices import (AXIsProcessTrusted, AXIsProcessTrustedWithOptions, AXUIElementCopyAttributeValue,
                                 AXUIElementCreateApplication, AXUIElementIsAttributeSettable,
                                 AXUIElementSetAttributeValue)
from AVFoundation import AVCaptureDevice, AVMediaTypeAudio
from PyObjCTools.AppHelper import callAfter
from Quartz import CGEventCreateKeyboardEvent, CGEventPost, CGEventSetFlags, kCGEventFlagMaskCommand, kCGHIDEventTap

from ui import Window

SR = 16000
MODEL = os.environ.get("WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
# Option droite. pynput est abandonné : il appelle des API clavier hors du thread principal et macOS tue l'app.
HOTKEY_KEYCODE = 61
HOTKEY_FLAG = 0x40  # NX_DEVICERALTKEYMASK : Option droite enfoncée
V_KEYCODE = 9
OUT_DIR = Path.home() / "Transcriptions"
OUT_DIR.mkdir(exist_ok=True)

SETTINGS_FILE = OUT_DIR / "reglages.json"
DEFAULT_SETTINGS = {
    "paste": True,  # coller dans le champ de texte actif
    "restore_clipboard": True,  # remettre l'ancien presse-papiers après collage
    "language": os.environ.get("WHISPER_LANG", "fr"),  # "" = détection automatique
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


def transcribe(audio: np.ndarray) -> dict:
    kwargs = dict(path_or_hf_repo=MODEL, condition_on_previous_text=False,
                  no_speech_threshold=0.6, compression_ratio_threshold=2.4)
    if SETTINGS["language"]:
        kwargs["language"] = SETTINGS["language"]
    prompt = vocab_prompt()
    if prompt:
        kwargs["initial_prompt"] = prompt
    with _model_lock:
        res = mlx_whisper.transcribe(audio, **kwargs)
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
    subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=False)


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


def permissions():
    mic = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeAudio)  # 0 pas demandé, 3 accordé
    return {"micro": {0: "inconnu", 3: "ok"}.get(mic, "refusé"),
            "accessibility": "ok" if AXIsProcessTrusted() else "refusé"}


PRIVACY_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_"


def fmt(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


# ----------------------------------------------------------------------------- enregistrements
class Dictation:
    """Enregistrement court tant que la touche est maintenue."""

    def __init__(self):
        self.frames = []
        self.stream = None

    def start(self):
        self.frames = []
        self.stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                                     callback=lambda d, f, t, s: self.frames.append(d.copy()))
        self.stream.start()

    def stop(self) -> np.ndarray:
        self.stream.stop()
        self.stream.close()
        self.stream = None
        if not self.frames:
            return np.zeros(0, dtype="float32")
        return np.concatenate(self.frames)[:, 0]


class LongRecording:
    """Sans limite de durée : WAV écrit en continu + transcription par blocs coupés sur les silences."""

    def __init__(self):
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

    @staticmethod
    def _tail_is_silent(buf):
        tail = np.concatenate(buf[-5:]).astype("float32")  # 0,5 s
        return float(np.sqrt(np.mean(tail ** 2))) < SILENCE_RMS

    def _run(self):
        wf = wave.open(str(self.wav_path), "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        buf, n_buf, offset = [], 0, 0.0
        while True:
            item = self.q.get()
            if item is not None:
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

    def _transcribe_chunk(self, audio, offset):
        res = transcribe(audio)
        with open(self.txt_path, "a", encoding="utf-8") as f:
            for s in res["segments"]:
                a, b, text = s["start"] + offset, s["end"] + offset, s["text"].strip()
                self.segments.append((a, b, text))
                f.write(f"[{fmt(a)}] {text}\n")

    def label_speakers(self):
        """Optionnel : pyannote (nécessite HF_TOKEN + conditions du modèle acceptées)."""
        if not HF_TOKEN:
            return None
        from pyannote.audio import Pipeline

        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=HF_TOKEN)
        try:
            import torch

            if torch.backends.mps.is_available():
                pipe.to(torch.device("mps"))  # GPU Apple : bien plus rapide
        except Exception:
            pass
        diar = pipe(str(self.wav_path))
        turns = [(t.start, t.end, spk) for t, _, spk in diar.itertracks(yield_label=True)]
        out = self.txt_path.with_name(self.txt_path.stem + "_locuteurs.txt")
        with open(out, "w", encoding="utf-8") as f:
            for a, b, text in self.segments:
                best, best_ov = "?", 0.0
                for ta, tb, spk in turns:
                    ov = min(b, tb) - max(a, ta)
                    if ov > best_ov:
                        best, best_ov = spk, ov
                f.write(f"[{fmt(a)}] {best} : {text}\n")
        return out


# ----------------------------------------------------------------------------- réunions (pour la fenêtre)
RECORDING_STEM = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}h\d{2}$")


def list_recordings():
    out = []
    for txt in sorted(OUT_DIR.glob("*.txt"), reverse=True):
        if not RECORDING_STEM.match(txt.stem):
            continue
        wav = txt.with_suffix(".wav")
        speakers = txt.with_name(txt.stem + "_locuteurs.txt")
        lines = [l for l in txt.read_text(encoding="utf-8").splitlines()[2:] if l.strip()]
        out.append({
            "stem": txt.stem,
            "txt": str(txt),
            "wav": str(wav) if wav.exists() else None,
            "speakers": str(speakers) if speakers.exists() else None,
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
STATUS_TITLES = {"pret": "🎙", "ecoute": "🔴", "transcription": "⏳", "reunion": "🔴 réunion"}


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
        self.dictating = False
        self.paste_gen = 0
        self.long = None
        self.status = "pret"
        self.model_ready = False
        self._monitors = []
        rumps.events.before_start.register(self._on_start)
        threading.Thread(target=self._warm_up, daemon=True).start()

    def _on_start(self):
        # écoute d'Option droite sur le thread principal (nécessite l'autorisation Accessibilité)
        self._monitors = [
            NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(NSEventMaskFlagsChanged, self.on_flags),
            NSEvent.addLocalMonitorForEventsMatchingMask_handler_(NSEventMaskFlagsChanged, self._on_local_flags),
        ]
        perms = permissions()
        if perms["accessibility"] != "ok":
            AXIsProcessTrustedWithOptions({"AXTrustedCheckOptionPrompt": True})
        if perms["micro"] == "inconnu":
            AVCaptureDevice.requestAccessForMediaType_completionHandler_(AVMediaTypeAudio, lambda ok: self.push_state())
        if set(perms.values()) != {"ok"}:
            self.window.show()  # premier lancement ou autorisation manquante : on montre où ça coince

    def _warm_up(self):
        transcribe(np.zeros(SR, dtype="float32"))  # la 1re dictée est ensuite instantanée
        self.model_ready = True
        self.push_state()

    # --- état partagé avec la fenêtre
    def set_status(self, status):
        self.status = status
        callAfter(setattr, self, "title", STATUS_TITLES[status])  # appelé depuis des threads
        self.push_state()

    def snapshot(self):
        bundle = str(NSBundle.mainBundle().bundlePath())
        return {
            "status": self.status,
            "model_ready": self.model_ready,
            "model": MODEL,
            "permissions": permissions(),
            "settings": SETTINGS,
            "long": {"since": self.long.started.isoformat(timespec="seconds"), "txt": str(self.long.txt_path)}
                    if self.long else None,
            "out_dir": str(OUT_DIR),
            "can_restart": bundle.endswith(".app"),
        }

    def push_state(self):
        self.window.emit("state", self.snapshot())

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
            return list_recordings()
        if action == "recording_text":
            if not RECORDING_STEM.match(args["stem"]):
                raise ValueError("Nom de fichier invalide")
            txt = OUT_DIR / f"{args['stem']}.txt"
            speakers = OUT_DIR / f"{args['stem']}_locuteurs.txt"
            return (speakers if speakers.exists() else txt).read_text(encoding="utf-8")
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
            SETTINGS[args["key"]] = args["value"]
            save_settings()
            self.sync_menu()
            return self.snapshot()
        if action == "privacy":
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
        if event.keyCode() != HOTKEY_KEYCODE:
            return
        if event.modifierFlags() & HOTKEY_FLAG:
            self.on_press()
        else:
            self.on_release()

    def _on_local_flags(self, event):
        self.on_flags(event)  # quand la fenêtre MacDictée a le focus, le moniteur global ne voit rien
        return event

    def on_press(self):
        if not self.dictating and self.long is None:
            self.dictating = True
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
        if self.dictating:
            self.dictating = False
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
            self.long = LongRecording()
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
            out = None
            try:
                out = rec.label_speakers()
            except Exception as e:  # la diarisation est facultative
                notify("MacDictée", "Locuteurs non identifiés", str(e)[:200])
            notify("MacDictée", "Transcription terminée", str(out or rec.txt_path))
        finally:
            self.set_status("pret")
            self.window.emit("recordings", list_recordings())


if __name__ == "__main__":
    App().run()
