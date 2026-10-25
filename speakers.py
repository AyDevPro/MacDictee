"""Séparation des intervenants d'une réunion : « Intervenant 1, 2, 3… », renommables ensuite.

sherpa-onnx (segmentation pyannote 3.0 + empreinte vocale WeSpeaker, en ONNX) : ~35 Mo de modèles
téléchargés à la première utilisation, sans compte Hugging Face ni torch.
Fichiers produits à côté de la transcription :
  <réunion>_intervenants.json  source de vérité (segments + noms)
  <réunion>_intervenants.txt   version lisible, régénérée à chaque renommage (à donner à Claude pour un résumé)
Avec <réunion>_mots.jsonl (horodatage de chaque mot, écrit pendant la réunion), chaque MOT va au bon intervenant :
deux personnes dans une même phrase de Whisper sont séparées. Sans ce fichier (anciennes réunions), phrase par phrase.
Visio (<réunion>_systeme.wav = son du Mac enregistré à part) : un mot prononcé pendant que le son du Mac est actif
vient de la visio, sinon c'est « Moi » ; seuls les gens de la visio passent par la diarisation, sur leur piste propre.
"""
import json
import os
import re
import tarfile
import tempfile
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx

from updates import download

SR = 16000
MODELS_DIR = Path.home() / "Library" / "Application Support" / "MacDictee" / "modeles"
_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEG_URL = f"{_RELEASES}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
EMB_URL = f"{_RELEASES}/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx"
SEG_MODEL = MODELS_DIR / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
EMB_MODEL = MODELS_DIR / "wespeaker_en_voxceleb_resnet34_LM.onnx"

LINE = re.compile(r"^\[(\d+):(\d\d):(\d\d)\]\s*(.*)$")


def ensure_models(progress=lambda f: None):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not SEG_MODEL.exists():
        with tempfile.TemporaryDirectory() as tmp:
            archive = download(SEG_URL, Path(tmp) / "seg.tar.bz2", lambda f: progress(0.2 * f))
            with tarfile.open(archive) as tar:
                tar.extractall(MODELS_DIR, filter="data")
    if not EMB_MODEL.exists():
        download(EMB_URL, EMB_MODEL, lambda f: progress(0.2 + 0.8 * f))


def load_wav(path) -> np.ndarray:
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="int16").astype("float32") / 32768.0


def diarize(audio: np.ndarray, num_speakers=None, progress=lambda f: None):
    """[(début, fin, n° d'intervenant)] ; num_speakers=None : nombre deviné automatiquement."""
    threads = max(1, (os.cpu_count() or 2) // 2)
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(SEG_MODEL)),
            num_threads=threads),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(EMB_MODEL), num_threads=threads),
        # 0,5 (valeur par défaut) regroupe des voix différentes avec ce modèle ; 0,3-0,4 sépare bien sans couper une même voix
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers or -1, threshold=0.4),
        min_duration_on=0.3, min_duration_off=0.5)
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)

    def callback(done, total):
        progress(done / max(total, 1))
        return 0

    result = sd.process(audio, callback=callback).sort_by_start_time()
    turns = [(s.start, s.end, s.speaker) for s in result]
    return turns if num_speakers else merge_phantoms(audio, turns)


# Une même personne est parfois coupée en deux groupes (distance au micro, rire, changement de ton).
# Empreintes calculées sur tout ce qu'a dit chaque groupe. Mesures : voix de synthèse, même voix ≥ 0,92 et
# voix différentes ≤ 0,75 ; vraie réunion à deux (même pièce, même micro), même personne coupée en deux 0,90
# et personnes différentes 0,79-0,81 : le micro tasse tout vers le haut, d'où 0,85.
# Un groupe qui parle très peu (sous les deux seuils) est rattaché à la voix la plus proche.
MERGE_SIMILARITY = 0.85
MIN_SHARE, MIN_SECONDS = 0.10, 30
EMBED_MAX_SECONDS = 120


def _embedding(extractor, audio, turns):
    pieces, total = [], 0
    for a, b, _ in sorted(turns, key=lambda t: t[0] - t[1]):  # les tours les plus longs d'abord
        if b - a < 0.5 or total >= EMBED_MAX_SECONDS:
            continue
        pieces.append(audio[int(a * SR):int(b * SR)])
        total += b - a
    if not pieces:
        return None
    stream = extractor.create_stream()
    stream.accept_waveform(SR, np.concatenate(pieces))
    stream.input_finished()
    if not extractor.is_ready(stream):
        return None
    v = np.array(extractor.compute(stream))
    return v / np.linalg.norm(v)


def merge_phantoms(audio, turns):
    if len({s for _, _, s in turns}) < 2:
        return turns
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(EMB_MODEL), num_threads=max(1, (os.cpu_count() or 2) // 2)))
    groups = {}
    for t in turns:
        groups.setdefault(t[2], []).append(t)
    emb = {s: _embedding(extractor, audio, g) for s, g in groups.items()}
    talk = lambda s: sum(b - a for a, b, _ in groups[s])

    def merge(src, dst):
        groups[dst] += groups.pop(src)
        emb.pop(src)
        emb[dst] = _embedding(extractor, audio, groups[dst])

    while len(groups) > 1:  # 1) voix quasi identiques
        pairs = [(float(emb[x] @ emb[y]), x, y) for x in groups for y in groups
                 if x < y and emb[x] is not None and emb[y] is not None]
        best = max(pairs, default=None)
        if not best or best[0] < MERGE_SIMILARITY:
            break
        _, x, y = best
        merge(*((x, y) if talk(x) < talk(y) else (y, x)))

    total = sum(talk(s) for s in groups)
    for s in sorted(groups, key=talk):  # 2) groupes minuscules, du plus petit au plus grand
        if len(groups) < 2 or talk(s) >= MIN_SHARE * total or talk(s) >= MIN_SECONDS:
            continue
        others = [o for o in groups if o != s]
        target = max(others, key=lambda o: float(emb[s] @ emb[o]) if emb[s] is not None and emb[o] is not None else -1)
        merge(s, target)
    return sorted((a, b, s) for s, g in groups.items() for a, b, _ in g)


def parse_transcript(txt_path: Path, duration: float):
    """Lignes « [hh:mm:ss] texte » -> [(début, fin, texte)] ; la fin est le début de la ligne suivante."""
    lines = []
    for raw in txt_path.read_text(encoding="utf-8").splitlines():
        m = LINE.match(raw)
        if m and m[4].strip():
            lines.append((int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]), m[4].strip()))
    return [(t, lines[i + 1][0] if i + 1 < len(lines) else max(duration, t + 1), text)
            for i, (t, text) in enumerate(lines)]


def words_path(txt_path: Path) -> Path:
    return Path(f"{txt_path.with_suffix('')}_mots.jsonl")


def load_words(txt_path: Path):
    """[(mot, début, fin)] écrits pendant la réunion, ou None (anciennes réunions)."""
    path = words_path(txt_path)
    if not path.exists():
        return None
    words = [tuple(w) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
             for w in json.loads(line)["words"]]
    return words or None


def _speaker_at(a, b, turns):
    """L'intervenant qui parle le plus longtemps entre a et b (ou le plus proche dans le temps)."""
    best, best_ov = None, 0.0
    for ta, tb, spk in turns:
        ov = min(b, tb) - max(a, ta)
        if ov > best_ov:
            best, best_ov = spk, ov
    if best is None and turns:
        best = min(turns, key=lambda t: min(abs(t[0] - a), abs(t[1] - a), abs(t[0] - b)))[2]
    return best if best is not None else 0


ME = "moi"  # étiquette provisoire de l'utilisateur en visio ; devient l'intervenant 0, nommé « Moi »
SYSTEM_ACTIVE_RMS = 0.01  # au-dessus, le son du Mac contient quelqu'un qui parle


def system_path(txt_path: Path) -> Path:
    return Path(f"{txt_path.with_suffix('')}_systeme.wav")


def _number_by_appearance(segments):
    """« Moi » (visio) devient 0 ; les autres sont numérotés dans l'ordre où ils prennent la parole."""
    has_me = any(seg["speaker"] == ME for seg in segments)
    order = {}
    for seg in segments:
        seg["speaker"] = 0 if seg["speaker"] == ME else order.setdefault(seg["speaker"], len(order) + has_me)
    return segments


def assign(transcript, label):
    """Phrase par phrase (anciennes réunions sans horodatage des mots). label(début, fin) -> intervenant."""
    return _number_by_appearance([{"start": a, "end": b, "speaker": label(a, b), "text": text}
                                  for a, b, text in transcript])


SMOOTH_WORDS = 2  # un îlot de 1-2 mots entre deux passages du même intervenant est une erreur de frontière


def assign_words(words, label, fix_text=lambda t: t):
    """Mot par mot, puis regroupement des mots consécutifs du même intervenant."""
    labels = [label(a, b) for _, a, b in words]
    runs = []  # [intervenant, indice début, indice fin exclu]
    for i, spk in enumerate(labels):
        if runs and runs[-1][0] == spk:
            runs[-1][2] = i + 1
        else:
            runs.append([spk, i, i + 1])
    for k in range(1, len(runs) - 1):
        if runs[k][2] - runs[k][1] <= SMOOTH_WORDS and runs[k - 1][0] == runs[k + 1][0]:
            runs[k][0] = runs[k - 1][0]
    segments = []
    for spk, i, j in runs:
        if segments and segments[-1]["speaker"] == spk:
            seg = segments[-1]
        else:
            seg = {"start": words[i][1], "end": words[i][2], "speaker": spk, "text": ""}
            segments.append(seg)
        seg["text"] += "".join(w for w, _, _ in words[i:j])
        seg["end"] = words[j - 1][2]
    for seg in segments:
        seg["text"] = fix_text(seg["text"].strip())
    return _number_by_appearance([s for s in segments if s["text"]])


def fmt(sec) -> str:
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def paths(txt_path: Path):
    stem = txt_path.with_suffix("")
    return Path(f"{stem}_intervenants.json"), Path(f"{stem}_intervenants.txt")


def write(txt_path: Path, data: dict):
    """Enregistre le JSON et régénère le texte lisible (paragraphes regroupés par intervenant)."""
    json_path, out_path = paths(txt_path)
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    header = txt_path.read_text(encoding="utf-8").splitlines()[0]
    blocks, current = [], None
    for seg in data["segments"]:
        name = data["names"].get(str(seg["speaker"]), f"Intervenant {seg['speaker'] + 1}")
        if name != current:
            blocks.append([f"[{fmt(seg['start'])}] {name} :", []])
            current = name
        blocks[-1][1].append(seg["text"])
    body = "\n\n".join(f"{title}\n{' '.join(texts)}" for title, texts in blocks)
    out_path.write_text(f"{header}\n\n{body}\n", encoding="utf-8")


def read(txt_path: Path):
    json_path, _ = paths(txt_path)
    return json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else None


def save_words(txt_path: Path, segments):
    """Écrit <réunion>_mots.jsonl à partir de segments mlx-whisper transcrits avec word_timestamps."""
    with open(words_path(txt_path), "w", encoding="utf-8") as f:
        for s in segments:
            words = [[w["word"], round(float(w["start"]), 2), round(float(w["end"]), 2)] for w in s.get("words", [])]
            f.write(json.dumps({"start": round(float(s["start"]), 2), "end": round(float(s["end"]), 2),
                                "words": words}, ensure_ascii=False) + "\n")


def run(txt_path: Path, wav_path: Path, num_speakers=None, progress=lambda f: None, fix_text=lambda t: t,
        transcribe_words=None) -> dict:
    """Modèles (10 %), diarisation (jusqu'à 90 %), puis écriture des fichiers.
    transcribe_words(audio) -> segments avec mots : pour une ancienne réunion sans _mots.jsonl."""
    ensure_models(lambda f: progress(0.1 * f))
    audio = load_wav(wav_path)
    diar_progress = lambda f: progress(0.1 + 0.8 * f)
    if system_path(txt_path).exists():  # visio : « Moi » quand le son du Mac est silencieux
        system = load_wav(system_path(txt_path))
        others = max(1, num_speakers - 1) if num_speakers else None  # le nombre choisi m'inclut
        turns = diarize(system, others, diar_progress)

        def label(a, b):
            span = system[int(a * SR):int(b * SR) + 1]
            active = len(span) and float(np.sqrt(np.mean(span ** 2))) > SYSTEM_ACTIVE_RMS
            return _speaker_at(a, b, turns) if active else ME
    else:
        turns = diarize(audio, num_speakers, diar_progress)

        def label(a, b):
            return _speaker_at(a, b, turns)
    words = load_words(txt_path)
    if words is None and transcribe_words is not None:
        save_words(txt_path, transcribe_words(audio))
        words = load_words(txt_path)
    progress(1.0)
    segments = (assign_words(words, label, fix_text) if words
                else assign(parse_transcript(txt_path, len(audio) / SR), label))
    previous = read(txt_path)
    speakers = {s["speaker"] for s in segments}
    count = len(speakers)
    if system_path(txt_path).exists() and 0 in speakers:
        names = {"0": "Moi", **{str(i): f"Intervenant {i}" for i in range(1, max(speakers) + 1)}}
    else:
        names = {str(i): f"Intervenant {i + 1}" for i in range(count)}
    if previous and len(previous["names"]) == count:
        names = previous["names"]  # relance avec le même nombre : on garde les noms déjà saisis
    data = {"names": names, "segments": segments}
    write(txt_path, data)
    return data
