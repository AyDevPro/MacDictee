"""Séparation des intervenants d'une réunion : « Intervenant 1, 2, 3… », renommables ensuite.

sherpa-onnx (segmentation pyannote 3.0 + empreinte vocale WeSpeaker, en ONNX) : ~35 Mo de modèles
téléchargés à la première utilisation, sans compte Hugging Face ni torch.
Fichiers produits à côté de la transcription :
  <réunion>_intervenants.json  source de vérité (segments + noms)
  <réunion>_intervenants.txt   version lisible, régénérée à chaque renommage (à donner à Claude pour un résumé)
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
    return [(s.start, s.end, s.speaker) for s in result]


def parse_transcript(txt_path: Path, duration: float):
    """Lignes « [hh:mm:ss] texte » -> [(début, fin, texte)] ; la fin est le début de la ligne suivante."""
    lines = []
    for raw in txt_path.read_text(encoding="utf-8").splitlines():
        m = LINE.match(raw)
        if m and m[4].strip():
            lines.append((int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]), m[4].strip()))
    return [(t, lines[i + 1][0] if i + 1 < len(lines) else max(duration, t + 1), text)
            for i, (t, text) in enumerate(lines)]


def assign(transcript, turns):
    """Chaque phrase va à l'intervenant qui parle le plus longtemps pendant celle-ci (ou au plus proche)."""
    out = []
    for a, b, text in transcript:
        best, best_ov = None, 0.0
        for ta, tb, spk in turns:
            ov = min(b, tb) - max(a, ta)
            if ov > best_ov:
                best, best_ov = spk, ov
        if best is None and turns:
            best = min(turns, key=lambda t: min(abs(t[0] - a), abs(t[1] - a)))[2]
        out.append({"start": a, "end": b, "speaker": best if best is not None else 0, "text": text})
    # numérotation dans l'ordre d'apparition : le premier qui parle est « Intervenant 1 »
    order = {}
    for seg in out:
        seg["speaker"] = order.setdefault(seg["speaker"], len(order))
    return out


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


def run(txt_path: Path, wav_path: Path, num_speakers=None, progress=lambda f: None) -> dict:
    """Tout le traitement : modèles (10 %), diarisation (90 %), puis écriture des fichiers."""
    ensure_models(lambda f: progress(0.1 * f))
    audio = load_wav(wav_path)
    turns = diarize(audio, num_speakers, lambda f: progress(0.1 + 0.9 * f))
    segments = assign(parse_transcript(txt_path, len(audio) / SR), turns)
    previous = read(txt_path)
    count = len({s["speaker"] for s in segments})
    names = {str(i): f"Intervenant {i + 1}" for i in range(count)}
    if previous and len(previous["names"]) == count:
        names = previous["names"]  # relance avec le même nombre : on garde les noms déjà saisis
    data = {"names": names, "segments": segments}
    write(txt_path, data)
    return data
