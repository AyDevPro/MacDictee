"""Mises à jour : compare la version installée à la dernière version publiée sur GitHub Releases."""
import json
import re
import ssl
import urllib.request
from pathlib import Path

import certifi

REPO = "AyDevPro/MacDictee"
LATEST_URL = f"https://api.github.com/repos/{REPO}/releases/latest"
# certificats explicites : dans l'app autonome, le Python embarqué cherche ceux de Homebrew, absents chez les autres
_SSL = ssl.create_default_context(cafile=certifi.where())


def parse_version(v: str):
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def latest_release() -> dict:
    req = urllib.request.Request(LATEST_URL, headers={"Accept": "application/vnd.github+json",
                                                      "User-Agent": "MacDictee"})
    with urllib.request.urlopen(req, timeout=15, context=_SSL) as r:
        data = json.load(r)
    dmg = next((a for a in data.get("assets", []) if a["name"].endswith(".dmg")), None)
    return {
        "version": data["tag_name"].lstrip("v"),
        "notes": (data.get("body") or "").strip(),
        "page": data["html_url"],
        "dmg": dmg["browser_download_url"] if dmg else None,
        "size": dmg["size"] if dmg else None,
    }


def is_newer(release: dict, current: str) -> bool:
    return bool(release.get("dmg")) and parse_version(release["version"]) > parse_version(current)


def download(url: str, dest: Path, progress) -> Path:
    """Télécharge par blocs de 1 Mo ; progress(fraction) est appelé au fil de l'eau."""
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(url, headers={"User-Agent": "MacDictee"})
    with urllib.request.urlopen(req, timeout=30, context=_SSL) as r, open(tmp, "wb") as f:
        total, done = int(r.headers.get("Content-Length") or 0), 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                progress(done / total)
    tmp.replace(dest)
    return dest
