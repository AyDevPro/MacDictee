"""Mise en forme des dictées, par règles (instantané, 100 % local, sans IA) :

- « point numéro 1 … point numéro 2 … », « premièrement … deuxièmement … », « étape 1 … étape 2 … »
  -> liste numérotée (1. 2. 3.)
- « tiret … tiret … » -> liste à puces (Whisper écrit souvent « tiret » prononcé comme « – » après un point)
- « à la ligne » / « nouvelle ligne » -> retour à la ligne ; « nouveau paragraphe » -> ligne vide

Une liste n'est créée que s'il y a au moins deux éléments qui se suivent (1, 2, 3…) : une mention isolée
(« le point numéro 3 de l'ordre du jour ») reste telle quelle.
"""
import re

NUMBERS = {"un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6, "sept": 7, "huit": 8,
           "neuf": 9, "dix": 10}
ORDINALS = {"premier": 1, "première": 1, "deuxième": 2, "second": 2, "seconde": 2, "troisième": 3,
            "quatrième": 4, "cinquième": 5, "sixième": 6, "septième": 7, "huitième": 8, "neuvième": 9,
            "dixième": 10}
ADVERBS = {"premièrement": 1, "deuxièmement": 2, "secondement": 2, "troisièmement": 3, "quatrièmement": 4,
           "cinquièmement": 5, "sixièmement": 6, "septièmement": 7, "huitièmement": 8, "neuvièmement": 9,
           "dixièmement": 10}

_NUM = r"(\d{1,2}|" + "|".join(NUMBERS) + r")"
_ORD = "|".join(sorted(ORDINALS, key=len, reverse=True))
_ADV = "|".join(ADVERBS)
# chaque marqueur capture le numéro dans un groupe nommé différent
MARKER = re.compile(
    rf"(?<![\w-])(?:"
    rf"(?:(?:le|la)\s+)?(?:point|étape|partie|item|élément)\s+(?:numéro\s+|n°\s*)?(?P<num>{_NUM[1:-1]})"
    rf"|(?:(?:le|la|en)\s+)?(?P<ord>{_ORD})\s+(?:point|étape|partie|élément)"
    rf"|(?P<adv>{_ADV})"
    rf")(?![\w-])\s*[,.:;]?",
    re.IGNORECASE)
DASH = re.compile(r"(?<![\w-])tiret(?![\w-])\s*[,.:;]?", re.IGNORECASE)
# tiret écrit par Whisper : seulement après une fin de phrase ou « : » (« Paul – mon collègue – » reste une incise)
DASH_MARK = re.compile(r"(?:(?<=[.:!?])|^)\s*[–—-]\s+")
# avant la commande, on retire une virgule (« Paul, à la ligne ») mais pas le point d'une phrase finie
NEW_PARAGRAPH = re.compile(r"\s*[,;:]?\s*(?<![\w-])nouveau paragraphe(?![\w-])\s*[,.;:]?\s*", re.IGNORECASE)
NEW_LINE = re.compile(r"\s*[,;:]?\s*(?<![\w-])(?:à la ligne|nouvelle ligne)(?![\w-])\s*[,.;:]?\s*", re.IGNORECASE)


def _value(match) -> int:
    if match["num"]:
        n = match["num"].lower()
        return int(n) if n.isdigit() else NUMBERS[n]
    if match["ord"]:
        return ORDINALS[match["ord"].lower()]
    return ADVERBS[match["adv"].lower()]


def _clean_item(text: str, last: bool) -> str:
    text = text.strip().lstrip(",.;:-– ").rstrip(",;: ")
    if not last:
        text = text.rstrip(".")
    text = re.sub(r"\s+(?:et|puis)$", "", text, flags=re.IGNORECASE)  # « … et point numéro 3 » (mot entier)
    return text[:1].upper() + text[1:]


def _clean_intro(text: str) -> str:
    text = text.rstrip().rstrip(",;")
    if not text:
        return ""
    return text if text.endswith((":", ".", "!", "?")) else text + " :"


def _numbered(text: str) -> str:
    matches = list(MARKER.finditer(text))
    # plus longue suite de marqueurs consécutifs 1, 2, 3… (la liste doit commencer à 1)
    best = []
    for i, m in enumerate(matches):
        if _value(m) != 1:
            continue
        run = [m]
        for nxt in matches[i + 1:]:
            if _value(nxt) == _value(run[-1]) + 1:
                run.append(nxt)
        if len(run) > len(best):
            best = run
    if len(best) < 2:
        return text
    intro = _clean_intro(text[:best[0].start()])
    items = []
    for k, m in enumerate(best):
        end = best[k + 1].start() if k + 1 < len(best) else len(text)
        items.append(f"{k + 1}. {_clean_item(text[m.end():end], last=k + 1 == len(best))}")
    return (intro + "\n" if intro else "") + "\n".join(items)


def _bullets(text: str) -> str:
    parts = DASH.split(text)
    if len(parts) < 3:
        parts = DASH_MARK.split(text)
    if len(parts) < 3:  # au moins deux « tiret »
        return text
    intro = _clean_intro(parts[0])
    items = [f"- {_clean_item(p, last=k == len(parts) - 2)}" for k, p in enumerate(parts[1:]) if p.strip()]
    return (intro + "\n" if intro else "") + "\n".join(items)


def format_dictation(text: str) -> str:
    text = NEW_PARAGRAPH.sub("\n\n", text)
    text = NEW_LINE.sub("\n", text)
    blocks = [_bullets(_numbered(block.strip())) for block in text.split("\n")]
    return "\n".join(_capitalize(b) for b in blocks).strip()


def _capitalize(line: str) -> str:
    return line[:1].upper() + line[1:]
