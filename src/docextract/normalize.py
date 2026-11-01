"""Normalisation des valeurs extraites, pour comparer prediction et verite terrain
sur une forme unique :

    date   -> "YYYY-MM-DD"          ("22 MAR 18" -> "2018-03-22")
    montant -> Decimal a 2 decimales ("RM 1,234.5" -> Decimal("1234.50"))
    texte  -> MAJUSCULES, sans ponctuation, espaces simples

Toutes les fonctions renvoient None si la valeur est vide ou illisible :
une valeur non normalisable compte comme "non extraite".
"""
from __future__ import annotations

import re
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

# ------------------------------------------------------------------ dates
# Mois en anglais + abreviations malaises (recus de Malaisie : MAC, MEI, OGOS, OKT, DIS)
_MONTHS = {
    "JAN": 1, "JANUARY": 1, "FEB": 2, "FEBRUARY": 2, "MAR": 3, "MARCH": 3, "MAC": 3,
    "APR": 4, "APRIL": 4, "MAY": 5, "MEI": 5, "JUN": 6, "JUNE": 6, "JUL": 7, "JULY": 7,
    "AUG": 8, "AUGUST": 8, "OGO": 8, "OGOS": 8, "SEP": 9, "SEPT": 9, "SEPTEMBER": 9,
    "OCT": 10, "OCTOBER": 10, "OKT": 10, "NOV": 11, "NOVEMBER": 11,
    "DEC": 12, "DECEMBER": 12, "DIS": 12,
}
_MON = "|".join(sorted(_MONTHS, key=len, reverse=True))
_SEP = r"[/\-.]"

# Ordre = priorite : formats sans ambiguite d'abord
_DATE_PATTERNS = [
    # 2018-01-27, 2018/01/27
    ("ymd", re.compile(rf"(?<!\d)(\d{{4}}){_SEP}(\d{{1,2}}){_SEP}(\d{{1,2}})(?!\d)")),
    # 22 MAR 18, 22-Mar-2018, 22MAR2018
    ("d_mon_y", re.compile(rf"(?<!\d)(\d{{1,2}})\s*[-/.\s]?\s*({_MON})\.?\s*[-/.,\s]?\s*(\d{{2,4}})(?!\d)", re.I)),
    # MAR 22, 2018
    ("mon_d_y", re.compile(rf"\b({_MON})\.?\s+(\d{{1,2}}),?\s+(\d{{2,4}})(?!\d)", re.I)),
    # 27/01/2018, 27-01-18, 27.01.2018 (jour en premier : convention malaise)
    ("dmy", re.compile(rf"(?<!\d)(\d{{1,2}}){_SEP}(\d{{1,2}}){_SEP}(\d{{2,4}})(?!\d)")),
]


def _year(y: str) -> int:
    n = int(y)
    return 2000 + n if len(y) == 2 else n


def _safe_date(y: int, m: int, d: int) -> str | None:
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


_COMPACT_DATE = re.compile(r"\d{8}")


def _compact_date(text: str) -> str | None:
    """'20180304' (YYYYMMDD) ou '25032018' (DDMMYYYY). Accepte SEULEMENT si la valeur entiere
    fait 8 chiffres : dans une ligne de texte, 8 chiffres peuvent etre un telephone."""
    if not _COMPACT_DATE.fullmatch(text):
        return None
    y = int(text[:4])
    if 1990 <= y <= 2035:
        return _safe_date(y, int(text[4:6]), int(text[6:]))
    return _safe_date(int(text[4:]), int(text[2:4]), int(text[:2]))


def normalize_date(value: str | None) -> str | None:
    """Premiere date reconnue dans la chaine -> 'YYYY-MM-DD', sinon None."""
    if not value:
        return None
    text = str(value).strip()
    compact = _compact_date(text)
    if compact and 1990 <= int(compact[:4]) <= 2035:
        return compact
    for kind, pattern in _DATE_PATTERNS:
        for m in pattern.finditer(text):
            if kind == "ymd":
                iso = _safe_date(int(m[1]), int(m[2]), int(m[3]))
            elif kind == "d_mon_y":
                iso = _safe_date(_year(m[3]), _MONTHS[m[2].upper()], int(m[1]))
            elif kind == "mon_d_y":
                iso = _safe_date(_year(m[3]), _MONTHS[m[1].upper()], int(m[2]))
            else:  # dmy, ou mdy si le "mois" depasse 12 (ex. 01/27/2018)
                d, mo, y = int(m[1]), int(m[2]), _year(m[3])
                iso = _safe_date(y, mo, d) or _safe_date(y, d, mo)
            if iso and 1990 <= int(iso[:4]) <= 2035:
                return iso
    return None


# ------------------------------------------------------------------ montants
_AMOUNT_RE = re.compile(r"\d[\d.,\s]*")
_THOUSANDS_ONLY = re.compile(r"^[1-9]\d{0,2}([.,]\d{3})+$")  # 63,800 / 1.234.567
_CENT = Decimal("0.01")


def normalize_amount(value: str | float | int | Decimal | None) -> Decimal | None:
    """Montant -> Decimal a 2 decimales. Gere 'RM 9.00', '1,234.50', '63,800', '9,50'."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        try:
            return Decimal(str(value)).quantize(_CENT, rounding=ROUND_HALF_UP)
        except InvalidOperation:
            return None

    match = _AMOUNT_RE.search(str(value))
    if not match:
        return None
    raw = re.sub(r"\s+", "", match.group()).rstrip(".,")
    if not raw:
        return None

    if _THOUSANDS_ONLY.match(raw):          # separateurs de milliers uniquement
        number = re.sub(r"[.,]", "", raw)
    elif "," in raw and "." in raw:          # le dernier separateur = decimales
        dec = max(raw.rfind(","), raw.rfind("."))
        number = re.sub(r"[.,]", "", raw[:dec]) + "." + raw[dec + 1:]
    elif "," in raw:                         # 9,50 -> decimale ; 1,2,3 -> illisible
        head, _, tail = raw.rpartition(",")
        number = head.replace(",", "") + "." + tail
    else:
        head, _, tail = raw.rpartition(".")
        number = (head.replace(".", "") + "." + tail) if head else tail
    try:
        return Decimal(number).quantize(_CENT, rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return None


# ------------------------------------------------------------------ texte
_GLUE = re.compile(r"[.'’`]")                 # S.B. -> SB, KING'S -> KINGS
_PUNCT = re.compile(r"[^\w\s&]", re.UNICODE)  # le reste -> espace


def normalize_text(value: str | None) -> str | None:
    """MAJUSCULES, ponctuation retiree (sauf &), espaces simples."""
    if not value:
        return None
    text = _PUNCT.sub(" ", _GLUE.sub("", str(value).upper()))
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


# ------------------------------------------------------------------ par champ
NORMALIZERS = {
    "company": normalize_text,
    "address": normalize_text,
    "date": normalize_date,
    "total": normalize_amount,
    "subtotal": normalize_amount,
    "tax": normalize_amount,
}


def normalize_field(field: str, value):
    return NORMALIZERS[field](value)
