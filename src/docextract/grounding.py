"""Verification des reponses du LLM contre le texte OCR (pas d'appel au modele).

Un LLM peut reecrire un nombre ("23.000" -> "23.00"), prendre la monnaie rendue pour le total ou inventer un nom.
Ici, chaque champ doit etre retrouve dans le texte lu par l'OCR, sinon il est rejete :

    - montant : un nombre imprime de memes chiffres (le format exact de l'OCR est rendu) ; un total ne doit pas
                se trouver seulement sur une ligne cash / monnaie ;
    - societe : son texte (espaces ignores) doit se trouver dans une ligne de l'OCR ;
    - date    : doit etre une vraie date.

Ce sont des verifications, pas des regles d'extraction : rien n'est invente, une valeur est conservee, corrigee
dans son format, ou supprimee.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Sequence

from .normalize import normalize_date
from .ocr import Line

_NUMBER = re.compile(r"\d+(?:[.,:\-] ?\d+)*")
_PAYMENT = re.compile(r"CASH|TUNAI|CHANGE|KEMBALI|BAYAR|TENDER|\bPAID\b|CARD|DEBIT|CREDIT|\bPAY\b|PAYCASH", re.I)
_TOTAL_WORD = re.compile(r"TOTAL|AMOUNT|DUE|JUMLAH", re.I)
_TWO_DECIMALS = re.compile(r"[.,]\d{2}$")


@dataclass
class Verdict:
    """Ce que la verification a fait sur une reponse (pour les statistiques)."""
    kept: list[str] = field(default_factory=list)
    fixed: list[str] = field(default_factory=list)       # format remplace par celui imprime sur le recu
    rejected: list[str] = field(default_factory=list)    # valeur absente du texte -> None


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def _compact(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def verify_amount(value: str | None, lines: Sequence[Line], *, is_total: bool) -> tuple[str | None, str]:
    """(valeur, 'kept' | 'fixed' | 'rejected' | 'none')."""
    if value is None:
        return None, "none"
    wanted = _digits(value)
    if not wanted:
        return None, "rejected"
    exact, snapped = [], []
    for line in lines:
        text = line.text
        blocked = is_total and _PAYMENT.search(text) and not _TOTAL_WORD.search(text)
        if blocked:
            continue
        for m in _NUMBER.finditer(text):
            token = m.group().replace(" ", "")
            digits = _digits(token)
            if token == value.replace(" ", ""):
                exact.append(token)
            elif digits == wanted or (_TWO_DECIMALS.search(value) and digits == wanted + "0"):
                snapped.append(token)          # memes chiffres, ou milliers reecrits en decimales
    if exact:
        return value, "kept"
    if snapped:
        return snapped[-1], "fixed"
    return None, "rejected"


_WINDOW = 3          # un nom peut etre coupe sur 3 lignes OCR
_MIN_BLOCK = 3       # un morceau commun de moins de 3 lettres ne compte pas (evite d'accepter des lettres eparpillees)
_COMPANY_RATIO = 0.8


def _coverage(wanted: str, text: str) -> float:
    """Part de `wanted` retrouvee dans `text` en morceaux d'au moins 3 caracteres consecutifs."""
    blocks = SequenceMatcher(None, wanted, text, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks if b.size >= _MIN_BLOCK) / len(wanted)


def verify_company(value: str | None, lines: Sequence[Line]) -> tuple[str | None, str]:
    """Le nom doit se retrouver dans 1 a 3 lignes OCR consecutives (espaces ignores, petites fautes d'OCR tolerees)."""
    if value is None:
        return None, "none"
    wanted = _compact(value)
    if len(wanted) < 3:
        return None, "rejected"
    compact = [_compact(line.text) for line in lines]
    for i in range(len(compact)):
        for size in range(1, _WINDOW + 1):
            text = "".join(compact[i:i + size])
            if text and (wanted in text or _coverage(wanted, text) >= _COMPANY_RATIO):
                return value, "kept"
    return None, "rejected"


def verify_date(value: str | None, lines: Sequence[Line]) -> tuple[str | None, str]:
    if value is None:
        return None, "none"
    if normalize_date(value) is None:
        return None, "rejected"
    return value, "kept"


def verify_fields(fields: dict[str, str | None], lines: Sequence[Line]) -> tuple[dict[str, str | None], Verdict]:
    """Applique la verification aux 4 champs. Renvoie les champs verifies et le bilan."""
    out, verdict = dict(fields), Verdict()
    checks = {
        "company": lambda v: verify_company(v, lines),
        "date": lambda v: verify_date(v, lines),
        "total": lambda v: verify_amount(v, lines, is_total=True),
        "tax": lambda v: verify_amount(v, lines, is_total=False),
    }
    for name, check in checks.items():
        if name not in fields:
            continue
        value, status = check(fields[name])
        out[name] = value
        if status != "none":
            getattr(verdict, {"kept": "kept", "fixed": "fixed", "rejected": "rejected"}[status]).append(name)
    return out, verdict
