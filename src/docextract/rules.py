"""Approche 1 : regles et heuristiques sur les lignes OCR (aucun apprentissage).

Chaque extracteur renvoie un `Found` (valeur + ligne qui l'a produite + nom de la regle),
ou None. La ligne et la regle servent a l'analyse d'erreurs : quand une valeur est fausse,
on voit tout de suite quelle regle s'est trompee.

    fournisseur : ligne du haut qui ressemble a un nom d'entreprise (SDN BHD, ENTERPRISE...),
                  reunie avec la ligne voisine quand le nom est coupe sur deux lignes
    date        : ligne avec le mot DATE, sinon premiere date lisible (jamais un code produit)
    total       : dernier mot-cle TOTAL / DUE (montant arrondi), recoupe avec especes - monnaie
    TVA         : ligne avec GST / TAX / PPN / PB1 / PAJAK, montant le plus a droite
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

from .normalize import normalize_amount, normalize_date
from .ocr import Line
from .schema import ReceiptFields


@dataclass
class Found:
    value: str | Decimal
    evidence: str       # texte de la ligne source
    rule: str           # nom de la regle qui a decide


# ================================================================== montants
_AMOUNT_RE = re.compile(
    r"(?<![\d.,])"
    r"(?:\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{2})?"      # 1,234.50  63,800  1.234.567
    r"|\d+[.,]\d{1,2}"                              # 4.90  9,5
    r"|[1-9]\d{3,8})"                               # 108000 (entier ; un 0 initial = numero, pas montant)
    r"(?!\d)(?!\s*%)")                              # pas un pourcentage
_SPLIT_DECIMAL = re.compile(r"(\d)\s+([.,])\s*(\d{2})(?!\d)")   # "4 .90" -> "4.90"


_COLON_DECIMAL = re.compile(r"(?<![\d:])(\d{1,4}):(\d{2})(?![\d:])")   # "7:20" lu a la place de "7.20"
_GAP_AFTER_SEP = re.compile(r"(\d)([.,])\s+(\d{2,3})(?!\d)")              # "88. 17" -> "88.17", "7. 400" -> "7.400"
_ODD_THOUSANDS = re.compile(r"(?<![\d.,:\-])(\d{1,3})[:\-](\d{3})(?!\d)")   # "8:000", "287-760" -> milliers


def _clean_line(text: str, keyword_line: bool = False) -> str:
    """Repare les erreurs d'OCR courantes dans les montants. Le ":" n'est lu comme une virgule
    que sur une ligne a mot-cle (TOTAL, TAX...) : ailleurs, 12:44 est une heure."""
    text = _SPLIT_DECIMAL.sub(r"\1\2\3", text)
    if not keyword_line:
        return text
    text = _GAP_AFTER_SEP.sub(r"\1\2\3", text)
    text = _ODD_THOUSANDS.sub(r"\1.\2", text)
    return _COLON_DECIMAL.sub(r"\1.\2", text)


def amounts_in(text: str, start: int = 0, allow_int: bool = False) -> list[Decimal]:
    """Montants lisibles dans text[start:], de gauche a droite.

    Un entier sans separateur (108000) n'est accepte que si allow_int : sinon un numero de
    GST ou de telephone serait pris pour un montant.
    """
    out = []
    for m in _AMOUNT_RE.finditer(_clean_line(text, keyword_line=allow_int)[start:]):
        if not allow_int and not re.search(r"[.,]", m.group()):
            continue
        v = normalize_amount(m.group())
        if v is not None:
            out.append(v)
    return out


def _amount_after(pattern: re.Pattern, text: str) -> Decimal | None:
    """Dernier montant situe apres le mot-cle `pattern` dans la ligne."""
    text = _clean_line(text, keyword_line=True)
    m = pattern.search(text)
    if not m:
        return None
    vals = amounts_in(text, m.end(), allow_int=True)
    return vals[-1] if vals else None


# ================================================================== fournisseur
_COMPANY_LEGAL = re.compile(                      # forme juridique : indice tres fort
    r"\b(SDN\.?\s*BHD|S/B|SB|BHD|ENTERPRISES?|TRADING|PLT|LTD|LIMITED|CORP(ORATION)?|"
    r"HOLDINGS?|INDUSTRIES|PT|CV)\b", re.I)
_COMPANY_BUSINESS = re.compile(                   # mot de commerce : indice faible
    r"\b(RESTAURANT|RESTORAN|STORES?|SHOP|MART|SUPERMARKET|CAFE|COFFEE|BAKERY|CENTRE|CENTER|"
    r"HARDWARE|STATIONERY|PHARMACY|MARKET|SERVICES?|BOOKS?|TRAVEL|PETROL|MOTOR|OPTICAL|HOTEL|"
    r"GALLERY|TOKO|WARUNG)\b", re.I)
_COMPANY_JUNK = re.compile(
    r"\b(TAX\s*INVOICE|INVOICE|RECEIPT|BILL|CASH\s*SALES?|CASH\s*BILL|TEL|FAX|PHONE|"
    r"GST|SST|REG|ROC|CO\.?\s*NO|NO\.|DATE|WELCOME|THANK|WWW|HTTP|EMAIL|@|DUPLICATE|COPY|"
    r"SIMPLIFIED|DOCUMENT|CUSTOMER|CASHIER|TABLE|ORDER|SLIP|COUNTER)\b", re.I)
_ADDRESS_WORDS = re.compile(
    r"\b(JALAN|JLN|LORONG|TAMAN|BANDAR|KAWASAN|LEBUH|PERSIARAN|KOMPLEKS|LOT|BLOCK|BLK|"
    r"LEVEL|FLOOR|SELANGOR|JOHOR|PENANG|PERAK|KEDAH|SABAH|SARAWAK|MALAYSIA|KUALA|LUMPUR|"
    r"JAKARTA)\b", re.I)
_POSTCODE_START = re.compile(r"^\W*\d{5}\b")                    # "41150 KLANG" : adresse
# numero d'enregistrement colle au nom : "(716269-X)", "273500-U", "SA0127959-D", "1070687"
_REG_NO = re.compile(
    r"\(\s*\d{4,}[\s\-]*[A-Z]?\s*\)|\(?\s*\d{5,}\s*-\s*[A-Z]\b\s*\)?|\b[A-Z]{1,2}\d{5,}(?:-[A-Z])?\b|\b\d{6,}\b", re.I)
_TRAILING_GROUP = re.compile(r"\s*[(\[{]([^()\[\]{}]*)([)\]}]?)\s*$")
_STREET_WORDS = re.compile(                      # adresse "sure" : un nom d'entreprise peut contenir MALAYSIA
    r"\b(JALAN|JLN|LORONG|TAMAN|BANDAR|KAWASAN|LEBUH|PERSIARAN|KOMPLEKS|LOT|BLOCK|BLK|LEVEL|FLOOR)\b", re.I)
_LEGAL_AT_END = re.compile(r"(SDN\.?\s*BHD|S/B|BHD|ENTERPRISES?|LTD|TRADING)\s*$", re.I)
# debut de ligne qui ne peut pas etre un nom complet : suite d'un nom coupe sur deux lignes
_FRAGMENT_START = re.compile(
    r"^\W*(?:C[OP]\.?\s*\(M\)|\(M\)|S[A-Z]N\.?\s*BHD|S/B\b|BHD\b|ENTERPRISES?\b|TRADING\b|&)", re.I)


def _digit_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    return sum(c.isdigit() for c in chars) / len(chars) if chars else 1.0


def _strip_trailing_groups(name: str) -> str:
    """Retire les groupes entre parentheses en fin de nom : numero (139386 X), sigle (KSB)."""
    while True:
        m = _TRAILING_GROUP.search(name)
        if not m:
            return name
        inner = m.group(1).strip()
        head = name[:m.start()]
        if sum(c.isdigit() for c in inner) >= 3:
            name = head                                        # (139386 X), (5i9537-¥)
        elif m.group(2) and re.fullmatch(r"[A-Z]{2,5}", inner) and _LEGAL_AT_END.search(head):
            name = head                                        # S/B (KSB)
        else:
            return name


def _clean_company(text: str) -> str:
    text = _REG_NO.sub(" ", text)                      # (716269-X) -> retire
    text = re.sub(r"^[^A-Za-z0-9(]+|[^A-Za-z0-9).]+$", "", text.strip())
    text = _strip_trailing_groups(re.sub(r"\s+", " ", text).strip())
    return re.sub(r"\s+", " ", text).strip()


def _name_part(raw: str, strict: bool = True) -> str | None:
    """Nom nettoye si la ligne peut faire partie d'un nom d'entreprise, sinon None.

    strict=False (lignes voisines d'un nom deja trouve) : MALAYSIA, KUALA... sont permis.
    """
    raw = raw.strip()
    address = _ADDRESS_WORDS if strict else _STREET_WORDS
    if _COMPANY_JUNK.search(raw) or address.search(raw) or _POSTCODE_START.search(raw):
        return None
    name = _clean_company(raw)
    if len(name) < 3 or sum(c.isalpha() for c in name) < 3 or _digit_ratio(name) > 0.35:
        return None
    return name


def _merge_split_name(parts: list[tuple[int, str]], best: int, lines: Sequence[Line]) -> tuple[list[int], str]:
    """Reunit le nom quand il est coupe sur plusieurs lignes ('NANDO'S CHICKENLAND' / 'SDN BHD')."""
    first = last = best
    name = parts_by_index = dict(parts)[best]
    # vers le bas : une parenthese ouverte "ENTERPRISE (SETIA" continue sur la ligne suivante
    while name.count("(") > name.count(")") and last + 1 < len(lines):
        nxt = _name_part(lines[last + 1].text, strict=False)
        if nxt is None:
            break
        name, last = f"{name} {nxt}", last + 1
    # vers le haut : une ligne qui commence par SDN BHD / CO. (M) / ENTERPRISE n'est qu'une suite
    if first > 0 and _FRAGMENT_START.search(name):
        prev = _name_part(lines[first - 1].text, strict=False)
        if prev is not None and not _FRAGMENT_START.search(prev):
            name, first = f"{prev} {name}", first - 1
    return list(range(first, last + 1)), _clean_company(name)


def extract_company(lines: Sequence[Line], top: int = 10) -> Found | None:
    parts: list[tuple[int, str]] = []
    best: tuple[float, int] | None = None                # (score, index)
    for i, line in enumerate(lines[:top]):
        name = _name_part(line.text)
        if name is None:
            continue
        parts.append((i, name))
        score = 0.0
        if _COMPANY_LEGAL.search(name):
            score += 5
        elif _COMPANY_BUSINESS.search(name):
            score += 2
        if name.upper() == name:
            score += 0.5
        if name.startswith("("):
            score -= 3                                   # "(TONYMOLY SHOP (95874))" : sous-titre
        score += max(0.0, 2.0 - 0.4 * i)                # les premieres lignes sont favorisees
        if best is None or score > best[0]:
            best = (score, i)
    if best is None:
        return None
    indexes, name = _merge_split_name(parts, best[1], lines)
    rule = "company:multi-line" if len(indexes) > 1 else "company:top-line-score"
    evidence = " | ".join(lines[k].text.strip() for k in indexes)
    return Found(name, evidence, rule)


# ================================================================== date
_DATE_KEY = re.compile(r"\bDATE(?![A-Za-z])|\bDT\b|TARIKH|\bTGL\b", re.I)
_DATE_NOT = re.compile(r"DUE|EXPIR|VALID|BEST\s*BEFORE|\bEXP\b", re.I)
_DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1", "S": "5", "B": "8"})
_DATE_LIKE = re.compile(r"[\dOolI|SB]{1,4}\s*[/\-.]\s*[\dOolI|SB]{1,2}\s*[/\-.]\s*[\dOolI|SB]{2,4}")
# "HD03-04-06", "MI11/3-21" : code produit (1 a 3 lettres collees aux chiffres), pas une date
_CODE_LIKE = re.compile(r"\b([A-Za-z]{1,3})(\d+[/\-.]\S*)")
_DATE_PREFIX = re.compile(r"^(dt|tgl|on|at)$", re.I)
# RapidOCR perd les espaces : "11/03/201806:20", "28Mar1818:32:36", "Date:2016-07-3113:48:37"
_GLUED_DATE_TIME = [
    (re.compile(r"(\d{1,2}[/\-.]\d{1,2}[/\-.])(20\d{2})(?=\d{1,2}:\d{2})"), r"\1\2 "),
    (re.compile(r"(\d{1,2}[/\-.]\d{1,2}[/\-.])(1[5-9]|2\d)(?=\d{2}:\d{2})"), r"\1\2 "),
    (re.compile(r"(\d{1,2}\s*[A-Za-z]{3,9}\.?\s*)(20\d{2})(?=\d{1,2}:\d{2})"), r"\1\2 "),
    (re.compile(r"(\d{1,2}\s*[A-Za-z]{3,9}\.?\s*)(1[5-9]|2\d)(?=\d{2}:\d{2})"), r"\1\2 "),
    (re.compile(r"(20\d{2}[/\-.]\d{1,2}[/\-.]\d{2})(?=\d{1,2}:\d{2})"), r"\1 "),
    (re.compile(r"(?<=\d{3})(\d{2}[/\-.]\d{2}[/\-.]20\d{2})"), r" \1"),     # numero de commande colle a la date
]


def _mask_codes(text: str) -> str:
    return _CODE_LIKE.sub(lambda m: m.group(0) if _DATE_PREFIX.match(m.group(1)) else " ", text)


def _split_glued(text: str) -> str:
    for pattern, repl in _GLUED_DATE_TIME:
        text = pattern.sub(repl, text)
    return text


def _plausible(iso: str) -> bool:
    return 2000 <= int(iso[:4]) <= 2035


def _repair_date_text(text: str) -> str:
    """Corrige les confusions chiffre/lettre dans les motifs qui ressemblent a une date."""
    return _DATE_LIKE.sub(lambda m: m.group().translate(_DIGIT_FIX), text)


def extract_date(lines: Sequence[Line]) -> Found | None:
    best: tuple[float, int, str, str, str] | None = None   # (score, -index, iso, ligne, regle)
    for i, line in enumerate(lines):
        text = _split_glued(_mask_codes(line.text))
        for variant, rule in ((text, "date:line"), (_repair_date_text(text), "date:line-repaired")):
            iso = normalize_date(variant)
            if not iso or not _plausible(iso):
                continue
            score = 1.0 - min(i, 40) * 0.01
            if _DATE_KEY.search(line.text):
                score += 3
            if _DATE_NOT.search(line.text):
                score -= 3
            if rule.endswith("repaired"):
                score -= 0.5
            if best is None or score > best[0]:
                best = (score, -i, iso, line.text, rule)
            break                                          # la variante brute suffit si lisible
    if best is None:
        return None
    return Found(best[2], best[3], best[4])


# ================================================================== total
# "total" tolere les fautes d'OCR : Iotal, lotal, fotal, Tatai, Poial
_TOT = r"(?:[TIl1fP][oa0][t1ila]?[a4][l1i])"
_T1 = re.compile(
    rf"ROUNDED\s*(TOTAL|AMOUNT)|{_TOT}\s*(AFTER\s*)?(ROUND|ADJ)|AMOUNT\s*DUE|AMT\.?\s*DUE|"
    rf"GRAND\s*{_TOT}|{_TOT}\s*(PAYABLE|BAYAR)|BALANCE\s*DUE|JUMLAH\s*BAYAR|{_TOT}\s*DUE|"
    r"AMOUNT\s*PAYABLE|\bDUE\b", re.I)
_T2 = re.compile(rf"{_TOT}|JUMLAH", re.I)
_TOTAL_NOT = re.compile(                           # sans \\b : certains OCR collent les mots (TOTALTAX)
    r"SU[BR8]\s*-?\s*[TIl1fP][oae0]|EXCL|BEFORE|QTY|QUANTITY|ITEM|DISC|SAVING|TAX|GST|SST|PPN|SERVICE|"
    r"POINT|CHANGE|ROUNDING\s*ADJ|GROSS|PAID|ACCEPTED|TENDER|RECEIVED|CASH|CARD|COUNT", re.I)
_TAX_DETAIL = re.compile(r"INCLUDED\s*GST\s*@", re.I)           # "(TOTAL INCLUDED GST @ 6% : 1.42)"
_DUE_DATE = re.compile(r"DUE\s*DATE|DATE\s*DUE", re.I)
_INCL = re.compile(r"INCL|INC\.|\bINC\b|INCLUSIVE", re.I)
_SUMMARY = re.compile(r"(GST|TAX|SST)\s*(SUMMARY|SUMMERY|ANALYSIS)", re.I)
_CASH = re.compile(r"\b(CASH|TUNAI|TENDER(ED)?|PAID|PAY|PAYMENT|ACCEPTED|VISA|MASTER(CARD)?|DEBIT|CREDIT|"
                   r"CARD|AMT\s*TENDERED|AMOUNT\s*PAID)\b", re.I)
_CHANGE = re.compile(r"\b(CHANGE|KEMBALI(AN)?)\b", re.I)
_ROUND = re.compile(r"ROUND", re.I)
_SMALL = re.compile(r"(?<![\d.,])([-−–~]\s*)?(\d?)[.,]\s?(\d{1,2})(?!\d)")   # -0.02, ~0,02, .01
_HALF_UNIT = Decimal("0.10")


def _body(lines: Sequence[Line]) -> list[Line]:
    """Lignes avant le recapitulatif de taxe (le TOTAL du recap n'est pas le total a payer)."""
    for i, line in enumerate(lines):
        if _SUMMARY.search(line.text):
            return list(lines[:i])
    return list(lines)


def _excluded_total_line(text: str) -> bool:
    if _TAX_DETAIL.search(text):
        return True
    if not _TOTAL_NOT.search(text):
        return False
    # "TOTAL RM INC. GST" = total TTC : les mots taxe sont alors permis
    stripped = re.sub(r"GST|SST|TAX", "", text, flags=re.I) if _INCL.search(text) else text
    return bool(_TOTAL_NOT.search(stripped))


def _keyword_total(body: Sequence[Line]) -> tuple[int, Found] | None:
    """Dernier montant derriere un mot-cle de total (le total arrondi est imprime en dernier).

    On regarde d'abord les mots-cles forts (DUE, GRAND TOTAL, TOTAL AFTER ROUNDING...), puis
    TOTAL. Une ligne qui porte plusieurs montants (ligne de recap : net, taxe, total) ne passe
    qu'en dernier recours.
    """
    for pattern, rule in ((_T1, "total:strong-keyword"), (_T2, "total:keyword")):
        singles: list[tuple[int, Found]] = []
        multis: list[tuple[int, Found]] = []
        for idx, line in enumerate(body):
            text = line.text
            if not pattern.search(text):
                continue
            if pattern is _T1 and (_CHANGE.search(text) or _DUE_DATE.search(text)):
                continue
            if pattern is _T2 and _excluded_total_line(text):
                continue
            cleaned = _clean_line(text, keyword_line=True)
            vals = [v for v in amounts_in(cleaned, pattern.search(cleaned).end(), allow_int=True) if v > 0]
            if vals:
                (singles if len(vals) == 1 else multis).append((idx, Found(vals[-1], text, rule)))
        hits = singles or multis
        if hits:
            return hits[-1]
    return None


def _rounding_lines(body: Sequence[Line]) -> list[tuple[int, Decimal | None, Decimal | None]]:
    """Lignes d'arrondi : (index, ajustement signe < 1, montant arrondi imprime >= 1)."""
    out = []
    for idx, line in enumerate(body):
        if not _ROUND.search(line.text) or _T1.search(line.text) or _CHANGE.search(line.text):
            continue
        cleaned = _clean_line(line.text, keyword_line=True)
        printed = [v for v in amounts_in(cleaned, allow_int=False) if v >= 1]
        adj = None
        for m in _SMALL.finditer(cleaned):
            value = Decimal(f"{m.group(2) or '0'}.{m.group(3).ljust(2, '0')}")
            if value < 1:
                adj = -value if m.group(1) else value
        out.append((idx, adj, printed[-1] if printed else None))
    return out


def _cash_and_change(body: Sequence[Line]) -> tuple[Found | None, Decimal | None]:
    cash = change = None
    for line in body:
        if cash is None and _CASH.search(line.text):
            v = _amount_after(_CASH, line.text)
            if v is not None and v > 0:
                cash = Found(v, line.text, "total:cash-minus-change")
        if change is None and _CHANGE.search(line.text):
            change = _amount_after(_CHANGE, line.text)
    return cash, change


def extract_total(lines: Sequence[Line]) -> Found | None:
    body = _body(lines)
    cash, change = _cash_and_change(body)
    net = cash.value - change if cash is not None and change is not None and 0 <= change < cash.value else None
    rounds = _rounding_lines(body)

    hit = _keyword_total(body)
    if hit is not None:
        idx, found = hit
        # 1) le total est imprime avant la ligne d'arrondi : on applique l'arrondi (65.53 + 0.02)
        for ridx, adj, printed in rounds:
            if printed is not None and printed != found.value and abs(printed - found.value) <= _HALF_UNIT:
                found = Found(printed, body[ridx].text, "total:rounded")
                break
            if adj and ridx > idx:
                found = Found(found.value + adj, body[ridx].text, "total:rounded")
                break
        # 2) especes - monnaie rendue : le montant reellement paye, s'il est imprime ou tout proche
        if net is not None and net > 0 and net != found.value:
            printed_net = any(net in amounts_in(line.text) for line in body)
            if printed_net or abs(net - found.value) <= _HALF_UNIT:
                found = Found(net, cash.evidence, "total:cash-minus-change")
        return found

    # pas de mot-cle : montant arrondi imprime, puis especes - monnaie, puis especes
    for ridx, _adj, printed in rounds:
        if printed is not None:
            return Found(printed, body[ridx].text, "total:rounded")
    if cash is not None:
        if net is not None:
            return Found(net, cash.evidence, "total:cash-minus-change")
        return Found(cash.value, cash.evidence, "total:cash")

    # dernier recours : plus gros montant du recu (hors lignes especes / monnaie)
    best = None
    for line in body:
        if _CASH.search(line.text) or _CHANGE.search(line.text):
            continue
        for v in amounts_in(line.text):
            if best is None or v > best.value:
                best = Found(v, line.text, "total:max-amount")
    return best


# ================================================================== TVA
_TAX_KEY = re.compile(r"\b(GST|SST|TAX|VAT|PPN|PB\s?1|PAJAK|P\.?\s?REST[O0]?)\b|TAX\s*AMT", re.I)
_TAX_STRONG = re.compile(
    r"TOTAL\s*(TAX|GST|SST)|(TAX|GST|SST)\s*AMOUNT|TAX\s*AMT|(TAX|GST)\s*PAYABLE|PB\s?1|PPN|PAJAK", re.I)
_TAX_NOT = re.compile(r"INVOICE|\bID\b|\bNO\b|REG|SUMMARY|INCLUSIVE|INCLUDED|\bRATE\b|TAXABLE|CODE", re.I)
_TAX_INCLUDED_AMOUNT = re.compile(r"TAX\s*INCLUDED\s*:?\s*(RP\.?)?\s*\d", re.I)   # "Total Tax Included: Rp5272.73"
_TAX_ZERO = re.compile(r"PB\s?1\s*:?\s*0\s*$", re.I)                                # "PB1: 0" = pas de taxe
_TAX_SPACED = re.compile(r"(?<![\d.,])(\d{1,3})\s(\d{3})\s*$")                      # "16 170"
_TAX_TRAILING_INT = re.compile(r"(?<![\d.,%/\-])(\d{2,9})\s*$")                      # "TAX AMT 727", "Resto(10%) 750"


def _tax_amount(text: str, start: int) -> Decimal | None:
    vals = amounts_in(text, start, allow_int=True)
    if vals:
        return vals[-1]
    tail = text[start:]
    m = _TAX_SPACED.search(tail)
    if m:
        return Decimal(m.group(1) + m.group(2))
    m = _TAX_TRAILING_INT.search(tail)
    return Decimal(m.group(1)) if m else None


def extract_tax(lines: Sequence[Line]) -> Found | None:
    for pattern, rule in ((_TAX_STRONG, "tax:strong-keyword"), (_TAX_KEY, "tax:keyword")):
        for line in _body(lines):
            if not pattern.search(line.text):
                continue
            if _TAX_NOT.search(line.text) and not _TAX_INCLUDED_AMOUNT.search(line.text):
                continue
            if _TAX_ZERO.search(line.text):
                return Found(Decimal("0"), line.text, "tax:zero")
            cleaned = _clean_line(line.text, keyword_line=True)
            val = _tax_amount(cleaned, pattern.search(cleaned).start())
            if val is not None:
                return Found(val, line.text, rule)
    return None


# ================================================================== point d'entree
@dataclass
class RulesOutput:
    fields: ReceiptFields
    found: dict[str, Found | None]
    ms: float

    def evidence(self, field: str) -> str:
        f = self.found.get(field)
        return f"{f.rule} | {f.evidence}" if f else ""


def extract_rules(lines: Sequence[Line]) -> RulesOutput:
    t0 = time.perf_counter()
    found = {
        "company": extract_company(lines),
        "date": extract_date(lines),
        "total": extract_total(lines),
        "tax": extract_tax(lines),
    }
    values = {k: (None if v is None else str(v.value)) for k, v in found.items()}
    fields = ReceiptFields(**values)
    return RulesOutput(fields, found, (time.perf_counter() - t0) * 1000)
