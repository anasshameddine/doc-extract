"""Approche 2 : OCR + LLM local (Ollama), sortie JSON validee par Pydantic.

    lignes OCR ─► prompt (consignes + 2 exemples) ─► Ollama (JSON impose par un schema) ─► Pydantic ─► champs

Pas de dependance en plus : le client parle a l'API HTTP d'Ollama avec la bibliotheque standard.
Le LLM recoit le texte deja lu par l'OCR (cache de l'etape 2) : on mesure le LLM, pas l'OCR.
"""
from __future__ import annotations

import json
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from . import config
from .ocr import Line
from .schema import ExtractionResult, ReceiptFields

PROMPT_VERSION = "v2"          # a changer a chaque modification du prompt : sert de cle de cache
MAX_CHARS = 4000               # texte OCR envoye au plus (le debut et la fin sont gardes)
FIELDS = ("company", "date", "total", "tax")


class OllamaError(RuntimeError):
    """Ollama injoignable, modele absent ou reponse HTTP en erreur (message lisible pour l'utilisateur)."""


# ================================================================== sortie attendue
_EMPTY = {"", "null", "none", "n/a", "na", "-", "--", "unknown"}


class LLMOutput(BaseModel):
    """JSON renvoye par le LLM. Tout est du texte : la normalisation (dates, montants) est commune aux 3 approches."""
    model_config = ConfigDict(extra="ignore")

    company: str | None = None
    date: str | None = None
    total: str | None = None
    tax: str | None = None

    @field_validator("*", mode="before")
    @classmethod
    def _as_text(cls, v):
        if v is None or isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            v = f"{v:g}" if isinstance(v, float) else str(v)
        if isinstance(v, str):
            v = " ".join(v.split())
            return None if v.lower() in _EMPTY else v
        return str(v)

    def to_fields(self) -> ReceiptFields:
        return ReceiptFields(**self.model_dump())


def output_schema() -> dict:
    """Schema JSON donne a Ollama : le modele ne peut produire que ces 4 cles."""
    nullable = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    return {"type": "object", "properties": {k: nullable for k in FIELDS}, "required": list(FIELDS)}


_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


def parse_reply(raw: str) -> LLMOutput:
    """Valide la reponse ; tolere du texte autour de l'objet JSON (modeles bavards)."""
    try:
        return LLMOutput.model_validate_json(raw)
    except ValidationError:
        m = _JSON_OBJECT.search(raw)
        if not m:
            raise
        return LLMOutput.model_validate_json(m.group())


# ================================================================== prompt
SYSTEM_PROMPT = """You extract key fields from the OCR text of a receipt (Malaysian or Indonesian; English, Malay or Indonesian). \
The OCR may contain typos and may have lost the spaces between words.
Reply with one JSON object with exactly these keys: company, date, total, tax.
- company: the business that issued the receipt, near the top. If a legal entity name is printed (it ends with SDN BHD, S/B, BHD, \
ENTERPRISE, TRADING, LTD...), use it rather than a brand, slogan or shop-name line. Restore the spaces between words if they are missing \
(for example "PASARAYABORONGPINTAR SDNBHD" becomes "PASARAYA BORONG PINTAR SDN BHD"). \
Remove the registration number such as "(519537-X)", and do not include the address or phone numbers.
- date: the date of the purchase, copied as printed, every digit of the year included (for example "12/03/2017" or "28 MAR 18"). \
Not a due date or an expiry date.
- total: the final amount the customer has to pay, tax included and after rounding. Copy the number exactly as printed, without \
currency symbol. It is never the cash given (Cash, Tunai, Pay, Payment, Bayar, Tendered, card) and never the change (Change, Kembali). \
It is not the subtotal when a total is printed. If a rounded total is printed after the total, use the rounded one. \
If no total is printed but cash and change are, the total is cash minus change.
- tax: the tax amount (GST, SST, PB1, PPN, Pajak, Tax), copied as printed. A rate alone ("10%") or an empty "PB1:" is not an amount: \
use null.
In Indonesian receipts "." and "," separate thousands: 72.000 means seventy-two thousand. Copy every digit exactly; \
never drop a zero and never round (72.000 stays "72.000", 304.326 stays "304.326").
Use null for any field you cannot find. Never invent a value."""

# Trois exemples inventes (aucun recu du jeu de donnees) : arrondi malaisien, recu indonesien (milliers avec un point),
# nom colle avec numero d'enregistrement et nom juridique sous l'enseigne. Les annees different pour ne pas en favoriser une.
FEW_SHOT: list[tuple[str, dict]] = [
    ("GOLDEN MART SDN BHD (123456-X)\nNO 5 JALAN TEST 50000 KUALA LUMPUR\nGST ID: 001234567890\nDate: 12/03/2019 10:15\n"
     "MILK 2L 1 x 7.50 7.50\nBREAD 1 x 3.20 3.20\nTotal (Excl GST) 10.12\nGST 6% 0.61\nTotal (Incl GST) 10.73\n"
     "Rounding Adj -0.03\nTOTAL 10.70\nCASH 20.00\nCHANGE 9.30",
     {"company": "GOLDEN MART SDN BHD", "date": "12/03/2019", "total": "10.70", "tax": "0.61"}),
    ("WARUNGSEDERHANA\nJLBUNGA12JAKARTA\n28/09/2016 12:40\nNASIGORENG 2 50.000\nESTEH 2 16.000\nSUBTOTAL 66.000\n"
     "PAJAK 10% 6.600\nTOTAL 72.600\nBAYAR 100.000\nKEMBALI 27.400",
     {"company": "WARUNG SEDERHANA", "date": "28/09/2016", "total": "72.600", "tax": "6.600"}),
    ("SUPER BAKERY\nLUCKYSTARTRADINGSDNBHD(556677-K)\nLOT3,JALANMAJU5,40000SHAHALAM\nGSTID:001122334455\n"
     "Date:03/11/201718:22\nBREAD 1 4.50\nTotalSales(InclusiveGST) 4.50\nRoundingAdj 0.00\nTotal 4.50\nPay 10.00\n"
     "Change 5.50\nGST6% 0.25",
     {"company": "LUCKY STAR TRADING SDN BHD", "date": "03/11/2017", "total": "4.50", "tax": "0.25"}),
]


def format_receipt(lines: Sequence[Line], max_chars: int = MAX_CHARS) -> str:
    """Une ligne OCR par ligne de texte. Trop long : on garde le haut (fournisseur) et le bas (total)."""
    text = "\n".join(t for t in (line.text.strip() for line in lines) if t)
    if len(text) <= max_chars:
        return text
    head = max_chars * 2 // 5
    return text[:head] + "\n[...]\n" + text[-(max_chars - head):]


def build_messages(receipt_text: str, few_shot: bool = True) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if few_shot:
        for text, answer in FEW_SHOT:
            messages += [{"role": "user", "content": text},
                         {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}]
    messages.append({"role": "user", "content": receipt_text})
    return messages


# ================================================================== client Ollama
Post = Callable[[str, dict, float], dict]
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))      # Ollama est local : jamais de proxy


def _request(req: urllib.request.Request, base_url: str, timeout: float) -> dict:
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:300]
        raise OllamaError(f"Ollama a repondu {e.code} : {body}") from e
    except (urllib.error.URLError, ConnectionError, socket.timeout, TimeoutError) as e:
        raise OllamaError(f"Impossible de joindre Ollama sur {base_url} ({e}). "
                          "Lance l'application Ollama (icone dans la barre des taches) puis reessaie.") from e


def http_post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    return _request(req, url, timeout)


def installed_models(base_url: str | None = None, timeout: float = 5.0) -> list[str]:
    base_url = base_url or config.OLLAMA_URL
    data = _request(urllib.request.Request(base_url.rstrip("/") + "/api/tags"), base_url, timeout)
    return [m.get("name", "") for m in data.get("models", [])]


def loaded_model_info(model: str, base_url: str | None = None, timeout: float = 5.0) -> dict | None:
    """Ce qu'Ollama garde en memoire pour `model` : taille totale, part en VRAM (Go) et part sur GPU. None si inconnu."""
    base_url = base_url or config.OLLAMA_URL
    try:
        data = _request(urllib.request.Request(base_url.rstrip("/") + "/api/ps"), base_url, timeout)
    except OllamaError:
        return None
    for m in data.get("models", []):
        if model in (m.get("name"), m.get("model")) and m.get("size"):
            size, vram = float(m["size"]), float(m.get("size_vram") or 0)
            return {"size_gb": round(size / 1e9, 2), "vram_gb": round(vram / 1e9, 2), "gpu_share": round(vram / size, 2)}
    return None


def warm_up(model: str | None = None, base_url: str | None = None, timeout: float = 300.0,
            post: Post | None = None) -> float:
    """Charge le modele en memoire (le premier appel prend 30 a 70 s) pour ne pas fausser les temps mesures ensuite."""
    model = model or config.LLM_MODEL
    base_url = (base_url or config.OLLAMA_URL).rstrip("/")
    t0 = time.perf_counter()
    (post or http_post)(f"{base_url}/api/chat", {"model": model, "messages": [{"role": "user", "content": "ok"}],
                        "stream": False, "keep_alive": "30m", "options": {"num_predict": 1, "num_ctx": 4096}}, timeout)
    return (time.perf_counter() - t0) * 1000


def has_model(model: str, installed: Sequence[str]) -> bool:
    return model in installed or f"{model}:latest" in installed


@dataclass
class LLMRun:
    """Resultat brut d'un appel : ce qu'il faut pour le cache, les scores et l'analyse d'erreurs."""
    output: LLMOutput | None
    raw: str
    ms: float                        # temps total (tentatives comprises)
    attempts: int
    error: str | None = None
    meta: dict = field(default_factory=dict)      # jetons et durees rapportes par Ollama

    def to_dict(self) -> dict:
        return {"output": self.output.model_dump() if self.output else None, "raw": self.raw, "ms": self.ms,
                "attempts": self.attempts, "error": self.error, "meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict) -> "LLMRun":
        out = LLMOutput(**d["output"]) if d.get("output") is not None else None
        return cls(out, d.get("raw", ""), d["ms"], d.get("attempts", 1), d.get("error"), d.get("meta", {}))


def extract_llm(lines: Sequence[Line], *, model: str | None = None, base_url: str | None = None,
                few_shot: bool = True, use_schema: bool = True, retries: int = 1, timeout: float = 300.0,
                post: Post | None = None) -> LLMRun:
    """Une extraction. Reponse invalide : `retries` nouvelles tentatives, puis erreur enregistree (pas d'exception).
    Ollama injoignable ou modele absent : OllamaError (inutile de continuer sur les autres recus)."""
    model = model or config.LLM_MODEL
    base_url = (base_url or config.OLLAMA_URL).rstrip("/")
    post = post or http_post
    messages = build_messages(format_receipt(lines), few_shot)
    payload = {"model": model, "messages": messages, "stream": False, "keep_alive": "30m",
               "format": output_schema() if use_schema else "json",
               "options": {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 200}}
    t0 = time.perf_counter()
    raw, meta, error = "", {}, None
    for attempt in range(1, retries + 2):
        try:
            data = post(f"{base_url}/api/chat", payload, timeout)
        except OllamaError as e:
            if "repeat limit" not in str(e):
                raise
            # Le modele boucle et Ollama coupe la generation : reponse invalide pour CE recu (deterministe,
            # inutile de reessayer), pas une panne. Le recu est compte comme rate et l'evaluation continue.
            return LLMRun(None, "", (time.perf_counter() - t0) * 1000, attempt,
                          "generation interrompue : boucle de repetition", {})
        raw = (data.get("message") or {}).get("content", "")
        meta = {"prompt_tokens": data.get("prompt_eval_count"), "output_tokens": data.get("eval_count"),
                "eval_ms": (data.get("eval_duration") or 0) / 1e6, "load_ms": (data.get("load_duration") or 0) / 1e6}
        try:
            output = parse_reply(raw)
            return LLMRun(output, raw, (time.perf_counter() - t0) * 1000, attempt, None, meta)
        except ValidationError as e:
            error = f"reponse invalide : {e.errors()[0]['type']}" if e.errors() else "reponse invalide"
    return LLMRun(None, raw, (time.perf_counter() - t0) * 1000, retries + 1, error, meta)


def verified_fields(run: LLMRun, lines: Sequence[Line]):
    """(champs verifies contre le texte OCR, bilan) ; le LLM n'est pas rappele."""
    from .grounding import verify_fields
    raw = {k: getattr(run.output, k) for k in FIELDS} if run.output else {k: None for k in FIELDS}
    return verify_fields(raw, lines)


def to_result(doc_id: str | None, run: LLMRun, lines: Sequence[Line] | None = None) -> ExtractionResult:
    """Au format commun aux 3 approches (c'est ce que l'API renverra). Avec `lines` : champs verifies (grounding)."""
    if lines is not None and run.output:
        fields = ReceiptFields(**verified_fields(run, lines)[0])
    else:
        fields = run.output.to_fields() if run.output else ReceiptFields()
    return ExtractionResult(doc_id=doc_id, method="llm", fields=fields, latency_ms=run.ms, error=run.error)


# ================================================================== cache
def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def cache_file(doc_id: str, model: str, ocr_engine: str, few_shot: bool = True, use_schema: bool = True):
    variant = f"{PROMPT_VERSION}{'' if few_shot else '-zeroshot'}{'' if use_schema else '-json'}"
    return config.LLM_CACHE_DIR / safe_name(model) / variant / ocr_engine / f"{doc_id}.json"


def cached_llm(doc_id: str, lines: Sequence[Line], *, model: str, ocr_engine: str, force: bool = False,
               few_shot: bool = True, use_schema: bool = True, **kwargs) -> tuple[LLMRun, bool]:
    """(run, from_cache). Une reponse deja obtenue n'est jamais redemandee sauf force=True."""
    path = cache_file(doc_id, model, ocr_engine, few_shot, use_schema)
    if path.exists() and not force:
        return LLMRun.from_dict(json.loads(path.read_text(encoding="utf-8"))), True
    run = extract_llm(lines, model=model, few_shot=few_shot, use_schema=use_schema, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run.to_dict(), ensure_ascii=False), encoding="utf-8")
    return run, False
