"""Logique de la demo Streamlit (sans interface, donc testable) : lancer une ou plusieurs approches sur une image."""
from __future__ import annotations

import base64
import html
import io
import logging
import time
from dataclasses import dataclass
from typing import Any, Sequence

from . import config
from .pipelines import DonutPipeline, EngineUnavailable, LLMPipeline, RulesPipeline
from .schema import ExtractionResult

log = logging.getLogger(__name__)

METHODS = {"docai": "Donut (Document AI)", "rules": "OCR + règles", "llm": "OCR + LLM local"}
FIELD_LABELS = {"company": "Fournisseur", "date": "Date", "total": "Total", "tax": "Taxe"}
DATASETS = {"sroie": "Reçu malaisien (SROIE)", "cord": "Reçu indonésien (CORD)"}

_PIPELINES: dict[tuple, Any] = {}          # une approche chargee reste en memoire entre deux clics


def get_pipeline(method: str, dataset: str, model: str):
    key = (method, dataset if method == "docai" else "", model if method == "llm" else "")
    if key not in _PIPELINES:
        _PIPELINES[key] = {"rules": lambda: RulesPipeline(), "llm": lambda: LLMPipeline(model=model),
                           "docai": lambda: DonutPipeline(dataset=dataset)}[method]()
    return _PIPELINES[key]


@dataclass
class Outcome:
    method: str
    result: ExtractionResult | None = None
    message: str | None = None            # pourquoi l'approche n'a pas pu repondre
    wall_s: float = 0.0
    detail: str = ""                      # ce qui a ete utilise (modele, OCR), affiche sous le titre du ticket

    @property
    def label(self) -> str:
        return METHODS[self.method]


def describe(method: str, dataset: str, model: str) -> str:
    return {"docai": f"lit l'image directement · modèle {dataset.upper()}",
            "rules": "OCR, puis règles écrites à la main",
            "llm": f"OCR, puis {model} en local"}[method]


def run_method(method: str, image, dataset: str = "sroie", model: str | None = None, doc_id: str | None = None) -> Outcome:
    model = model or config.LLM_MODEL
    detail, t0 = describe(method, dataset, model), time.perf_counter()
    try:
        result = get_pipeline(method, dataset, model).extract(image, doc_id)
        return Outcome(method, result=result, wall_s=time.perf_counter() - t0, detail=detail)
    except EngineUnavailable as e:
        return Outcome(method, message=str(e), wall_s=time.perf_counter() - t0, detail=detail)
    except Exception as e:                                           # la demo ne doit jamais planter devant le public
        log.exception("echec de %s", method)
        return Outcome(method, message=f"Erreur inattendue : {type(e).__name__}: {e}",
                       wall_s=time.perf_counter() - t0, detail=detail)


def run_methods(methods: Sequence[str], image, dataset: str = "sroie", model: str | None = None,
                doc_id: str | None = None) -> list[Outcome]:
    return [run_method(m, image, dataset, model, doc_id) for m in methods]


MONTHS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
          "novembre", "décembre"]


def format_date(d) -> str:
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}"


def format_amount(value) -> str:
    """9.5 -> '9,50' ; 91000 -> '91 000,00' (espace insecable, virgule decimale)."""
    return f"{value:,.2f}".replace(",", "\u00a0").replace(".", ",")


def format_duration(ms: float) -> str:
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f}".replace(".", ",") + " s"


def ticket_html(outcome: Outcome, fresh: bool = False) -> str:
    """Un ticket de caisse par approche. Tout texte venant du modele est echappe (un nom de magasin peut contenir < ou &)."""
    e = html.escape
    head = f'<div class="t-title">{e(outcome.label)}</div><div class="t-sub">{e(outcome.detail)}</div><hr>'
    if outcome.result is None:
        body = (f'<div class="t-down">Cette méthode n\'a pas répondu.</div><div class="t-why">{e(outcome.message or "")}</div>')
        foot = '<div class="t-foot"><span class="stamp">hors service</span></div>'
    else:
        f = outcome.result.fields
        values = {"company": f.company, "date": format_date(f.date) if f.date else None,
                  "total": format_amount(f.total) if f.total is not None else None,
                  "tax": format_amount(f.tax) if f.tax is not None else None}
        rows = "".join(
            f'<div class="t-row"><span>{FIELD_LABELS[k]}</span>'
            + (f'<b><span class="hit">{e(v)}</span></b>' if v else '<i class="none">non trouvé</i>') + "</div>"
            for k, v in values.items())
        note = f'<div class="t-why">{e(outcome.result.error)}</div>' if outcome.result.error else ""
        body = rows + note
        foot = f'<div class="t-foot">lu en {format_duration(outcome.result.latency_ms)}</div>'
    return f'<div class="ticket{" fresh" if fresh else ""}"><div class="t-paper">{head}{body}<hr>{foot}</div></div>'


def photo_html(image, name: str, max_side: int = 900) -> str:
    """La photo du recu, posee de travers sur le bureau (incrustee en base64 pour que le CSS puisse la styler)."""
    img = image.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    data = base64.b64encode(buf.getvalue()).decode()
    return f'<div class="photo"><img src="data:image/jpeg;base64,{data}" alt="{html.escape(name)}"></div>'


def benchmark_markdown(role: str = "eval") -> str | None:
    path = config.OUTPUTS_DIR / "benchmark" / f"benchmark_{role}.md"
    return path.read_text(encoding="utf-8") if path.exists() else None
