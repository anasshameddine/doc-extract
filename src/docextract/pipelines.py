"""Les 3 approches vues de l'API : une image PIL entre, un ExtractionResult sort.

Chaque pipeline charge ses ressources a la premiere utilisation (pas au demarrage) et leve
EngineUnavailable quand il ne peut pas travailler (Ollama eteint, modele absent, torch non installe) :
l'API repond alors 503 avec un message clair, au lieu d'une erreur 500.
"""
from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable

from . import config, llm
from .ocr import OcrResult, run_ocr
from .rules import extract_rules
from .schema import ExtractionResult

_OCR_LOCK = threading.Lock()          # le moteur OCR est partage : un recu a la fois


class EngineUnavailable(RuntimeError):
    """L'approche demandee ne peut pas tourner sur cette machine (message lisible par l'utilisateur)."""


def ocr_image(image, engine: str, ocr: Callable[..., OcrResult] = run_ocr) -> OcrResult:
    """OCR d'une image PIL (les moteurs lisent un fichier : on passe par un fichier temporaire)."""
    fd, name = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    path = Path(name)
    try:
        image.convert("RGB").save(path)
        with _OCR_LOCK:
            return ocr(path, engine)
    except RuntimeError as e:                                   # moteur OCR non installe
        raise EngineUnavailable(str(e)) from e
    finally:
        path.unlink(missing_ok=True)


class RulesPipeline:
    """Approche 1 : OCR + regles."""

    def __init__(self, engine: str | None = None, ocr: Callable[..., OcrResult] = run_ocr):
        self.engine, self.ocr = engine or config.API_OCR_ENGINE, ocr

    def extract(self, image, doc_id: str | None = None) -> ExtractionResult:
        res = ocr_image(image, self.engine, self.ocr)
        out = extract_rules(res.lines())
        return ExtractionResult(doc_id=doc_id, method="rules", fields=out.fields, latency_ms=res.ms + out.ms)


class LLMPipeline:
    """Approche 2 : OCR + LLM local (Ollama), champs verifies contre le texte OCR."""

    def __init__(self, model: str | None = None, engine: str | None = None, base_url: str | None = None,
                 ocr: Callable[..., OcrResult] = run_ocr, post: llm.Post | None = None):
        self.model, self.engine = model or config.LLM_MODEL, engine or config.API_OCR_ENGINE
        self.base_url, self.ocr, self.post = base_url, ocr, post

    def extract(self, image, doc_id: str | None = None) -> ExtractionResult:
        res = ocr_image(image, self.engine, self.ocr)
        lines = res.lines()
        try:
            run = llm.extract_llm(lines, model=self.model, base_url=self.base_url, post=self.post)
        except llm.OllamaError as e:
            raise EngineUnavailable(str(e)) from e
        result = llm.to_result(doc_id, run, lines)
        return result.model_copy(update={"latency_ms": res.ms + run.ms})


class DonutPipeline:
    """Approche 3 : Donut, image -> JSON sans OCR. `dataset` choisit le modele et les champs (sroie ou cord)."""

    def __init__(self, dataset: str = "sroie", source: str | None = None, loader: Callable[..., Any] | None = None):
        self.dataset = dataset
        self.source = source or (config.DONUT_SROIE_MODEL if dataset == "sroie" else config.DONUT_CORD_MODEL)
        self._loader, self._model = loader, None

    def _load(self):
        if self._model is None:
            if self.dataset == "sroie" and not Path(self.source).exists():
                raise EngineUnavailable(f"Modele Donut introuvable : {self.source}. "
                                        "Copie le dossier donut_sroie/best de Google Drive dans data/donut_models/sroie_best.")
            try:
                if self._loader is None:
                    from .donut_infer import DonutExtractor as loader
                else:
                    loader = self._loader
                model = loader(self.source, self.dataset)
                getattr(model, "warm_up", lambda: None)()        # l'initialisation CUDA ne retombe pas sur la 1re requete
                self._model = model
            except ImportError as e:
                raise EngineUnavailable(f"Donut n'a pas pu s'importer ({type(e).__name__}: {e}). Si torch ou transformers manque : "
                                        "pip install -r requirements-donut.txt") from e
        return self._model

    def extract(self, image, doc_id: str | None = None) -> ExtractionResult:
        return self._load().extract(image, doc_id)
