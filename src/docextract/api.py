"""API FastAPI : POST /extract avec choix de l'approche.

    uvicorn docextract.api:app --port 8000        puis ouvrir http://localhost:8000/docs

Les approches se chargent a la premiere requete qui les utilise ; /health dit lesquelles sont en memoire.
"""
from __future__ import annotations

import io
import threading
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from . import config
from .pipelines import DonutPipeline, EngineUnavailable, LLMPipeline, RulesPipeline
from .schema import ExtractionResult, Method

Factory = Callable[[str, str], Any]       # (dataset, llm_model) -> objet avec .extract(image, doc_id)


def default_factories() -> dict[str, Factory]:
    return {
        "rules": lambda dataset, model: RulesPipeline(),
        "llm": lambda dataset, model: LLMPipeline(model=model),
        "docai": lambda dataset, model: DonutPipeline(dataset=dataset),
    }


class Health(BaseModel):
    status: str
    methods: list[str]
    loaded: list[str]


def create_app(factories: dict[str, Factory] | None = None, max_bytes: int | None = None) -> FastAPI:
    factories = factories or default_factories()
    limit = max_bytes or config.API_MAX_BYTES
    cache: dict[tuple, tuple[Any, threading.Lock]] = {}
    guard = threading.Lock()

    def pipeline(method: str, dataset: str, model: str):
        # rules et llm ne dependent pas du jeu ; llm depend du modele ; docai du jeu
        key = (method, dataset if method == "docai" else "", model if method == "llm" else "")
        with guard:
            if key not in cache:
                cache[key] = (factories[method](dataset, model), threading.Lock())
            return cache[key]

    app = FastAPI(
        title="DOC-EXTRACT",
        version="1.0",
        description="Extraction du fournisseur, de la date, du total et de la taxe depuis une image de recu. "
                    "Trois approches : rules (OCR + regles), llm (OCR + LLM local), docai (Donut).",
    )

    @app.get("/health", response_model=Health)
    def health() -> Health:
        names = {"rules": "rules", "llm": "llm", "docai": "docai"}
        loaded = sorted({names[k[0]] for k in cache})
        return Health(status="ok", methods=list(factories), loaded=loaded)

    @app.post("/extract", response_model=ExtractionResult,
              responses={400: {"description": "image illisible"}, 413: {"description": "image trop lourde"},
                         503: {"description": "approche indisponible (Ollama eteint, modele absent...)"}})
    def extract(
        file: UploadFile = File(..., description="Image du recu (JPEG, PNG...)"),
        method: Method = Form("docai", description="rules, llm ou docai (Donut, par defaut)"),
        dataset: Literal["sroie", "cord"] = Form("sroie", description="docai seulement : sroie = fournisseur/date/total, "
                                                                     "cord = total/taxe"),
        llm_model: str = Form(config.LLM_MODEL, description="llm seulement : modele Ollama"),
    ) -> ExtractionResult:
        data = file.file.read(limit + 1)
        if not data:
            raise HTTPException(400, "Fichier vide.")
        if len(data) > limit:
            raise HTTPException(413, f"Image trop lourde (max {limit // (1024 * 1024)} Mo).")
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise HTTPException(400, "Image illisible : envoie un JPEG ou un PNG.") from None
        doc_id = Path(file.filename or "").stem or None
        if method not in factories:
            raise HTTPException(503, f"Approche non configuree sur ce serveur : {method}.")
        pipe, lock = pipeline(method, dataset, llm_model)
        try:
            with lock:
                return pipe.extract(image, doc_id)
        except EngineUnavailable as e:
            raise HTTPException(503, str(e)) from e

    return app


app = create_app()
