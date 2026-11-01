"""OCR commun aux approches 1 et 2 : deux moteurs derriere la meme interface.

    moteur "tesseract" (defaut) : garde bien les espaces       -> pytesseract + programme Tesseract
    moteur "rapidocr"           : modeles inclus, tres leger    -> rapidocr-onnxruntime
    "moteur" "gt"               : le texte annote du jeu (OCR parfait) = borne haute pour les regles

Chaque moteur produit des SEGMENTS (texte + boite + confiance). `build_lines` les regroupe
en LIGNES de lecture (etiquette a gauche + montant a droite = une ligne), ce qui est la seule
forme que les regles et le LLM voient.

Les sorties OCR sont mises en cache (data/ocr_cache/<moteur>/<doc_id>.json) : on lance l'OCR une
seule fois, puis on regle les regles autant qu'on veut sans refaire l'OCR.
"""
from __future__ import annotations

import json
import os
import shutil
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from PIL import Image

from . import config

TESSERACT_WINDOWS_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)


# ------------------------------------------------------------------ structures
@dataclass
class Segment:
    """Un morceau de texte lu par l'OCR, boite en pixels de l'image d'origine."""
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float | None = None

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def h(self) -> float:
        return max(self.y1 - self.y0, 1.0)


@dataclass
class Line:
    """Une ligne de lecture : segments voisins sur la meme hauteur, de gauche a droite."""
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float | None = None


@dataclass
class OcrResult:
    engine: str
    segments: list[Segment]
    ms: float = 0.0                      # temps OCR (chargement + moteur), en millisecondes
    width: int = 0
    height: int = 0
    params: dict[str, Any] = field(default_factory=dict)

    def lines(self) -> list[Line]:
        return build_lines(self.segments)

    def text(self) -> str:
        return "\n".join(line.text for line in self.lines())

    def to_dict(self) -> dict:
        return {"engine": self.engine, "ms": round(self.ms, 1), "width": self.width,
                "height": self.height, "params": self.params,
                "segments": [[s.text, round(s.x0, 1), round(s.y0, 1), round(s.x1, 1),
                              round(s.y1, 1), s.conf] for s in self.segments]}

    @classmethod
    def from_dict(cls, d: dict) -> "OcrResult":
        segs = [Segment(t, x0, y0, x1, y1, c) for t, x0, y0, x1, y1, c in d["segments"]]
        return cls(d["engine"], segs, d.get("ms", 0.0), d.get("width", 0),
                   d.get("height", 0), d.get("params", {}))


# ------------------------------------------------------------------ lignes
def build_lines(segments: Sequence[Segment]) -> list[Line]:
    """Regroupe les segments en lignes (meme hauteur), triees de haut en bas puis de gauche a droite.

    Deux segments sont sur la meme ligne si l'ecart de leurs centres verticaux est inferieur a
    ~60 % de la plus petite hauteur : c'est ce qui rassemble « Total » et « 30.25 », que
    les moteurs rendent souvent comme deux boites separees.
    """
    segs = [s for s in segments if s.text and s.text.strip()]
    segs.sort(key=lambda s: s.cy)
    groups: list[list[Segment]] = []
    for s in segs:
        if groups:
            cy = statistics.fmean(g.cy for g in groups[-1])
            h = min(statistics.fmean(g.h for g in groups[-1]), s.h)
            if abs(s.cy - cy) <= 0.6 * h:
                groups[-1].append(s)
                continue
        groups.append([s])

    lines = []
    for g in groups:
        g.sort(key=lambda s: s.x0)
        confs = [s.conf for s in g if s.conf is not None]
        lines.append(Line(
            text=" ".join(s.text.strip() for s in g),
            x0=min(s.x0 for s in g), y0=min(s.y0 for s in g),
            x1=max(s.x1 for s in g), y1=max(s.y1 for s in g),
            conf=statistics.fmean(confs) if confs else None))
    return lines


# ------------------------------------------------------------------ verite terrain comme OCR parfait
def box_to_xyxy(box: Any) -> tuple[float, float, float, float] | None:
    """Accepte [x0,y0,x1,y1], 8 nombres (4 coins) ou 4 paires [[x,y]...]."""
    if box is None:
        return None
    flat: list[float] = []
    for v in box:
        if isinstance(v, (list, tuple)):
            flat.extend(float(x) for x in v)
        else:
            flat.append(float(v))
    if len(flat) == 4:
        x0, y0, x1, y1 = flat
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
    if len(flat) == 8:
        xs, ys = flat[0::2], flat[1::2]
        return min(xs), min(ys), max(xs), max(ys)
    return None


def segments_from_gt(words: Sequence[str], bboxes: Sequence[Any]) -> list[Segment]:
    """Mots annotes + boites du jeu de donnees -> segments (OCR « parfait »)."""
    out = []
    for text, box in zip(words, bboxes):
        xyxy = box_to_xyxy(box)
        if xyxy and str(text).strip():
            out.append(Segment(str(text), *xyxy, conf=1.0))
    return out


# ------------------------------------------------------------------ pretraitement image
def prepare_image(path: str | Path, min_width: int = 0, max_side: int = 3000) -> tuple[Image.Image, float]:
    """Charge l'image en RGB, la reduit si trop grande (max_side) ou l'agrandit si trop etroite
    (min_width). Retourne (image, echelle) : x_original = x_image / echelle."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = 1.0
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
    elif min_width and w < min_width:
        scale = min_width / w
    if scale != 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return img, scale


# ------------------------------------------------------------------ moteurs
def find_tesseract() -> str | None:
    """Chemin de tesseract : variable TESSERACT_CMD, sinon PATH, sinon dossiers Windows usuels."""
    env = os.environ.get("TESSERACT_CMD")
    if env and Path(env).exists():
        return env
    found = shutil.which("tesseract")
    if found:
        return found
    for p in TESSERACT_WINDOWS_PATHS:
        if Path(p).exists():
            return p
    return None


def ocr_tesseract(path: str | Path, psm: int = 4, min_width: int = 1400, max_side: int = 3000) -> OcrResult:
    try:
        import pytesseract
        from pytesseract import Output
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pytesseract manquant : pip install pytesseract") from e
    cmd = find_tesseract()
    if not cmd:
        raise RuntimeError(
            "Tesseract introuvable. Installe-le (PowerShell) :\n"
            "    winget install --id UB-Mannheim.TesseractOCR -e\n"
            "puis rouvre PowerShell, ou definis TESSERACT_CMD vers tesseract.exe.")
    pytesseract.pytesseract.tesseract_cmd = cmd

    t0 = time.perf_counter()
    img, scale = prepare_image(path, min_width=min_width, max_side=max_side)
    data = pytesseract.image_to_data(img.convert("L"), lang="eng", config=f"--psm {psm}",
                                     output_type=Output.DICT)
    segs = []
    for i, text in enumerate(data["text"]):
        text = (text or "").strip()
        conf = float(data["conf"][i])
        if not text or conf < 0:
            continue
        x, y, w, h = (data[k][i] for k in ("left", "top", "width", "height"))
        segs.append(Segment(text, x / scale, y / scale, (x + w) / scale, (y + h) / scale, conf / 100))
    ms = (time.perf_counter() - t0) * 1000
    ow, oh = (round(v / scale) for v in img.size)
    return OcrResult("tesseract", segs, ms, ow, oh, {"psm": psm, "min_width": min_width, "max_side": max_side})


_rapid_engine = None


def ocr_rapidocr(path: str | Path, max_side: int = 2000) -> OcrResult:
    global _rapid_engine
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("RapidOCR manquant : pip install -r requirements-rapidocr.txt") from e
    import numpy as np

    if _rapid_engine is None:
        _rapid_engine = RapidOCR()
    t0 = time.perf_counter()
    img, scale = prepare_image(path, max_side=max_side)
    result, _ = _rapid_engine(np.asarray(img)[:, :, ::-1])   # RGB -> BGR (OpenCV)
    segs = []
    for box, text, conf in result or []:
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        segs.append(Segment(text, min(xs) / scale, min(ys) / scale, max(xs) / scale,
                            max(ys) / scale, float(conf)))
    ms = (time.perf_counter() - t0) * 1000
    ow, oh = (round(v / scale) for v in img.size)
    return OcrResult("rapidocr", segs, ms, ow, oh, {"max_side": max_side})


ENGINES: dict[str, Callable[..., OcrResult]] = {
    "tesseract": ocr_tesseract,
    "rapidocr": ocr_rapidocr,
}


def run_ocr(image_path: str | Path, engine: str = "tesseract", **params) -> OcrResult:
    if engine not in ENGINES:
        raise ValueError(f"Moteur inconnu : {engine!r} (choix : {', '.join(ENGINES)})")
    return ENGINES[engine](image_path, **params)


# ------------------------------------------------------------------ cache
def cache_path(engine: str, doc_id: str) -> Path:
    return Path(config.OCR_CACHE_DIR) / engine / f"{doc_id}.json"


def cached_ocr(doc_id: str, image_path: str | Path, engine: str = "tesseract",
               force: bool = False, **params) -> tuple[OcrResult, bool]:
    """OCR avec cache. Retourne (resultat, vient_du_cache)."""
    path = cache_path(engine, doc_id)
    if path.exists() and not force:
        return OcrResult.from_dict(json.loads(path.read_text(encoding="utf-8"))), True
    result = run_ocr(image_path, engine, **params)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"doc_id": doc_id, "image": str(image_path), **result.to_dict()},
                               ensure_ascii=False), encoding="utf-8")
    return result, False


def load_cached(doc_id: str, engine: str) -> OcrResult | None:
    path = cache_path(engine, doc_id)
    if not path.exists():
        return None
    return OcrResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
