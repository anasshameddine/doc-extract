"""Chemins et constantes du projet."""
import os
from pathlib import Path

# Racine du projet = dossier qui contient src/
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # images exportees
PROCESSED_DIR = DATA_DIR / "processed"  # annotations JSONL + echantillon d'eval
HF_CACHE_DIR = DATA_DIR / "hf_cache"   # cache Hugging Face (supprimable)
OUTPUTS_DIR = ROOT / "outputs"

# Sources Hugging Face (verifiees le 26/10/2026)
SROIE_REPO = "jsdnrs/ICDAR2019-SROIE"   # CC BY 4.0 - train 626 / test 361
CORD_REPO = "naver-clova-ix/cord-v2"     # CC BY 4.0 - train 800 / val 100 / test 100

# Echantillon d'evaluation fixe (identique pour les 3 approches)
EVAL_SAMPLE_SIZE = 100
EVAL_SEED = 42
EVAL_SAMPLE_FILE = PROCESSED_DIR / "eval_sample.json"

# Jeu de DEVELOPPEMENT : sert a regler les regles, le prompt, etc. Jamais le jeu d'evaluation.
# SROIE : 100 recus tires dans le split train. CORD : tout le split validation (100 recus).
DEV_SAMPLE_SIZE = 100
DEV_SEED = 7
DEV_SAMPLE_FILE = PROCESSED_DIR / "dev_sample.json"

# Sorties OCR mises en cache (une fois par moteur et par recu)
OCR_CACHE_DIR = DATA_DIR / "ocr_cache"
RULES_OUTPUT_DIR = OUTPUTS_DIR / "rules"

# Approche 2 : LLM local via Ollama (https://ollama.com)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.environ.get("DOCEXTRACT_LLM_MODEL", "qwen2.5:3b")    # ~2 Go : tient dans 4 Go de VRAM
LLM_CACHE_DIR = DATA_DIR / "llm_cache"
LLM_OUTPUT_DIR = OUTPUTS_DIR / "llm"

# Approche 3 : Donut (fine-tuning sur Google Colab, predictions copiees ici puis notees en local)
DONUT_PRED_DIR = DATA_DIR / "donut_preds"      # predictions_<jeu>_<dev|eval>.json telechargees depuis Drive
DONUT_OUTPUT_DIR = OUTPUTS_DIR / "donut"
DONUT_MODEL_DIR = DATA_DIR / "donut_models"     # sroie_best/ : notre modele fine-tune (copie de Drive)
DONUT_LOCAL_PRED_DIR = DATA_DIR / "donut_preds_local"   # predictions refaites sur ce PC
DONUT_CORD_MODEL = "naver-clova-ix/donut-base-finetuned-cord-v2"   # checkpoint public (telecharge au 1er usage)


def jsonl_path(dataset: str, split: str) -> Path:
    return PROCESSED_DIR / f"{dataset}_{split}.jsonl"

# API (etape 6)
API_MAX_BYTES = 10 * 1024 * 1024                  # taille max d'une image envoyee a l'API
API_OCR_ENGINE = os.environ.get("DOCEXTRACT_OCR", "rapidocr")
DONUT_SROIE_MODEL = os.environ.get("DOCEXTRACT_DONUT_SROIE", str(DONUT_MODEL_DIR / "sroie_best"))
