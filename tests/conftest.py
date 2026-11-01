"""Fixtures : faux jeux SROIE / CORD au meme format que Hugging Face (sans reseau)."""
import importlib.util
import json
from pathlib import Path

import pytest
from datasets import Dataset, Features, Image as HFImage, Sequence, Value
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def _img(w=120, h=300):
    return Image.new("RGB", (w, h), "white")


def fake_sroie(n=4):
    feats = Features({
        "image": HFImage(),
        "key": Value("string"),
        "image_size": {"width": Value("int32"), "height": Value("int32")},
        "entities": {"company": Value("string"), "date": Value("string"),
                     "address": Value("string"), "total": Value("string")},
        "words": Sequence(Value("string")),
        "bboxes": Sequence(Sequence(Value("int32"))),
    })
    data = {
        "image": [_img(100 + i, 300) for i in range(n)],
        "key": [f"X5100{i:07d}" for i in range(n)],
        "image_size": [{"width": 100 + i, "height": 300} for i in range(n)],
        "entities": [{"company": f"SHOP {i} SDN BHD", "date": "25/12/2018",
                      "address": "NO 1 JALAN", "total": "9.00" if i else ""}
                     for i in range(n)],
        "words": [["SHOP", "TOTAL", "9.00"] for _ in range(n)],
        "bboxes": [[[1, 2, 30, 12], [1, 20, 30, 30], [40, 20, 60, 30]] for _ in range(n)],
    }
    return Dataset.from_dict(data, features=feats)


CORD_GT = {
    "gt_parse": {
        "menu": [{"nm": "NASI GORENG", "cnt": "2", "price": "50,000"},
                 {"nm": "ES TEH", "cnt": "1", "price": "8,000"}],
        "sub_total": {"subtotal_price": "58,000", "tax_price": "5,800"},
        "total": {"total_price": "63,800", "cashprice": "100,000"},
    },
    "valid_line": [
        {"category": "total.total_price", "words": [
            {"quad": {"x1": 10, "y1": 5, "x2": 50, "y2": 5, "x3": 50, "y3": 15, "x4": 10, "y4": 15},
             "text": "63,800"}]},
    ],
}


def fake_cord(n=3):
    feats = Features({"image": HFImage(), "ground_truth": Value("string")})
    gts = [CORD_GT] * (n - 1) + [{"gt_parse": {"menu": {"nm": "KOPI", "price": "10"},
                                               "total": {"total_price": "10"}}}]
    return Dataset.from_dict(
        {"image": [_img() for _ in range(n)], "ground_truth": [json.dumps(g) for g in gts]},
        features=feats)


@pytest.fixture
def tmp_project(tmp_path, monkeypatch):
    """Redirige tous les chemins de config vers un dossier temporaire."""
    from docextract import config
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "data" / "raw")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "data" / "processed")
    monkeypatch.setattr(config, "HF_CACHE_DIR", tmp_path / "data" / "hf_cache")
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(config, "EVAL_SAMPLE_FILE",
                        tmp_path / "data" / "processed" / "eval_sample.json")
    monkeypatch.setattr(config, "DEV_SAMPLE_FILE",
                        tmp_path / "data" / "processed" / "dev_sample.json")
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path / "data" / "ocr_cache")
    monkeypatch.setattr(config, "RULES_OUTPUT_DIR", tmp_path / "outputs" / "rules")
    monkeypatch.setattr(config, "LLM_CACHE_DIR", tmp_path / "data" / "llm_cache")
    monkeypatch.setattr(config, "LLM_OUTPUT_DIR", tmp_path / "outputs" / "llm")
    monkeypatch.setattr(config, "DONUT_PRED_DIR", tmp_path / "data" / "donut_preds")
    monkeypatch.setattr(config, "DONUT_OUTPUT_DIR", tmp_path / "outputs" / "donut")
    monkeypatch.setattr(config, "DONUT_MODEL_DIR", tmp_path / "data" / "donut_models")
    monkeypatch.setattr(config, "DONUT_LOCAL_PRED_DIR", tmp_path / "data" / "donut_preds_local")
    return tmp_path


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name.replace("0", "s0", 1), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
