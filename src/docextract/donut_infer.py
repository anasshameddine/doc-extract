"""Approche 3 en local : fait tourner Donut sur ce PC (c'est aussi ce que l'API utilisera a l'etape 6).

    image ─► Donut (encodeur Swin + decodeur) ─► JSON ─► champs

torch et transformers ne sont importes qu'a l'utilisation : le reste du projet marche sans eux
(installation : requirements-donut.txt).
"""
from __future__ import annotations

import re
import time
from typing import Any

from . import config
from .donut import fields_from_prediction
from .schema import ExtractionResult, ReceiptFields

TASK_PROMPTS = {"sroie": "<s_receipt>", "cord": "<s_cord-v2>"}
MAX_LENGTHS = {"sroie": 128, "cord": 768}


def clean_sequence(seq: str, tokenizer) -> str:
    """Retire eos / pad et le premier jeton (le prompt de tache) avant token2json."""
    seq = seq.replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "")
    return re.sub(r"<.*?>", "", seq, count=1).strip()


class DonutExtractor:
    """Un modele Donut charge en memoire. `dataset` choisit le prompt de tache et le schema de sortie."""

    def __init__(self, source: str, dataset: str = "sroie", device: str | None = None, amp: bool = True):
        import torch
        from transformers import DonutProcessor, VisionEncoderDecoderModel
        if dataset not in TASK_PROMPTS:
            raise ValueError(f"jeu inconnu : {dataset!r}")
        self.dataset, self.source = dataset, str(source)
        self.task_prompt, self.max_length = TASK_PROMPTS[dataset], MAX_LENGTHS[dataset]
        t0 = time.perf_counter()
        self.processor = DonutProcessor.from_pretrained(source)
        self.model = VisionEncoderDecoderModel.from_pretrained(source).eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.amp = amp and self.device == "cuda"          # fp16 sur GPU : poids en fp32, calculs en fp16
        self.load_seconds = round(time.perf_counter() - t0, 1)
        self._torch = torch

    @property
    def gpu(self) -> str:
        return self._torch.cuda.get_device_name(0) if self.device == "cuda" else "cpu"

    def run(self, image) -> dict[str, Any]:
        """image PIL -> {'parsed', 'sequence', 'ms' (generation seule), 'ms_total' (preparation de l'image comprise)}."""
        torch, tok = self._torch, self.processor.tokenizer
        t_start = time.perf_counter()
        pixel = self.processor(image.convert("RGB"), return_tensors="pt").pixel_values.to(self.device)
        prompt = tok(self.task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        bad = [[tok.unk_token_id]] if tok.unk_token_id is not None else None
        with torch.no_grad():
            if self.device == "cuda":
                torch.cuda.synchronize()
            t_gen = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp):
                out = self.model.generate(pixel, decoder_input_ids=prompt, max_length=self.max_length, use_cache=True,
                                          pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, num_beams=1,
                                          bad_words_ids=bad, return_dict_in_generate=True)
            if self.device == "cuda":
                torch.cuda.synchronize()
            now = time.perf_counter()
        seq = clean_sequence(tok.batch_decode(out.sequences)[0], tok)
        try:
            parsed = self.processor.token2json(seq)
        except Exception as e:                           # sequence mal formee : on garde le texte brut
            parsed = {"_error": str(e)}
        return {"parsed": parsed, "sequence": seq, "ms": round((now - t_gen) * 1000, 1),
                "ms_total": round((now - t_start) * 1000, 1)}

    def warm_up(self) -> None:
        """Premiere inference jetable (initialisation CUDA) : a faire avant de mesurer des temps."""
        from PIL import Image
        self.run(Image.new("RGB", (320, 480), "white"))

    def peak_vram_gb(self) -> float | None:
        return round(self._torch.cuda.max_memory_allocated() / 1e9, 2) if self.device == "cuda" else None

    def extract(self, image, doc_id: str | None = None) -> ExtractionResult:
        """Au format commun aux 3 approches (c'est ce que l'API renverra)."""
        res = self.run(image)
        error = res["parsed"].get("_error") if isinstance(res["parsed"], dict) else None
        fields = ReceiptFields(**fields_from_prediction(self.dataset, res["parsed"]))
        return ExtractionResult(doc_id=doc_id, method="docai", fields=fields, latency_ms=res["ms_total"], error=error)
