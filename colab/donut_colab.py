"""Etape 4b - Fine-tuning de Donut sur SROIE + predictions (a lancer sur Google Colab, GPU T4).

Donut (naver-clova-ix/donut-base) lit l'image du recu et ecrit directement le JSON : pas d'OCR.

    image ─► encodeur Swin ─► decodeur (texte) ─► <s_company>...</s_company><s_date>...</s_date><s_total>...</s_total>

Ce fichier est autonome (aucun import du projet) : le notebook le copie tel quel sur Colab.
Les imports lourds sont en haut : Colab a deja torch ; transformers est installe par le notebook (version fixee).
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

TASK_PROMPT = "<s_receipt>"
SROIE_FIELDS = ("company", "date", "total")
BASE_MODEL = "naver-clova-ix/donut-base"
CORD_MODEL = "naver-clova-ix/donut-base-finetuned-cord-v2"
IMAGE_SIZE = (1280, 960)        # (hauteur, largeur) : 2 fois moins que le defaut de Donut, tient sur un T4
MAX_LENGTH = 128                # jetons de la sequence cible (3 champs courts)


# ------------------------------------------------------------------ cible
def json2token(obj) -> str:
    """{'company': 'A', 'total': '5'} -> '<s_company>A</s_company><s_total>5</s_total>' (comme Donut)."""
    if isinstance(obj, dict):
        return "".join(f"<s_{k}>{json2token(v)}</s_{k}>" for k, v in obj.items())
    if isinstance(obj, list):
        return "<sep/>".join(json2token(v) for v in obj)
    return str(obj)


def special_tokens(fields=SROIE_FIELDS, task_prompt: str = TASK_PROMPT) -> list[str]:
    toks = [task_prompt]
    for f in fields:
        toks += [f"<s_{f}>", f"</s_{f}>"]
    return toks


def read_set(folder) -> list[dict]:
    """metadata.jsonl d'un jeu -> [{'path', 'doc_id', 'gt' (dict ou None)}]."""
    folder = Path(folder)
    rows = []
    for line in (folder / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        gt = json.loads(m["ground_truth"])["gt_parse"] if m.get("ground_truth") else None
        rows.append({"path": folder / m["file_name"], "doc_id": m["doc_id"], "gt": gt})
    return rows


# ------------------------------------------------------------------ donnees
class DonutDataset(torch.utils.data.Dataset):
    """Un item = image redimensionnee + sequence cible. decoder_input = ids[:-1], labels = ids[1:] (pad = -100)."""

    def __init__(self, rows, processor, max_length: int = MAX_LENGTH, task_prompt: str = TASK_PROMPT):
        self.rows = [r for r in rows if r["gt"] is not None]
        self.processor, self.max_length, self.task_prompt = processor, max_length, task_prompt

    def __len__(self):
        return len(self.rows)

    def target_ids(self, gt: dict) -> list[int]:
        tok = self.processor.tokenizer
        text = self.task_prompt + json2token(gt) + tok.eos_token
        ids = tok(text, add_special_tokens=False).input_ids[: self.max_length + 1]
        return ids + [tok.pad_token_id] * (self.max_length + 1 - len(ids))

    def __getitem__(self, i):
        row = self.rows[i]
        image = Image.open(row["path"]).convert("RGB")
        pixel = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        ids = torch.tensor(self.target_ids(row["gt"]))
        labels = ids[1:].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel, "decoder_input_ids": ids[:-1], "labels": labels}


# ------------------------------------------------------------------ modele
def prepare(model, processor, tokens, image_size=IMAGE_SIZE, task_prompt: str = TASK_PROMPT) -> None:
    """Ajoute les jetons `<s_company>`... et regle la taille d'image / jetons de depart. Modifie model et processor."""
    tok = processor.tokenizer
    if tok.add_tokens(tokens):
        model.decoder.resize_token_embeddings(len(tok))
    processor.image_processor.size = {"height": image_size[0], "width": image_size[1]}
    processor.image_processor.do_align_long_axis = False
    model.config.pad_token_id = tok.pad_token_id
    model.config.decoder_start_token_id = tok.convert_tokens_to_ids(task_prompt)


def load_base(source: str = BASE_MODEL, image_size=IMAGE_SIZE, tokens=None, max_length: int = MAX_LENGTH):
    """Charge donut-base (ou un checkpoint sauvegarde) avec une image plus petite. Renvoie (model, processor)."""
    from transformers import DonutProcessor, VisionEncoderDecoderConfig, VisionEncoderDecoderModel
    config = VisionEncoderDecoderConfig.from_pretrained(source)
    config.encoder.image_size = list(image_size)
    config.decoder.max_length = max_length
    model = VisionEncoderDecoderModel.from_pretrained(source, config=config)
    processor = DonutProcessor.from_pretrained(source)
    prepare(model, processor, tokens or special_tokens(), image_size)
    return model, processor


# ------------------------------------------------------------------ entrainement
def _loss(model, batch, device, amp: bool) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        out = model(pixel_values=batch["pixel_values"].to(device), decoder_input_ids=batch["decoder_input_ids"].to(device))
    logits = out.logits.float()
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch["labels"].to(device).reshape(-1), ignore_index=-100)


@torch.no_grad()
def dev_loss(model, loader, device, amp: bool) -> float:
    model.eval()
    losses = [_loss(model, b, device, amp).item() for b in loader]
    return sum(losses) / max(len(losses), 1)


def train(model, processor, train_rows, dev_rows, out_dir, *, epochs: int = 15, batch_size: int = 1, accum: int = 8,
          lr: float = 3e-5, device: str = "cuda", amp: bool = True, checkpointing: bool = True,
          limit: int = 0, save_last_every: int = 5, num_workers: int = 2, log=print) -> dict:
    """Fine-tuning. Sauvegarde `best/` quand la perte dev baisse et `last/` toutes les `save_last_every` epoques (et a la fin).
    Chaque sauvegarde pese ~0,8 Go sur Drive : on limite leur nombre. Renvoie un resume."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    amp = amp and device == "cuda"
    train_ds, dev_ds = DonutDataset(train_rows, processor), DonutDataset(dev_rows, processor)
    if limit:
        train_ds.rows, dev_ds.rows = train_ds.rows[:limit], dev_ds.rows[:max(2, limit // 4)]
    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dev_loader = torch.utils.data.DataLoader(dev_ds, batch_size=batch_size, num_workers=num_workers)

    model.to(device)
    if checkpointing:
        try:
            model.gradient_checkpointing_enable()       # ~30 % plus lent, mais evite le manque de memoire
        except Exception as e:                           # pragma: no cover - depend de la version
            log(f"checkpointing indisponible ({e}) : on continue sans")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = max(1, epochs * math.ceil(len(loader) / accum))
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    history, best, t_start = [], float("inf"), time.time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, epochs + 1):
        model.train()
        t0, running, n = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader, 1):
            loss = _loss(model, batch, device, amp)
            scaler.scale(loss / accum).backward()
            running, n = running + loss.item(), n + 1
            if i % accum == 0 or i == len(loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
        val = dev_loss(model, dev_loader, device, amp)
        secs = time.time() - t0
        history.append({"epoch": epoch, "train_loss": running / max(n, 1), "dev_loss": val, "seconds": round(secs, 1)})
        log(f"epoque {epoch}/{epochs} | perte train {running / max(n, 1):.4f} | perte dev {val:.4f} | "
            f"{secs / max(len(train_ds), 1):.2f} s/recu | {secs / 60:.1f} min")
        if math.isnan(val):
            log("ATTENTION : perte NaN (debordement fp16). Relance avec amp=False dans dc.train(...).")
        if epoch % save_last_every == 0 or epoch == epochs:
            model.save_pretrained(out_dir / "last")
            processor.save_pretrained(out_dir / "last")
        if val < best:
            best = val
            model.save_pretrained(out_dir / "best")
            processor.save_pretrained(out_dir / "best")
    info = {"epochs": epochs, "batch_size": batch_size, "accum": accum, "lr": lr, "n_train": len(train_ds),
            "best_dev_loss": best, "train_minutes": round((time.time() - t_start) / 60, 1), "history": history,
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}
    (out_dir / "train_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


# ------------------------------------------------------------------ prediction
def clean_sequence(seq: str, tokenizer) -> str:
    """Retire eos / pad et le premier jeton (le prompt de tache) avant token2json."""
    seq = seq.replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "")
    return re.sub(r"<.*?>", "", seq, count=1).strip()


@torch.no_grad()
def predict(model, processor, rows, *, task_prompt: str = TASK_PROMPT, max_length: int = MAX_LENGTH,
            device: str = "cuda", amp: bool = True, limit: int = 0, log=print) -> dict:
    """{doc_id: {'parsed': dict, 'sequence': str, 'ms': float}} ; ms = temps de generation d'un recu (hors lecture disque)."""
    tok = processor.tokenizer
    model.to(device).eval()
    amp = amp and device == "cuda"
    prompt_ids = tok(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    bad = [[tok.unk_token_id]] if tok.unk_token_id is not None else None
    results = {}
    for k, row in enumerate(rows[:limit] if limit else rows, 1):
        image = Image.open(row["path"]).convert("RGB")
        pixel = processor(image, return_tensors="pt").pixel_values.to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            out = model.generate(pixel, decoder_input_ids=prompt_ids, max_length=max_length, early_stopping=True,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, use_cache=True,
                                 num_beams=1, bad_words_ids=bad, return_dict_in_generate=True)
        if device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        seq = clean_sequence(tok.batch_decode(out.sequences)[0], tok)
        try:
            parsed = processor.token2json(seq)
        except Exception as e:                           # sequence mal formee : on garde le texte brut
            parsed = {"_error": str(e)}
        results[row["doc_id"]] = {"parsed": parsed, "sequence": seq, "ms": round(ms, 1)}
        if k % 20 == 0:
            log(f"  {k}/{len(rows)} recus")
    return results


def save_predictions(results: dict, path, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta or {}, "predictions": results}, indent=1, ensure_ascii=False), encoding="utf-8")
