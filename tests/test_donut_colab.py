"""Teste le code Colab (colab/donut_colab.py) avec un MINI modele aleatoire : memes appels que sur Colab, en quelques secondes.

Necessite torch + transformers (absents du venv du projet : le fichier est alors ignore)."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers  # noqa: E402
from transformers import (DonutImageProcessor, DonutProcessor, DonutSwinConfig, MBartConfig,  # noqa: E402
                          PreTrainedTokenizerFast, VisionEncoderDecoderConfig, VisionEncoderDecoderModel)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("donut_colab", ROOT / "colab" / "donut_colab.py")
dc = importlib.util.module_from_spec(spec)
sys.modules["donut_colab"] = dc
spec.loader.exec_module(dc)

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,/-&'()"


def tiny_tokenizer():
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
    vocab.update({c: i + 4 for i, c in enumerate(CHARS)})
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")      # un jeton par caractere
    tk.decoder = decoders.Fuse()                                                  # decode sans espaces en trop
    return PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>")


@pytest.fixture
def tiny(tmp_path):
    """Mini Donut sauvegarde sur disque (comme un checkpoint Hugging Face)."""
    tok = tiny_tokenizer()
    enc = DonutSwinConfig(image_size=[64, 48], patch_size=4, embed_dim=16, depths=[1, 1, 1, 1], num_heads=[1, 1, 2, 2],
                          window_size=2, num_channels=3)
    dec = MBartConfig(vocab_size=len(tok), d_model=32, encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
                      decoder_attention_heads=2, decoder_ffn_dim=64, encoder_ffn_dim=64, max_position_embeddings=256,
                      is_decoder=True, add_cross_attention=True, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    cfg = VisionEncoderDecoderConfig.from_encoder_decoder_configs(enc, dec)
    model = VisionEncoderDecoderModel(cfg)
    proc = DonutProcessor(image_processor=DonutImageProcessor(), tokenizer=tok)
    src = tmp_path / "tiny_donut"
    model.save_pretrained(src)
    proc.save_pretrained(src)
    return src


def write_set(folder: Path, n: int, truth: bool):
    folder.mkdir(parents=True)
    lines = []
    for i in range(n):
        Image.new("RGB", (60, 120), (255 - i * 10, 255, 255)).save(folder / f"d{i}.jpg")
        line = {"file_name": f"d{i}.jpg", "doc_id": f"d{i}"}
        if truth:
            line["ground_truth"] = json.dumps({"gt_parse": {"company": f"SHOP {i} SDN BHD", "date": "25/12/2018",
                                                              "total": "9.00"}})
        lines.append(json.dumps(line))
    (folder / "metadata.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_json2token_matches_donut_format():
    assert dc.json2token({"company": "A B", "total": "5"}) == "<s_company>A B</s_company><s_total>5</s_total>"
    assert dc.json2token({"menu": [{"nm": "x"}, {"nm": "y"}]}) == "<s_menu><s_nm>x</s_nm><sep/><s_nm>y</s_nm></s_menu>"


def test_read_set_with_and_without_truth(tmp_path):
    write_set(tmp_path / "a", 2, True)
    write_set(tmp_path / "b", 2, False)
    assert dc.read_set(tmp_path / "a")[0]["gt"]["company"] == "SHOP 0 SDN BHD"
    assert all(r["gt"] is None for r in dc.read_set(tmp_path / "b"))


def test_load_base_adds_tokens_and_resizes_image(tiny):
    model, proc = dc.load_base(str(tiny), image_size=(96, 64), max_length=40)
    tok = proc.tokenizer
    assert tok.convert_tokens_to_ids(dc.TASK_PROMPT) != tok.unk_token_id
    assert model.config.decoder_start_token_id == tok.convert_tokens_to_ids(dc.TASK_PROMPT)
    assert model.decoder.get_input_embeddings().num_embeddings == len(tok)
    assert list(model.config.encoder.image_size) == [96, 64]
    pix = proc(Image.new("RGB", (60, 120)), return_tensors="pt").pixel_values
    assert tuple(pix.shape[-2:]) == (96, 64)                     # (hauteur, largeur)


def test_dataset_shifts_labels_and_masks_padding(tiny, tmp_path):
    write_set(tmp_path / "t", 1, True)
    model, proc = dc.load_base(str(tiny), image_size=(96, 64), max_length=40)
    ds = dc.DonutDataset(dc.read_set(tmp_path / "t"), proc, max_length=40)
    item = ds[0]
    tok = proc.tokenizer
    di, lab = item["decoder_input_ids"], item["labels"]
    assert di.shape == lab.shape == (40,)
    assert di[0].item() == tok.convert_tokens_to_ids(dc.TASK_PROMPT)           # commence par le prompt
    assert lab[0].item() == tok.convert_tokens_to_ids("<s_company>")           # label = jeton suivant
    real = lab[lab != -100]
    assert real[-1].item() == tok.eos_token_id and (lab == -100).any()         # eos a la fin, padding ignore
    text = tok.decode(real, skip_special_tokens=False)
    assert "SHOP 0 SDN BHD" in text.replace("<pad>", "")


def test_train_predict_roundtrip_on_cpu(tiny, tmp_path):
    write_set(tmp_path / "train", 4, True)
    write_set(tmp_path / "dev", 2, True)
    model, proc = dc.load_base(str(tiny), image_size=(96, 64), max_length=40)
    logs = []
    info = dc.train(model, proc, dc.read_set(tmp_path / "train"), dc.read_set(tmp_path / "dev"), tmp_path / "out",
                    epochs=2, batch_size=2, accum=2, device="cpu", checkpointing=True, num_workers=0, log=logs.append)
    assert len(info["history"]) == 2 and info["gpu"] == "cpu" and len(logs) == 2
    assert (tmp_path / "out" / "last" / "config.json").exists() and (tmp_path / "out" / "best" / "config.json").exists()
    assert (tmp_path / "out" / "train_info.json").exists()
    # le checkpoint sauvegarde se recharge et predit (c'est ce que fera la suite sur Colab)
    model2, proc2 = dc.load_base(str(tmp_path / "out" / "best"), image_size=(96, 64), max_length=40)
    res = dc.predict(model2, proc2, dc.read_set(tmp_path / "dev"), max_length=40, device="cpu")
    assert set(res) == {"d0", "d1"}
    for v in res.values():
        assert isinstance(v["parsed"], dict) and v["ms"] >= 0 and "<s_receipt>" not in v["sequence"]
    dc.save_predictions(res, tmp_path / "p" / "pred.json", meta={"model": "tiny"})
    saved = json.loads((tmp_path / "p" / "pred.json").read_text(encoding="utf-8"))
    assert saved["meta"]["model"] == "tiny" and set(saved["predictions"]) == {"d0", "d1"}


def test_training_can_memorize_one_receipt(tiny, tmp_path):
    """Si la boucle est correcte, la perte baisse nettement et le modele recopie la cible apprise."""
    write_set(tmp_path / "train", 1, True)
    rows = dc.read_set(tmp_path / "train")
    model, proc = dc.load_base(str(tiny), image_size=(96, 64), max_length=40)
    info = dc.train(model, proc, rows, rows, tmp_path / "out", epochs=120, batch_size=1, accum=1, lr=1e-2,
                    device="cpu", checkpointing=False, num_workers=0, log=lambda *_: None)
    assert info["history"][-1]["train_loss"] < 0.3 * info["history"][0]["train_loss"]
    res = dc.predict(model, proc, rows, max_length=40, device="cpu")
    assert res["d0"]["parsed"].get("company") == "SHOP 0 SDN BHD" and res["d0"]["parsed"].get("total") == "9.00"


def test_clean_sequence_removes_prompt_and_special_tokens():
    tok = tiny_tokenizer()
    assert dc.clean_sequence("<s_receipt><s_total>5</s_total></s><pad>", tok) == "<s_total>5</s_total>"
