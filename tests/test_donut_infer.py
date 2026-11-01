"""Donut en local : DonutExtractor et scripts/09_run_donut.py, avec un MINI modele aleatoire (pas de telechargement).

Necessite torch + transformers (absents du venv du projet : le fichier est alors ignore)."""
import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from PIL import Image  # noqa: E402
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers  # noqa: E402
from transformers import (DonutImageProcessor, DonutProcessor, DonutSwinConfig, MBartConfig,  # noqa: E402
                          PreTrainedTokenizerFast, VisionEncoderDecoderConfig, VisionEncoderDecoderModel)

from conftest import fake_cord, fake_sroie, load_script  # noqa: E402
from docextract import config  # noqa: E402
from docextract.donut_infer import DonutExtractor, clean_sequence  # noqa: E402

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,/-&'()_"


def tiny_model_dir(path, prompts=("<s_receipt>", "<s_cord-v2>")):
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
    vocab.update({c: i + 4 for i, c in enumerate(CHARS)})
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    tk.decoder = decoders.Fuse()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>")
    tok.add_tokens(list(prompts))
    enc = DonutSwinConfig(image_size=[96, 64], patch_size=4, embed_dim=16, depths=[1, 1, 1, 1], num_heads=[1, 1, 2, 2],
                          window_size=2, num_channels=3)
    dec = MBartConfig(vocab_size=len(tok), d_model=32, encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
                      decoder_attention_heads=2, decoder_ffn_dim=64, encoder_ffn_dim=64, max_position_embeddings=1024,
                      is_decoder=True, add_cross_attention=True, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    model = VisionEncoderDecoderModel(VisionEncoderDecoderConfig.from_encoder_decoder_configs(enc, dec))
    proc = DonutProcessor(image_processor=DonutImageProcessor(size={"height": 96, "width": 64}, do_align_long_axis=False),
                          tokenizer=tok)
    model.save_pretrained(path)
    proc.save_pretrained(path)
    return path


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return tiny_model_dir(tmp_path_factory.mktemp("tiny_donut"))


def test_clean_sequence_removes_prompt_and_special_tokens(tiny):
    tok = DonutProcessor.from_pretrained(tiny).tokenizer
    assert clean_sequence("<s_receipt><s_total>5</s_total></s><pad>", tok) == "<s_total>5</s_total>"


def test_extractor_runs_and_returns_the_common_result(tiny):
    ex = DonutExtractor(str(tiny), "sroie", device="cpu")
    res = ex.run(Image.new("RGB", (60, 120), "white"))
    assert set(res) == {"parsed", "sequence", "ms", "ms_total"} and res["ms_total"] >= res["ms"] >= 0
    out = ex.extract(Image.new("RGB", (60, 120)), doc_id="d1")
    assert out.method == "docai" and out.doc_id == "d1" and out.latency_ms > 0
    assert ex.gpu == "cpu" and ex.peak_vram_gb() is None and ex.load_seconds >= 0


def test_extractor_cord_uses_its_own_prompt_and_schema(tiny):
    ex = DonutExtractor(str(tiny), "cord", device="cpu")
    assert ex.task_prompt == "<s_cord-v2>" and ex.max_length == 768
    with pytest.raises(ValueError):
        DonutExtractor(str(tiny), "other", device="cpu")


def test_extractor_error_is_reported_not_raised(tiny, monkeypatch):
    ex = DonutExtractor(str(tiny), "sroie", device="cpu")
    monkeypatch.setattr(ex.processor, "token2json", lambda s: (_ for _ in ()).throw(ValueError("bad")))
    out = ex.extract(Image.new("RGB", (60, 120)))
    assert out.error == "bad" and out.fields.total is None


# ------------------------------------------------------------------ script
@pytest.fixture
def project(tmp_project, monkeypatch):
    prepare = load_script("00_prepare_data")
    monkeypatch.setattr(prepare, "load_split",
                        lambda repo, split, limit: fake_sroie(130) if "SROIE" in repo else fake_cord(4))
    assert prepare.main([]) == 0
    return tmp_project


def test_script_writes_local_predictions_with_hardware_info(project, tiny, capsys):
    argv = ["--set", "dev", "--limit", "3", "--device", "cpu", "--sroie-model", str(tiny), "--cord-model", str(tiny)]
    assert load_script("09_run_donut").main(argv) == 0
    out = capsys.readouterr().out
    assert "[sroie] modele charge" in out and "[cord] modele charge" in out and "--preds-dir data/donut_preds_local" in out
    data = json.loads((config.DONUT_LOCAL_PRED_DIR / "predictions_sroie_dev.json").read_text(encoding="utf-8"))
    assert len(data["predictions"]) == 3 and data["meta"]["local"] is True and data["meta"]["device"] == "cpu"
    assert all({"parsed", "sequence", "ms", "ms_total"} <= set(v) for v in data["predictions"].values())


def test_local_predictions_can_be_scored_by_script_08(project, tiny, capsys):
    load_script("09_run_donut").main(["--set", "dev", "--limit", "3", "--device", "cpu", "--sroie-model", str(tiny),
                                      "--cord-model", str(tiny)])
    capsys.readouterr()
    code = load_script("08_eval_donut").main(["--set", "dev", "--preds-dir", str(config.DONUT_LOCAL_PRED_DIR)])
    out = capsys.readouterr().out
    assert code == 0 and "sur ton PC" in out and (config.DONUT_OUTPUT_DIR / "dev_sroie_local.json").exists()


def test_script_missing_model_explains_where_to_copy_it(project, capsys):
    assert load_script("09_run_donut").main(["--set", "dev", "--dataset", "sroie"]) == 1
    assert "data/donut_models/sroie_best" in capsys.readouterr().out


def test_script_eval_needs_final(project, capsys):
    assert load_script("09_run_donut").main(["--set", "eval"]) == 3
