import json

import pytest
from conftest import fake_cord, fake_sroie, load_script

from docextract import config, samples


def _prepare(tmp_project, monkeypatch, n_sroie=130):
    prepare = load_script("00_prepare_data")
    monkeypatch.setattr(prepare, "load_split",
                        lambda repo, split, limit: fake_sroie(n_sroie) if "SROIE" in repo else fake_cord(5))
    assert prepare.main([]) == 0


def test_prepare_exports_cord_validation_and_skip_sroie(tmp_project, monkeypatch):
    prepare = load_script("00_prepare_data")
    monkeypatch.setattr(prepare, "load_split", lambda repo, split, limit: fake_cord(4))
    assert prepare.main(["--skip-sroie"]) == 0
    processed = tmp_project / "data" / "processed"
    assert (processed / "cord_test.jsonl").exists() and (processed / "cord_validation.jsonl").exists()
    assert not (processed / "sroie_train.jsonl").exists()


def test_dev_and_eval_are_disjoint_and_stable(tmp_project, monkeypatch):
    _prepare(tmp_project, monkeypatch)
    dev = [r["doc_id"] for r in samples.get_docs("sroie", "dev")]
    ev = [r["doc_id"] for r in samples.get_docs("sroie", "eval")]
    assert len(dev) == 100 and len(ev) == 100
    assert all(d.startswith("sroie_train_") for d in dev) and all(e.startswith("sroie_test_") for e in ev)
    assert not set(dev) & set(ev)
    assert dev == [r["doc_id"] for r in samples.get_docs("sroie", "dev")]      # meme tirage a chaque appel
    saved = json.loads(config.DEV_SAMPLE_FILE.read_text(encoding="utf-8"))
    assert saved["seed"] == config.DEV_SEED and saved["sroie_train"] == sorted(dev)


def test_cord_uses_whole_splits(tmp_project, monkeypatch):
    _prepare(tmp_project, monkeypatch)
    assert len(samples.get_docs("cord", "dev")) == 5
    assert all("validation" in r["doc_id"] for r in samples.get_docs("cord", "dev"))
    assert all("_test_" in r["doc_id"] for r in samples.get_docs("cord", "eval"))
    assert samples.available("dev") == ["sroie", "cord"]


def test_missing_data_message(tmp_project):
    with pytest.raises(FileNotFoundError, match="00_prepare_data"):
        samples.get_docs("cord", "dev")
    assert samples.available("dev") == []
    with pytest.raises(ValueError):
        samples.get_docs("sroie", "autre")
