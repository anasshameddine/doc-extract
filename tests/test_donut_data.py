import json
import zipfile

from conftest import fake_cord, fake_sroie, load_script
from docextract import config
from docextract.donut_data import export_all, metadata_line, sroie_target


def prepare(tmp_project, monkeypatch):
    mod = load_script("00_prepare_data")
    monkeypatch.setattr(mod, "load_split", lambda repo, split, limit: fake_sroie(130) if "SROIE" in repo else fake_cord(4))
    assert mod.main([]) == 0


def read(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def test_target_keeps_raw_text_and_empty_string_for_missing():
    gt = {"company": "SHOP SDN BHD", "date": "25/12/2018", "total": None, "address": "x", "tax": "1"}
    assert sroie_target(gt) == {"company": "SHOP SDN BHD", "date": "25/12/2018", "total": ""}


def test_metadata_line_has_truth_only_for_train_and_dev():
    rec = {"doc_id": "d1", "gt": {"company": "A", "date": "1/1/2018", "total": "5"}}
    assert json.loads(metadata_line(rec, "d1.jpg", True)["ground_truth"]) == {
        "gt_parse": {"company": "A", "date": "1/1/2018", "total": "5"}}
    assert "ground_truth" not in metadata_line(rec, "d1.jpg", False)


def test_export_separates_train_dev_eval(tmp_project, monkeypatch):
    prepare(tmp_project, monkeypatch)
    counts = export_all()
    assert counts == {"sroie_train": 30, "sroie_dev": 100, "sroie_eval": 100, "cord_dev": 4, "cord_eval": 4}
    d = config.DATA_DIR / "donut"
    train, dev, ev = (read(d / n / "metadata.jsonl") for n in ("sroie_train", "sroie_dev", "sroie_eval"))
    ids = lambda rows: {r["doc_id"] for r in rows}
    assert not ids(train) & ids(dev) and not ids(train) & ids(ev) and not ids(dev) & ids(ev)
    assert all("ground_truth" in r for r in train + dev)
    assert all("ground_truth" not in r for r in ev)                          # l'eval n'embarque aucune verite
    assert all((d / "sroie_train" / r["file_name"]).exists() for r in train)
    for name in ("cord_dev", "cord_eval"):
        assert all("ground_truth" not in r for r in read(d / name / "metadata.jsonl"))


def test_export_zip_contains_every_set(tmp_project, monkeypatch):
    prepare(tmp_project, monkeypatch)
    export_all()
    with zipfile.ZipFile(config.DATA_DIR / "donut_data.zip") as z:
        names = z.namelist()
    for folder in ("sroie_train", "sroie_dev", "sroie_eval", "cord_dev", "cord_eval"):
        assert f"{folder}/metadata.jsonl" in names


def test_script_runs_and_reports(tmp_project, monkeypatch, capsys):
    prepare(tmp_project, monkeypatch)
    assert load_script("07_export_donut").main([]) == 0
    out = capsys.readouterr().out
    assert "sroie_train" in out and "donut_data.zip" in out and "aucune verite" in out


def test_script_without_data(tmp_project, capsys):
    assert load_script("07_export_donut").main([]) == 1
    assert "00_prepare_data" in capsys.readouterr().out
