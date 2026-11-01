"""Test de bout en bout des scripts de l'etape 0, avec faux jeux (sans reseau)."""
import json

from conftest import fake_cord, fake_sroie, load_script


def _fake_loader(repo, split, limit):
    ds = fake_sroie(130) if "SROIE" in repo else fake_cord(5)
    return ds.select(range(min(limit, len(ds)))) if limit else ds


def test_prepare_then_inspect(tmp_project, monkeypatch, capsys):
    prepare = load_script("00_prepare_data")
    monkeypatch.setattr(prepare, "load_split", _fake_loader)
    assert prepare.main([]) == 0

    processed = tmp_project / "data" / "processed"
    for name in ("sroie_train", "sroie_test", "cord_test"):
        assert (processed / f"{name}.jsonl").exists()

    sample = json.loads((processed / "eval_sample.json").read_text(encoding="utf-8"))
    assert sample["seed"] == 42
    assert len(sample["sroie_test"]) == 100     # 130 dispo -> 100 tires
    assert len(sample["cord_test"]) == 5        # moins de 100 -> tous

    inspect = load_script("01_inspect_data")
    assert inspect.main() == 0
    out = capsys.readouterr().out
    assert "images manquantes sur disque : 0" in out
    assert "'99/99/9999'" in out                # format de date detecte
    assert (tmp_project / "outputs" / "inspect" / "apercu_sroie.jpg").exists()
    report = json.loads((tmp_project / "outputs" / "inspect" / "rapport_donnees.json")
                        .read_text(encoding="utf-8"))
    assert report["sroie_test"]["n"] == 130


def test_prepare_skip_cord_and_limit(tmp_project, monkeypatch):
    prepare = load_script("00_prepare_data")
    monkeypatch.setattr(prepare, "load_split", _fake_loader)
    assert prepare.main(["--skip-cord", "--limit", "3"]) == 0
    processed = tmp_project / "data" / "processed"
    assert not (processed / "cord_test.jsonl").exists()
    sample = json.loads((processed / "eval_sample.json").read_text(encoding="utf-8"))
    assert len(sample["sroie_test"]) == 3 and "cord_test" not in sample


def test_inspect_without_data(tmp_project, capsys):
    inspect = load_script("01_inspect_data")
    assert inspect.main() == 1
    assert "Aucune donnee" in capsys.readouterr().out
