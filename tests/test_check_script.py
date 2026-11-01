"""Test de bout en bout du script de l'etape 1 (faux jeux, sans reseau)."""
import csv

from conftest import fake_cord, fake_sroie, load_script


def test_check_ground_truth(tmp_project, monkeypatch, capsys):
    prepare = load_script("00_prepare_data")
    monkeypatch.setattr(prepare, "load_split",
                        lambda repo, split, limit: fake_sroie(6) if "SROIE" in repo else fake_cord(3))
    assert prepare.main([]) == 0

    check = load_script("02_check_ground_truth")
    assert check.main() == 0
    out = capsys.readouterr().out
    assert "controle evaluation : OK" in out
    assert "date     normalises    6/6" in out
    fails = tmp_project / "outputs" / "ground_truth" / "echecs_normalisation.csv"
    assert fails.exists()
    assert list(csv.DictReader(fails.open(encoding="utf-8"))) == []


def test_check_without_data(tmp_project, capsys):
    check = load_script("02_check_ground_truth")
    assert check.main() == 1
