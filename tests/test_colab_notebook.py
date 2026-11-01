"""Le notebook Colab est genere a partir de colab/donut_colab.py : il ne doit jamais diverger du code teste."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_notebook", ROOT / "colab" / "build_notebook.py")
nb_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nb_mod)


def cells():
    return json.loads((ROOT / "colab" / "DONUT_COLAB.ipynb").read_text(encoding="utf-8"))["cells"]


def source(cell):
    return "".join(cell["source"])


def test_committed_notebook_is_up_to_date():
    assert json.loads((ROOT / "colab" / "DONUT_COLAB.ipynb").read_text(encoding="utf-8")) == json.loads(
        json.dumps(nb_mod.build(), ensure_ascii=False))


def test_notebook_embeds_the_tested_script_verbatim():
    script = (ROOT / "colab" / "donut_colab.py").read_text(encoding="utf-8")
    writefile = [source(c) for c in cells() if source(c).startswith("%%writefile donut_colab.py")]
    assert len(writefile) == 1 and writefile[0].split("\n", 1)[1].strip() == script.strip()


def test_code_cells_are_valid_python_once_magics_are_removed():
    for c in cells():
        if c["cell_type"] != "code" or source(c).startswith("%%writefile"):
            continue
        lines = [ln for ln in source(c).splitlines() if not ln.lstrip().startswith(("!", "%"))]
        compile("\n".join(lines), "<cell>", "exec")


def test_eval_cell_is_commented_out():
    last = source(cells()[-1])
    assert "sroie_eval" in last and all(ln.startswith("#") or not ln.strip() for ln in last.splitlines())
