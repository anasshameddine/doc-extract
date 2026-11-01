import re

import pytest
from PIL import Image

from docextract import config, demo
from docextract.pipelines import EngineUnavailable
from docextract.schema import ExtractionResult, ReceiptFields

IMG = Image.new("RGB", (60, 90), "white")


class Fake:
    def __init__(self, method, fail=None):
        self.method, self.fail = method, fail

    def extract(self, image, doc_id=None):
        if self.fail:
            raise self.fail
        return ExtractionResult(doc_id=doc_id, method=self.method, latency_ms=2500.0 if self.method == "llm" else 400.0,
                                fields=ReceiptFields(company="SHOP", date="2019-03-12", total="9.50"))


@pytest.fixture(autouse=True)
def fake_pipelines(monkeypatch):
    fails = {"llm": EngineUnavailable("Ollama est eteint")}
    monkeypatch.setattr(demo, "get_pipeline", lambda method, dataset, model: Fake(method, fails.get(method)))
    return fails


def test_run_method_returns_result_or_a_message_never_raises(fake_pipelines):
    ok = demo.run_method("docai", IMG, doc_id="r1")
    assert ok.result.fields.company == "SHOP" and ok.message is None and ok.label == "Donut (Document AI)"
    down = demo.run_method("llm", IMG)
    assert down.result is None and "Ollama" in down.message
    fake_pipelines["rules"] = ValueError("boom")
    crash = demo.run_method("rules", IMG)
    assert crash.result is None and "ValueError" in crash.message


def test_formats_follow_french_conventions():
    import datetime as dt
    from decimal import Decimal
    assert demo.format_date(dt.date(2019, 3, 12)) == "12 mars 2019" and demo.format_date(dt.date(2018, 8, 1)) == "1 août 2018"
    assert demo.format_amount(Decimal("9.5")) == "9,50" and demo.format_amount(Decimal("91000")) == "91\u00a0000,00"
    assert demo.format_duration(400.0) == "400 ms" and demo.format_duration(2540.0) == "2,5 s"


def test_ticket_shows_found_values_highlighted_and_missing_ones_plainly():
    ok = demo.run_method("docai", IMG, "cord")
    ticket = demo.ticket_html(ok)
    assert '<b><span class="hit">SHOP</span></b>' in ticket and '<b><span class="hit">12 mars 2019</span></b>' in ticket
    assert '<b><span class="hit">9,50</span></b>' in ticket and "lu en 400 ms" in ticket
    assert '<i class="none">non trouvé</i>' in ticket                              # taxe absente : pas de surlignage
    assert "modèle CORD" in ticket and "fresh" not in ticket and 'class="ticket fresh"' in demo.ticket_html(ok, fresh=True)


def test_ticket_of_an_unavailable_method_explains_and_stamps():
    ticket = demo.ticket_html(demo.run_method("llm", IMG, model="qwen2.5:7b"))
    assert "Cette méthode n'a pas répondu." in ticket and "Ollama est eteint" in ticket
    assert "hors service" in ticket and "qwen2.5:7b" in ticket and 'class="hit"' not in ticket


def test_ticket_escapes_text_coming_from_the_model():
    class Evil(Fake):
        def extract(self, image, doc_id=None):
            return ExtractionResult(doc_id=doc_id, method="rules", latency_ms=1.0, error="<script>x</script>",
                                    fields=ReceiptFields(company="A&B <b>SDN</b>"))
    import docextract.demo as d
    d.get_pipeline = lambda method, dataset, model: Evil("rules")           # fixture autouse restaure ensuite
    ticket = demo.ticket_html(demo.run_method("rules", IMG))
    assert "A&amp;B &lt;b&gt;SDN&lt;/b&gt;" in ticket and "<script>" not in ticket and "&lt;script&gt;" in ticket


def test_photo_is_embedded_and_named_safely():
    out = demo.photo_html(Image.new("RGB", (2000, 3000), "white"), 'x" onerror="y')
    assert out.startswith('<div class="photo"><img src="data:image/jpeg;base64,') and 'onerror="y' not in out.split("alt=")[1][1:]


def test_pipelines_are_built_once_per_configuration(monkeypatch):
    monkeypatch.undo()
    demo._PIPELINES.clear()
    a = demo.get_pipeline("docai", "sroie", "m1")
    assert demo.get_pipeline("docai", "sroie", "m2") is a                          # le modele LLM n'influe pas sur Donut
    assert demo.get_pipeline("docai", "cord", "m1") is not a
    assert demo.get_pipeline("llm", "sroie", "m1") is not demo.get_pipeline("llm", "sroie", "m2")
    demo._PIPELINES.clear()


def test_benchmark_markdown(tmp_project):
    assert demo.benchmark_markdown("eval") is None
    out = config.OUTPUTS_DIR / "benchmark"
    out.mkdir(parents=True)
    (out / "benchmark_eval.md").write_text("# Tableau", encoding="utf-8")
    assert demo.benchmark_markdown("eval") == "# Tableau"


# ------------------------------------------------------------ l'interface elle-meme
def test_streamlit_app_extracts_an_uploaded_image(monkeypatch):
    import io

    import streamlit as st
    from streamlit.testing.v1 import AppTest

    buf = io.BytesIO()
    IMG.save(buf, format="PNG")
    buf.seek(0)
    buf.name = "ticket_7.png"
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: buf)
    at = AppTest.from_file(str(config.ROOT / "app" / "streamlit_app.py"), default_timeout=30).run()
    assert not at.exception
    assert any('class="photo"' in m.value for m in at.markdown)                       # la photo est posee sur le bureau
    at.button[0].click().run()                                                          # methode par defaut : les trois
    assert not at.exception
    tickets = [m.value for m in at.markdown if 'class="tickets"' in m.value][0]
    assert len(re.findall(r'class="ticket( fresh)?"', tickets)) == 3 and "SHOP" in tickets and "hors service" in tickets
    at.radio[1].set_value("docai").run()                                                # changer de methode garde les tickets
    assert any('class="tickets"' in m.value for m in at.markdown)
    at.button[0].click().run()
    tickets = [m.value for m in at.markdown if 'class="tickets"' in m.value][0]
    assert len(re.findall(r'class="ticket( fresh)?"', tickets)) == 1 and "Donut" in tickets


def test_streamlit_app_without_image_invites_to_drop_one():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(config.ROOT / "app" / "streamlit_app.py"), default_timeout=30).run()
    assert not at.exception and at.button[0].disabled
    assert any("Déposez une image" in m.value for m in at.markdown)


def test_stylesheet_has_the_pieces_the_markup_uses():
    css = (config.ROOT / "app" / "style.css").read_text(encoding="utf-8")
    for cls in (".ticket", ".t-paper", ".t-row b .hit", ".stamp", ".photo", ".tickets", ".ticket.fresh"):
        assert cls in css
    assert "prefers-reduced-motion" in css
