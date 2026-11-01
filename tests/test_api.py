import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from docextract import api, config
from docextract.ocr import OcrResult, Segment
from docextract.pipelines import DonutPipeline, EngineUnavailable, LLMPipeline, RulesPipeline
from docextract.schema import ExtractionResult, ReceiptFields


def png(w=60, h=90) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, format="PNG")
    return buf.getvalue()


def fake_ocr(path, engine):
    lines = ["GOLDEN MART SDN BHD", "Date: 12/03/2019", "TOTAL 23.50", "GST 6% 1.33"]
    return OcrResult(engine, [Segment(t, 10, 20 * i, 200, 20 * i + 14) for i, t in enumerate(lines)], ms=12.0)


class Fake:
    def __init__(self, method, dataset="sroie", fail=None):
        self.method, self.dataset, self.fail, self.calls = method, dataset, fail, 0

    def extract(self, image, doc_id=None):
        self.calls += 1
        if self.fail:
            raise self.fail
        return ExtractionResult(doc_id=doc_id, method=self.method, latency_ms=5.0,
                                fields=ReceiptFields(company=f"{self.method}-{self.dataset}", total="9.00"))


@pytest.fixture
def made():
    return []


@pytest.fixture
def client(made):
    def factory(method):
        def build(dataset, model):
            f = Fake(method, dataset)
            made.append((method, dataset, model, f))
            return f
        return build
    app = api.create_app({m: factory(m) for m in ("rules", "llm", "docai")}, max_bytes=2000)
    return TestClient(app)


def post(client, data=None, content=None, name="ticket_001.png"):
    return client.post("/extract", files={"file": (name, content if content is not None else png(), "image/png")},
                       data=data or {})


def test_default_method_is_donut_and_response_is_the_common_schema(client):
    r = post(client)
    assert r.status_code == 200
    body = r.json()
    assert body["method"] == "docai" and body["doc_id"] == "ticket_001"
    assert body["fields"]["company"] == "docai-sroie" and body["fields"]["total"] == "9.00"
    assert set(body["fields"]) == {"company", "date", "total", "tax"} and body["latency_ms"] == 5.0


def test_each_method_and_dataset_gets_its_own_pipeline_loaded_once(client, made):
    for data in ({"method": "rules"}, {"method": "rules"}, {"method": "llm", "llm_model": "qwen2.5:7b"},
                 {"method": "docai", "dataset": "cord"}, {"method": "docai", "dataset": "cord"}, {"method": "docai"}):
        assert post(client, data).status_code == 200
    assert [(m, d, mod) for m, d, mod, _ in made] == [("rules", "sroie", config.LLM_MODEL),
                                                        ("llm", "sroie", "qwen2.5:7b"),
                                                        ("docai", "cord", config.LLM_MODEL),
                                                        ("docai", "sroie", config.LLM_MODEL)]
    assert made[0][3].calls == 2 and made[2][3].calls == 2
    assert client.get("/health").json() == {"status": "ok", "methods": ["rules", "llm", "docai"],
                                            "loaded": ["docai", "llm", "rules"]}


def test_health_before_any_request(client):
    assert client.get("/health").json()["loaded"] == []


@pytest.mark.parametrize("data", [{"method": "ocr"}, {"dataset": "funsd"}])
def test_unknown_method_or_dataset_is_422(client, data):
    assert post(client, data).status_code == 422


def test_bad_uploads(client):
    assert post(client, content=b"").status_code == 400
    assert post(client, content=b"ceci n'est pas une image").status_code == 400
    assert post(client, content=b"x" * 2001).status_code == 413
    assert client.post("/extract").status_code == 422                                  # pas de fichier


def test_unavailable_engine_is_503_with_message():
    app = api.create_app({"docai": lambda d, m: Fake("docai", fail=EngineUnavailable("Ollama est eteint"))})
    r = TestClient(app).post("/extract", files={"file": ("a.png", png(), "image/png")})
    assert r.status_code == 503 and "Ollama est eteint" in r.json()["detail"]


# ------------------------------------------------------------ vrais pipelines, OCR et LLM simules
def test_rules_pipeline_runs_ocr_then_rules():
    out = RulesPipeline(engine="fake", ocr=fake_ocr).extract(Image.new("RGB", (60, 90)), "d1")
    assert out.method == "rules" and out.doc_id == "d1"
    assert str(out.fields.date) == "2019-03-12" and str(out.fields.total) == "23.50"
    assert out.latency_ms >= 12.0                                    # l'OCR compte dans le temps


def test_llm_pipeline_verifies_against_ocr_and_counts_ocr_time():
    def post_(url, payload, timeout):
        reply = '{"company": "GOLDEN MART SDN BHD", "date": "12/03/2019", "total": "99.99", "tax": "1.33"}'
        return {"message": {"content": reply}, "eval_count": 5}
    out = LLMPipeline(model="m", engine="fake", ocr=fake_ocr, post=post_).extract(Image.new("RGB", (60, 90)), "d2")
    assert out.method == "llm" and out.fields.company == "GOLDEN MART SDN BHD"
    assert out.fields.total is None and str(out.fields.tax) == "1.33"          # 99.99 absent de l'OCR : rejete
    assert out.latency_ms >= 12.0


def test_llm_pipeline_ollama_down_is_unavailable():
    def down(url, payload, timeout):
        from docextract.llm import OllamaError
        raise OllamaError("Impossible de joindre Ollama")
    with pytest.raises(EngineUnavailable, match="Ollama"):
        LLMPipeline(engine="fake", ocr=fake_ocr, post=down).extract(Image.new("RGB", (60, 90)))


def test_missing_ocr_engine_is_unavailable():
    def broken(path, engine):
        raise RuntimeError("RapidOCR manquant")
    with pytest.raises(EngineUnavailable, match="RapidOCR"):
        RulesPipeline(engine="fake", ocr=broken).extract(Image.new("RGB", (60, 90)))


def test_donut_pipeline_missing_model_and_lazy_loading(tmp_path):
    with pytest.raises(EngineUnavailable, match="sroie_best"):
        DonutPipeline("sroie", source=str(tmp_path / "absent")).extract(Image.new("RGB", (60, 90)))
    loads = []

    def loader(source, dataset):
        loads.append((source, dataset))
        model = Fake("docai", dataset)
        model.warm_up = lambda: loads.append("warm")
        return model
    pipe = DonutPipeline("cord", source="hub/cord", loader=loader)
    assert loads == []                                               # rien n'est charge a la creation
    pipe.extract(Image.new("RGB", (60, 90)))
    pipe.extract(Image.new("RGB", (60, 90)))
    assert loads == [("hub/cord", "cord"), "warm"]               # charge et chauffe une seule fois


def test_donut_without_torch_is_unavailable(tmp_path):
    def loader(source, dataset):
        raise ImportError("No module named 'torch'")
    with pytest.raises(EngineUnavailable, match=r"requirements-donut.*|ImportError: No module named .torch."):
        DonutPipeline("cord", source="hub/cord", loader=loader).extract(Image.new("RGB", (60, 90)))


def test_try_api_script_against_a_live_server(tmp_path):
    import threading
    import time

    import uvicorn
    from conftest import load_script
    server = uvicorn.Server(uvicorn.Config(api.create_app({"docai": lambda d, m: Fake("docai", d)}),
                                           host="127.0.0.1", port=8765, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    try:
        image = tmp_path / "recu.png"
        image.write_bytes(png())
        script = load_script("11_try_api")
        status, body = script.post_image("http://127.0.0.1:8765", image, {"method": "docai", "dataset": "cord"})
        assert status == 200 and body["fields"]["company"] == "docai-cord" and body["doc_id"] == "recu"
        assert script.main([str(image), "--url", "http://127.0.0.1:8765"]) == 0
        status, body = script.post_image("http://127.0.0.1:8765", image, {"method": "rules"})
        assert status == 503 and "non configuree" in body["detail"]
    finally:
        server.should_exit = True
        thread.join(5)
    assert script.main([str(image), "--url", "http://127.0.0.1:8765"]) == 1             # serveur arrete


def test_donut_import_error_message_keeps_the_real_cause():
    def loader(source, dataset):
        raise ImportError("DLL load failed while importing _C")
    with pytest.raises(EngineUnavailable) as err:
        DonutPipeline("cord", source="hub/cord", loader=loader).extract(Image.new("RGB", (60, 90)))
    assert "ImportError: DLL load failed while importing _C" in str(err.value)
