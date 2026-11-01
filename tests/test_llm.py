"""Approche 2 (OCR + LLM) : client Ollama, prompt, validation Pydantic, cache, script.

Un faux serveur HTTP joue le role d'Ollama : aucun modele n'est necessaire pour lancer ces tests.
"""
import csv
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from conftest import fake_cord, fake_sroie, load_script

from docextract import llm, ocr
from docextract.ocr import Line, OcrResult, Segment


def L(*texts):
    return [Line(t, 0, 30 * i, 100, 30 * i + 20) for i, t in enumerate(texts)]


GOOD = {"company": "SHOP 1 SDN BHD", "date": "25/12/2018", "total": "9.00", "tax": None}


class FakeOllama:
    """Serveur /api/tags + /api/chat. `replies` : contenus renvoyes tour a tour (le dernier se repete)."""

    def __init__(self, replies=None, models=("qwen2.5:3b",)):
        self.replies = list(replies or [json.dumps(GOOD)])
        self.models = list(models)
        self.chat_calls: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/api/ps":
                    return self._send(200, {"models": [{"name": m, "size": 2_200_000_000, "size_vram": 2_200_000_000}
                                                       for m in outer.models]})
                self._send(200, {"models": [{"name": m} for m in outer.models]})

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.chat_calls.append(payload)
                if payload["model"] not in outer.models:
                    return self._send(404, {"error": f"model '{payload['model']}' not found"})
                i = min(len(outer.chat_calls) - 1, len(outer.replies) - 1)
                self._send(200, {"message": {"role": "assistant", "content": outer.replies[i]},
                                 "prompt_eval_count": 500, "eval_count": 40,
                                 "eval_duration": 2_000_000_000, "load_duration": 1_000_000})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def ollama():
    servers = []

    def make(**kw):
        servers.append(FakeOllama(**kw))
        return servers[-1]

    yield make
    for s in servers:
        s.close()


# ------------------------------------------------------------------ sortie attendue (Pydantic)
def test_output_accepts_numbers_and_empty_values():
    out = llm.LLMOutput(company=" GOLDEN   MART ", date=None, total=10.7, tax="null", extra="ignore me")
    assert out.company == "GOLDEN MART" and out.total == "10.7" and out.tax is None and out.date is None
    assert llm.LLMOutput(total=8000.0).total == "8000"
    assert llm.LLMOutput(total=45500).total == "45500"
    assert llm.LLMOutput(company="N/A", date="", total="-").model_dump() == \
        {"company": None, "date": None, "total": None, "tax": None}


def test_output_converts_to_normalized_fields():
    f = llm.LLMOutput(company="SHOP", date="12/03/2018", total="RM 10.70", tax="6,600").to_fields()
    assert str(f.date) == "2018-03-12" and str(f.total) == "10.70" and str(f.tax) == "6600.00"
    assert llm.LLMOutput(date="pas une date", total="beaucoup").to_fields().date is None


def test_schema_requires_the_four_keys():
    schema = llm.output_schema()
    assert schema["required"] == ["company", "date", "total", "tax"] and set(schema["properties"]) == set(llm.FIELDS)


def test_parse_reply_tolerates_text_around_json():
    assert llm.parse_reply('Voici : {"company": "A", "total": "1.00"} voila').total == "1.00"
    with pytest.raises(ValueError):
        llm.parse_reply("pas de json ici")
    with pytest.raises(ValueError):
        llm.parse_reply('{"company": ')


# ------------------------------------------------------------------ prompt
def test_format_receipt_keeps_head_and_tail_when_too_long():
    lines = L("TOP COMPANY", *[f"ITEM {i} 1.00" for i in range(500)], "TOTAL 99.00")
    text = llm.format_receipt(lines, max_chars=300)
    assert len(text) < 330 and text.startswith("TOP COMPANY") and text.endswith("TOTAL 99.00") and "[...]" in text
    assert llm.format_receipt(L("A", "  ", "B")) == "A\nB"


def test_messages_have_system_examples_then_receipt():
    msgs = llm.build_messages("SHOP\nTOTAL 5.00")
    assert [m["role"] for m in msgs] == ["system"] + ["user", "assistant"] * len(llm.FEW_SHOT) + ["user"]
    assert len(llm.FEW_SHOT) == 3 and msgs[-1]["content"] == "SHOP\nTOTAL 5.00"
    for m in msgs[2:-1:2]:
        assert set(json.loads(m["content"])) == set(llm.FIELDS)          # exemples = JSON valide
    assert [m["role"] for m in llm.build_messages("x", few_shot=False)] == ["system", "user"]
    assert "rounded" in msgs[0]["content"] and "null" in msgs[0]["content"]
    assert "thousands" in msgs[0]["content"] and "cash minus change" in msgs[0]["content"]


# ------------------------------------------------------------------ client
def test_extract_llm_sends_schema_and_parses(ollama):
    srv = ollama()
    run = llm.extract_llm(L("SHOP 1 SDN BHD", "TOTAL 9.00"), model="qwen2.5:3b", base_url=srv.url)
    assert run.output.total == "9.00" and run.attempts == 1 and run.error is None
    assert run.meta["output_tokens"] == 40 and run.meta["eval_ms"] == 2000.0
    sent = srv.chat_calls[0]
    assert sent["format"] == llm.output_schema() and sent["stream"] is False
    assert sent["options"]["temperature"] == 0 and sent["messages"][-1]["content"] == "SHOP 1 SDN BHD\nTOTAL 9.00"
    r = llm.to_result("doc1", run)
    assert r.method == "llm" and r.doc_id == "doc1" and str(r.fields.total) == "9.00" and r.error is None


def test_extract_llm_json_mode(ollama):
    srv = ollama()
    llm.extract_llm(L("A"), model="qwen2.5:3b", base_url=srv.url, use_schema=False)
    assert srv.chat_calls[0]["format"] == "json"


def test_invalid_reply_is_retried_then_accepted(ollama):
    srv = ollama(replies=["desole, voici du texte", json.dumps(GOOD)])
    run = llm.extract_llm(L("A"), model="qwen2.5:3b", base_url=srv.url)
    assert run.output is not None and run.attempts == 2 and len(srv.chat_calls) == 2


def test_invalid_replies_end_as_recorded_error_not_exception(ollama):
    srv = ollama(replies=["rien de valide"])
    run = llm.extract_llm(L("A"), model="qwen2.5:3b", base_url=srv.url, retries=1)
    assert run.output is None and run.attempts == 2 and run.error.startswith("reponse invalide")
    r = llm.to_result("d", run)
    assert r.fields.total is None and r.error


def test_unreachable_ollama_has_actionable_message():
    with pytest.raises(llm.OllamaError, match="Lance l'application Ollama"):
        llm.extract_llm(L("A"), model="m", base_url="http://127.0.0.1:9", timeout=2)


def test_missing_model_is_an_ollama_error(ollama):
    srv = ollama(models=("autre:1b",))
    with pytest.raises(llm.OllamaError, match="404"):
        llm.extract_llm(L("A"), model="qwen2.5:3b", base_url=srv.url)


def test_installed_models_and_has_model(ollama):
    srv = ollama(models=("qwen2.5:3b", "llama3.2:latest"))
    names = llm.installed_models(srv.url)
    assert llm.has_model("qwen2.5:3b", names) and llm.has_model("llama3.2", names) and not llm.has_model("qwen2.5:7b", names)


def test_few_shot_examples_are_consistent_with_the_rules_of_the_prompt():
    years = {json.loads(json.dumps(a))["date"][-4:] for _, a in llm.FEW_SHOT}
    assert len(years) == len(llm.FEW_SHOT)                       # une annee par exemple : aucune annee favorisee
    for text, answer in llm.FEW_SHOT:
        assert answer["total"] in text and answer["tax"] in text   # la reponse est copiee, pas inventee


def test_warm_up_sends_a_tiny_request(ollama):
    srv = ollama()
    assert llm.warm_up("qwen2.5:3b", srv.url) >= 0
    assert srv.chat_calls[0]["options"]["num_predict"] == 1 and srv.chat_calls[0]["keep_alive"] == "30m"


# ------------------------------------------------------------------ cache
def test_cache_avoids_second_call_and_force_bypasses(tmp_project, ollama):
    srv = ollama()
    kw = dict(model="qwen2.5:3b", ocr_engine="rapidocr", base_url=srv.url)
    run, cached = llm.cached_llm("d1", L("A"), **kw)
    assert not cached and len(srv.chat_calls) == 1
    run2, cached2 = llm.cached_llm("d1", L("A"), **kw)
    assert cached2 and len(srv.chat_calls) == 1 and run2.output == run.output and run2.ms == run.ms
    llm.cached_llm("d1", L("A"), force=True, **kw)
    assert len(srv.chat_calls) == 2
    llm.cached_llm("d1", L("A"), few_shot=False, **kw)           # autre variante de prompt = autre cache
    assert len(srv.chat_calls) == 3
    assert (tmp_project / "data" / "llm_cache" / "qwen2.5_3b" / f"{llm.PROMPT_VERSION}" / "rapidocr" / "d1.json").exists()


# ------------------------------------------------------------------ script
@pytest.fixture
def project(tmp_project, monkeypatch):
    prepare = load_script("00_prepare_data")
    monkeypatch.setattr(prepare, "load_split",
                        lambda repo, split, limit: fake_sroie(130) if "SROIE" in repo else fake_cord(4))
    assert prepare.main([]) == 0

    def fake_engine(path, **params):
        return OcrResult("fake", [Segment("SHOP SDN BHD", 0, 0, 90, 10), Segment("TOTAL", 0, 30, 40, 40),
                                  Segment("9.00", 80, 31, 110, 41)], ms=7.0, width=100, height=300)

    monkeypatch.setitem(ocr.ENGINES, "fake", fake_engine)
    assert load_script("03_run_ocr").main(["--engine", "fake", "--set", "dev"]) == 0
    return tmp_project


def run_script(args, srv):
    return load_script("06_eval_llm").main([*args, "--url", srv.url, "--model", "qwen2.5:3b", "--engine", "fake"])


def test_script_scores_dev_and_writes_outputs(project, ollama, capsys):
    srv = ollama(replies=[json.dumps({"company": "SHOP 1 SDN BHD", "date": "25/12/2018", "total": "9.00", "tax": "5,800"})])
    assert run_script(["--set", "dev", "--limit", "6"], srv) == 0
    out = capsys.readouterr().out
    assert "=== sroie (train) | LLM qwen2.5:3b" in out and "=== cord (validation)" in out
    assert "| date | 100.0%" in out                                    # la date de tous les faux recus est 25/12/2018
    assert "Temps LLM par recu" in out and "20.0 jetons/s" in out and "Reponses invalides : 0" in out
    assert len(srv.chat_calls) == 1 + 6 + 4                            # echauffement + 6 SROIE (limite) + 4 CORD
    assert "Modele charge en memoire" in out
    d = project / "outputs" / "llm"
    stem = "dev_qwen2.5_3b_fake_sroie"
    summary = json.loads((d / f"{stem}.json").read_text(encoding="utf-8"))
    assert summary["model"] == "qwen2.5:3b" and summary["prompt"] == llm.PROMPT_VERSION and summary["temps"]["reponses_invalides"] == 0
    rows = list(csv.DictReader((d / f"{stem}_erreurs.csv").open(encoding="utf-8")))
    assert rows and {"verite", "prediction", "reponse_llm", "debut_ocr"} <= set(rows[0])
    text = (d / f"explication_{stem}.txt").read_text(encoding="utf-8")
    assert "reponse LLM:" in text and "lignes OCR :" in text


def test_script_second_run_uses_cache(project, ollama, capsys):
    srv = ollama()
    run_script(["--set", "dev", "--limit", "3"], srv)
    calls = len(srv.chat_calls)
    capsys.readouterr()
    assert run_script(["--set", "dev", "--limit", "3"], srv) == 0
    assert len(srv.chat_calls) == calls + 1 and "depuis le cache : 3/3" in capsys.readouterr().out


def test_script_eval_needs_final(project, ollama, capsys):
    srv = ollama()
    assert run_script(["--set", "eval"], srv) == 3
    assert "reserve au score final" in capsys.readouterr().out and not srv.chat_calls


def test_script_without_ollama(project, capsys):
    assert load_script("06_eval_llm").main(["--engine", "fake", "--url", "http://127.0.0.1:9"]) == 2
    out = capsys.readouterr().out
    assert "Impossible de joindre Ollama" in out and "winget install --id Ollama.Ollama" in out


def test_script_missing_model(project, ollama, capsys):
    srv = ollama(models=("llama3.2:latest",))
    assert run_script(["--set", "dev"], srv) == 2
    assert "ollama pull qwen2.5:3b" in capsys.readouterr().out and not srv.chat_calls


def test_script_without_ocr_cache(project, ollama, capsys):
    srv = ollama()
    code = load_script("06_eval_llm").main(["--engine", "tesseract", "--url", srv.url, "--model", "qwen2.5:3b"])
    assert code == 1 and "aucun OCR en cache pour tesseract" in capsys.readouterr().out


def test_script_with_perfect_ocr_engine(project, ollama, capsys):
    srv = ollama()
    assert load_script("06_eval_llm").main(["--engine", "gt", "--set", "dev", "--limit", "2", "--url", srv.url,
                                            "--model", "qwen2.5:3b"]) == 0
    assert "OCR gt" in capsys.readouterr().out


def test_script_verify_uses_cached_answers_and_reports_checks(project, ollama, capsys):
    reply = json.dumps({"company": "INVENTE SDN BHD", "date": "25/12/2018", "total": "9.00", "tax": "5,800"})
    srv = ollama(replies=[reply])
    run_script(["--set", "dev", "--limit", "3"], srv)
    calls = len(srv.chat_calls)
    capsys.readouterr()
    assert run_script(["--set", "dev", "--limit", "3", "--verify"], srv) == 0
    out = capsys.readouterr().out
    assert len(srv.chat_calls) == calls + 1                      # seul l'echauffement : reponses prises dans le cache
    assert "Verification : gardes" in out and "rejetes [" in out
    assert (project / "outputs" / "llm" / "dev_qwen2.5_3b_fake_verif_sroie.json").exists()


def test_to_result_with_lines_applies_the_verification():
    from docextract.ocr import Line
    run = llm.LLMRun(output=llm.LLMOutput(company="Darepes", total="23.00"), raw="{}", ms=1.0, attempts=1)
    lines = [Line("TOTAL 23.000", 0, 0, 1, 1)]
    from decimal import Decimal
    assert llm.to_result("d", run).fields.total == Decimal("23.00")
    checked = llm.to_result("d", run, lines).fields
    assert checked.total == Decimal("23000.00") and checked.company is None


def test_loaded_model_info_reads_ollama_ps(ollama):
    srv = ollama()
    assert llm.loaded_model_info("qwen2.5:3b", srv.url) == {"size_gb": 2.2, "vram_gb": 2.2, "gpu_share": 1.0}
    assert llm.loaded_model_info("autre:7b", srv.url) is None
    assert llm.loaded_model_info("qwen2.5:3b", "http://127.0.0.1:9") is None       # Ollama injoignable : pas d'erreur


def test_script_records_ollama_memory(project, ollama, capsys):
    srv = ollama()
    assert run_script(["--set", "dev", "--limit", "2"], srv) == 0
    assert "Memoire Ollama : 2.2 Go dont 2.2 Go en VRAM (100% sur GPU)" in capsys.readouterr().out
    summary = json.loads((project / "outputs" / "llm" / "dev_qwen2.5_3b_fake_sroie.json").read_text(encoding="utf-8"))
    assert summary["memoire_ollama"]["vram_gb"] == 2.2


def test_repeat_limit_is_an_invalid_reply_not_an_outage():
    def looping(url, payload, timeout):
        raise llm.OllamaError('Ollama a repondu 500 : {"error":"prediction aborted, token repeat limit reached"}')

    run = llm.extract_llm([], post=looping)
    assert run.output is None and "boucle" in run.error and run.attempts == 1
    assert llm.to_result("x", run).error == run.error

    def down(url, payload, timeout):
        raise llm.OllamaError("Ollama a repondu 500 : autre panne")

    with pytest.raises(llm.OllamaError):
        llm.extract_llm([], post=down)
