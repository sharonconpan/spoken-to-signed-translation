import json
import threading
from http.server import ThreadingHTTPServer
from queue import Queue
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from avatar_http_api import create_avatar_request_handler


@pytest.fixture
def avatar_api():
    mode = {"value": "solo_reglas"}
    status = {"value": "disponible"}
    queue = Queue()
    refresh_count = []

    def process_text(text, final, enqueued_at_ms, selected_mode):
        assert selected_mode == mode["value"]
        return {
            "queued": 1,
            "glosses": ["hola"],
            "sin_sena": ["mundo"],
            "reglas_aplicadas": ["deixis"],
            "camino": selected_mode,
            "descartada_llm": ["inventada"],
            "descartada_sin_relacion": [],
            "timing_ms": {"total_ms": 2.4},
        }

    handler = create_avatar_request_handler(
        avatar_dir=".",
        index={
            "clips": {
                "hola": {"gloss": "HOLA", "file": "hola.json"},
            }
        },
        pending_glosses=queue,
        process_text=process_text,
        get_mode=lambda: mode["value"],
        set_mode=lambda value: mode.update(value=value),
        get_ollama_status=lambda: status["value"],
        refresh_ollama_status=lambda: refresh_count.append(True),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", mode, queue, refresh_count
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def _request(base_url, path, payload=None):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{base_url}{path}",
        data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method="POST" if body is not None else "GET",
    )
    try:
        with urlopen(request, timeout=2) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def test_say_text_response_includes_translation_metadata(avatar_api):
    base_url, _, _, _ = avatar_api

    status, payload = _request(
        base_url,
        "/say_text",
        {"text": "Hola mundo", "final": True, "enqueued_at_ms": 123.0},
    )

    assert status == 202
    assert payload["text"] == "Hola mundo"
    assert payload["texto_recibido"] == "Hola mundo"
    assert payload["queued"] == 1
    assert payload["glosses"] == ["hola"]
    assert payload["glosas_encoladas"] == ["hola"]
    assert payload["sin_sena"] == ["mundo"]
    assert payload["reglas_aplicadas"] == ["deixis"]
    assert payload["camino"] == "solo_reglas"
    assert payload["descartadas_llm"] == ["inventada"]
    assert payload["timing_ms"] == {"total_ms": 2.4}


def test_mode_endpoint_reads_and_changes_mode_without_restarting(avatar_api):
    base_url, mode, _, refresh_count = avatar_api

    assert _request(base_url, "/config") == (200, {"mode": "solo_reglas"})
    assert _request(base_url, "/config", {"mode": "solo_si_hace_falta"}) == (
        200,
        {"mode": "solo_si_hace_falta"},
    )
    assert _request(base_url, "/config") == (200, {"mode": "solo_si_hace_falta"})
    status, payload = _request(
        base_url,
        "/say_text",
        {"text": "Hola", "final": True},
    )
    assert status == 202
    assert payload["camino"] == "solo_si_hace_falta"
    assert mode["value"] == "solo_si_hace_falta"
    assert len(refresh_count) == 1


@pytest.mark.parametrize("mode", ["other", "", None, 1, [], {}])
def test_mode_endpoint_rejects_invalid_modes(avatar_api, mode):
    base_url, current_mode, _, _ = avatar_api

    status, payload = _request(base_url, "/config", {"mode": mode})

    assert status == 400
    assert payload["mode"] == "solo_reglas"
    assert current_mode["value"] == "solo_reglas"


def test_ollama_status_endpoint_is_nonblocking_status_snapshot(avatar_api):
    base_url, _, _, _ = avatar_api

    assert _request(base_url, "/ollama_status") == (200, {"status": "disponible"})
