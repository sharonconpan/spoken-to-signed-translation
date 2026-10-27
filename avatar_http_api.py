import json
import math
from http.server import SimpleHTTPRequestHandler
from queue import Empty, Queue
from typing import Any, Callable


VALID_MODES = {"solo_reglas", "solo_si_hace_falta", "siempre"}


def create_avatar_request_handler(
    *,
    avatar_dir: str,
    index: dict[str, Any],
    pending_glosses: Queue,
    process_text: Callable[[str, bool, float | None, str], dict[str, Any]],
    get_mode: Callable[[], str],
    set_mode: Callable[[str], None],
    get_ollama_status: Callable[[], str],
    refresh_ollama_status: Callable[[], None],
) -> type[SimpleHTTPRequestHandler]:
    by_gloss: dict[str, str] = {}
    for key, clip in index["clips"].items():
        gloss = clip["gloss"].strip().upper()
        if gloss in by_gloss and by_gloss[gloss] != key:
            raise ValueError(f"La glosa {gloss!r} apunta a más de un clip del cache")
        by_gloss[gloss] = key

    class AvatarRequestHandler(SimpleHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=avatar_dir, **kwargs)

        def end_headers(self):
            path = self.path.split("?", 1)[0]
            if path in {
                "/",
                "/index.html",
                "/say",
                "/say_text",
                "/config",
                "/ollama_status",
                "/next",
            }:
                self.send_header("Cache-Control", "no-store")
            elif path == "/avatar.vrm" or path.endswith(".js"):
                self.send_header("Cache-Control", "public, max-age=86400")
            else:
                self.send_header("Cache-Control", "public, max-age=300")
            super().end_headers()

        def send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def read_json_body(self) -> dict[str, Any] | None:
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_json(400, {"error": "Content-Length inválido"})
                return None
            if content_length <= 0 or content_length > 65536:
                self.send_json(413, {"error": "El cuerpo debe tener entre 1 y 65536 bytes"})
                return None
            try:
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.send_json(400, {"error": "Se esperaba un cuerpo JSON UTF-8 válido"})
                return None
            if not isinstance(payload, dict):
                self.send_json(400, {"error": "El cuerpo JSON debe ser un objeto"})
                return None
            return payload

        def do_POST(self):
            request_path = self.path.split("?", 1)[0]
            if request_path not in {"/say", "/say_text", "/config"}:
                self.send_json(404, {"error": "Endpoint no encontrado"})
                return
            payload = self.read_json_body()
            if payload is None:
                return

            if request_path == "/config":
                mode = payload.get("mode")
                if not isinstance(mode, str) or mode not in VALID_MODES:
                    self.send_json(
                        400,
                        {
                            "error": "Modo inválido. Use solo_reglas, "
                            "solo_si_hace_falta o siempre.",
                            "mode": get_mode(),
                        },
                    )
                    return
                set_mode(mode)
                refresh_ollama_status()
                self.send_json(200, {"mode": mode})
                return

            if request_path == "/say_text":
                text = payload.get("text")
                final = payload.get("final")
                enqueued_at_ms = payload.get("enqueued_at_ms")
                if (
                    not isinstance(text, str)
                    or len(text) > 2000
                    or not isinstance(final, bool)
                ):
                    self.send_json(
                        400,
                        {"error": 'Se esperaba {"text": string, "final": boolean}'},
                    )
                    return
                if enqueued_at_ms is not None and (
                    isinstance(enqueued_at_ms, bool)
                    or not isinstance(enqueued_at_ms, (int, float))
                    or not math.isfinite(enqueued_at_ms)
                ):
                    self.send_json(400, {"error": "enqueued_at_ms debe ser un número finito"})
                    return
                try:
                    result = process_text(text, final, enqueued_at_ms, get_mode())
                except ValueError as error:
                    self.send_json(400, {"error": str(error)})
                    return
                except Exception as error:
                    print(f"Error en /say_text: {error}")
                    self.send_json(500, {"error": f"No se pudo procesar el texto: {error}"})
                    return
                response = {
                    **result,
                    "text": text,
                    "texto_recibido": text,
                    "glosas_encoladas": result.get("glosses", []),
                    "descartadas_llm": result.get(
                        "descartadas_llm",
                        result.get("descartada_llm", []),
                    ),
                    "camino": result.get("camino", "reglas"),
                    "sin_sena": result.get("sin_sena", []),
                    "reglas_aplicadas": result.get("reglas_aplicadas", []),
                    "timing_ms": result.get("timing_ms", {}),
                }
                status = 422 if result.get("not_found") else 202
                if status == 422:
                    response["error"] = "Ninguna palabra está en el léxico."
                self.send_json(status, response)
                return

            glosses = payload.get("glosses")
            if not isinstance(glosses, list) or len(glosses) > 256 or any(
                not isinstance(gloss, str) or not gloss.strip() or len(gloss) > 64
                for gloss in glosses
            ):
                self.send_json(400, {"error": "Lista de glosas inválida"})
                return
            enqueued_at_ms = payload.get("enqueued_at_ms")
            if (
                isinstance(enqueued_at_ms, bool)
                or not isinstance(enqueued_at_ms, (int, float))
                or not math.isfinite(enqueued_at_ms)
            ):
                self.send_json(400, {"error": "Falta enqueued_at_ms numérico"})
                return

            queue_items = []
            for gloss in glosses:
                normalized_gloss = gloss.strip().upper()
                clip_key = by_gloss.get(normalized_gloss)
                if clip_key is None:
                    self.send_json(404, {"error": f"Glosa no encontrada: {normalized_gloss}"})
                    return
                clip = index["clips"][clip_key]
                queue_items.append(
                    {
                        "key": clip_key,
                        "gloss": clip["gloss"],
                        "file": f"data/lexicon/{clip['file']}",
                        "enqueued_at_ms": enqueued_at_ms,
                    }
                )
            for queue_item in queue_items:
                pending_glosses.put(queue_item)
            self.send_json(
                202,
                {"queued": len(queue_items), "glosses": [item["gloss"] for item in queue_items]},
            )

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/next":
                try:
                    queue_item = pending_glosses.get_nowait()
                except Empty:
                    self.send_response(204)
                    self.end_headers()
                    return
                self.send_json(200, queue_item)
                return
            if path == "/config":
                self.send_json(200, {"mode": get_mode()})
                return
            if path == "/ollama_status":
                self.send_json(200, {"status": get_ollama_status()})
                return
            super().do_GET()

    return AvatarRequestHandler
