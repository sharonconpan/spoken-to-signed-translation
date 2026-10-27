"""Streaming Ollama integration for closed-vocabulary LSP gloss generation."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Iterator

import requests
import yaml

LOGGER = logging.getLogger(__name__)
_WARNING_LOCK = threading.Lock()
_WARNED_OLLAMA_UNAVAILABLE = False

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config_lsp.yaml"
VALID_MODES = {"solo_reglas", "solo_si_hace_falta", "siempre"}


@dataclass(frozen=True)
class LLMConfig:
    mode: str = "solo_reglas"
    model: str = "qwen2.5:3b"
    base_url: str = "http://127.0.0.1:11434"
    first_gloss_timeout_seconds: float = 1.5
    request_timeout_seconds: float = 30.0
    connect_timeout_seconds: float = 3.0
    temperature: float = 0.0
    seed: int = 0
    num_predict: int = 48
    num_ctx: int = 2048
    keep_alive: str = "30m"
    cache_size: int = 512

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> LLMConfig:
        with Path(path).open(encoding="utf-8") as config_file:
            raw = yaml.safe_load(config_file) or {}
        if not isinstance(raw, dict):
            raise ValueError("config_lsp.yaml debe contener un objeto YAML.")

        parameters = raw.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError("La sección parameters debe ser un objeto YAML.")

        config = cls(
            mode=str(raw.get("mode", cls.mode)),
            model=str(raw.get("model", cls.model)),
            base_url=str(raw.get("base_url", cls.base_url)).rstrip("/"),
            first_gloss_timeout_seconds=float(
                raw.get("first_gloss_timeout_seconds", cls.first_gloss_timeout_seconds)
            ),
            request_timeout_seconds=float(
                raw.get("request_timeout_seconds", cls.request_timeout_seconds)
            ),
            connect_timeout_seconds=float(
                raw.get("connect_timeout_seconds", cls.connect_timeout_seconds)
            ),
            temperature=float(parameters.get("temperature", cls.temperature)),
            seed=int(parameters.get("seed", cls.seed)),
            num_predict=int(parameters.get("num_predict", cls.num_predict)),
            num_ctx=int(parameters.get("num_ctx", cls.num_ctx)),
            keep_alive=str(raw.get("keep_alive", cls.keep_alive)),
            cache_size=int(raw.get("cache_size", cls.cache_size)),
        )
        if config.mode not in VALID_MODES:
            raise ValueError(f"Modo LLM inválido: {config.mode!r}")
        if min(
            config.first_gloss_timeout_seconds,
            config.request_timeout_seconds,
            config.connect_timeout_seconds,
        ) <= 0:
            raise ValueError("Los timeouts del LLM deben ser positivos.")
        if config.num_predict <= 0 or config.num_ctx <= 0 or config.cache_size < 0:
            raise ValueError("num_predict y num_ctx deben ser positivos; cache_size no negativo.")
        return config


@dataclass
class LLMResult:
    glosas: list[str] = field(default_factory=list)
    descartada_llm: list[str] = field(default_factory=list)
    descartada_sin_relacion: list[str] = field(default_factory=list)
    first_gloss_ms: float | None = None
    total_ms: float = 0.0
    timed_out: bool = False
    error: str | None = None
    from_cache: bool = False


class _CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[], None]] = []

    def is_set(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float) -> bool:
        return self._event.wait(timeout)

    def add_callback(self, callback: Callable[[], None]) -> None:
        with self._lock:
            if self._event.is_set():
                callback()
            else:
                self._callbacks.append(callback)

    def cancel(self) -> None:
        self._event.set()
        with self._lock:
            callbacks, self._callbacks = self._callbacks, []
        for callback in callbacks:
            try:
                callback()
            except requests.RequestException:
                LOGGER.debug("Error al cerrar una petición Ollama cancelada.", exc_info=True)


class OllamaHTTPClient:
    """Small streaming client using Ollama's local HTTP API."""

    def stream_chat(
        self,
        *,
        system_prompt: str,
        text: str,
        config: LLMConfig,
        cancellation: _CancellationToken,
    ) -> Iterator[str]:
        response = requests.post(
            f"{config.base_url}/api/chat",
            json={
                "model": config.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                "stream": True,
                "keep_alive": config.keep_alive,
                "options": {
                    "temperature": config.temperature,
                    "seed": config.seed,
                    "num_predict": config.num_predict,
                    "num_ctx": config.num_ctx,
                },
            },
            stream=True,
            timeout=(
                min(
                    config.connect_timeout_seconds,
                    config.first_gloss_timeout_seconds,
                ),
                min(
                    config.first_gloss_timeout_seconds,
                    config.request_timeout_seconds,
                ),
            ),
        )
        cancellation.add_callback(response.close)
        try:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True, chunk_size=1):
                if cancellation.is_set():
                    return
                if not line:
                    continue
                payload = json.loads(line)
                content = payload.get("message", {}).get("content", "")
                if content:
                    yield content
                if payload.get("done"):
                    return
        finally:
            response.close()

    def warmup(self, config: LLMConfig) -> None:
        response = requests.post(
            f"{config.base_url}/api/chat",
            json={
                "model": config.model,
                "messages": [{"role": "user", "content": "Responde OK."}],
                "stream": False,
                "keep_alive": config.keep_alive,
                "options": {
                    "temperature": 0,
                    "seed": config.seed,
                    "num_predict": 1,
                    "num_ctx": config.num_ctx,
                },
            },
            timeout=(config.connect_timeout_seconds, config.request_timeout_seconds),
        )
        try:
            response.raise_for_status()
        finally:
            response.close()


def _normalizar(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())
    without_accents = "".join(
        character for character in decomposed
        if unicodedata.category(character) != "Mn"
    )
    without_punctuation = "".join(
        " " if unicodedata.category(character).startswith(("P", "S")) else character
        for character in without_accents
    )
    return re.sub(r"\s+", " ", without_punctuation).strip()


def _warn_ollama_once(error: str) -> None:
    global _WARNED_OLLAMA_UNAVAILABLE
    with _WARNING_LOCK:
        if _WARNED_OLLAMA_UNAVAILABLE:
            return
        _WARNED_OLLAMA_UNAVAILABLE = True
    LOGGER.warning("Ollama no responde; se usarán las reglas LSP: %s", error)


def _active_rule_lines(rules: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    activation_keys = {
        "activo",
        "usar_infinitivo",
        "usar_deixis",
        "expresion_facial_obligatoria",
        "tratar_con_llm_o_revision_humana",
    }

    def visit(node: Any, path: tuple[str, ...]) -> None:
        if not isinstance(node, dict):
            return
        is_active = any(node.get(key) is True for key in activation_keys)
        if is_active:
            evidence = node.get("evidencia", "no especificada")
            settings = {
                key: value
                for key, value in node.items()
                if key not in {"nota", "evidencia"}
            }
            lines.append(
                f"- {'.'.join(path)} (evidencia: {evidence}): "
                f"{json.dumps(settings, ensure_ascii=False, sort_keys=True)}"
            )
        for key, value in node.items():
            if isinstance(value, dict):
                visit(value, (*path, str(key)))

    for key, value in rules.items():
        visit(value, (str(key),))
    return lines


def build_system_prompt(
    *,
    rules_path: str | Path,
    examples_path: str | Path,
    available_glosses: set[str] | list[str],
    conservative: bool = True,
) -> str:
    with Path(rules_path).open(encoding="utf-8") as rules_file:
        rules = yaml.safe_load(rules_file) or {}
    with Path(examples_path).open(encoding="utf-8") as examples_file:
        examples_data = yaml.safe_load(examples_file) or {}
    if not isinstance(rules, dict) or not isinstance(examples_data, dict):
        raise ValueError("Los archivos de reglas y ejemplos deben contener objetos YAML.")

    examples = [
        example for example in examples_data.get("ejemplos", [])
        if isinstance(example, dict)
        and example.get("estado") not in {"EN_REVISION", "SIN_RESOLVER"}
        and isinstance(example.get("glosas"), list)
    ][:12]
    example_lines = [
        f"- {example['es']} -> {' '.join(map(str, example['glosas']))} "
        f"(estado: {example.get('estado', 'PROVISIONAL')})"
        for example in examples
    ]
    if len(example_lines) < 8:
        LOGGER.warning("El prompt LSP tiene solo %d ejemplos elegibles.", len(example_lines))

    glosses = sorted({_normalizar(str(gloss)) for gloss in available_glosses if str(gloss).strip()})
    if not glosses:
        raise ValueError("El índice léxico no contiene glosas disponibles.")
    instructions = [
        "Instrucción central: responde SOLO con glosas exactas de esa lista, separadas "
        "por espacios, en el orden final y sin explicaciones. Si una palabra no tiene "
        "glosa disponible, omítela. No inventes glosas.",
    ]
    if conservative:
        instructions.insert(
            0,
            "Ante cualquier duda, OMITE la glosa en vez de rellenar o adivinar. "
            "PROHIBIDO agregar glosas que no correspondan a palabras o alias explícitos "
            "del texto original. No añadas sujetos, lugares, acciones ni conceptos "
            "implícitos que no aparezcan en el texto.",
        )
    return "\n".join(
        [
            "Eres un generador de glosas para Lengua de Señas Peruana (LSP).",
            "Las reglas y ejemplos son provisionales; respétalos como configuración, "
            "sin afirmar que sean gramática validada.",
            "Reglas activas con evidencia:",
            *_active_rule_lines(rules),
            "Ejemplos provisionales:",
            *example_lines,
            "Vocabulario cerrado de glosas disponibles:",
            " ".join(glosses),
            *instructions,
        ]
    )


def build_gloss_aliases(
    lookup: Any,
    avatar_index: dict[str, Any],
    *,
    spoken_language: str = "es",
    signed_language: str = "prl",
) -> dict[str, list[str]]:
    by_path = avatar_index.get("by_path", {})
    clips = avatar_index.get("clips", {})
    aliases_by_gloss: dict[str, set[str]] = {}
    for dictionary_name in ("words_index", "glosses_index"):
        dictionary = getattr(lookup, dictionary_name, {})
        signed_index = dictionary.get(spoken_language, {}).get(signed_language, {})
        for alias, rows in signed_index.items():
            if not rows:
                continue
            best_row = lookup.get_best_row(rows, alias)
            clip_key = by_path.get(best_row.get("path"))
            clip = clips.get(clip_key, {})
            canonical = clip.get("gloss")
            if isinstance(canonical, str) and canonical.strip():
                normalized_gloss = _normalizar(canonical)
                aliases_by_gloss.setdefault(normalized_gloss, set()).add(_normalizar(alias))
    return {gloss: sorted(aliases) for gloss, aliases in aliases_by_gloss.items()}


class LLMGlosser:
    """Streams validated glosses from Ollama and caches complete responses."""

    def __init__(
        self,
        *,
        system_prompt: str,
        available_glosses: set[str] | list[str],
        gloss_aliases: dict[str, list[str]] | None = None,
        config: LLMConfig | None = None,
        client: Any = None,
        enforce_text_relation: bool = True,
    ) -> None:
        self.system_prompt = system_prompt
        self.config = config or LLMConfig()
        self.client = client or OllamaHTTPClient()
        self.available_glosses = {
            _normalizar(str(gloss)) for gloss in available_glosses
        }
        self.gloss_aliases = {
            _normalizar(gloss): {_normalizar(alias) for alias in aliases}
            for gloss, aliases in (gloss_aliases or {}).items()
        }
        self.enforce_text_relation = enforce_text_relation
        self._cache: OrderedDict[str, LLMResult] = OrderedDict()
        self._cache_lock = threading.Lock()

    def warmup_async(self) -> threading.Thread:
        def warmup() -> None:
            try:
                self.client.warmup(self.config)
            except Exception as error:  # noqa: BLE001 - Ollama is optional
                _warn_ollama_once(str(error))

        thread = threading.Thread(target=warmup, name="ollama-lsp-warmup", daemon=True)
        thread.start()
        return thread

    def generate(
        self,
        text: str,
        *,
        on_gloss: Callable[[str], None] | None = None,
        use_cache: bool = True,
    ) -> LLMResult:
        started = time.perf_counter()
        normalized_text = _normalizar(text)
        if use_cache and self.config.cache_size:
            with self._cache_lock:
                cached = self._cache.get(normalized_text)
                if cached is not None:
                    self._cache.move_to_end(normalized_text)
                    result = self._copy_result(cached)
                    result.from_cache = True
                else:
                    result = None
            if result is not None:
                result.total_ms = (time.perf_counter() - started) * 1000
                result.first_gloss_ms = 0.0 if result.glosas else None
                if result.glosas and on_gloss:
                    for gloss in result.glosas:
                        if gloss not in result.descartada_sin_relacion:
                            on_gloss(gloss)
                return result

        events: Queue[tuple[str, str | None]] = Queue()
        cancellation = _CancellationToken()

        def worker() -> None:
            buffer = ""
            try:
                for chunk in self.client.stream_chat(
                    system_prompt=self.system_prompt,
                    text=text,
                    config=self.config,
                    cancellation=cancellation,
                ):
                    if cancellation.is_set():
                        break
                    buffer += chunk
                    while True:
                        separator = re.search(r"\s", buffer)
                        if separator is None:
                            break
                        candidate = buffer[:separator.start()]
                        buffer = buffer[separator.end():]
                        if candidate:
                            events.put(("token", candidate))
                if buffer and not cancellation.is_set():
                    events.put(("token", buffer))
                if not cancellation.is_set():
                    events.put(("done", None))
            except Exception as error:  # noqa: BLE001 - returned for rules fallback
                if not cancellation.is_set():
                    events.put(("error", str(error)))

        worker_thread = threading.Thread(target=worker, name="ollama-lsp-stream", daemon=True)
        worker_thread.start()
        result = LLMResult()
        last_emitted_gloss: str | None = None
        first_deadline = started + self.config.first_gloss_timeout_seconds
        total_deadline = started + self.config.request_timeout_seconds
        completed = False

        while not completed:
            deadline = total_deadline if result.first_gloss_ms is not None else first_deadline
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                cancellation.cancel()
                result.timed_out = result.first_gloss_ms is None
                result.error = (
                    "Timeout esperando la primera glosa."
                    if result.timed_out
                    else "Timeout durante el streaming de glosas."
                )
                break
            try:
                event, value = events.get(timeout=remaining)
            except Empty:
                cancellation.cancel()
                result.timed_out = result.first_gloss_ms is None
                result.error = (
                    "Timeout esperando la primera glosa."
                    if result.timed_out
                    else "Timeout durante el streaming de glosas."
                )
                break

            if event == "done":
                completed = True
                continue
            if event == "error":
                result.error = value
                if not result.glosas:
                    _warn_ollama_once(value or "error desconocido")
                completed = True
                continue
            if event != "token" or value is None:
                continue

            gloss = self._normalize_gloss_token(value)
            if gloss is None:
                continue
            if gloss not in self.available_glosses:
                result.descartada_llm.append(gloss)
                continue
            if self.enforce_text_relation and not self._related_to_text(gloss, normalized_text):
                result.descartada_sin_relacion.append(gloss)
                continue
            if gloss == last_emitted_gloss:
                continue

            if result.first_gloss_ms is None:
                result.first_gloss_ms = (time.perf_counter() - started) * 1000
            result.glosas.append(gloss)
            last_emitted_gloss = gloss
            if on_gloss:
                on_gloss(gloss)

        result.total_ms = (time.perf_counter() - started) * 1000
        if (
            use_cache
            and completed
            and not result.error
            and not result.timed_out
            and self.config.cache_size
        ):
            with self._cache_lock:
                self._cache[normalized_text] = self._copy_result(result)
                self._cache.move_to_end(normalized_text)
                while len(self._cache) > self.config.cache_size:
                    self._cache.popitem(last=False)
        if result.error:
            _warn_ollama_once(result.error)
        return result

    @staticmethod
    def _normalize_gloss_token(token: str) -> str | None:
        normalized = _normalizar(token).strip()
        if not normalized or len(normalized.split()) != 1:
            return None
        return normalized

    def _related_to_text(self, gloss: str, normalized_text: str) -> bool:
        text_words = set(normalized_text.split())
        if gloss in text_words:
            return True
        return any(
            self._contains_phrase(normalized_text, alias)
            for alias in self.gloss_aliases.get(gloss, set())
        )

    @staticmethod
    def _contains_phrase(text: str, phrase: str) -> bool:
        return f" {phrase} " in f" {text} "

    @staticmethod
    def _copy_result(result: LLMResult) -> LLMResult:
        return LLMResult(
            glosas=list(result.glosas),
            descartada_llm=list(result.descartada_llm),
            descartada_sin_relacion=list(result.descartada_sin_relacion),
            first_gloss_ms=result.first_gloss_ms,
            total_ms=result.total_ms,
            timed_out=result.timed_out,
            error=result.error,
            from_cache=result.from_cache,
        )


def should_use_llm(mode: str, text: str, rules_result: Any) -> bool:
    if mode == "solo_reglas":
        return False
    if mode == "siempre":
        return True
    if mode != "solo_si_hace_falta":
        raise ValueError(f"Modo LLM desconocido: {mode!r}")
    return bool(text.strip()) and (
        getattr(rules_result, "estado", None) == "SIN_RESOLVER"
        or not getattr(rules_result, "glosas", [])
    )


def choose_llm_or_rules(llm_result: LLMResult, rule_glosses: list[str]) -> tuple[list[str], str]:
    """Use streamed LLM output if any was queued; otherwise fall back to rules."""
    if llm_result.glosas:
        return list(llm_result.glosas), "llm"
    return list(rule_glosses), "reglas"


_WARNED_EXPORT = False


def warn_ollama_unavailable_once(error: str) -> None:
    """Expose one-time startup warning behavior to the Streamlit integration."""
    global _WARNED_EXPORT
    if _WARNED_EXPORT:
        return
    _WARNED_EXPORT = True
    _warn_ollama_once(error)
