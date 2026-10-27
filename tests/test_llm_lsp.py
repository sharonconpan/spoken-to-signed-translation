import json
from types import SimpleNamespace
from pathlib import Path

from gramatica_lsp import GramaticaLSP
from llm_lsp import (
    LLMConfig,
    LLMGlosser,
    build_system_prompt,
    build_gloss_aliases,
    choose_llm_or_rules,
    should_use_llm,
)
from spoken_to_signed.gloss_to_pose import CSVPoseLookup

ROOT = Path(__file__).resolve().parents[1]


class FakeStreamingClient:
    def __init__(self, chunks=None, *, wait_before_end=0, error=None):
        self.chunks = chunks or []
        self.wait_before_end = wait_before_end
        self.error = error

    def stream_chat(self, *, cancellation, system_prompt, text, config):
        assert system_prompt and text and config
        for chunk in self.chunks:
            if cancellation.is_set():
                return
            yield chunk
        if self.wait_before_end:
            if cancellation.wait(self.wait_before_end):
                return
        if self.error:
            raise self.error

    def warmup(self, config):
        assert config
        return None


def _config(**overrides):
    values = {
        "first_gloss_timeout_seconds": 0.2,
        "request_timeout_seconds": 1.0,
        "connect_timeout_seconds": 0.1,
    }
    values.update(overrides)
    return LLMConfig(**values)


def _service(client, *, glosses=("hola", "tu", "vivir"), config=None):
    return LLMGlosser(
        system_prompt="Prompt de prueba",
        available_glosses=set(glosses),
        config=config or _config(),
        client=client,
    )


def test_stream_parser_validates_and_emits_glosses_incrementally():
    service = _service(
        FakeStreamingClient(["ho", "la ", "inventada ", "tu ", "viv", "ir"])
    )
    emitted = []

    result = service.generate("hola tú vivir", on_gloss=emitted.append)

    assert result.glosas == ["hola", "tu", "vivir"]
    assert result.descartada_llm == ["inventada"]
    assert emitted == result.glosas
    assert result.first_gloss_ms is not None


def test_prompt_uses_active_rules_examples_and_closed_lexicon():
    index_path = ROOT / "avatar" / "data" / "lexicon" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    available = {clip["gloss"].lower() for clip in index["clips"].values()}
    prompt = build_system_prompt(
        rules_path=ROOT / "reglas_lsp.yaml",
        examples_path=ROOT / "tests" / "ejemplos_lsp.yaml",
        available_glosses=available,
    )

    assert "orden_constituyentes" in prompt
    assert "evidencia" in prompt
    assert "hola tu vivir" not in prompt
    assert "La mujer que vive aquí es mi tía" not in prompt
    assert "¿Vives en Lima?" not in prompt
    assert "porfavor" in prompt
    assert "responde SOLO" in prompt
    assert "Ante cualquier duda, OMITE" in prompt
    assert "PROHIBIDO agregar glosas" in prompt


def test_prompt_can_build_previous_and_conservative_versions():
    index = json.loads(
        (ROOT / "avatar" / "data" / "lexicon" / "index.json").read_text(encoding="utf-8")
    )
    glosses = {clip["gloss"].lower() for clip in index["clips"].values()}
    before = build_system_prompt(
        rules_path=ROOT / "reglas_lsp.yaml",
        examples_path=ROOT / "tests" / "ejemplos_lsp.yaml",
        available_glosses=glosses,
        conservative=False,
    )
    after = build_system_prompt(
        rules_path=ROOT / "reglas_lsp.yaml",
        examples_path=ROOT / "tests" / "ejemplos_lsp.yaml",
        available_glosses=glosses,
        conservative=True,
    )

    assert "Ante cualquier duda, OMITE" not in before
    assert "Ante cualquier duda, OMITE" in after
    assert "PROHIBIDO agregar glosas" in after


def test_streaming_deduplicates_only_consecutive_glosses():
    service = LLMGlosser(
        system_prompt="test",
        available_glosses={"tu", "me", "vivir"},
        gloss_aliases={"tu": ["tu"], "me": ["me"], "vivir": ["vivir", "vivo"]},
        config=_config(),
        client=FakeStreamingClient(["tu ", "tu ", "me ", "tu ", "vivir ", "vivir"]),
    )
    emitted = []

    result = service.generate("tú me tú vivir vivo", on_gloss=emitted.append)

    assert result.glosas == ["tu", "me", "tu", "vivir"]
    assert emitted == result.glosas


def test_timeout_without_first_gloss_returns_rules_without_emitting():
    client = FakeStreamingClient(wait_before_end=0.5)
    service = _service(
        client,
        config=_config(first_gloss_timeout_seconds=0.03, request_timeout_seconds=0.2),
    )
    emitted = []

    result = service.generate("hola tu vivir", on_gloss=emitted.append)
    selected, route = choose_llm_or_rules(result, ["hola", "tu", "vivir"])

    assert result.timed_out
    assert not emitted
    assert selected == ["hola", "tu", "vivir"]
    assert route == "reglas"


def test_partial_stream_timeout_never_appends_rule_glosses():
    service = _service(
        FakeStreamingClient(["hola ", "tu "], wait_before_end=0.5),
        config=_config(first_gloss_timeout_seconds=0.2, request_timeout_seconds=0.03),
    )
    emitted = []

    result = service.generate("hola tu vivir", on_gloss=emitted.append)
    selected, route = choose_llm_or_rules(result, ["hola", "tu", "vivir"])

    assert result.glosas == ["hola", "tu"]
    assert emitted == ["hola", "tu"]
    assert selected == emitted
    assert route == "llm"


def test_connection_error_falls_back_to_rules():
    service = _service(FakeStreamingClient(error=ConnectionError("offline")))
    result = service.generate("hola tu vivir")
    selected, route = choose_llm_or_rules(result, ["hola", "tu", "vivir"])

    assert result.error == "offline"
    assert selected == ["hola", "tu", "vivir"]
    assert route == "reglas"


def test_mode_only_uses_llm_when_configured_or_unresolved():
    grammar = GramaticaLSP(reglas_path=ROOT / "reglas_lsp.yaml")
    simple = grammar.texto_a_glosas("hola tu vivir", use_cache=False)
    unresolved = grammar.texto_a_glosas(
        "La mujer que vive aquí es mi tía",
        use_cache=False,
    )

    assert not should_use_llm("solo_reglas", "hola tu vivir", simple)
    assert not should_use_llm("solo_si_hace_falta", "hola tu vivir", simple)
    assert should_use_llm("solo_si_hace_falta", "frase compleja", unresolved)
    assert should_use_llm(
        "solo_si_hace_falta",
        "frase con palabras desconocidas",
        SimpleNamespace(estado=None, glosas=[]),
    )
    assert should_use_llm("siempre", "hola tu vivir", simple)


def test_cached_output_is_replayed_to_stream_callback():
    service = _service(FakeStreamingClient(["hola ", "tu "]))
    first = service.generate("hola tu")
    replayed = []

    second = service.generate("hola tu", on_gloss=replayed.append)

    assert second.from_cache
    assert second.glosas == first.glosas == ["hola", "tu"]
    assert replayed == ["hola", "tu"]


def test_unrelated_lexicon_gloss_is_discarded_before_callback():
    alias_map = {"tu": ["tu", "te"], "vivir": ["vivo", "vives", "vivir"]}
    service = LLMGlosser(
        system_prompt="test",
        available_glosses={"tu", "vivir"},
        gloss_aliases=alias_map,
        config=_config(),
        client=FakeStreamingClient(["tu ", "vivir "]),
    )
    emitted = []

    result = service.generate("Tú estás bien", on_gloss=emitted.append)

    assert result.glosas == ["tu"]
    assert result.descartada_sin_relacion == ["vivir"]
    assert emitted == ["tu"]


def test_lexicon_aliases_build_from_lookup():
    lookup = CSVPoseLookup(str(ROOT / "assets" / "lexicon_lsp"))
    index = json.loads(
        (ROOT / "avatar" / "data" / "lexicon" / "index.json").read_text(encoding="utf-8")
    )

    aliases = build_gloss_aliases(lookup, index)

    assert "donde" in aliases["donde"]
    assert "vives" in aliases["vivir"]
    assert "favor" in aliases["porfavor"]


def test_config_defaults_to_rules_only():
    config = LLMConfig.load(ROOT / "config_lsp.yaml")

    assert config.mode == "solo_reglas"
    assert config.seed == 0
