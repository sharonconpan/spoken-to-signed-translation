import json
import os
import unicodedata
from pathlib import Path

import pytest
import yaml

from gramatica_lsp import GramaticaLSP, _Yaml12SafeLoader
from spoken_to_signed.gloss_to_pose import CSVPoseLookup

ROOT = Path(__file__).resolve().parents[1]
RULES_PATH = ROOT / "reglas_lsp.yaml"
EXAMPLES_PATH = ROOT / "tests" / "ejemplos_lsp.yaml"
LEXICON_PATH = ROOT / "assets" / "lexicon_lsp"
AVATAR_INDEX_PATH = ROOT / "avatar" / "data" / "lexicon" / "index.json"


def _normalizar_glosa(glosa):
    normalized = unicodedata.normalize("NFD", glosa.lower())
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


@pytest.fixture(scope="module")
def grammar_engine():
    lookup = CSVPoseLookup(str(LEXICON_PATH))
    avatar_index = json.loads(AVATAR_INDEX_PATH.read_text(encoding="utf-8"))
    return GramaticaLSP(
        reglas_path=RULES_PATH,
        lookup=lookup,
        indice_lexico=avatar_index,
    )


def _load_examples():
    with EXAMPLES_PATH.open(encoding="utf-8") as examples_file:
        return yaml.load(examples_file, Loader=_Yaml12SafeLoader)["ejemplos"]


def test_provisional_examples_and_print_comparison_table(grammar_engine):
    rows = []
    failures = []
    expected_missing = []
    result_missing = []

    for example in _load_examples():
        result = grammar_engine.texto_a_glosas(
            example["es"],
            sin_filtro_lexico=True,
            use_cache=False,
        )
        if result.sin_sena:
            result_missing.append((example["es"], result.sin_sena))
        expected = example.get("glosas")
        state = example.get("estado")
        if state == "EN_REVISION":
            rows.append((example["es"], result.glosas, expected, "EN_REVISION"))
            continue
        if expected is None or state == "SIN_RESOLVER":
            rows.append((example["es"], result.glosas, expected, "SIN_COMPARAR"))
            continue

        normalized_expected = [_normalizar_glosa(glosa) for glosa in expected]
        outcome = "OK" if result.glosas == normalized_expected else "FALLA"
        rows.append((example["es"], result.glosas, normalized_expected, outcome))
        if outcome == "FALLA":
            failures.append(example["id"])

        available_glosses = grammar_engine._glosas_disponibles
        missing_expected = [
            gloss for gloss in normalized_expected if gloss not in available_glosses
        ]
        if missing_expected:
            expected_missing.append((example["id"], missing_expected))

    print("| Ejemplo | Resultado | Esperado | Estado |")
    print("|---|---|---|---|")
    for sentence, result, expected, outcome in rows:
        print(f"| {sentence} | {result} | {expected} | {outcome} |")
    print(f"Sin seña reportadas por el motor: {result_missing}")
    print(f"Glosas esperadas sin seña en el léxico: {expected_missing}")

    assert not failures, f"Fallaron ejemplos provisionales: {failures}"


def test_filtered_results_match_lexicon_and_report_missing_glosses(grammar_engine):
    expected = ["yo", "pizza", "comer"]
    unfiltered = grammar_engine.texto_a_glosas(
        "Yo como pizza",
        sin_filtro_lexico=True,
        use_cache=False,
    )
    filtered = grammar_engine.texto_a_glosas("Yo como pizza", use_cache=False)
    available = grammar_engine._glosas_disponibles

    assert unfiltered.glosas == expected
    assert filtered.glosas == [gloss for gloss in expected if gloss in available]
    assert filtered.sin_sena == [
        gloss for gloss in expected if gloss not in available
    ]


def test_hola_tu_vivir_remains_lexicon_compatible(grammar_engine):
    result = grammar_engine.texto_a_glosas("hola tu vivir", use_cache=False)

    assert result.glosas == ["hola", "tu", "vivir"]


def test_hola_is_not_misclassified_as_a_clitic_verb(grammar_engine):
    result = grammar_engine.texto_a_glosas("Hola", use_cache=False)

    assert result.glosas == ["hola"]
    assert result.sin_sena == []
    assert "sin_resolver" not in result.reglas_aplicadas


def test_only_consecutive_duplicate_glosses_are_removed(grammar_engine):
    repeated = grammar_engine.texto_a_glosas(
        "Yo vivo donde tú vives",
        sin_filtro_lexico=True,
        use_cache=False,
    )
    nonconsecutive = grammar_engine.texto_a_glosas(
        "Yo tú yo",
        sin_filtro_lexico=True,
        use_cache=False,
    )

    assert repeated.glosas == ["yo", "donde", "tu", "vivir"]
    assert nonconsecutive.glosas == ["yo", "tu", "yo"]


def test_rules_reload_and_lru_results_are_isolated(grammar_engine, tmp_path):
    rules_path = tmp_path / "reglas_lsp.yaml"
    rules_path.write_bytes(RULES_PATH.read_bytes())
    engine = GramaticaLSP(
        reglas_path=rules_path,
        lookup=grammar_engine.lookup,
        indice_lexico=grammar_engine.indice_lexico,
        cache_size=2,
    )

    initial = engine.texto_a_glosas("Yo como pizza", sin_filtro_lexico=True)
    initial.glosas.clear()
    cached = engine.texto_a_glosas("Yo como pizza", sin_filtro_lexico=True)
    assert cached.glosas == ["yo", "pizza", "comer"]

    rules = rules_path.read_text(encoding="utf-8")
    rules_path.write_text(rules.replace("valor: SOV", "valor: SVO"), encoding="utf-8")
    old_mtime_ns = rules_path.stat().st_mtime_ns
    os.utime(rules_path, ns=(old_mtime_ns + 2_000_000_000, old_mtime_ns + 2_000_000_000))

    reloaded = engine.texto_a_glosas(
        "Yo como pizza",
        sin_filtro_lexico=True,
        use_cache=False,
    )
    assert reloaded.glosas == ["yo", "comer", "pizza"]
