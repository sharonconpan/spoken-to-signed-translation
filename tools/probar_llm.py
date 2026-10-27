"""Compara reglas y Ollama en cinco frases, incluyendo una relativa."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gramatica_lsp import GramaticaLSP  # noqa: E402
from llm_lsp import (
    LLMConfig,
    LLMGlosser,
    build_gloss_aliases,
    build_system_prompt,
)  # noqa: E402
from spoken_to_signed.gloss_to_pose import CSVPoseLookup  # noqa: E402

PHRASES = [
    "Hola tú vives aquí",
    "Yo como pizza",
    "Mañana voy al colegio",
    "La manzana la come Juan",
    "La mujer que vive aquí es mi tía",
]


def main() -> None:
    lexicon_dir = ROOT / "assets" / "lexicon_lsp"
    index_path = ROOT / "avatar" / "data" / "lexicon" / "index.json"
    rules_path = ROOT / "reglas_lsp.yaml"
    examples_path = ROOT / "tests" / "ejemplos_lsp.yaml"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    lookup = CSVPoseLookup(str(lexicon_dir))
    grammar = GramaticaLSP(
        reglas_path=rules_path,
        lookup=lookup,
        indice_lexico=index,
    )
    glosses = {clip["gloss"].lower() for clip in index["clips"].values()}
    config = LLMConfig.load(ROOT / "config_lsp.yaml")
    service = LLMGlosser(
        system_prompt=build_system_prompt(
            rules_path=rules_path,
            examples_path=examples_path,
            available_glosses=glosses,
        ),
        available_glosses=glosses,
        gloss_aliases=build_gloss_aliases(lookup, index),
        config=config,
    )
    grammar.texto_a_glosas("hola tu vivir", use_cache=False)
    warmup = service.warmup_async()
    warmup.join(timeout=config.request_timeout_seconds + config.connect_timeout_seconds)

    print("| Frase | Glosas reglas | Glosas LLM | Primera glosa (ms) | Total LLM (ms) |")
    print("|---|---|---|---:|---:|")
    for phrase in PHRASES:
        rules_start = time.perf_counter()
        rules_result = grammar.texto_a_glosas(phrase, use_cache=False)
        rules_elapsed = (time.perf_counter() - rules_start) * 1000
        llm_result = service.generate(phrase)
        if llm_result.first_gloss_ms is not None:
            first_gloss = f"{llm_result.first_gloss_ms:.1f}"
        elif llm_result.timed_out:
            first_gloss = "timeout"
        else:
            first_gloss = "sin glosa válida"
        print(
            f"| {phrase} | {rules_result.glosas} ({rules_elapsed:.1f} ms) "
            f"| {llm_result.glosas} | {first_gloss} | {llm_result.total_ms:.1f} |"
        )
        if llm_result.error:
            print(f"  Ollama: {llm_result.error}; descartadas={llm_result.descartada_llm}")


if __name__ == "__main__":
    main()
