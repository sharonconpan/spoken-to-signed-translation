"""Audits LLM outputs against Spanish lexical aliases, before and after prompt hardening."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gramatica_lsp import GramaticaLSP  # noqa: E402
from llm_lsp import (  # noqa: E402
    LLMConfig,
    LLMGlosser,
    build_gloss_aliases,
    build_system_prompt,
    _normalizar,
)
from spoken_to_signed.gloss_to_pose import CSVPoseLookup  # noqa: E402

PHRASES = [
    ("simple", "Hola"),
    ("simple", "Yo estoy bien"),
    ("simple", "Tú estás bien"),
    ("simple", "Mucho gusto"),
    ("simple", "Yo te llamo"),
    ("orden distinto", "Me llamas tú"),
    ("simple", "Hasta luego"),
    ("relativa", "Yo vivo donde tú vives"),
]
REPETITIONS = 3


def _service(prompt, glosses, aliases, config, *, enforce_text_relation):
    return LLMGlosser(
        system_prompt=prompt,
        available_glosses=glosses,
        gloss_aliases=aliases,
        config=config,
        enforce_text_relation=enforce_text_relation,
    )


def _covered_glosses(text, aliases_by_gloss):
    normalized_text = _normalizar(text)
    return {
        gloss
        for gloss, aliases in aliases_by_gloss.items()
        if any(
            f" {alias} " in f" {normalized_text} "
            for alias in aliases
        )
    }


def _result_summary(result):
    return str(result.glosas)


def main() -> None:
    lexicon_dir = ROOT / "assets" / "lexicon_lsp"
    index_path = ROOT / "avatar" / "data" / "lexicon" / "index.json"
    rules_path = ROOT / "reglas_lsp.yaml"
    examples_path = ROOT / "tests" / "ejemplos_lsp.yaml"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    lookup = CSVPoseLookup(str(lexicon_dir))
    glosses = {clip["gloss"].lower() for clip in index["clips"].values()}
    aliases = build_gloss_aliases(lookup, index)
    config = LLMConfig.load(ROOT / "config_lsp.yaml")
    grammar = GramaticaLSP(
        reglas_path=rules_path,
        lookup=lookup,
        indice_lexico=index,
    )

    print("VOCABULARIO CERRADO (glosa -> palabras/alias)")
    for gloss in sorted(glosses):
        print(f"- {gloss}: {', '.join(aliases.get(gloss, [])) or '(sin alias)'}")

    prompt_before = build_system_prompt(
        rules_path=rules_path,
        examples_path=examples_path,
        available_glosses=glosses,
        conservative=False,
    )
    prompt_after = build_system_prompt(
        rules_path=rules_path,
        examples_path=examples_path,
        available_glosses=glosses,
        conservative=True,
    )
    service_previous_prompt = _service(
        prompt_before,
        glosses,
        aliases,
        config,
        enforce_text_relation=True,
    )
    service_current_prompt = _service(
        prompt_after,
        glosses,
        aliases,
        config,
        enforce_text_relation=True,
    )
    warmup = service_current_prompt.warmup_async()
    warmup.join(timeout=config.request_timeout_seconds + config.connect_timeout_seconds)

    print(
        "\n| Frase | Reglas | LLM (prompt anterior + filtro) "
        "| LLM (prompt actual) | Diferencias |"
    )
    print("|---|---|---|---|---|")
    for _, phrase in PHRASES:
        rules_result = grammar.texto_a_glosas(phrase, use_cache=False)
        previous_prompt_results = [
            service_previous_prompt.generate(phrase, use_cache=False)
            for _ in range(REPETITIONS)
        ]
        current_prompt_results = [
            service_current_prompt.generate(phrase, use_cache=False)
            for _ in range(REPETITIONS)
        ]
        previous_text = " / ".join(
            _result_summary(result) for result in previous_prompt_results
        )
        current_text = " / ".join(
            _result_summary(result) for result in current_prompt_results
        )
        differences = [
            f"rep{run}: {previous.glosas} -> {current.glosas}"
            for run, (previous, current) in enumerate(
                zip(previous_prompt_results, current_prompt_results),
                1,
            )
            if previous.glosas != current.glosas
        ]
        difference_text = "; ".join(differences) if differences else "sin diferencias"
        print(
            f"| {phrase} | {rules_result.glosas} | {previous_text} "
            f"| {current_text} | {difference_text} |"
        )

        covered = _covered_glosses(phrase, aliases)
        for index_run, (previous, current) in enumerate(
            zip(previous_prompt_results, current_prompt_results),
            1,
        ):
            unrelated_previous = [gloss for gloss in previous.glosas if gloss not in covered]
            unrelated_current = [gloss for gloss in current.glosas if gloss not in covered]
            omitted_previous = sorted(covered - set(previous.glosas))
            omitted_current = sorted(covered - set(current.glosas))
            print(
                f"  {phrase!r} repetición {index_run}: "
                f"anterior ajenas={unrelated_previous}, omisiones léxicas={omitted_previous}; "
                f"actual ajenas={unrelated_current}, omisiones léxicas={omitted_current}"
            )


if __name__ == "__main__":
    main()
