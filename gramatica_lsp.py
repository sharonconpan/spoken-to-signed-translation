"""Reglas configurables para transformar español a glosas LSP."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from simplemma import simple_tokenizer, text_lemmatizer

ROOT = Path(__file__).resolve().parent
DEFAULT_RULES_PATH = ROOT / "reglas_lsp.yaml"
DEFAULT_CACHE_SIZE = 512
VERB_ENDINGS = ("ar", "er", "ir")
FIRST_PERSON_ENDINGS = ("o", "oy", "e", "i")
SECOND_PERSON_ENDINGS = ("as", "es", "aste", "iste")
CLITIC_ENDINGS = ("me", "te", "se", "lo", "la", "los", "las", "les")
COMPLEX_CLAUSE_MARKERS = {"que", "porque", "aunque", "cuando", "mientras"}
VERB_LEMMA_OVERRIDES = {"como": "comer"}


class _Yaml12SafeLoader(yaml.SafeLoader):
    pass


_Yaml12SafeLoader.yaml_implicit_resolvers = {
    first: [
        (tag, regexp)
        for tag, regexp in resolvers
        if tag != "tag:yaml.org,2002:bool"
    ]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_Yaml12SafeLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|false)$", re.IGNORECASE),
    list("tTfF"),
)


@dataclass
class ResultadoGlosas:
    glosas: list[str]
    sin_sena: list[str]
    reglas_aplicadas: list[str]
    estado: str | None = None


@dataclass
class _Token:
    texto: str
    lema: str
    glosa: str


def _sin_tildes(texto: str) -> str:
    return "".join(
        caracter
        for caracter in unicodedata.normalize("NFD", texto)
        if unicodedata.category(caracter) != "Mn"
    )


def _normalizar(texto: str) -> str:
    minusculas = texto.lower()
    sin_signos = "".join(
        " " if unicodedata.category(caracter).startswith(("P", "S")) else caracter
        for caracter in minusculas
    )
    return re.sub(r"\s+", " ", _sin_tildes(sin_signos)).strip()


def _tokens_lematizados(texto: str) -> list[_Token]:
    texto_limpio = "".join(
        " " if unicodedata.category(caracter).startswith(("P", "S")) else caracter
        for caracter in texto.lower()
    )
    palabras = simple_tokenizer(texto_limpio)
    lemas = text_lemmatizer(texto_limpio, lang="es")
    tokens = []
    for palabra, lema in zip(palabras, lemas):
        palabra_normalizada = _sin_tildes(palabra.lower())
        lema_normalizado = _sin_tildes(lema.lower())
        if palabra_normalizada.isalpha():
            tokens.append(_Token(palabra_normalizada, lema_normalizado, lema_normalizado))
    return tokens


def _lista(regla: dict[str, Any] | None, campo: str) -> set[str]:
    if not isinstance(regla, dict):
        return set()
    valor = regla.get(campo, [])
    if not isinstance(valor, list):
        return set()
    return {_normalizar(str(elemento)) for elemento in valor}


class GramaticaLSP:
    """Carga reglas recargables y procesa frases con una caché LRU acotada."""

    def __init__(
        self,
        *,
        reglas_path: str | Path = DEFAULT_RULES_PATH,
        lookup: Any = None,
        indice_lexico: dict[str, Any] | None = None,
        spoken_language: str = "es",
        signed_language: str = "prl",
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        self.reglas_path = Path(reglas_path)
        self.lookup = lookup
        self.indice_lexico = indice_lexico or {}
        self.spoken_language = spoken_language
        self.signed_language = signed_language
        self.cache_size = max(0, cache_size)
        self._lock = threading.RLock()
        self._cache: OrderedDict[tuple[str, str, bool], ResultadoGlosas] = OrderedDict()
        self._reglas: dict[str, Any] = {}
        self._reglas_mtime_ns: int | None = None
        self._reglas_digest = ""
        self._alias_a_glosa = self._crear_mapa_lexico()
        self._glosas_disponibles = {
            _normalizar(str(clip["gloss"]))
            for clip in self.indice_lexico.get("clips", {}).values()
            if isinstance(clip, dict) and isinstance(clip.get("gloss"), str)
        }
        self._alias_digest = hashlib.sha256(
            json.dumps(self._alias_a_glosa, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self._cargar_reglas()

    def _crear_mapa_lexico(self) -> dict[str, str]:
        if self.lookup is None:
            return {}

        by_path = self.indice_lexico.get("by_path", {})
        clips = self.indice_lexico.get("clips", {})
        alias_a_glosa: dict[str, str] = {}
        for dictionary_name in ("words_index", "glosses_index"):
            dictionary = getattr(self.lookup, dictionary_name, {})
            language_index = dictionary.get(self.spoken_language, {})
            signed_index = language_index.get(self.signed_language, {})
            for alias, rows in signed_index.items():
                if not rows:
                    continue
                best_row = self.lookup.get_best_row(rows, alias)
                clip_key = by_path.get(best_row["path"])
                clip = clips.get(clip_key, {})
                canonical = clip.get("gloss")
                if isinstance(canonical, str) and canonical.strip():
                    alias_a_glosa[_normalizar(alias)] = _normalizar(canonical)
        return alias_a_glosa

    def _cargar_reglas(self) -> dict[str, Any]:
        with self._lock:
            stat = self.reglas_path.stat()
            if self._reglas_mtime_ns == stat.st_mtime_ns:
                return self._reglas
            with self.reglas_path.open(encoding="utf-8") as rules_file:
                reglas = yaml.load(rules_file, Loader=_Yaml12SafeLoader) or {}
            if not isinstance(reglas, dict):
                raise ValueError(f"El archivo de reglas debe contener un objeto YAML: {self.reglas_path}")
            self._reglas = reglas
            self._reglas_mtime_ns = stat.st_mtime_ns
            self._reglas_digest = hashlib.sha256(
                json.dumps(reglas, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            self._cache.clear()
            return self._reglas

    def texto_a_glosas(
        self,
        texto: str,
        *,
        sin_filtro_lexico: bool = False,
        use_cache: bool = True,
    ) -> ResultadoGlosas:
        reglas = self._cargar_reglas()
        frase_normalizada = _normalizar(texto)
        key = (frase_normalizada, self._reglas_digest + self._alias_digest, sin_filtro_lexico)
        if use_cache and self.cache_size:
            with self._lock:
                cached = self._cache.get(key)
                if cached is not None:
                    self._cache.move_to_end(key)
                    return self._copiar_resultado(cached)

        resultado = self._procesar(texto, reglas, sin_filtro_lexico)
        if use_cache and self.cache_size:
            with self._lock:
                self._cache[key] = self._copiar_resultado(resultado)
                self._cache.move_to_end(key)
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
        return resultado

    @staticmethod
    def _copiar_resultado(resultado: ResultadoGlosas) -> ResultadoGlosas:
        return ResultadoGlosas(
            glosas=list(resultado.glosas),
            sin_sena=list(resultado.sin_sena),
            reglas_aplicadas=list(resultado.reglas_aplicadas),
            estado=resultado.estado,
        )

    def _resolver_glosa(self, texto: str, lema: str) -> str:
        return self._alias_a_glosa.get(texto, self._alias_a_glosa.get(lema, lema))

    def _combinar_expresiones(self, tokens: list[_Token]) -> list[_Token]:
        aliases = tuple(self._alias_a_glosa)
        max_expression_length = max(
            (len(alias.split()) if " " in alias else len(alias) for alias in aliases),
            default=1,
        )
        combined: list[_Token] = []
        index = 0
        while index < len(tokens):
            match = None
            max_words = min(max_expression_length, len(tokens) - index)
            for word_count in range(max_words, 1, -1):
                sequence = tokens[index:index + word_count]
                spaced_alias = " ".join(token.texto for token in sequence)
                compact_alias = "".join(token.texto for token in sequence)
                gloss = self._alias_a_glosa.get(spaced_alias)
                if gloss is None:
                    gloss = self._alias_a_glosa.get(compact_alias)
                if gloss is not None:
                    match = _Token(
                        texto=spaced_alias,
                        lema=compact_alias,
                        glosa=gloss,
                    )
                    index += word_count
                    break
            if match is None:
                combined.append(tokens[index])
                index += 1
            else:
                combined.append(match)
        return combined

    @staticmethod
    def _es_infinitivo(palabra: str) -> bool:
        return palabra.endswith(VERB_ENDINGS)

    @staticmethod
    def _lema_forma_verbal(token: _Token) -> str:
        if token.glosa in VERB_LEMMA_OVERRIDES:
            return VERB_LEMMA_OVERRIDES[token.glosa]
        if GramaticaLSP._es_infinitivo(token.glosa):
            return token.glosa

        for clitic in CLITIC_ENDINGS:
            if token.texto.endswith(clitic) and len(token.texto) > len(clitic) + 2:
                base = token.texto[: -len(clitic)]
                base_lemma = _sin_tildes(text_lemmatizer(base, lang="es")[0].lower())
                if base_lemma.endswith("a"):
                    return f"{base_lemma[:-1]}ar"
                if base_lemma.endswith("e"):
                    return f"{base_lemma[:-1]}er"
                return base_lemma

        if token.texto.endswith("a"):
            return f"{token.texto[:-1]}ar"
        if token.texto.endswith("e"):
            return f"{token.texto[:-1]}er"
        if token.texto.endswith("o"):
            return VERB_LEMMA_OVERRIDES.get(token.texto, token.lema)
        return token.lema

    @staticmethod
    def _persona_implicita(palabra: str) -> str | None:
        if palabra.endswith(FIRST_PERSON_ENDINGS):
            return "yo"
        if palabra.endswith(SECOND_PERSON_ENDINGS):
            return "tu"
        return None

    def _procesar(  # noqa: C901 - keep the rule pipeline explicit and ordered
        self,
        texto: str,
        reglas: dict[str, Any],
        sin_filtro_lexico: bool,
    ) -> ResultadoGlosas:
        tokens = _tokens_lematizados(texto)
        if not tokens:
            return ResultadoGlosas([], [], [])

        for token in tokens:
            token.glosa = self._resolver_glosa(token.texto, token.lema)
        tokens = self._combinar_expresiones(tokens)

        subordinadas = reglas.get("oraciones_subordinadas", {})
        has_complex_clause = (
            isinstance(subordinadas, dict)
            and subordinadas.get("tratar_con_llm_o_revision_humana", False)
            and any(token.texto in COMPLEX_CLAUSE_MARKERS for token in tokens)
        )
        if has_complex_clause:
            return self._resultado_filtrado(
                [token.glosa for token in tokens],
                ["sin_resolver"],
                sin_filtro_lexico,
                estado="SIN_RESOLVER",
            )

        omitir = reglas.get("omitir", {})
        articulos_regla = omitir.get("articulos", {})
        copula_regla = omitir.get("copula_ser_estar", {})
        omitir_articulos = isinstance(articulos_regla, dict) and articulos_regla.get("activo", False)
        omitir_copula = isinstance(copula_regla, dict) and copula_regla.get("activo", False)
        articulos = _lista(articulos_regla, "lista")
        copulas = _lista(copula_regla, "lista")
        if omitir_articulos and "el" in articulos:
            articulos.update({"al", "del"})

        reglas_aplicadas: list[str] = []
        verb_rule = reglas.get("verbos", {})
        infinitive_active = (
            not isinstance(verb_rule, dict)
            or verb_rule.get("usar_infinitivo", True)
        )
        tiempo = reglas.get("tiempo", {})
        time_active = isinstance(tiempo, dict) and tiempo.get("activo", False)
        time_markers: set[str] = set()
        if time_active and isinstance(tiempo.get("marcadores"), dict):
            for markers in tiempo["marcadores"].values():
                if isinstance(markers, list):
                    time_markers.update(_normalizar(str(marker)) for marker in markers)
        pronombres = reglas.get("pronombres", {})
        pronouns_active = isinstance(pronombres, dict) and pronombres.get("usar_deixis", False)
        equivalences = pronombres.get("equivalencias", {}) if pronouns_active else {}
        normalized_equivalences = (
            {
                _normalizar(str(source)): _normalizar(str(target))
                for source, target in equivalences.items()
            }
            if isinstance(equivalences, dict)
            else {}
        )
        verb_positions = [
            index
            for index, token in enumerate(tokens)
            if (
                token.texto not in time_markers
                and token.texto not in normalized_equivalences
                and token.texto not in articulos
                and (
                    self._es_infinitivo(token.glosa)
                    or self._es_infinitivo(token.lema)
                )
            )
        ]
        if not verb_positions:
            fallback_candidates = []
            for index, token in enumerate(tokens):
                has_left = index > 0
                has_right = index + 1 < len(tokens)
                has_clitic = any(
                    token.texto.endswith(clitic)
                    and len(token.texto) > len(clitic) + 2
                    for clitic in CLITIC_ENDINGS
                )
                is_conjugated_candidate = token.texto.endswith(
                    ("o", "a", "e", "as", "es", "an", "en", "amos", "emos", "imos")
                )
                if has_clitic or (has_left and has_right and is_conjugated_candidate):
                    fallback_candidates.append(index)
            if fallback_candidates:
                verb_positions.append(
                    min(
                        fallback_candidates,
                        key=lambda index: (
                            0 if tokens[index].texto.endswith(("a", "as", "an", "amos")) else 1,
                            index,
                        ),
                    )
                )

        verb_position = verb_positions[0] if verb_positions else None
        verb_token = tokens[verb_position] if verb_position is not None else None
        verb_gloss = None
        if verb_token is not None:
            verb_gloss = (
                self._lema_forma_verbal(verb_token)
                if infinitive_active
                else verb_token.glosa
            )
            if infinitive_active and verb_gloss != verb_token.texto:
                reglas_aplicadas.append("infinitivo")

        retained: list[_Token] = []
        for token in tokens:
            if omitir_articulos and (token.texto in articulos or token.lema in articulos):
                if "omitir_articulos" not in reglas_aplicadas:
                    reglas_aplicadas.append("omitir_articulos")
                continue
            if omitir_copula and (token.lema in copulas or token.glosa in copulas):
                if "omitir_copula" not in reglas_aplicadas:
                    reglas_aplicadas.append("omitir_copula")
                continue
            retained.append(token)

        time_tokens = [
            token for token in retained
            if time_active and (token.texto in time_markers or token.lema in time_markers)
        ]
        retained = [token for token in retained if token not in time_tokens]
        if time_tokens:
            reglas_aplicadas.append("tiempo")

        for token in retained:
            replacement = normalized_equivalences.get(token.texto)
            if replacement is not None:
                token.glosa = self._resolver_glosa(replacement, replacement)
                if "deixis" not in reglas_aplicadas:
                    reglas_aplicadas.append("deixis")

        negacion = reglas.get("negacion", {})
        negation_active = isinstance(negacion, dict) and negacion.get("activo", True)
        negation_tokens = [
            token for token in retained
            if negation_active and token.texto == "no"
        ]
        retained = [token for token in retained if token not in negation_tokens]

        retained_verb_position = retained.index(verb_token) if verb_token in retained else None
        implicit_subject = None
        if verb_token is not None and pronouns_active:
            has_explicit_subject = any(
                token.texto in normalized_equivalences
                and (retained_verb_position is None or index < retained_verb_position)
                for index, token in enumerate(retained)
            )
            if not has_explicit_subject:
                if not verb_token.texto.endswith(CLITIC_ENDINGS) and verb_token.texto != "es":
                    implicit_subject = self._persona_implicita(verb_token.texto)
                if verb_token.texto == "voy":
                    implicit_subject = "yo"
                if implicit_subject in normalized_equivalences and "deixis" not in reglas_aplicadas:
                    reglas_aplicadas.append("deixis")

        output_tokens = [token.glosa for token in retained]
        if (
            verb_token is not None
            and retained_verb_position is None
            and omitir_copula
            and implicit_subject in normalized_equivalences
        ):
            insertion = next(
                (index + 1 for index, token in enumerate(retained) if token.glosa == "hola"),
                min(verb_position or len(retained), len(retained)),
            )
            output_tokens.insert(
                insertion,
                self._resolver_glosa(implicit_subject, implicit_subject),
            )
        if retained_verb_position is not None and verb_gloss is not None:
            explicit_subject_positions = [
                index for index, token in enumerate(retained)
                if token.texto in normalized_equivalences and index < retained_verb_position
            ]
            if explicit_subject_positions:
                subject_start = explicit_subject_positions[0]
                prefix_tokens = [
                    token.glosa for token in retained[:subject_start]
                    if token.texto not in time_markers
                ]
                subject_tokens = [
                    token.glosa for token in retained[subject_start:retained_verb_position]
                    if token.texto not in time_markers
                ]
                object_tokens = [
                    token.glosa for token in retained[retained_verb_position + 1:]
                    if token.texto not in time_markers
                ]
            else:
                prefix_tokens = []
                before_verb = [
                    token for token in retained[:retained_verb_position]
                    if token.texto not in time_markers
                ]
                object_tokens = [
                    token.glosa for token in retained[retained_verb_position + 1:]
                    if token.texto not in time_markers
                ]
                if implicit_subject:
                    subject_tokens = [self._resolver_glosa(implicit_subject, implicit_subject)]
                    greeting_tokens = [token.glosa for token in before_verb if token.glosa == "hola"]
                    prefix_tokens = greeting_tokens
                    object_tokens = [
                        token.glosa for token in before_verb
                        if token.glosa not in greeting_tokens
                    ] + object_tokens
                else:
                    subject_tokens = [token.glosa for token in before_verb]

            order_rule = reglas.get("orden_constituyentes", {})
            order_active = isinstance(order_rule, dict) and order_rule.get("activo", False)
            order = str(order_rule.get("valor", "SOV")).upper() if isinstance(order_rule, dict) else "SOV"
            if order_active and order in {"SOV", "SVO"}:
                if order == "SOV":
                    output_tokens = prefix_tokens + subject_tokens + object_tokens + [verb_gloss]
                else:
                    output_tokens = prefix_tokens + subject_tokens + [verb_gloss] + object_tokens
                reglas_aplicadas.append(f"orden_{order}")

        if time_tokens:
            time_position = str(tiempo.get("posicion_marcador", "inicio")).lower()
            if time_position == "inicio":
                output_tokens = [token.glosa for token in time_tokens] + output_tokens
                reglas_aplicadas.append("tiempo_inicio")
            elif time_position == "despues_del_verbo" and verb_gloss in output_tokens:
                insertion = output_tokens.index(verb_gloss) + 1
                output_tokens[insertion:insertion] = [token.glosa for token in time_tokens]
                reglas_aplicadas.append("tiempo_despues_del_verbo")

        if negation_tokens:
            position = str(negacion.get("posicion", "despues_del_verbo")).lower()
            if position in {"despues_del_verbo", "después_del_verbo"} and verb_gloss in output_tokens:
                insertion = output_tokens.index(verb_gloss) + 1
            elif position in {"inicio", "antes"}:
                insertion = 0
            else:
                insertion = len(output_tokens)
            output_tokens[insertion:insertion] = [token.glosa for token in negation_tokens]
            reglas_aplicadas.append(f"negacion_{position}")

        return self._resultado_filtrado(output_tokens, reglas_aplicadas, sin_filtro_lexico)

    def _resultado_filtrado(
        self,
        glosas_candidatas: list[str],
        reglas_aplicadas: list[str],
        sin_filtro_lexico: bool,
        estado: str | None = None,
    ) -> ResultadoGlosas:
        glosas: list[str] = []
        sin_sena: list[str] = []
        previous_gloss: str | None = None
        for gloss in glosas_candidatas:
            normalized = _normalizar(gloss)
            if not normalized:
                continue
            exists = not self._glosas_disponibles or normalized in self._glosas_disponibles
            if (exists or sin_filtro_lexico) and normalized != previous_gloss:
                glosas.append(normalized)
            if self._glosas_disponibles and not exists and normalized not in sin_sena:
                sin_sena.append(normalized)
            previous_gloss = normalized
        return ResultadoGlosas(
            glosas=glosas,
            sin_sena=sin_sena,
            reglas_aplicadas=list(dict.fromkeys(reglas_aplicadas)),
            estado=estado,
        )


_ENGINES: dict[tuple[str, int, int, int, str, str], GramaticaLSP] = {}
_ENGINES_LOCK = threading.Lock()


def texto_a_glosas(
    texto: str,
    *,
    reglas_path: str | Path = DEFAULT_RULES_PATH,
    lookup: Any = None,
    indice_lexico: dict[str, Any] | None = None,
    spoken_language: str = "es",
    signed_language: str = "prl",
    sin_filtro_lexico: bool = False,
    use_cache: bool = True,
    cache_size: int = DEFAULT_CACHE_SIZE,
) -> ResultadoGlosas:
    """Convierte texto español a glosas usando las reglas configuradas."""
    engine_key = (
        str(reglas_path),
        id(lookup),
        id(indice_lexico),
        cache_size,
        spoken_language,
        signed_language,
    )
    with _ENGINES_LOCK:
        engine = _ENGINES.get(engine_key)
        if engine is None:
            engine = GramaticaLSP(
                reglas_path=reglas_path,
                lookup=lookup,
                indice_lexico=indice_lexico,
                spoken_language=spoken_language,
                signed_language=signed_language,
                cache_size=cache_size,
            )
            _ENGINES[engine_key] = engine
    return engine.texto_a_glosas(
        texto,
        sin_filtro_lexico=sin_filtro_lexico,
        use_cache=use_cache,
    )
