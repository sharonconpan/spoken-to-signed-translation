"""
Dashboard local: Texto -> Glosa -> Avatar 3D
Basado en el paquete `spoken-to-signed-translation`.

Ejecutar con:
    streamlit run app.py
"""

import importlib
import json
import math
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from importlib.metadata import entry_points
from pathlib import Path
from queue import Queue
from urllib.request import Request, urlopen

import streamlit as st
import streamlit.components.v1 as components

from avatar_http_api import create_avatar_request_handler
from gramatica_lsp import GramaticaLSP
from llm_lsp import (
    LLMConfig,
    LLMGlosser,
    build_gloss_aliases,
    build_system_prompt,
    choose_llm_or_rules,
    should_use_llm,
    warn_ollama_unavailable_once,
)

st.set_page_config(
    page_title="Texto -> Lengua de Señas",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      header, footer, #MainMenu, .stAppHeader, .stAppToolbar, .stDecoration,
      [data-testid="stHeader"], [data-testid="stToolbar"],
      [data-testid="stStatusWidget"], [data-testid="stSidebar"],
      h1, div[data-testid="stTextArea"], div[data-testid="stButton"],
      div[data-testid="stAlert"], div[data-testid="stCaptionContainer"] {
        display: none !important;
      }
      [data-testid="stAppViewContainer"], [data-testid="stMain"],
      section.main, section.main > div, [data-testid="stMainBlockContainer"] {
        width: 100% !important;
        max-width: none !important;
        margin: 0 !important;
        padding: 0 !important;
      }
      div[data-testid="stHorizontalBlock"]:has(iframe) {
        display: block !important;
        width: 100vw !important;
        max-width: none !important;
        gap: 0 !important;
        margin: 0 !important;
      }
      div[data-testid="stColumn"]:not(:has(iframe)) {
        display: none !important;
      }
      div[data-testid="stColumn"]:has(iframe),
      div[data-testid="stElementContainer"]:has(iframe),
      div[data-testid="stIFrame"] {
        display: block !important;
        width: 100vw !important;
        max-width: none !important;
        margin: 0 !important;
        padding: 0 !important;
      }
      div[data-testid="stIFrame"] {
        height: 100vh !important;
        min-height: 100vh !important;
        border: 0 !important;
      }
      iframe {
        position: fixed !important;
        inset: 0 !important;
        display: block !important;
        width: 100vw !important;
        height: 100vh !important;
        min-height: 100vh !important;
        margin: 0 !important;
        border: 0 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline_main():
    """
    Localiza la función real que ejecuta la CLI `text_to_gloss_to_pose`
    (el mismo entry point que usas por consola) sin lanzar un subproceso.
    """
    eps = entry_points()
    try:
        scripts = eps.select(group="console_scripts")
    except AttributeError:
        scripts = eps.get("console_scripts", [])

    for ep in scripts:
        if ep.name == "text_to_gloss_to_pose":
            return ep.load()

    raise RuntimeError(
        "No se encontró el entry point 'text_to_gloss_to_pose'. "
        "Corre en tu venv: python -c \"from importlib.metadata import entry_points; "
        "print([e.name for e in entry_points().select(group='console_scripts')])\" "
        "para ver el nombre real, y ajusta este script."
    )


def translate_es_to_de(text_es: str) -> str:
    """Traduce español -> alemán usando Google Translate (sin API key)."""
    from deep_translator import GoogleTranslator
    return GoogleTranslator(source="es", target="de").translate(text_es)


def run_text_to_pose(text, glosser, lexicon, spoken_language, signed_language, pose_path):
    main_fn = get_pipeline_main()
    old_argv = sys.argv
    try:
        sys.argv = [
            "text_to_gloss_to_pose",
            "--text", text,
            "--glosser", glosser,
            "--lexicon", str(lexicon),
            "--spoken-language", spoken_language,
            "--signed-language", signed_language,
            "--pose", str(pose_path),
        ]
        try:
            main_fn()
        except SystemExit:
            # argparse suele llamar sys.exit(0) al terminar bien
            pass
    finally:
        sys.argv = old_argv


def pose_to_video(pose_path, raw_video_path):
    """Genera el video con PoseVisualizer (queda con códec mp4v/MPEG-4)."""
    from pose_format.pose import Pose
    from pose_format.pose_visualizer import PoseVisualizer
    from pose_format.utils.generic import pose_normalization_info, normalize_pose_size

    with open(pose_path, "rb") as f:
        pose = Pose.read(f.read())
    pose = pose.normalize(pose_normalization_info(pose.header))
    normalize_pose_size(pose)
    v = PoseVisualizer(pose)
    v.save_video(str(raw_video_path), v.draw())


def reencode_to_h264(raw_video_path, final_video_path):
    """
    Reencodea de mp4v (MPEG-4 Part 2, se ve/etiqueta como M4V y no lo
    reproducen bien los navegadores) a H.264 real, compatible con
    st.video y cualquier reproductor.
    """
    import subprocess
    import imageio_ffmpeg

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-y",
        "-i", str(raw_video_path),
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        str(final_video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg falló:\n{result.stderr[-1000:]}")


def find_lexicon_row(lookup, word, gloss, spoken_language, signed_language):
    """Busca una fila con el mismo orden de prioridades que PoseLookup.lookup."""
    from spoken_to_signed.gloss_to_pose.languages import LANGUAGE_BACKUP

    language = signed_language
    while True:
        lookup_list = (
            (lookup.words_index, word),
            (lookup.glosses_index, word),
            (lookup.glosses_index, gloss),
        )
        for dictionary, term in lookup_list:
            if term is None:
                continue
            if (
                spoken_language in dictionary
                and language in dictionary[spoken_language]
                and term.lower() in dictionary[spoken_language][language]
            ):
                rows = dictionary[spoken_language][language][term.lower()]
                return lookup.get_best_row(rows, term)

        if language not in LANGUAGE_BACKUP:
            return None
        language = LANGUAGE_BACKUP[language]


@st.cache_resource
def get_cached_glosser(glosser_name, spoken_language):
    module = importlib.import_module(f"spoken_to_signed.text_to_gloss.{glosser_name}")
    gloss_function = module.text_to_gloss
    gloss_function("hola tu vivir", language=spoken_language)
    return gloss_function


@st.cache_resource
def get_cached_pose_lookup(lexicon_dir):
    from spoken_to_signed.gloss_to_pose import CSVPoseLookup

    return CSVPoseLookup(lexicon_dir)


@st.cache_resource
def get_avatar_lexicon_index(index_path):
    with Path(index_path).open(encoding="utf-8") as index_file:
        index = json.load(index_file)

    if index.get("version") != 1 or not isinstance(index.get("by_path"), dict):
        raise ValueError(f"Índice de cache de avatar inválido: {index_path}")

    cache_dir = Path(index_path).parent
    for clip in index.get("clips", {}).values():
        clip_path = cache_dir / clip["file"]
        if not clip_path.is_file():
            raise FileNotFoundError(f"Falta el clip convertido del cache: {clip_path}")
    return index


@st.cache_resource
def get_cached_grammar_engine(rules_path, _lookup, _avatar_index):
    return GramaticaLSP(
        reglas_path=rules_path,
        lookup=_lookup,
        indice_lexico=_avatar_index,
        spoken_language="es",
        signed_language="prl",
        cache_size=512,
    )


@st.cache_resource
def get_cached_llm_service(config_path, rules_path, examples_path, _lookup, _avatar_index):
    config = LLMConfig.load(config_path)
    glosses = {
        str(clip["gloss"]).strip().lower()
        for clip in _avatar_index["clips"].values()
        if isinstance(clip, dict) and clip.get("gloss")
    }
    system_prompt = build_system_prompt(
        rules_path=rules_path,
        examples_path=examples_path,
        available_glosses=glosses,
    )
    service = LLMGlosser(
        system_prompt=system_prompt,
        available_glosses=glosses,
        gloss_aliases=build_gloss_aliases(_lookup, _avatar_index),
        config=config,
    )
    service.warmup_async()
    return config, service


def clear_phrase_json_cache(avatar_data_dir):
    """Borra los JSON temporales de frases para mantener el directorio limpio."""
    for old_json in avatar_data_dir.glob("frase_*.json"):
        try:
            old_json.unlink()
        except OSError:
            pass


def write_phrase_json(glosses, avatar_data_dir, avatar_index):
    """Guarda una vista estática de la frase para compatibilidad con los loaders legacy."""
    clear_phrase_json_cache(avatar_data_dir)
    timestamp = int(time.time_ns() / 1_000_000)
    phrase_path = avatar_data_dir / f"frase_{timestamp}.json"
    by_gloss = {clip["gloss"].upper(): key for key, clip in avatar_index["clips"].items()}
    payload = {
        "version": 1,
        "glosses": glosses,
        "clips": [],
    }
    for gloss in glosses:
        key = by_gloss.get(gloss.upper())
        if key is None or key not in avatar_index["clips"]:
            continue
        payload["clips"].append(
            {
                "gloss": gloss,
                "file": f"data/lexicon/{avatar_index['clips'][key]['file']}",
            }
        )
    phrase_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return f"data/{phrase_path.name}"


def send_glosses_to_avatar(glosses, enqueued_at_ms):
    request = Request(
        "http://127.0.0.1:8765/say",
        data=json.dumps(
            {"glosses": glosses, "enqueued_at_ms": enqueued_at_ms}
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=2) as response:
        payload = json.load(response)
        if response.status != 202:
            raise RuntimeError(f"El servidor del avatar respondió {response.status}: {payload}")
        return payload


def _glosas_con_glosser_anterior(texto, glosser, lookup, avatar_index):
    sentences = glosser(texto, language=spoken_language)
    found_glosses = []
    for sentence in sentences:
        for item in sentence:
            if not item.word:
                continue
            row = find_lexicon_row(
                lookup,
                item.word,
                item.gloss,
                spoken_language,
                signed_language,
            )
            if row is None:
                continue
            clip_key = avatar_index["by_path"].get(row["path"])
            if clip_key is None:
                raise KeyError(
                    f"El clip {row['path']!r} no está en el cache de avatar. "
                    "Reconstruye el cache del léxico."
                )
            found_glosses.append(avatar_index["clips"][clip_key]["gloss"])
    return found_glosses


def _encolar_glosa(gloss, avatar_index, pending_glosses, enqueued_at_ms):
    by_gloss = {
        clip["gloss"].strip().upper(): key
        for key, clip in avatar_index["clips"].items()
    }
    clip_key = by_gloss.get(gloss.strip().upper())
    if clip_key is None:
        raise KeyError(f"La glosa {gloss!r} no está en el cache de avatar.")
    clip = avatar_index["clips"][clip_key]
    pending_glosses.put(
        {
            "key": clip_key,
            "gloss": clip["gloss"],
            "file": f"data/lexicon/{clip['file']}",
            "enqueued_at_ms": enqueued_at_ms,
        }
    )
    return clip["gloss"]


def _procesar_texto_y_encolar(
    texto,
    final,
    glosser,
    lookup,
    avatar_index,
    pending_glosses,
    grammar_engine,
    llm_service=None,
    llm_mode="solo_reglas",
    enqueued_at_ms=None,
):
    if not final:
        return {
            "queued": 0,
            "text": texto,
            "glosses": [],
            "glosas_encoladas": [],
            "sin_sena": [],
            "reglas_aplicadas": [],
            "descartada_llm": [],
            "descartadas_llm": [],
            "descartada_sin_relacion": [],
            "camino": "reglas",
            "ignored": True,
            "timing_ms": {},
        }
    if not isinstance(texto, str) or not texto.strip():
        raise ValueError("El texto está vacío.")

    started_at = time.perf_counter()
    grammar_ms = 0.0
    fallback_glosser_ms = 0.0
    llm_ms = 0.0
    found_glosses = []
    missing_glosses = []
    applied_rules = []
    rules_result = None
    try:
        grammar_result = grammar_engine.texto_a_glosas(texto)
        grammar_ms = (time.perf_counter() - started_at) * 1000
        rules_result = grammar_result
        found_glosses = list(grammar_result.glosas)
        missing_glosses = grammar_result.sin_sena
        applied_rules = grammar_result.reglas_aplicadas
        if missing_glosses:
            print(f"PALABRAS_SIN_SENA: {', '.join(missing_glosses)}")
        print(f"REGLAS_LSP_APLICADAS: {', '.join(applied_rules)}")
    except Exception as error:  # noqa: BLE001 - fall back for any grammar failure
        grammar_ms = (time.perf_counter() - started_at) * 1000
        print(f"Error en gramática LSP; usando glosser simple como fallback: {error}")
        fallback_started_at = time.perf_counter()
        found_glosses = _glosas_con_glosser_anterior(
            texto,
            glosser,
            lookup,
            avatar_index,
        )
        fallback_glosser_ms = (time.perf_counter() - fallback_started_at) * 1000

    grammar_finished_at = time.perf_counter()
    queued_at = enqueued_at_ms
    if queued_at is None:
        queued_at = time.time_ns() / 1_000_000
    elif (
        isinstance(queued_at, bool)
        or not isinstance(queued_at, (int, float))
        or not math.isfinite(queued_at)
    ):
        raise ValueError("enqueued_at_ms debe ser un número finito.")

    route = "reglas"
    discarded_llm = []
    discarded_unrelated = []
    if (
        rules_result is not None
        and llm_service is not None
        and should_use_llm(llm_mode, texto, rules_result)
    ):
        llm_started_at = time.perf_counter()

        def enqueue_streamed_gloss(gloss):
            _encolar_glosa(gloss, avatar_index, pending_glosses, queued_at)

        llm_result = llm_service.generate(texto, on_gloss=enqueue_streamed_gloss)
        llm_ms = (time.perf_counter() - llm_started_at) * 1000
        found_glosses, route = choose_llm_or_rules(llm_result, rules_result.glosas)
        discarded_llm = llm_result.descartada_llm
        discarded_unrelated = llm_result.descartada_sin_relacion
        if discarded_unrelated:
            print(f"DESCARTADA_SIN_RELACION: {', '.join(discarded_unrelated)}")
        if llm_result.error:
            print(f"LLM_LSP_FALLBACK: {llm_result.error}")
            if not llm_result.glosas:
                warn_ollama_unavailable_once(llm_result.error)
        if route == "llm":
            print(
                "LLM_LSP_TIMING "
                f"first_gloss_ms={llm_result.first_gloss_ms} "
                f"total_ms={llm_result.total_ms:.1f} "
                f"cached={llm_result.from_cache}"
            )
        elif llm_result.timed_out:
            print("LLM_LSP_TIMING first_gloss_timeout; se usan las reglas")

    timings = {
        "grammar_ms": round(grammar_ms, 1),
        "glosser_ms": round(fallback_glosser_ms, 1),
        "llm_ms": round(llm_ms, 1),
        "lookup_ms": 0.0,
    }
    if not found_glosses:
        timings["queue_ms"] = 0.0
        finished_at = time.perf_counter()
        timings["total_ms"] = round((finished_at - started_at) * 1000, 1)
        print(
            "ENVIAR_TEXTO_TIMING "
            f"camino={route} "
            f"grammar_ms={timings['grammar_ms']:.1f} "
            f"glosser_ms={timings['glosser_ms']:.1f} "
            f"llm_ms={timings['llm_ms']:.1f} "
            f"queue_ms=0.0 "
            f"total_ms={timings['total_ms']:.1f} glosses=0"
        )
        return {
            "queued": 0,
            "text": texto,
            "glosses": [],
            "glosas_encoladas": [],
            "sin_sena": missing_glosses,
            "reglas_aplicadas": applied_rules,
            "descartada_llm": discarded_llm,
            "descartadas_llm": discarded_llm,
            "descartada_sin_relacion": discarded_unrelated,
            "camino": route,
            "not_found": True,
            "timing_ms": timings,
        }

    queued_count = len(found_glosses) if route == "llm" else 0
    if route == "reglas":
        for gloss in found_glosses:
            _encolar_glosa(gloss, avatar_index, pending_glosses, queued_at)
            queued_count += 1

    finished_at = time.perf_counter()
    timings["queue_ms"] = round(
        max(0.0, (finished_at - grammar_finished_at) * 1000 - llm_ms),
        1,
    )
    timings["total_ms"] = round((finished_at - started_at) * 1000, 1)
    print(
        "ENVIAR_TEXTO_TIMING "
        f"camino={route} "
        f"grammar_ms={timings['grammar_ms']:.1f} "
        f"glosser_ms={timings['glosser_ms']:.1f} "
        f"llm_ms={timings['llm_ms']:.1f} "
        f"queue_ms={timings['queue_ms']:.1f} "
        f"total_ms={timings['total_ms']:.1f} "
        f"glosses={len(found_glosses)}"
    )
    return {
        "queued": queued_count,
        "text": texto,
        "glosses": found_glosses,
        "glosas_encoladas": found_glosses,
        "sin_sena": missing_glosses,
        "reglas_aplicadas": applied_rules,
        "descartada_llm": discarded_llm,
        "descartadas_llm": discarded_llm,
        "descartada_sin_relacion": discarded_unrelated,
        "camino": route,
        "timing_ms": timings,
    }


def crear_enviar_texto(
    glosser,
    lookup,
    avatar_index,
    pending_glosses,
    grammar_engine,
    llm_service=None,
    llm_mode="solo_reglas",
    enqueued_at_ms=None,
):
    def enviar_texto(texto, final):
        return _procesar_texto_y_encolar(
            texto,
            final,
            glosser,
            lookup,
            avatar_index,
            pending_glosses,
            grammar_engine,
            llm_service,
            llm_mode,
            enqueued_at_ms,
        )

    return enviar_texto


@st.cache_resource
def start_avatar_server(index_path, _glosser, _lookup, _grammar_engine, _llm_service, llm_mode):
    avatar_dir = Path(__file__).resolve().parent / "avatar"
    index = get_avatar_lexicon_index(index_path)
    pending_glosses = Queue()
    mode_lock = threading.Lock()
    current_mode = {"value": llm_mode}
    ollama_lock = threading.Lock()
    ollama_status = {"value": "comprobando", "checking": False}

    def get_mode():
        with mode_lock:
            return current_mode["value"]

    def set_mode(mode):
        with mode_lock:
            current_mode["value"] = mode

    def get_ollama_status():
        with ollama_lock:
            return ollama_status["value"]

    def refresh_ollama_status():
        with ollama_lock:
            if ollama_status["checking"]:
                return
            ollama_status["checking"] = True
            ollama_status["value"] = "comprobando"

        def check_status():
            base_url = getattr(getattr(_llm_service, "config", None), "base_url", None)
            base_url = base_url or "http://127.0.0.1:11434"
            try:
                with urlopen(Request(f"{base_url.rstrip('/')}/api/tags"), timeout=0.8) as response:
                    available = response.status == 200
            except Exception:
                available = False
            with ollama_lock:
                ollama_status["value"] = "disponible" if available else "no disponible"
                ollama_status["checking"] = False

        threading.Thread(target=check_status, name="ollama-status-check", daemon=True).start()

    def process_text(text, final, enqueued_at_ms, mode):
        return _procesar_texto_y_encolar(
            text,
            final,
            _glosser,
            _lookup,
            index,
            pending_glosses,
            _grammar_engine,
            _llm_service,
            mode,
            enqueued_at_ms,
        )

    AvatarRequestHandler = create_avatar_request_handler(
        avatar_dir=str(avatar_dir),
        index=index,
        pending_glosses=pending_glosses,
        process_text=process_text,
        get_mode=get_mode,
        set_mode=set_mode,
        get_ollama_status=get_ollama_status,
        refresh_ollama_status=refresh_ollama_status,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 8765), AvatarRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    server.pending_glosses = pending_glosses
    server.get_mode = get_mode
    refresh_ollama_status()
    return server


st.title("Texto -> Lengua de Señas")

# --- Configuración avanzada (deshabilitada por ahora) ---
# with st.expander("Configuración avanzada"):
#     glosser = st.selectbox("Glosser", ["simple", "spacylemma", "nmt"], index=0)
#     lexicon = st.text_input("Ruta del lexicón", value="assets/dummy_lexicon")
#     spoken_language = st.text_input("Idioma hablado (spoken)", value="de")
#     signed_language = st.text_input("Idioma de señas (signed)", value="sgg")

# Parámetros fijos por ahora:
glosser = "simple"
lexicon = "assets/lexicon_lsp"
spoken_language = "es"
signed_language = "prl"         # Lengua de Señas Peruana

project_dir = Path(__file__).resolve().parent
lexicon_dir = project_dir / lexicon
avatar_lexicon_dir = project_dir / "avatar" / "data" / "lexicon"
avatar_index_path = avatar_lexicon_dir / "index.json"
rules_path = project_dir / "reglas_lsp.yaml"
examples_path = project_dir / "tests" / "ejemplos_lsp.yaml"
llm_config_path = project_dir / "config_lsp.yaml"
startup_error = None
server_error = None
llm_config = LLMConfig(mode="solo_reglas")
cached_llm_service = None
try:
    cached_glosser = get_cached_glosser(glosser, spoken_language)
    cached_lookup = get_cached_pose_lookup(str(lexicon_dir))
    avatar_lexicon_index = get_avatar_lexicon_index(str(avatar_index_path))
    try:
        cached_grammar_engine = get_cached_grammar_engine(
            str(rules_path),
            cached_lookup,
            avatar_lexicon_index,
        )
    except Exception as e:  # noqa: BLE001 - keep the app available without a valid grammar file
        cached_grammar_engine = None
        print(f"No se pudo cargar la gramática LSP; se usará el glosser simple: {e}")
    try:
        llm_config, cached_llm_service = get_cached_llm_service(
            str(llm_config_path),
            str(rules_path),
            str(examples_path),
            cached_lookup,
            avatar_lexicon_index,
        )
    except Exception as e:  # noqa: BLE001 - LLM is optional
        warn_ollama_unavailable_once(str(e))
except Exception as e:
    startup_error = str(e)
else:
    try:
        start_avatar_server(
            str(avatar_index_path),
            cached_glosser,
            cached_lookup,
            cached_grammar_engine,
            cached_llm_service,
            llm_config.mode,
        )
    except Exception as e:
        server_error = str(e)

col_input, col_output = st.columns(2)

with col_input:
    text_es = st.text_area(
        "Escribe una oración en español",
        value="Los niños pequeños comen pizza",
        height=150,
    )
    traducir = st.button("Traducir", type="primary", use_container_width=True)

with col_output:
    components.iframe("http://localhost:8765/", height=620)

    if startup_error:
        st.error(
            "No se pudieron cargar el glosser, el léxico y su cache de avatar. "
            "Construye el cache con `python tools/build_lexicon_cache.py` y reinicia la app. "
            f"Detalle: {startup_error}"
        )
    elif server_error:
        st.error(f"No se pudo iniciar el servidor local del avatar: {server_error}")
    elif traducir:
        if not text_es.strip():
            st.warning("Escribe una oración primero.")
        else:
            translation_started_at_ms = time.time_ns() / 1_000_000
            with st.spinner("Generando..."):
                try:
                    avatar_server = start_avatar_server(
                        str(avatar_index_path),
                        cached_glosser,
                        cached_lookup,
                        cached_grammar_engine,
                        cached_llm_service,
                        llm_config.mode,
                    )
                    enviar_texto = crear_enviar_texto(
                        cached_glosser,
                        cached_lookup,
                        avatar_lexicon_index,
                        avatar_server.pending_glosses,
                        cached_grammar_engine,
                        cached_llm_service,
                        avatar_server.get_mode(),
                        translation_started_at_ms,
                    )
                    result = enviar_texto(text_es, True)
                    found_glosses = result["glosses"]
                    if result.get("not_found") or not found_glosses:
                        st.error("Ninguna palabra de la oración está en el léxico.")
                    else:
                        st.success(f"Enviadas a la cola del avatar: {result['queued']}")
                except Exception as e:
                    st.error(f"Error enviando glosas al avatar: {e}")
                else:
                    if found_glosses:
                        st.caption(f"Glosas encontradas: {' · '.join(found_glosses)}")
    else:
        st.info("Escribe una oración y pulsa Traducir; el visor permanece listo.")
