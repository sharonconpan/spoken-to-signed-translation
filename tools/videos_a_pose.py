#!/usr/bin/env python3
"""Convert named sign videos to MediaPipe Holistic .pose files and update index.csv."""

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "videos_nuevos"
DEFAULT_OUTPUT = ROOT / "assets" / "lexicon_lsp" / "prl"
DEFAULT_INDEX = ROOT / "assets" / "lexicon_lsp" / "index.csv"
CSV_FIELDS = (
    "path",
    "spoken_language",
    "signed_language",
    "start",
    "end",
    "words",
    "glosses",
    "priority",
)

VERB_ALIASES = {
    "aprender": ("aprendo", "aprendes", "aprende"),
    "atender": ("atiendo", "atiendes", "atiende"),
    "ayudar": ("ayudo", "ayudas", "ayuda"),
    "beber": ("bebo", "bebes"),
    "caminar": ("camino", "caminas", "camina"),
    "comer": ("comes", "come"),
    "comprar": ("compro", "compras", "compra"),
    "dar": ("doy", "das", "da"),
    "decir": ("digo", "dices", "dice"),
    "dormir": ("duermo", "duermes", "duerme"),
    "empezar": ("empiezo", "empiezas", "empieza"),
    "ensenar": ("enseño", "enseñas", "enseña"),
    "entender": ("entiendo", "entiendes", "entiende"),
    "escribir": ("escribo", "escribes", "escribe"),
    "escuchar": ("escucho", "escuchas", "escucha"),
    "estudiar": ("estudio", "estudias", "estudia"),
    "explicar": ("explico", "explicas", "explica"),
    "hablar": ("hablo", "hablas", "habla"),
    "ir": ("voy", "vas", "va"),
    "jugar": ("juego", "juegas", "juega"),
    "leer": ("leo", "lees", "lee"),
    "necesitar": ("necesito", "necesitas", "necesita"),
    "practicar": ("practico", "practicas", "practica"),
    "preguntar": ("pregunto", "preguntas", "pregunta"),
    "querer": ("quiero", "quieres", "quiere"),
    "responder": ("respondo", "respondes", "responde"),
    "saber": ("sabes", "sabe"),
    "sentir": ("siento", "sientes", "siente"),
    "tener": ("tengo", "tienes", "tiene"),
    "tomar": ("toma", "tomas", "tomo"),
    "terminar": ("termino", "terminas", "termina"),
    "ver": ("veo", "ves"),
}


def pose_stem(video_stem: str) -> str:
    lowered = video_stem.lower()
    if lowered == "letra enie ñ":
        return "letra_enie"
    without_enye = lowered.replace("ñ", "n")
    decomposed = unicodedata.normalize("NFD", without_enye)
    ascii_stem = "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    )
    return re.sub(r"[^a-z0-9]+", "_", ascii_stem).strip("_")


def canonical_words(video_stem: str) -> str:
    if video_stem.lower().startswith("letra "):
        if video_stem.lower() == "letra enie ñ":
            return "letra enie"
        return video_stem.lower().replace("ñ", "n")
    return video_stem.lower()


def make_row(
    path: str,
    words: str,
    gloss: str,
    priority: int,
    spoken_language: str,
    signed_language: str,
) -> dict[str, str]:
    return {
        "path": path,
        "spoken_language": spoken_language,
        "signed_language": signed_language,
        "start": "0",
        "end": "0",
        "words": words,
        "glosses": gloss,
        "priority": str(priority),
    }


def load_rows(index_path: Path) -> list[dict[str, str]]:
    if not index_path.exists():
        return []
    with index_path.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError(f"Esquema CSV inesperado en {index_path}")
        return [
            {field: row[field] for field in CSV_FIELDS}
            for row in reader
        ]


def convert_video(video_path: Path, pose_path: Path) -> tuple[bool, str]:
    if pose_path.is_file():
        return False, "ya existía; omitido"

    converter = Path(sys.executable).with_name(
        "video_to_pose.exe" if os.name == "nt" else "video_to_pose"
    )
    if not converter.is_file():
        return False, f"no se encontró el CLI de pose-format: {converter}"

    pose_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=pose_path.parent,
            prefix=f".{pose_path.stem}_",
            suffix=".pose.tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        result = subprocess.run(
            [
                str(converter),
                "--format",
                "mediapipe",
                "-i",
                str(video_path.resolve()),
                "-o",
                str(temporary_path.resolve()),
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()
            return False, f"exit {result.returncode}: {detail[-1200:]}"
        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            return False, "el conversor no produjo un .pose válido"

        temporary_path.replace(pose_path)
        return True, "generado"
    except (OSError, subprocess.SubprocessError) as error:
        return False, f"{type(error).__name__}: {error}"
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def update_index(
    index_path: Path,
    output_dir: Path,
    videos: list[Path],
    spoken_language: str,
    signed_language: str,
) -> int:
    rows = load_rows(index_path)
    canonical_paths: set[str] = set()
    new_rows: list[dict[str, str]] = []

    for video_path in videos:
        stem = pose_stem(video_path.stem)
        pose_path = output_dir / f"{stem}.pose"
        if not pose_path.is_file():
            continue

        relative_path = Path(
            os.path.relpath(pose_path.resolve(), index_path.parent.resolve())
        ).as_posix()
        gloss = stem.upper()
        canonical_paths.add(relative_path)
        words = canonical_words(video_path.stem)
        new_rows.append(
            make_row(relative_path, words, gloss, 0, spoken_language, signed_language)
        )

        if video_path.stem.lower() == "letra enie ñ" and words != "letra enie":
            new_rows.append(
                make_row(relative_path, "letra enie", gloss, 1, spoken_language, signed_language)
            )

        for priority, alias in enumerate(VERB_ALIASES.get(stem, ()), start=1):
            new_rows.append(
                make_row(relative_path, alias, gloss, priority, spoken_language, signed_language)
            )

    existing_preserved = [
        row
        for row in rows
        if row["path"] not in canonical_paths
        or int(row["priority"]) != 0
    ]
    merged = existing_preserved + new_rows

    unique_rows: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for row in merged:
        identity = tuple(row[field] for field in CSV_FIELDS)
        if identity not in seen:
            seen.add(identity)
            unique_rows.append(row)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_index = index_path.with_name(f".{index_path.name}.tmp")
    with temporary_index.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(unique_rows)
    temporary_index.replace(index_path)
    return len(new_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--language", default="prl")
    parser.add_argument("--spoken-language", default="es")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        parser.error(f"No existe la carpeta de entrada: {args.input_dir}")

    videos = sorted(
        (path for path in args.input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".mp4"),
        key=lambda path: path.name.casefold(),
    )
    destinations: dict[str, Path] = {}
    for video_path in videos:
        stem = pose_stem(video_path.stem)
        if not stem:
            parser.error(f"El video no produce un nombre de destino válido: {video_path.name}")
        if stem in destinations:
            parser.error(
                f"Colisión de destino .pose para {destinations[stem].name} y {video_path.name}"
            )
        destinations[stem] = video_path

    successes = 0
    skipped = 0
    failures: list[tuple[str, str]] = []
    for index, video_path in enumerate(videos, start=1):
        destination = args.output_dir / f"{pose_stem(video_path.stem)}.pose"
        created, message = convert_video(video_path, destination)
        if created:
            successes += 1
        elif message.startswith("ya existía"):
            skipped += 1
        else:
            failures.append((video_path.name, message))
        print(f"[{index}/{len(videos)}] {video_path.name}: {message}", flush=True)

    indexed_rows = update_index(
        args.index,
        args.output_dir,
        videos,
        args.spoken_language,
        args.language,
    )
    print(
        f"Resumen: {len(videos)} videos, {successes} generados, "
        f"{skipped} omitidos, {len(failures)} fallidos; "
        f"{indexed_rows} filas nuevas para videos."
    )
    for video_name, error in failures:
        print(f"ERROR {video_name}: {error}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
