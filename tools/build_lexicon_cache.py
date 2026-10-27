#!/usr/bin/env python3
"""Prebuild avatar-ready JSON clips from the original pose lexicon."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import numpy as np

DEFAULT_LEXICON = ROOT / "assets" / "lexicon_lsp"
DEFAULT_OUTPUT = ROOT / "avatar" / "data" / "lexicon"
DEFAULT_LANGUAGE = "prl"
FPS = 30.0
POSE_WRISTS = (15, 16)
POSE_HIPS = (23, 24)


def get_pose_converter():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import pose_to_json

    return pose_to_json


def rest_trim_bounds(
    pose,
    confidence,
    fps,
    confidence_threshold=0.5,
    motion_threshold=0.08,
    below_hips_margin=0.0,
    consecutive_frames=6,
):
    """Return leading/trailing rest bounds; interior frames are never removed."""
    frame_count = len(pose)
    if frame_count == 0:
        return 0, 0

    wrists = pose[:, POSE_WRISTS, :]
    wrist_confidence = confidence[:, POSE_WRISTS]
    hips_y = pose[:, POSE_HIPS, 1]
    low_confidence = np.all(wrist_confidence < confidence_threshold, axis=1)
    hands_below_hips = np.all(
        wrists[:, :, 1] <= hips_y + below_hips_margin,
        axis=1,
    )

    if frame_count > 1:
        wrist_speed = np.linalg.norm(np.diff(wrists, axis=0), axis=2) * fps
        wrist_speed = np.vstack((wrist_speed[:1], wrist_speed))
    else:
        wrist_speed = np.zeros((1, len(POSE_WRISTS)))
    stationary = np.max(wrist_speed, axis=1) <= motion_threshold
    rest = (low_confidence | hands_below_hips) & stationary

    leading = 0
    while leading < frame_count and rest[leading]:
        leading += 1
    trailing = 0
    while trailing < frame_count - leading and rest[frame_count - trailing - 1]:
        trailing += 1

    start = leading if leading >= consecutive_frames else 0
    end = frame_count - trailing if trailing >= consecutive_frames else frame_count
    if start >= end:
        return 0, frame_count
    return start, end


def convert_clip(
    source_path,
    gloss,
    trim_rest=False,
    compare_trimming=False,
    confidence_threshold=0.5,
    motion_threshold=0.08,
    below_hips_margin=0.0,
    consecutive_frames=6,
):
    converter = get_pose_converter()
    pose_data = converter.read_pose(str(source_path))
    extracted = converter.extract(pose_data)
    arrays = {}
    for key in ("pose", "lh", "rh"):
        points, confidence = converter.clean(*extracted[key])
        arrays[key] = converter.resample(points, confidence, pose_data["fps"], FPS)

    original_frames = len(arrays["pose"][0])
    candidate_start, candidate_end = 0, original_frames
    if trim_rest or compare_trimming:
        candidate_start, candidate_end = rest_trim_bounds(
            arrays["pose"][0],
            arrays["pose"][1],
            FPS,
            confidence_threshold,
            motion_threshold,
            below_hips_margin,
            consecutive_frames,
        )
    trim_start, trim_end = (
        (candidate_start, candidate_end) if trim_rest else (0, original_frames)
    )

    for key, (points, confidence) in arrays.items():
        arrays[key] = (
            points[trim_start:trim_end],
            confidence[trim_start:trim_end],
        )

    frame_count = trim_end - trim_start
    payload = {
        "version": 1,
        "fps": FPS,
        "frames": frame_count,
        "glosses": [{"gloss": gloss, "start": 0, "end": frame_count}],
        "coords": "x=izquierda del avatar, y=arriba, z=adelante (three.js/VRM). pose en metros.",
        "pose_n": 33,
        "hand_n": 21,
    }
    for key in ("pose", "lh", "rh"):
        points, confidence = arrays[key]
        payload[key] = np.round(points.reshape(-1), 4).tolist()
        payload[f"{key}Conf"] = np.round(confidence.reshape(-1), 2).tolist()

    return (
        payload,
        original_frames,
        candidate_start,
        original_frames - candidate_end,
    )


def load_lexicon_rows(lexicon_dir, signed_language):
    with (lexicon_dir / "index.csv").open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    rows = [row for row in rows if row["signed_language"] == signed_language]
    if not rows:
        raise ValueError(f"No hay clips para el idioma de señas {signed_language!r} en {lexicon_dir}")
    return rows


def build_cache(
    lexicon_dir,
    output_dir,
    signed_language,
    trim_rest=False,
    confidence_threshold=0.5,
    motion_threshold=0.08,
    below_hips_margin=0.0,
    consecutive_frames=6,
    compare_trimming=False,
):
    rows = load_lexicon_rows(lexicon_dir, signed_language)
    rows_by_path = defaultdict(list)
    for row in rows:
        rows_by_path[row["path"]].append(row)

    clips = {}
    paths_by_key = {}
    by_path = {}
    trim_report = []
    for relative_path, clip_rows in sorted(rows_by_path.items()):
        source_path = lexicon_dir / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(f"No se encontró el clip original del léxico: {source_path}")

        clip_key = Path(relative_path).stem.lower()
        if clip_key in paths_by_key and paths_by_key[clip_key] != relative_path:
            raise ValueError(
                f"Dos clips distintos comparten el nombre de cache {clip_key!r}: "
                f"{paths_by_key[clip_key]!r} y {relative_path!r}"
            )
        paths_by_key[clip_key] = relative_path

        canonical_gloss = min(clip_rows, key=lambda row: int(row["priority"]))["glosses"]
        payload, original_frames, trim_start, trim_end = convert_clip(
            source_path,
            canonical_gloss,
            trim_rest=trim_rest,
            compare_trimming=compare_trimming,
            confidence_threshold=confidence_threshold,
            motion_threshold=motion_threshold,
            below_hips_margin=below_hips_margin,
            consecutive_frames=consecutive_frames,
        )
        output_name = f"{clip_key}.json"
        with (output_dir / output_name).open("w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, ensure_ascii=False, separators=(",", ":"))

        clips[clip_key] = {
            "file": output_name,
            "gloss": canonical_gloss,
            "frames": payload["frames"],
        }
        by_path[relative_path] = clip_key
        if compare_trimming:
            trim_report.append(
                {
                    "gloss": canonical_gloss,
                    "frames_before": original_frames,
                    "trim_start": trim_start,
                    "trim_end": trim_end,
                    "frames_after": original_frames - trim_start - trim_end,
                }
            )

    index = {
        "version": 1,
        "fps": FPS,
        "signed_language": signed_language,
        "trim_applied": trim_rest,
        "trim": {
            "confidence_threshold": confidence_threshold,
            "motion_threshold_mps": motion_threshold,
            "below_hips_margin_m": below_hips_margin,
            "consecutive_frames": consecutive_frames,
        },
        "by_path": by_path,
        "clips": clips,
    }
    with (output_dir / "index.json").open("w", encoding="utf-8") as index_file:
        json.dump(index, index_file, ensure_ascii=False, indent=2)

    if compare_trimming:
        print(
            "Recorte candidato (informativo; solo aplicado si se usó --trim-rest): "
            f"confianza<{confidence_threshold:g}, movimiento<={motion_threshold:g} m/s, "
            f"muñecas bajo cadera (margen {below_hips_margin:g} m), "
            f"racha mínima={consecutive_frames} frames."
        )
        print("Glosa | Antes | Recorte inicio | Recorte final | Después")
        for report in trim_report:
            print(
                f"{report['gloss']} | {report['frames_before']} | "
                f"{report['trim_start']} | {report['trim_end']} | "
                f"{report['frames_after']}"
            )
        if trim_rest:
            print("El recorte se aplicó al cache porque se indicó --trim-rest.")
        else:
            print("El cache se escribió sin recorte; el análisis no modificó los clips.")

    total_bytes = sum(path.stat().st_size for path in output_dir.glob("*.json"))
    print(
        f"Cache: {len(clips)} clips, {len(clips) + 1} JSON, "
        f"{total_bytes} bytes ({total_bytes / 1024:.1f} KiB) en {output_dir}"
    )
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--signed-language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--trim-rest", action="store_true", help="Apply edge-rest trimming (off by default).")
    parser.add_argument("--rest-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--rest-motion-threshold", type=float, default=0.08, help="Wrist speed in meters/second.")
    parser.add_argument("--below-hips-margin", type=float, default=0.0, help="Margin in meters.")
    parser.add_argument("--rest-consecutive-frames", type=int, default=6)
    parser.add_argument("--compare-trimming", action="store_true", help="Report candidate cuts for each clip.")
    args = parser.parse_args()

    if args.rest_consecutive_frames < 1:
        parser.error("--rest-consecutive-frames debe ser al menos 1")
    if args.rest_confidence_threshold < 0 or args.rest_confidence_threshold > 1:
        parser.error("--rest-confidence-threshold debe estar entre 0 y 1")
    if args.rest_motion_threshold < 0 or args.below_hips_margin < 0:
        parser.error("Los umbrales de movimiento y margen deben ser no negativos")

    args.output.mkdir(parents=True, exist_ok=True)
    build_cache(
        args.lexicon,
        args.output,
        args.signed_language,
        trim_rest=args.trim_rest,
        confidence_threshold=args.rest_confidence_threshold,
        motion_threshold=args.rest_motion_threshold,
        below_hips_margin=args.below_hips_margin,
        consecutive_frames=args.rest_consecutive_frames,
        compare_trimming=args.compare_trimming,
    )


if __name__ == "__main__":
    main()
