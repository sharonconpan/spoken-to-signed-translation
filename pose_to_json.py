#!/usr/bin/env python3
"""
pose_to_json.py  --  .pose (Holistic, 576 pts) -> JSON limpio para animar un avatar.

Uso:
    python pose_to_json.py hola.pose tu.pose vivir.pose -o data/hola_tu_vivir.json

IMPORTANTE: necesita los clips ORIGINALES del léxico (576 puntos, incluyen
POSE_WORLD_LANDMARKS). El .pose ya concatenado/reducido (178 pts) NO sirve:
reduce_holistic descarta POSE_WORLD y la normalización aplasta Z (~256 constante).

Qué hace
  1. Lee el binario .pose v0.2 (lector mínimo propio; con `pose-format` instalado
     equivale a Pose.read(...).body.data / .confidence).
  2. Corrige unidades/ejes -> sistema de three.js / VRM:  x = izquierda del avatar,
     y = arriba, z = hacia adelante.   (x, -y, -z)
       * POSE_WORLD: x,y vienen en ~mm y z en ~m (mezcla de unidades!). Se pasa todo a metros.
       * Manos: x,y en píxeles, z normalizado -> z * ancho_imagen. Solo se usan direcciones.
  3. Landmarks ausentes (xyz = 0 y conf = 0) -> confianza 0 (el solver los ignora).
  4. Rellena huecos cortos, suaviza (Savitzky-Golay) y remuestrea todo a 30 FPS.
  5. Concatena clips con N frames de transición lineal.
"""
import argparse, json, struct
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter


# ----------------------------------------------------------------- lectura .pose
def _s(b, o):
    n = struct.unpack_from('<H', b, o)[0]
    return b[o + 2:o + 2 + n].decode(), o + 2 + n


def read_pose(path):
    b = open(path, 'rb').read()
    o = 0
    version = struct.unpack_from('<f', b, o)[0]; o += 4
    if abs(version - 0.2) > 1e-4:
        raise ValueError(f'{path}: versión .pose {version} no soportada por este lector (esperado 0.2)')
    width, height, _depth, ncomp = struct.unpack_from('<HHHH', b, o); o += 8
    names = []
    for _ in range(ncomp):
        comp, o = _s(b, o); _fmt, o = _s(b, o)
        npts, nlimbs, ncols = struct.unpack_from('<HHH', b, o); o += 6
        for _ in range(npts):
            n, o = _s(b, o); names.append((comp, n))
        o += nlimbs * 4 + ncols * 6
    fps = struct.unpack_from('<f', b, o)[0]; o += 4
    frames = struct.unpack_from('<I', b, o)[0]; o += 4
    people = struct.unpack_from('<H', b, o)[0]; o += 2
    P = len(names)
    cnt = frames * people * P
    data = np.frombuffer(b, '<f4', cnt * 3, o).reshape(frames, people, P, 3)
    conf = np.frombuffer(b, '<f4', cnt, o + cnt * 12).reshape(frames, people, P)
    return dict(data=data[:, 0].astype(np.float64), conf=conf[:, 0].astype(np.float64),
                fps=float(fps), width=width, names=names)


def comp_idx(names, comp):
    idx = [i for i, (c, _) in enumerate(names) if c == comp]
    if not idx:
        raise ValueError(f'El archivo no contiene {comp}. ¿Es un .pose reducido (178 pts)? '
                         f'Usa los clips originales del léxico (576 pts).')
    return idx


# ----------------------------------------------------------------- extracción
def extract(p):
    d, c, names, W = p['data'], p['conf'], p['names'], p['width']
    iw = comp_idx(names, 'POSE_WORLD_LANDMARKS')
    pw = d[:, iw]
    pose = np.stack([pw[..., 0] / 1000.0, -pw[..., 1] / 1000.0, -pw[..., 2]], -1)  # metros
    out = {'pose': (pose, c[:, iw].copy())}
    for key, comp in (('lh', 'LEFT_HAND_LANDMARKS'), ('rh', 'RIGHT_HAND_LANDMARKS')):
        ih = comp_idx(names, comp)
        h = d[:, ih]
        hc = c[:, ih].copy()
        hc[np.all(h == 0, axis=-1)] = 0.0              # mano no detectada
        hand = np.stack([h[..., 0], -h[..., 1], -h[..., 2] * W], -1) / 1000.0
        out[key] = (hand, hc)
    return out


# ----------------------------------------------------------------- limpieza
def clean(x, c, thr=0.5, max_gap=6, win=7, order=2):
    F, N, _ = x.shape
    xo, co = x.copy(), np.zeros_like(c)
    t = np.arange(F)
    for n in range(N):
        idx = np.flatnonzero(c[:, n] >= thr)
        if idx.size == 0:
            continue
        for k in range(3):
            xo[:, n, k] = np.interp(t, idx, x[idx, n, k])
        co[idx, n] = c[idx, n]
        for a, b in zip(idx[:-1], idx[1:]):            # huecos cortos interiores
            if 1 < b - a <= max_gap + 1:
                co[a + 1:b, n] = min(c[a, n], c[b, n])
    if F >= win:
        xo = savgol_filter(xo, win, order, axis=0, mode='interp')
    return xo, co


def resample(x, c, fps_in, fps_out):
    F = len(x)
    n = int(round((F - 1) / fps_in * fps_out)) + 1
    f = np.arange(n) / fps_out * fps_in
    i0 = np.clip(np.floor(f).astype(int), 0, F - 1)
    i1 = np.minimum(i0 + 1, F - 1)
    a = (f - i0)[:, None, None]
    return x[i0] * (1 - a) + x[i1] * a, np.minimum(c[i0], c[i1])


def blend_frames(last_x, last_c, first_x, first_c, n):
    a = (np.arange(1, n + 1) / (n + 1))[:, None, None]
    return last_x[None] * (1 - a) + first_x[None] * a, np.repeat(np.minimum(last_c, first_c)[None], n, 0)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--blend', type=int, default=6, help='frames de transición entre clips (6 = 0.2 s)')
    a = ap.parse_args()

    keys = ['pose', 'lh', 'rh']
    acc = {k: ([], []) for k in keys}
    glosses, cursor = [], 0
    for path in a.inputs:
        p = read_pose(path)
        ex = extract(p)
        clip = {}
        for k in keys:
            x, c = clean(*ex[k])
            clip[k] = resample(x, c, p['fps'], a.fps)
        nF = len(clip['pose'][0])
        if cursor > 0 and a.blend > 0:
            for k in keys:
                bx, bc = blend_frames(acc[k][0][-1][-1], acc[k][1][-1][-1], *(v[0] for v in clip[k]), a.blend)
                acc[k][0].append(bx); acc[k][1].append(bc)
            cursor += a.blend
        for k in keys:
            acc[k][0].append(clip[k][0]); acc[k][1].append(clip[k][1])
        name = Path(path).stem
        glosses.append({'gloss': name, 'start': cursor, 'end': cursor + nF})
        cursor += nF
        print(f'{name}: {p["fps"]:.0f} fps -> {nF} frames | '
              f'RH válida {np.mean(clip["rh"][1][:, 0] > 0.5):.0%}  LH válida {np.mean(clip["lh"][1][:, 0] > 0.5):.0%}')

    r = lambda arr, d: np.round(np.concatenate(arr, 0).reshape(-1), d).tolist()
    out = {'version': 1, 'fps': a.fps, 'frames': cursor, 'glosses': glosses,
           'coords': 'x=izquierda del avatar, y=arriba, z=adelante (three.js/VRM). pose en metros.',
           'pose_n': 33, 'hand_n': 21}
    for k in keys:
        out[k] = r(acc[k][0], 4)
        out[k + 'Conf'] = r(acc[k][1], 2)
    json.dump(out, open(a.out, 'w'), separators=(',', ':'))
    print(f'-> {a.out}: {cursor} frames, {cursor / a.fps:.2f} s')


if __name__ == '__main__':
    main()