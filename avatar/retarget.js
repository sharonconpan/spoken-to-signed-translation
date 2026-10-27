// retarget.js -- posiciones XYZ (MediaPipe) -> rotaciones locales de huesos VRM.
// JS puro, sin dependencias (funciona en navegador y en Node). Cuaterniones = [x,y,z,w].
//
// Método: "aim" / swing por hueso.  Para cada hueso se conoce su dirección en reposo
// (T-pose, sacada del propio avatar) y la dirección objetivo (landmark_hijo - landmark_padre).
// Se calcula el arco mínimo que lleva la dirección actual (reposo rotado por el padre) a la objetivo:
//     world  = delta * parentWorld
//     local  = parentWorld^-1 * world
// Las manos usan un marco completo (muñeca, nudillos índice/meñique) para resolver también el giro.
// Supuesto: huesos NORMALIZADOS de VRM (rotación de reposo = identidad, T-pose).

export const ID = [0, 0, 0, 1];

// ---------- vectores
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export function norm(a, fb = null) {
  const l = Math.hypot(a[0], a[1], a[2]);
  return l < 1e-9 ? fb : [a[0] / l, a[1] / l, a[2] / l];
}
const lerp3 = (a, b, t) => add(mul(a, 1 - t), mul(b, t));
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

// ---------- cuaterniones
export const qMul = (a, b) => [
  a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
  a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
  a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
  a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]];
export const qInv = (q) => [-q[0], -q[1], -q[2], q[3]];
export function qRot(q, v) {
  const u = [q[0], q[1], q[2]];
  const t = mul(cross(u, v), 2);
  return add(add(v, mul(t, q[3])), cross(u, t));
}
export function qNorm(q) {
  const l = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
}
export function qSlerp(a, b, t) {
  let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  let bb = b;
  if (d < 0) { d = -d; bb = [-b[0], -b[1], -b[2], -b[3]]; }
  if (d > 0.9995) return qNorm([a[0] + (bb[0] - a[0]) * t, a[1] + (bb[1] - a[1]) * t, a[2] + (bb[2] - a[2]) * t, a[3] + (bb[3] - a[3]) * t]);
  const th = Math.acos(d), s = Math.sin(th);
  const wa = Math.sin((1 - t) * th) / s, wb = Math.sin(t * th) / s;
  return [a[0] * wa + bb[0] * wb, a[1] * wa + bb[1] * wb, a[2] * wa + bb[2] * wb, a[3] * wa + bb[3] * wb];
}
export function fromUnitVectors(u, v) {
  const d = dot(u, v);
  if (d > 0.999999) return ID;
  if (d < -0.999999) {                                   // 180°: eje perpendicular cualquiera
    let ax = cross([1, 0, 0], u);
    if (Math.hypot(...ax) < 1e-6) ax = cross([0, 1, 0], u);
    ax = norm(ax);
    return [ax[0], ax[1], ax[2], 0];
  }
  const c = cross(u, v);
  return qNorm([c[0], c[1], c[2], 1 + d]);
}
// Matriz de rotación con columnas c0,c1,c2 -> cuaternión
export function basisToQuat(c0, c1, c2) {
  const m00 = c0[0], m10 = c0[1], m20 = c0[2], m01 = c1[0], m11 = c1[1], m21 = c1[2], m02 = c2[0], m12 = c2[1], m22 = c2[2];
  const tr = m00 + m11 + m22;
  let x, y, z, w;
  if (tr > 0) { const s = Math.sqrt(tr + 1) * 2; w = 0.25 * s; x = (m21 - m12) / s; y = (m02 - m20) / s; z = (m10 - m01) / s; }
  else if (m00 > m11 && m00 > m22) { const s = Math.sqrt(1 + m00 - m11 - m22) * 2; w = (m21 - m12) / s; x = 0.25 * s; y = (m01 + m10) / s; z = (m02 + m20) / s; }
  else if (m11 > m22) { const s = Math.sqrt(1 + m11 - m00 - m22) * 2; w = (m02 - m20) / s; x = (m01 + m10) / s; y = 0.25 * s; z = (m12 + m21) / s; }
  else { const s = Math.sqrt(1 + m22 - m00 - m11) * 2; w = (m10 - m01) / s; x = (m02 + m20) / s; y = (m12 + m21) / s; z = 0.25 * s; }
  return qNorm([x, y, z, w]);
}

// ---------- definición del esqueleto
export const FINGERS = {
  Thumb: ['Metacarpal', 'Proximal', 'Distal'],
  Index: ['Proximal', 'Intermediate', 'Distal'],
  Middle: ['Proximal', 'Intermediate', 'Distal'],
  Ring: ['Proximal', 'Intermediate', 'Distal'],
  Little: ['Proximal', 'Intermediate', 'Distal'],
};
const MP_START = { Thumb: 1, Index: 5, Middle: 9, Ring: 13, Little: 17 }; // índice MediaPipe del 1er landmark

// Pose MediaPipe: 0 nariz, 7/8 orejas L/R, 11/12 hombros L/R, 13/14 codos, 15/16 muñecas
const PI = { nose: 0, earL: 7, earR: 8, shL: 11, shR: 12, elL: 13, elR: 14, wrL: 15, wrR: 16 };

export function boneNames() {
  const n = ['chest', 'head'];
  for (const s of ['left', 'right']) {
    n.push(`${s}UpperArm`, `${s}LowerArm`, `${s}Hand`);
    for (const [f, parts] of Object.entries(FINGERS)) for (const p of parts) n.push(`${s}${f}${p}`);
  }
  return n;
}

/** restPos: { nombreHueso: [x,y,z] } posiciones MUNDO del avatar en T-pose. */
export function buildRig(restPos) {
  const P = restPos;
  const dirTo = (a, b) => norm(sub(P[b], P[a]));
  const rig = { restShoulderDir: norm(sub(P.leftUpperArm, P.rightUpperArm)), sides: {} };
  for (const s of ['left', 'right']) {
    const side = { s, down: norm([s === 'left' ? 0.12 : -0.12, -1, 0]) };
    side.upper = { name: `${s}UpperArm`, restDir: dirTo(`${s}UpperArm`, `${s}LowerArm`) };
    side.lower = { name: `${s}LowerArm`, restDir: dirTo(`${s}LowerArm`, `${s}Hand`) };
    const hand = `${s}Hand`, mid = `${s}MiddleProximal`, idx = `${s}IndexProximal`, lit = `${s}LittleProximal`;
    const f = dirTo(hand, mid), sv = sub(P[lit], P[idx]);
    const sp = norm(sub(sv, mul(f, dot(sv, f))));
    side.hand = { name: hand, restQ: basisToQuat(f, sp, cross(f, sp)) };
    side.fingers = [];
    for (const [fname, parts] of Object.entries(FINGERS)) {
      const names = parts.map(p => `${s}${fname}${p}`);
      if (!names.every(n => P[n])) continue;                      // modelo sin ese hueso
      const d0 = dirTo(names[0], names[1]), d1 = dirTo(names[1], names[2]);
      const rest = [d0, d1, d1];                                  // la falange distal hereda la dirección
      names.forEach((n, j) => side.fingers.push({ name: n, restDir: rest[j], a: MP_START[fname] + j, b: MP_START[fname] + j + 1, first: j === 0 }));
    }
    rig.sides[s] = side;
  }
  return rig;
}

// ---------- solver por frame
function aim(parentQ, restDir, target) {
  const cur = qRot(parentQ, restDir);
  const world = qMul(fromUnitVectors(cur, target), parentQ);
  return { world, local: qMul(qInv(parentQ), world) };
}

/**
 * fr = { pose[33*3], poseConf[33], lh[21*3], lhConf[21], rh[21*3], rhConf[21] }  (ver sampleFrame)
 * opts: torso (0..1), head (0..1), binary (true => pesos 0/1, útil para tests)
 * state: objeto persistente entre frames (calibración neutra de la cabeza)
 * Devuelve { hueso: [x,y,z,w] } con rotaciones LOCALES.
 */
export function solveFrame(rig, fr, opts = {}, state = {}) {
  const o = { torso: 0.6, head: 0.5, binary: false, ...opts };
  const W = (c) => { if (o.binary) return c >= 0.5 ? 1 : 0; const t = clamp((c - 0.35) / 0.4, 0, 1); return t * t * (3 - 2 * t); };
  const pt = (a, i) => [a[3 * i], a[3 * i + 1], a[3 * i + 2]];
  const P = fr.pose, PC = fr.poseConf, out = {};

  // 1) pecho: alinea la línea de hombros
  let chestQ = ID;
  const wSh = W(Math.min(PC[PI.shL], PC[PI.shR]));
  if (wSh > 0) {
    const d = norm(sub(pt(P, PI.shL), pt(P, PI.shR)));
    if (d) chestQ = qSlerp(ID, fromUnitVectors(rig.restShoulderDir, d), wSh * o.torso);
  }
  out.chest = chestQ;

  // 2) cabeza: marco (izq, arriba, adelante) con orejas+nariz, relativo al primer frame válido
  let headQ = ID;
  const wH = W(Math.min(PC[PI.nose], PC[PI.earL], PC[PI.earR]));
  if (wH > 0) {
    const X = norm(sub(pt(P, PI.earL), pt(P, PI.earR)));
    const mid = mul(add(pt(P, PI.earL), pt(P, PI.earR)), 0.5);
    const fwd = sub(pt(P, PI.nose), mid);
    const Z = X && norm(sub(fwd, mul(X, dot(fwd, X))));
    if (X && Z) {
      const Qw = basisToQuat(X, cross(Z, X), Z);
      state.headNeutral ??= Qw;
      const rel = qSlerp(ID, qMul(Qw, qInv(state.headNeutral)), wH * o.head);
      headQ = qMul(qInv(chestQ), rel);
    }
  }
  out.head = headQ;

  // 3) brazos, manos y dedos
  for (const s of ['left', 'right']) {
    const L = s === 'left', side = rig.sides[s];
    const sh = pt(P, L ? PI.shL : PI.shR), el = pt(P, L ? PI.elL : PI.elR), wr = pt(P, L ? PI.wrL : PI.wrR);
    const cEl = PC[L ? PI.elL : PI.elR], cWr = PC[L ? PI.wrL : PI.wrR];

    const wU = W(cEl);
    const tU = norm(lerp3(side.down, norm(sub(el, sh), side.down), wU), side.down);
    const up = aim(chestQ, side.upper.restDir, tU);
    out[side.upper.name] = up.local;

    const wL = W(Math.min(cEl, cWr));
    const tL = norm(lerp3(side.down, norm(sub(wr, el), side.down), wL), side.down);
    const lo = aim(up.world, side.lower.restDir, tL);
    out[side.lower.name] = lo.local;

    const H = L ? fr.lh : fr.rh, HC = L ? fr.lhConf : fr.rhConf;
    const wHand = W(Math.min(HC[0], HC[5], HC[9], HC[17]));
    let handWorld = lo.world, handLocal = ID;
    if (wHand > 0) {
      const f = norm(sub(pt(H, 9), pt(H, 0)));
      const sv = sub(pt(H, 17), pt(H, 5));
      const sp = f && norm(sub(sv, mul(f, dot(sv, f))));
      if (f && sp) {
        const Qm = basisToQuat(f, sp, cross(f, sp));
        const worldMeas = qMul(Qm, qInv(side.hand.restQ));
        handLocal = qSlerp(ID, qMul(qInv(lo.world), worldMeas), wHand);
        handWorld = qMul(lo.world, handLocal);
      }
    }
    out[side.hand.name] = handLocal;

    let parentQ = handWorld;
    for (const fb of side.fingers) {
      if (fb.first) parentQ = handWorld;
      const w = wHand > 0 ? W(Math.min(HC[fb.a], HC[fb.b])) : 0;
      const tgt = w > 0 ? norm(sub(pt(H, fb.b), pt(H, fb.a))) : null;
      let local = ID;
      if (tgt) local = qSlerp(ID, aim(parentQ, fb.restDir, tgt).local, w);
      out[fb.name] = local;
      parentQ = qMul(parentQ, local);
    }
  }
  return out;
}

// ---------- datos
export function prepareClip(json) {
  const f32 = (a) => Float32Array.from(a);
  return { fps: json.fps, frames: json.frames, glosses: json.glosses,
    pose: f32(json.pose), poseConf: f32(json.poseConf), lh: f32(json.lh), lhConf: f32(json.lhConf),
    rh: f32(json.rh), rhConf: f32(json.rhConf) };
}
/** Interpola linealmente entre frames (f puede ser fraccionario). Confianza = mínimo de los dos. */
export function sampleFrame(clip, f) {
  const i0 = clamp(Math.floor(f), 0, clip.frames - 1), i1 = Math.min(i0 + 1, clip.frames - 1), a = clamp(f - i0, 0, 1);
  const get = (arr, n, conf) => {
    const o = new Float32Array(n);
    for (let i = 0; i < n; i++) o[i] = conf ? Math.min(arr[i0 * n + i], arr[i1 * n + i]) : arr[i0 * n + i] * (1 - a) + arr[i1 * n + i] * a;
    return o;
  };
  return { pose: get(clip.pose, 99), poseConf: get(clip.poseConf, 33, true),
    lh: get(clip.lh, 63), lhConf: get(clip.lhConf, 21, true), rh: get(clip.rh, 63), rhConf: get(clip.rhConf, 21, true) };
}