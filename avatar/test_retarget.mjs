// Valida el solver con cinemática directa (FK) sobre datos REALES:
//   node test_retarget.mjs data/hola_tu_vivir.json
import fs from 'node:fs';
import { buildRig, solveFrame, prepareClip, sampleFrame, boneNames, qMul, qRot, qInv, norm, ID, FINGERS } from './retarget.js';

const clip = prepareClip(JSON.parse(fs.readFileSync(process.argv[2] ?? 'data/hola_tu_vivir.json')));

// --- T-pose sintético (avatar mira a +Z, su izquierda = +X). Dedos a lo largo de ±X.
const pos = { chest: [0, 1.3, 0], head: [0, 1.55, 0] };
for (const [s, k] of [['left', 1], ['right', -1]]) {
  pos[`${s}UpperArm`] = [k * 0.17, 1.4, 0]; pos[`${s}LowerArm`] = [k * 0.45, 1.4, 0]; pos[`${s}Hand`] = [k * 0.70, 1.4, 0];
  const base = { Index: 0.015, Middle: 0.0, Ring: -0.012, Little: -0.025 };
  for (const f of ['Index', 'Middle', 'Ring', 'Little'])
    FINGERS[f].forEach((p, j) => { pos[`${s}${f}${p}`] = [k * (0.78 + 0.03 * j), 1.4, base[f]]; });
  [[0.72, 0.02], [0.75, 0.04], [0.78, 0.055]].forEach(([x, z], j) => { pos[`${s}Thumb${FINGERS.Thumb[j]}`] = [k * x, 1.4, z]; });
}
const rig = buildRig(pos);

const ang = (a, b) => Math.acos(Math.max(-1, Math.min(1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))) * 180 / Math.PI;
const pt = (a, i) => [a[3 * i], a[3 * i + 1], a[3 * i + 2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const err = { upper: [], lower: [], finger: [] };
let nan = 0;

for (let f = 0; f < clip.frames; f++) {
  const fr = sampleFrame(clip, f);
  const q = solveFrame(rig, fr, { binary: true }, {});
  for (const k of Object.values(q)) if (k.some(Number.isNaN)) nan++;
  for (const s of ['left', 'right']) {
    const L = s === 'left', S = rig.sides[s];
    const [iS, iE, iW] = L ? [11, 13, 15] : [12, 14, 16];
    const c = fr.poseConf;
    const wUp = qMul(q.chest, q[S.upper.name]), wLo = qMul(wUp, q[S.lower.name]);
    if (c[iE] >= 0.5) err.upper.push(ang(qRot(wUp, S.upper.restDir), norm(sub(pt(fr.pose, iE), pt(fr.pose, iS)))));
    if (c[iE] >= 0.5 && c[iW] >= 0.5) err.lower.push(ang(qRot(wLo, S.lower.restDir), norm(sub(pt(fr.pose, iW), pt(fr.pose, iE)))));
    const H = L ? fr.lh : fr.rh, HC = L ? fr.lhConf : fr.rhConf;
    if (Math.min(HC[0], HC[5], HC[9], HC[17]) >= 0.5) {
      const wHand = qMul(wLo, q[S.hand.name]);
      let parent = wHand;
      for (const fb of S.fingers) {
        if (fb.first) parent = wHand;
        parent = qMul(parent, q[fb.name]);
        if (Math.min(HC[fb.a], HC[fb.b]) >= 0.5) err.finger.push(ang(qRot(parent, fb.restDir), norm(sub(pt(H, fb.b), pt(H, fb.a)))));
      }
    }
  }
}
const st = (a) => a.length ? `n=${a.length}  media=${(a.reduce((x, y) => x + y, 0) / a.length).toExponential(2)}°  max=${Math.max(...a).toExponential(2)}°` : 'sin datos';
console.log('error angular FK vs. dirección medida (pesos binarios):');
console.log(' brazo superior :', st(err.upper));
console.log(' antebrazo      :', st(err.lower));
console.log(' falanges       :', st(err.finger));
console.log(' cuaterniones NaN:', nan);

let bad = 0, st2 = {};
for (let f = 0; f < clip.frames; f += 1) {
  const q = solveFrame(rig, sampleFrame(clip, f + 0.37), {}, st2);
  for (const k of Object.values(q)) { if (!(Math.abs(Math.hypot(...k) - 1) < 1e-6)) bad++; }
}
console.log(' modo suave: cuaterniones no unitarios/NaN =', bad, '| huesos por frame =', Object.keys(solveFrame(rig, sampleFrame(clip, 0), {}, {})).length, 'de', boneNames().length);