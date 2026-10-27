// Development helpers (used by tests/probe.mjs and tests/perf.mjs, exposed on window in main.js).
import { DEAD } from './simulation.js';

// GPU time per kernel in ms, summed over one frame's substeps (needs 'timestamp-query').
export async function benchKernels(sim) {
  const d = sim.device;
  if (!d.features.has('timestamp-query')) return { error: 'timestamp-query unsupported' };
  const MAX = 2048;
  const qs = d.createQuerySet({ type: 'timestamp', count: MAX });
  const res = d.createBuffer({ size: MAX * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const rd = d.createBuffer({ size: MAX * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = d.createCommandEncoder();
  const labels = [];
  let q = 0;
  const timed = (label, fn) => {
    const pass = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: q, endOfPassWriteIndex: q + 1 } });
    fn(pass);
    pass.end();
    labels.push(label);
    q += 2;
  };
  sim.step(null, 1 / 60, timed);
  enc.resolveQuerySet(qs, 0, q, res, 0);
  enc.copyBufferToBuffer(res, 0, rd, 0, q * 8);
  d.queue.submit([enc.finish()]);
  await rd.mapAsync(GPUMapMode.READ);
  const t = new BigUint64Array(rd.getMappedRange());
  const out = {};
  let total = 0;
  for (let k = 0; k < labels.length; k++) {
    const ms = Number(t[2 * k + 1] - t[2 * k]) / 1e6;
    out[labels[k]] = (out[labels[k]] || 0) + ms;
    total += ms;
  }
  rd.unmap();
  [qs, res, rd].forEach((x) => x.destroy());
  for (const k in out) out[k] = +out[k].toFixed(3);
  out.total = +total.toFixed(2);
  return out;
}

// Snapshot the particle state and summarise it: speeds, cloth strain, compression, leaks.
export async function debugStats(sim) {
  const d = sim.device, N = sim.N;
  const want = [sim.buf.pos, sim.buf.prev, sim.buf.edgeAlive];
  const sizes = [N * 16, N * 16, sim.mesh.numEdges * 4];
  // copy everything in one submit so all buffers describe the same instant
  const enc = d.createCommandEncoder();
  const staged = want.map((buf, k) => {
    const st = d.createBuffer({ size: sizes[k], usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyBufferToBuffer(buf, 0, st, 0, sizes[k]);
    return st;
  });
  d.queue.submit([enc.finish()]);
  const data = [];
  for (const st of staged) {
    await st.mapAsync(GPUMapMode.READ);
    data.push(new Float32Array(st.getMappedRange().slice(0)));
    st.destroy();
  }
  const [pos, prev, edgeF] = data;
  const dt = 1 / 60 / sim.cfg.substeps;
  const vel = pos.map((x, k) => (x - prev[k]) / dt);
  const E = new Uint32Array(edgeF.buffer);
  const S = sim.cfg.shape, r = sim.radius;

  // grain speeds + floor count
  const speeds = [];
  const mv = [0, 0, 0];
  let onFloor = 0, nan = 0;
  for (let i = sim.nCloth; i < N; i++) {
    if (!(pos[i * 4] === pos[i * 4])) { nan++; continue; }
    speeds.push(Math.hypot(vel[i * 4], vel[i * 4 + 1], vel[i * 4 + 2]));
    for (let a = 0; a < 3; a++) mv[a] += vel[i * 4 + a];
    if (pos[i * 4 + 1] < r * 1.5) onFloor++;
  }
  speeds.sort((a, b) => a - b);
  const pct = (a, q) => (a.length ? +a[Math.floor(q * (a.length - 1))].toFixed(3) : null);

  // cloth strain over live stretch edges
  const C = sim.mesh.stretch, cf = new Float32Array(C.buffer);
  const strains = [];
  let torn = 0, minYCloth = 1e9;
  for (const e of E) if (!e) torn++;
  for (let v = 0; v < sim.nCloth; v++) if (!(sim.clothFlags[v] & DEAD)) minYCloth = Math.min(minYCloth, pos[v * 4 + 1]);
  for (let k = 0; k < C.length; k += 4) {
    const a = C[k], b = C[k + 1];
    if (a === 0xffffffff || !E[C[k + 3]] || (sim.clothFlags[a] | sim.clothFlags[b]) & DEAD) continue;
    strains.push(Math.hypot(pos[a * 4] - pos[b * 4], pos[a * 4 + 1] - pos[b * 4 + 1], pos[a * 4 + 2] - pos[b * 4 + 2]) / cf[k + 2]);
  }
  strains.sort((a, b) => a - b);

  // compression: nearest grain-grain distance (in radii) for a sample of grains
  const g = 2 * r, grid = new Map();
  const key = (a, b, c) => (a * 73856093) ^ (b * 19349663) ^ (c * 83492791);
  for (let i = sim.nCloth; i < N; i++) {
    const k = key(Math.floor(pos[i * 4] / g), Math.floor(pos[i * 4 + 1] / g), Math.floor(pos[i * 4 + 2] / g));
    if (!grid.has(k)) grid.set(k, []);
    grid.get(k).push(i);
  }
  const nn = [0, 0, 0, 0]; // <0.5r, <1.2r, <1.8r, >=1.8r
  for (let i = sim.nCloth; i < N; i += 7) {
    const cx = Math.floor(pos[i * 4] / g), cy = Math.floor(pos[i * 4 + 1] / g), cz = Math.floor(pos[i * 4 + 2] / g);
    let best = 1e9;
    for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let c = -1; c <= 1; c++) {
      for (const j of grid.get(key(cx + a, cy + b, cz + c)) || []) {
        if (j !== i) best = Math.min(best, Math.hypot(pos[i * 4] - pos[j * 4], pos[i * 4 + 1] - pos[j * 4 + 1], pos[i * 4 + 2] - pos[j * 4 + 2]) / r);
      }
    }
    nn[best < 0.5 ? 0 : best < 1.2 ? 1 : best < 1.8 ? 2 : 3]++;
  }

  const ng = speeds.length || 1;
  return {
    v50: pct(speeds, 0.5),
    v99: pct(speeds, 0.99),
    maxV: pct(speeds, 1),
    vMean: mv.map((x) => +(x / ng).toFixed(3)),
    strain50: pct(strains, 0.5),
    strain99: pct(strains, 0.99),
    nn_lt05_lt12_lt18_ok: nn,
    onFloor,
    torn,
    nan,
    minYCloth: +minYCloth.toFixed(3),
    shapeTop: +(S.centerY + S.ay).toFixed(3),
  };
}

// Upward "fliers": particles rising faster than 1 m/s, and where they are relative to the skin.
export async function flierStats(sim) {
  const d = sim.device, N = sim.N;
  const enc = d.createCommandEncoder();
  const st = [sim.buf.pos, sim.buf.prev, sim.buf.vfix].map((buf) => {
    const b = d.createBuffer({ size: N * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyBufferToBuffer(buf, 0, b, 0, N * 16);
    return b;
  });
  d.queue.submit([enc.finish()]);
  const data = [];
  for (const b of st) { await b.mapAsync(GPUMapMode.READ); data.push(new Float32Array(b.getMappedRange().slice(0))); b.destroy(); }
  const [P, Q, VF] = data;
  const dt = 1 / 60 / sim.cfg.substeps, r = sim.radius;
  // grid of live skin particles to measure distance to the skin
  const g = 4 * r, grid = new Map();
  const key = (a, b, c) => `${a},${b},${c}`;
  for (let v = 0; v < sim.nCloth; v++) {
    if (sim.clothFlags[v] & 4) continue;
    const k = key(Math.floor(P[v * 4] / g), Math.floor(P[v * 4 + 1] / g), Math.floor(P[v * 4 + 2] / g));
    if (!grid.has(k)) grid.set(k, []);
    grid.get(k).push(v);
  }
  let n = 0, nearSkin = 0, maxVy = 0, at = null;
  for (let i = sim.nCloth; i < N; i++) {
    const vy = (P[i * 4 + 1] - Q[i * 4 + 1]) / dt - VF[i * 4 + 1]; // as the next substep sees it
    if (vy < 1) continue;
    n++;
    const cx = Math.floor(P[i * 4] / g), cy = Math.floor(P[i * 4 + 1] / g), cz = Math.floor(P[i * 4 + 2] / g);
    let best = 1e9;
    for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let c = -1; c <= 1; c++) {
      for (const v of grid.get(key(cx + a, cy + b, cz + c)) || []) {
        best = Math.min(best, Math.hypot(P[i * 4] - P[v * 4], P[i * 4 + 1] - P[v * 4 + 1], P[i * 4 + 2] - P[v * 4 + 2]));
      }
    }
    if (best < 3 * r) nearSkin++;
    if (vy > maxVy) { maxVy = vy; at = { y: +P[i * 4 + 1].toFixed(3), occ: P[i * 4 + 3], skinDist_r: +(best / r).toFixed(1) }; }
  }
  return { fliers: n, nearSkin, maxVy: +maxVy.toFixed(2), at };
}
