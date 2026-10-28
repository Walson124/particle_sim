// CPU-side scene construction: the bag skin (ring mesh on a necked superellipsoid) and
// the particle fill. Everything here runs once per (re)build.

const smoothstep = (a, b, x) => {
  const t = Math.min(Math.max((x - a) / (b - a), 0), 1);
  return t * t * (3 - 2 * t);
};

export function neckScale(yn, S) {
  return 1 + (S.neckScale - 1) * smoothstep(S.neckStart, 1, yn);
}

// Inside test for the fill region, shrunk by `inset` metres.
function insideShape(x, y, z, S, inset) {
  const s = neckScale(y / S.ay, S);
  const wx = S.ax * s - inset, wy = S.ay - inset, wz = S.az * s - inset;
  if (wx <= 0 || wy <= 0 || wz <= 0) return false;
  const e = S.exp;
  return Math.abs(x / wx) ** e + Math.abs(y / wy) ** e + Math.abs(z / wz) ** e < 1;
}

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Half-widths of the horizontal cross-section at normalised height yn.
function crossSection(yn, S) {
  const e = S.exp;
  const k = Math.pow(Math.max(0, 1 - Math.abs(yn) ** e), 1 / e);
  const s = neckScale(yn, S);
  return [S.ax * k * s, S.az * k * s];
}

// Points on a superellipse (half-widths wx, wz) at equal arc length; u = arc fraction from +x.
function ringPoints(wx, wz, e, spacing, phase) {
  const SAMPLES = 1024;
  const xs = new Float64Array(SAMPLES + 1), zs = new Float64Array(SAMPLES + 1), L = new Float64Array(SAMPLES + 1);
  for (let i = 0; i <= SAMPLES; i++) {
    const t = (i / SAMPLES) * Math.PI * 2;
    const c = Math.cos(t), s = Math.sin(t);
    xs[i] = wx * Math.sign(c) * Math.abs(c) ** (2 / e);
    zs[i] = wz * Math.sign(s) * Math.abs(s) ** (2 / e);
    if (i) L[i] = L[i - 1] + Math.hypot(xs[i] - xs[i - 1], zs[i] - zs[i - 1]);
  }
  const perim = L[SAMPLES];
  const m = Math.max(3, Math.round(perim / spacing));
  const pts = [];
  let j = 0;
  for (let k = 0; k < m; k++) {
    const target = (perim * (k + phase)) / m;
    while (j < SAMPLES - 1 && L[j + 1] < target) j++;
    const f = (target - L[j]) / Math.max(L[j + 1] - L[j], 1e-12);
    pts.push([xs[j] + (xs[j + 1] - xs[j]) * f, zs[j] + (zs[j + 1] - zs[j]) * f, (k + phase) / m]);
  }
  return pts;
}

// Triangulate the band between two rings (vertex ids + arc fractions), "zipper" style.
function zip(A, uA, B, uB, tris) {
  const m = A.length, n = B.length;
  if (m === 1) { for (let j = 0; j < n; j++) tris.push(A[0], B[j], B[(j + 1) % n]); return; }
  if (n === 1) { for (let i = 0; i < m; i++) tris.push(A[i], A[(i + 1) % m], B[0]); return; }
  let i = 0, j = 0;
  while (i < m || j < n) {
    const ua = i < m ? (i + 1 < m ? uA[i + 1] : 1 + uA[0]) : Infinity;
    const ub = j < n ? (j + 1 < n ? uB[j + 1] : 1 + uB[0]) : Infinity;
    if (ua <= ub) { tris.push(A[i % m], A[(i + 1) % m], B[j % n]); i++; }
    else { tris.push(A[i % m], B[(j + 1) % n], B[j % n]); j++; }
  }
}

function implicitF(x, y, z, S) {
  const yn = y / S.ay;
  const s = neckScale(yn, S);
  const e = S.exp;
  return (Math.abs(x / (S.ax * s)) ** e + Math.abs(yn) ** e + Math.abs(z / (S.az * s)) ** e) ** (1 / e);
}

// Ring mesh: rings spaced evenly in surface arc length up the profile, each ring sampled at
// equal arc length -> near-equilateral triangles with edge ~= spacing everywhere.
function genMesh(S, spacing) {
  // sample the profile with the superellipse angle parameter so flat caps are resolved too
  const M = 6000;
  const L = new Float64Array(M + 1);
  const YN = new Float64Array(M + 1);
  const profYn = (i) => {
    const s = Math.sin(-Math.PI / 2 + (Math.PI * i) / M);
    return Math.sign(s) * Math.abs(s) ** (2 / S.exp);
  };
  YN[0] = -1;
  let prev = crossSection(-1, S);
  for (let i = 1; i <= M; i++) {
    YN[i] = profYn(i);
    const cur = crossSection(YN[i], S);
    const dy = (YN[i] - YN[i - 1]) * S.ay;
    L[i] = L[i - 1] + Math.max(Math.hypot(dy, cur[0] - prev[0]), Math.hypot(dy, cur[1] - prev[1]));
    prev = cur;
  }
  const nRings = Math.max(2, Math.round(L[M] / (spacing * 0.866)));
  const positions = [];
  const tris = [];
  let prevIds = null, prevU = null;
  let j = 0;
  for (let r = 0; r <= nRings; r++) {
    const target = (L[M] * r) / nRings;
    while (j < M - 1 && L[j + 1] < target) j++;
    const f = (target - L[j]) / Math.max(L[j + 1] - L[j], 1e-12);
    const yn = Math.min(1, YN[j] + (YN[j + 1] - YN[j]) * f);
    const y = yn * S.ay + S.centerY;
    const ids = [], us = [];
    if (r === 0 || r === nRings) {
      ids.push(positions.length / 3);
      us.push(0);
      positions.push(0, y, 0);
    } else {
      const [wx, wz] = crossSection(yn, S);
      for (const [x, z, u] of ringPoints(wx, wz, S.exp, spacing, (r & 1) * 0.5)) {
        ids.push(positions.length / 3);
        us.push(u);
        positions.push(x, y, z);
      }
    }
    if (prevIds) zip(prevIds, prevU, ids, us, tris);
    prevIds = ids;
    prevU = us;
  }

  // orient every triangle along the implicit surface gradient (outward)
  const P = positions;
  for (let t = 0; t < tris.length; t += 3) {
    const a = tris[t] * 3, b = tris[t + 1] * 3, c = tris[t + 2] * 3;
    const e1 = [P[b] - P[a], P[b + 1] - P[a + 1], P[b + 2] - P[a + 2]];
    const e2 = [P[c] - P[a], P[c + 1] - P[a + 1], P[c + 2] - P[a + 2]];
    const nx = e1[1] * e2[2] - e1[2] * e2[1], ny = e1[2] * e2[0] - e1[0] * e2[2], nz = e1[0] * e2[1] - e1[1] * e2[0];
    const cx = (P[a] + P[b] + P[c]) / 3, cy = (P[a + 1] + P[b + 1] + P[c + 1]) / 3 - S.centerY, cz = (P[a + 2] + P[b + 2] + P[c + 2]) / 3;
    const h = 1e-4;
    const gx = implicitF(cx + h, cy, cz, S) - implicitF(cx - h, cy, cz, S);
    const gy = implicitF(cx, cy + h, cz, S) - implicitF(cx, cy - h, cz, S);
    const gz = implicitF(cx, cy, cz + h, S) - implicitF(cx, cy, cz - h, S);
    if (nx * gx + ny * gy + nz * gz < 0) {
      const tmp = tris[t + 1];
      tris[t + 1] = tris[t + 2];
      tris[t + 2] = tmp;
    }
  }

  let maxEdge = 0;
  for (let t = 0; t < tris.length; t += 3) {
    for (let k = 0; k < 3; k++) {
      const a = tris[t + k] * 3, b = tris[t + ((k + 1) % 3)] * 3;
      maxEdge = Math.max(maxEdge, Math.hypot(P[a] - P[b], P[a + 1] - P[b + 1], P[a + 2] - P[b + 2]));
    }
  }
  return { positions: new Float32Array(positions), tris: new Uint32Array(tris), maxEdge };
}

const dist3 = (P, a, b) =>
  Math.hypot(P[a * 3] - P[b * 3], P[a * 3 + 1] - P[b * 3 + 1], P[a * 3 + 2] - P[b * 3 + 2]);

// Build the cloth skin with max edge length <= spacing, plus all constraint topology.
export function buildBag(S, spacing, pinAbove) {
  let h = spacing * 0.92;
  let m = genMesh(S, h);
  for (let it = 0; it < 4 && m.maxEdge > spacing * 1.05; it++) {
    h *= (spacing * 1.05) / m.maxEdge;
    m = genMesh(S, h);
  }
  const { positions, tris } = m;
  const V = positions.length / 3;
  const T = tris.length / 3;

  // Unique edges + triangle->edge table.
  const edgeMap = new Map();
  const edgeA = [], edgeB = [], edgeTris = [];
  const triEdges = new Uint32Array(T * 4);
  for (let t = 0; t < T; t++) {
    for (let k = 0; k < 3; k++) {
      const a = tris[t * 3 + k], b = tris[t * 3 + ((k + 1) % 3)];
      const lo = Math.min(a, b), hi = Math.max(a, b);
      const key = lo * V + hi;
      let id = edgeMap.get(key);
      if (id === undefined) {
        id = edgeA.length;
        edgeMap.set(key, id);
        edgeA.push(lo);
        edgeB.push(hi);
        edgeTris.push([]);
      }
      edgeTris[id].push(t);
      triEdges[t * 4 + k] = id;
    }
  }
  const E = edgeA.length;

  // Stretch constraints, greedily edge-coloured so every colour can be projected in parallel
  // (Gauss-Seidel across colours). Each colour segment is padded to 16 records (256 bytes) so
  // it can be bound at a storage-buffer offset.
  const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
  const vColors = new Uint32Array(V);
  const color = new Int32Array(E);
  let numColors = 0;
  for (let e = 0; e < E; e++) {
    const used = vColors[edgeA[e]] | vColors[edgeB[e]];
    let c = 0;
    while (c < 31 && used & (1 << c)) c++;
    color[e] = c;
    vColors[edgeA[e]] |= 1 << c;
    vColors[edgeB[e]] |= 1 << c;
    numColors = Math.max(numColors, c + 1);
  }
  const colorCount = new Uint32Array(numColors);
  for (let e = 0; e < E; e++) colorCount[color[e]]++;
  const colorRanges = [];
  let total = 0;
  for (let c = 0; c < numColors; c++) {
    const padded = Math.ceil(colorCount[c] / 16) * 16;
    colorRanges.push({ offset: total, count: padded });
    total += padded;
  }
  const stretch = new Uint32Array(total * 4).fill(0xffffffff);
  const fill = colorRanges.map((r) => r.offset);
  for (let e = 0; e < E; e++) {
    const slot = fill[color[e]]++;
    f32[0] = dist3(positions, edgeA[e], edgeB[e]);
    stretch[slot * 4] = edgeA[e];
    stretch[slot * 4 + 1] = edgeB[e];
    stretch[slot * 4 + 2] = u32[0];
    stretch[slot * 4 + 3] = e;
  }

  // Bending constraints: [a, b, rest, edgeId, wings] across each shared edge (Jacobi, soft).
  const cons = [];
  for (let e = 0; e < E; e++) {
    const ts = edgeTris[e];
    if (ts.length !== 2) continue;
    const opp = ts.map((t) => {
      for (let k = 0; k < 3; k++) {
        const v = tris[t * 3 + k];
        if (v !== edgeA[e] && v !== edgeB[e]) return v;
      }
      return -1;
    });
    if (opp[0] < 0 || opp[1] < 0 || opp[0] === opp[1]) continue;
    // the other four edges of the two triangles: the spring only acts while both are intact
    const wings = [];
    for (const t of ts) for (let k = 0; k < 3; k++) if (triEdges[t * 4 + k] !== e) wings.push(triEdges[t * 4 + k]);
    cons.push([opp[0], opp[1], dist3(positions, opp[0], opp[1]), e, wings]);
  }

  // Per-vertex constraint lists (CSR) so the GPU solves each vertex without atomics.
  const deg = new Uint32Array(V);
  for (const c of cons) { deg[c[0]]++; deg[c[1]]++; }
  const adjStart = new Uint32Array(V * 2);
  let off = 0;
  for (let v = 0; v < V; v++) { adjStart[v * 2] = off; adjStart[v * 2 + 1] = 0; off += deg[v]; }
  // two vec4u per entry: [neighbour, shared edge, rest, -], [4 wing edges]
  const adj = new Uint32Array(Math.max(off, 1) * 8);
  for (const [a, b, rest, e, wings] of cons) {
    f32[0] = rest;
    for (const [i, j] of [[a, b], [b, a]]) {
      const slot = adjStart[i * 2] + adjStart[i * 2 + 1]++;
      adj[slot * 8] = j;
      adj[slot * 8 + 1] = e;
      adj[slot * 8 + 2] = u32[0];
      for (let k = 0; k < 4; k++) adj[slot * 8 + 4 + k] = wings[k];
    }
  }

  // Vertex -> triangle lists for normal accumulation.
  const vdeg = new Uint32Array(V);
  for (let i = 0; i < tris.length; i++) vdeg[tris[i]]++;
  const vtStart = new Uint32Array(V * 2);
  off = 0;
  for (let v = 0; v < V; v++) { vtStart[v * 2] = off; off += vdeg[v]; }
  const vtList = new Uint32Array(off);
  for (let t = 0; t < T; t++) {
    for (let k = 0; k < 3; k++) {
      const v = tris[t * 3 + k];
      vtList[vtStart[v * 2] + vtStart[v * 2 + 1]++] = t;
    }
  }

  // the (up to two) triangles of every edge, for pruning edges no intact triangle holds any more
  const edgeTris2 = new Uint32Array(E * 2).fill(0xffffffff);
  for (let e = 0; e < E; e++) for (let k = 0; k < Math.min(edgeTris[e].length, 2); k++) edgeTris2[e * 2 + k] = edgeTris[e][k];
  const tris4 = new Uint32Array(T * 4);
  for (let t = 0; t < T; t++) {
    tris4[t * 4] = tris[t * 3];
    tris4[t * 4 + 1] = tris[t * 3 + 1];
    tris4[t * 4 + 2] = tris[t * 3 + 2];
  }

  // Pins + long range attachments (geodesic distance to the nearest pin, via Dijkstra).
  const pinned = new Uint8Array(V);
  const lraDist = new Float32Array(V).fill(-1);
  const lraPin = new Int32Array(V).fill(-1);
  if (pinAbove > 0) {
    for (let v = 0; v < V; v++) {
      if ((positions[v * 3 + 1] - S.centerY) / S.ay > pinAbove) pinned[v] = 1;
    }
    const nbr = Array.from({ length: V }, () => []);
    for (let e = 0; e < E; e++) {
      const d = dist3(positions, edgeA[e], edgeB[e]);
      nbr[edgeA[e]].push(edgeB[e], d);
      nbr[edgeB[e]].push(edgeA[e], d);
    }
    const dist = new Float64Array(V).fill(Infinity);
    const heap = new MinHeap();
    for (let v = 0; v < V; v++) {
      if (pinned[v]) { dist[v] = 0; lraPin[v] = v; heap.push(0, v); }
    }
    while (heap.size) {
      const [d, v] = heap.pop();
      if (d > dist[v]) continue;
      const nb = nbr[v];
      for (let k = 0; k < nb.length; k += 2) {
        const u = nb[k], nd = d + nb[k + 1];
        if (nd < dist[u]) { dist[u] = nd; lraPin[u] = lraPin[v]; heap.push(nd, u); }
      }
    }
    for (let v = 0; v < V; v++) if (!pinned[v] && isFinite(dist[v])) lraDist[v] = dist[v];
  }

  return {
    numVerts: V, numTris: T, numEdges: E,
    positions, tris, tris4, triEdges, edgeTris: edgeTris2, adjStart, adj, vtStart, vtList, stretch, colorRanges,
    pinned, lraDist, lraPin, maxEdge: m.maxEdge,
  };
}

export class MinHeap {
  constructor() { this.k = []; this.v = []; }
  get size() { return this.k.length; }
  push(key, val) {
    const k = this.k, v = this.v;
    let i = k.length;
    k.push(key); v.push(val);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (k[p] <= k[i]) break;
      [k[p], k[i]] = [k[i], k[p]];
      [v[p], v[i]] = [v[i], v[p]];
      i = p;
    }
  }
  pop() {
    const k = this.k, v = this.v;
    const out = [k[0], v[0]];
    const lk = k.pop(), lv = v.pop();
    if (k.length) {
      k[0] = lk; v[0] = lv;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < k.length && k[l] < k[m]) m = l;
        if (r < k.length && k[r] < k[m]) m = r;
        if (m === i) break;
        [k[m], k[i]] = [k[i], k[m]];
        [v[m], v[i]] = [v[i], v[m]];
        i = m;
      }
    }
    return out;
  }
}

// Lattice sites (relative to the bag centre) inside the fill region.
function latticeSites(S, s, inset, fillTop) {
  const sites = [];
  const nx = Math.floor(S.ax / s), ny = Math.floor(S.ay / s), nz = Math.floor(S.az / s);
  const yMax = fillTop * S.ay;
  for (let iy = -ny; iy <= ny; iy++) {
    const y = iy * s;
    if (y > yMax) continue;
    for (let iz = -nz; iz <= nz; iz++) {
      for (let ix = -nx; ix <= nx; ix++) {
        if (insideShape(ix * s, y, iz * s, S, inset)) sites.push(ix, iy, iz);
      }
    }
  }
  return sites;
}

// Find the lattice spacing that yields ~targetCount sites. insetFactor is in units of spacing.
export function solveSpacing(S, targetCount, insetFactor, fillTop) {
  const volGuess = (4 / 3) * Math.PI * S.ax * S.ay * S.az * 0.5;
  let s = Math.cbrt(volGuess / targetCount);
  for (let it = 0; it < 5; it++) {
    const count = latticeSites(S, s, insetFactor * s, fillTop).length / 3;
    if (!count) { s *= 0.7; continue; }
    s *= Math.cbrt(count / targetCount);
  }
  return s;
}

// Single particles on a jittered lattice.
export function fillSingles(S, s, inset, fillTop, jitter, rand) {
  const sites = latticeSites(S, s, inset, fillTop);
  const n = sites.length / 3;
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    out[i * 3] = sites[i * 3] * s + (rand() - 0.5) * jitter * s;
    out[i * 3 + 1] = sites[i * 3 + 1] * s + S.centerY + (rand() - 0.5) * jitter * s;
    out[i * 3 + 2] = sites[i * 3 + 2] * s + (rand() - 0.5) * jitter * s;
  }
  return out;
}

// Rigid pairs ("dumbbells") from a random domino tiling of the lattice -> randomly oriented rice.
export function fillPairs(S, s, inset, fillTop, rand) {
  const sites = latticeSites(S, s, inset, fillTop);
  const n = sites.length / 3;
  const index = new Map();
  const key = (x, y, z) => `${x},${y},${z}`;
  for (let i = 0; i < n; i++) index.set(key(sites[i * 3], sites[i * 3 + 1], sites[i * 3 + 2]), i);
  const order = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  const partner = new Int32Array(n).fill(-1);
  const dirs = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
  const pairs = [];
  for (const i of order) {
    if (partner[i] >= 0) continue;
    const start = Math.floor(rand() * 6);
    for (let d = 0; d < 6; d++) {
      const D = dirs[(start + d) % 6];
      const j = index.get(key(sites[i * 3] + D[0], sites[i * 3 + 1] + D[1], sites[i * 3 + 2] + D[2]));
      if (j !== undefined && partner[j] < 0) {
        partner[i] = j;
        partner[j] = i;
        pairs.push(i, j);
        break;
      }
    }
  }
  const out = new Float32Array(pairs.length * 3);
  for (let k = 0; k < pairs.length; k++) {
    const i = pairs[k];
    out[k * 3] = sites[i * 3] * s;
    out[k * 3 + 1] = sites[i * 3 + 1] * s + S.centerY;
    out[k * 3 + 2] = sites[i * 3 + 2] * s;
  }
  return out;
}
