import { COMMON, KERNELS, bindingHeader, parseBinds } from './kernels.js';
import { MinHeap, buildBag, fillPairs, fillSingles, mulberry32, solveSpacing } from './mesh.js';

export const KIND_GRAIN = 0, KIND_FLUID = 1, KIND_CLOTH = 2, DEAD = 4, PINNED = 8, PAIRED = 16, FRAYED = 32;

const SU = GPUBufferUsage;

export class Simulation {
  constructor(device, cfg, targetCount) {
    this.device = device;
    this.cfg = cfg;
    this.targetCount = targetCount;
    this.time = 0;
    this.build();
    // hidden, heavily damped settling so the bag is at rest when it first appears
    this.warmup = cfg.warmup ?? 90;
    this.wakeFrames = 0;
  }

  // ---------------------------------------------------------------- scene construction
  build() {
    const cfg = this.cfg;
    const S = cfg.shape;
    const fluid = cfg.kind === 'fluid';
    const rand = mulberry32(1337);

    // lattice spacing -> particle radius
    const insetFactor = cfg.pairs ? 1.05 : 1.1;
    const s = solveSpacing(S, this.targetCount, insetFactor, cfg.fillTop);
    let r, grainPos;
    if (cfg.pairs) {
      r = s / 2.2;
      grainPos = fillPairs(S, s, insetFactor * s, cfg.fillTop, rand);
      this.pairRest = s;
    } else {
      r = fluid ? s * 0.5 : s * 0.5 * 0.985;
      grainPos = fillSingles(S, s, insetFactor * s, cfg.fillTop, fluid ? 0.02 : 0.04, rand);
      this.pairRest = 0;
    }
    this.radius = r;
    this.clothSpacing = cfg.clothSpacing * r;

    const mesh = buildBag(S, this.clothSpacing, cfg.pinAbove);
    this.mesh = mesh;
    const nCloth = mesh.numVerts;
    const nGrain = grainPos.length / 3;
    const N = nCloth + nGrain;
    this.nCloth = nCloth;
    this.nGrain = nGrain;
    this.N = N;

    // particle state
    const pos = new Float32Array(N * 4);
    const info = new Uint32Array(N);
    for (let v = 0; v < nCloth; v++) {
      pos[v * 4] = mesh.positions[v * 3];
      pos[v * 4 + 1] = mesh.positions[v * 3 + 1];
      pos[v * 4 + 2] = mesh.positions[v * 3 + 2];
      info[v] = KIND_CLOTH | (mesh.pinned[v] ? PINNED : 0);
    }
    const grainFlags = fluid ? KIND_FLUID : KIND_GRAIN | (cfg.pairs ? PAIRED : 0);
    for (let g = 0; g < nGrain; g++) {
      const i = nCloth + g;
      pos[i * 4] = grainPos[g * 3];
      pos[i * 4 + 1] = grainPos[g * 3 + 1];
      pos[i * 4 + 2] = grainPos[g * 3 + 2];
      info[i] = grainFlags;
    }
    this.clothFlags = info.slice(0, nCloth);

    // rest shape + long range attachments, interleaved: [rest.xyz, pinIndex] [pinPos.xyz, maxDist]
    const restLra = new Float32Array(nCloth * 8);
    const restU = new Uint32Array(restLra.buffer);
    for (let v = 0; v < nCloth; v++) {
      restLra[v * 8] = mesh.positions[v * 3];
      restLra[v * 8 + 1] = mesh.positions[v * 3 + 1];
      restLra[v * 8 + 2] = mesh.positions[v * 3 + 2];
      const pin = mesh.lraPin[v];
      restU[v * 8 + 3] = pin >= 0 && mesh.lraDist[v] > 0 ? pin : 0xffffffff;
      if (pin >= 0) {
        restLra[v * 8 + 4] = mesh.positions[pin * 3];
        restLra[v * 8 + 5] = mesh.positions[pin * 3 + 1];
        restLra[v * 8 + 6] = mesh.positions[pin * 3 + 2];
        restLra[v * 8 + 7] = mesh.lraDist[v];
      }
    }

    // rope anchor (top of the pinned neck)
    this.ropeTop = null;
    if (cfg.rope && cfg.pinAbove > 0) {
      let cx = 0, cz = 0, maxY = -Infinity, n = 0;
      for (let v = 0; v < nCloth; v++) {
        if (!mesh.pinned[v]) continue;
        cx += mesh.positions[v * 3];
        cz += mesh.positions[v * 3 + 2];
        maxY = Math.max(maxY, mesh.positions[v * 3 + 1]);
        n++;
      }
      if (n) this.ropeTop = [cx / n, maxY - r, cz / n];
    }

    // spatial hash
    this.cellSize = fluid ? cfg.hRatio * r : 2.05 * r;
    let H = 1 << 16;
    while (H < N * (this.cfg.hashLoad ?? 1) && H < 1 << 20) H <<= 1;
    this.H = H;
    this.numBlocks = H / 1024;

    // position based fluid constants
    this.fluidConsts = fluid ? computeFluidConstants(r, cfg) : null;

    this.restLraData = restLra;
    this.buildClothGraph(mesh);
    this.createBuffers({ pos, info, restLra, mesh });
    this.createPipelines();
  }

  createBuffers({ pos, info, restLra, mesh }) {
    const d = this.device;
    const N = this.N, H = this.H;
    const mk = (data, extra = 0) => {
      const size = Math.max(16, Math.ceil((typeof data === 'number' ? data : data.byteLength) / 4) * 4);
      const b = d.createBuffer({ size, usage: SU.STORAGE | SU.COPY_DST | SU.COPY_SRC | extra, mappedAtCreation: typeof data !== 'number' });
      if (typeof data !== 'number') {
        new Uint8Array(b.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
        b.unmap();
      }
      return b;
    };
    const edgeAlive = new Uint32Array(mesh.numEdges).fill(1);
    this.buf = {
      P: d.createBuffer({ size: 240, usage: SU.UNIFORM | SU.COPY_DST | SU.COPY_SRC }),
      pos: mk(pos),
      prev: mk(pos),
      posNew: mk(pos),
      info: mk(info),
      cellCount: mk(H * 4),
      cellStart: mk((H + 4) * 4),
      blockSums: mk(1028 * 4),
      pCellOff: mk(N * 8),
      sortedPos: mk(N * 16),
      sortedDisp: mk(N * 16),
      lambdaS: mk(N * 4),
      vfix: mk(N * 16),
      sleep: mk(this.initialSleepState(N)),
      wake: mk(N * 4),
      adjStart: mk(mesh.adjStart),
      adj: mk(mesh.adj),
      edgeAlive: mk(edgeAlive),
      restLra: mk(restLra),
      cons: mk(mesh.stretch),
      vtStart: mk(mesh.vtStart),
      vtList: mk(mesh.vtList),
      tris: mk(mesh.tris4),
      triEdges: mk(mesh.triEdges),
      normals: mk(this.nCloth * 16),
    };
    this.paramsData = new ArrayBuffer(240);
    this.paramsF = new Float32Array(this.paramsData);
    this.paramsU = new Uint32Array(this.paramsData);

    // ring of staging buffers for reading cloth positions back (mouse picking / cutting)
    this.staging = Array.from({ length: 3 }, () => ({
      buf: d.createBuffer({ size: this.nCloth * 16, usage: SU.MAP_READ | SU.COPY_DST }),
      state: 'idle',
    }));
    this.clothPos = null;
    this.pendingMap = null;
  }

  createPipelines() {
    const d = this.device;
    this.kernels = {};
    for (const [name, k] of Object.entries(KERNELS)) {
      const binds = parseBinds(this.cfg.kind !== 'fluid' && k.bindsGrain ? k.bindsGrain : k.binds);
      const layout = d.createBindGroupLayout({
        entries: binds.map((b, i) => ({
          binding: i,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: b.name === 'P' ? 'uniform' : b.mode === 'r' ? 'read-only-storage' : 'storage' },
        })),
      });
      // scene-specialised source: granular scenes compile without the fluid paths (fewer registers)
      const wg = k.wgVar ? this.cfg.contactWG ?? k.wg : k.wg;
      const head = `const FLUID_SCENE: bool = ${this.cfg.kind === 'fluid'};
const WG: u32 = ${wg}u;
`;
      const fluidScene = this.cfg.kind === 'fluid';
      const kb = !fluidScene && k.bindsGrain ? k.bindsGrain : k.binds;
      // lines tagged @fluid / @grain only exist in that kind of scene
      const kc = k.code.split('\n').filter((l) => !l.includes(fluidScene ? '// @grain' : '// @fluid')).join('\n');
      const module = d.createShaderModule({ label: name, code: head + COMMON + bindingHeader(kb) + kc });
      const pipeline = d.createComputePipeline({
        label: name,
        layout: d.createPipelineLayout({ bindGroupLayouts: [layout] }),
        compute: { module, entryPoint: 'main' },
      });
      this.kernels[name] = { pipeline, layout, binds, wg };
    }
    const bg = (name, alias = {}) => {
      const k = this.kernels[name];
      return d.createBindGroup({
        layout: k.layout,
        entries: k.binds.map((b, i) => ({ binding: i, resource: { buffer: alias[b.name] || this.buf[b.name] } })),
      });
    };
    this.bindGroups = {};
    for (const name of Object.keys(KERNELS)) if (name !== 'stretch') this.bindGroups[name] = bg(name);
    // one bind group per constraint colour, each viewing its own slice of the constraint buffer
    const sk = this.kernels.stretch;
    this.stretchColors = this.mesh.colorRanges.map(({ offset, count }) => ({
      count,
      bg: d.createBindGroup({
        layout: sk.layout,
        entries: sk.binds.map((b, i) => ({
          binding: i,
          resource: b.name === 'cons' ? { buffer: this.buf.cons, offset: offset * 16, size: count * 16 } : { buffer: this.buf[b.name] },
        })),
      }),
    }));
  }

  // ---------------------------------------------------------------- per frame
  writeParams(frameDt) {
    const c = this.cfg, f = this.paramsF, u = this.paramsU;
    const sub = c.substeps;
    const dt = frameDt / sub;
    const S = c.shape;
    const fc = this.fluidConsts;
    f[0] = dt; f[1] = this.radius; f[2] = this.cellSize; f[3] = c.gravity;
    u[4] = this.N; u[5] = this.nCloth; u[6] = this.H - 1; u[7] = this.numBlocks;
    f[8] = c.muS; f[9] = c.muK; f[10] = c.cohesion; f[11] = Math.exp(-(this.warmup > 0 ? (c.warmupDamping ?? 6) : c.damping) * dt);
    f[12] = c.clothInvMass; f[13] = 0; f[14] = 1.8; f[15] = c.sor ?? 1.4;
    f[16] = fc ? fc.h : this.cellSize;
    f[17] = fc ? fc.invRho0 : 0;
    f[18] = fc ? fc.eps : 0;
    f[19] = fc ? fc.scorrK : 0;
    f[20] = fc ? fc.invWdq : 0;
    f[21] = fc ? fc.poly6 : 0;
    f[22] = fc ? fc.spiky : 0;
    f[23] = c.viscosity || 0;
    f[24] = c.stretchK; f[25] = c.bendK; f[26] = c.compressK; f[27] = c.tear;
    f[28] = c.anchorK; f[29] = c.lraSlack; f[30] = this.pairRest; f[31] = 8.0;
    f[32] = S.ax; f[33] = S.ay; f[34] = S.az; f[35] = S.exp;
    f[36] = S.centerY; f[37] = S.neckStart; f[38] = S.neckScale; f[39] = c.pressure || 0;
    f[40] = this.heartPulse(); f[41] = this.time; f[42] = c.fluidCohesion || 0; f[43] = c.floorMu;
    f[44] = c.clothSor; f[45] = c.boundaryPsi || 1; f[46] = c.fluidOmega ?? 1; f[47] = c.fluidMaxStep || 0.5;
    f[48] = this.clothSpacing * 0.8;
    f[49] = c.contactDamp ?? 0;
    f[50] = c.maxCorr ?? (c.kind === 'fluid' ? 1 : 0.2);
    f[51] = this.wakeFrames > 0 ? 1 : 0;
    f[52] = (c.sleepTol ?? 0.3) * this.radius;
    f[53] = c.vWake ?? 0.3;
    // sleeping is a granular thing: fluids never sleep
    f[54] = c.kind === 'fluid' ? 1e9 : Math.round((c.sleepTime ?? 0.25) * 60 * sub);
    f[55] = c.vSepMax ?? 0.25;
    this.device.queue.writeBuffer(this.buf.P, 0, this.paramsData);
  }

  heartPulse() {
    if (!this.cfg.heartRate) return 0;
    const ph = (this.time * this.cfg.heartRate) % 1;
    const g = (x, m, w) => Math.exp(-(((x - m) / w) ** 2));
    return g(ph, 0.08, 0.05) + 0.55 * g(ph, 0.28, 0.06);
  }

  // `timed` (debug) wraps each kernel in its own timestamped pass; normally everything shares `pass`.
  step(pass, frameDt, timed = null) {
    if (this.warmup > 0) this.warmup--;
    else this.time += frameDt;
    if (this.wakeFrames > 0) this.wakeFrames--;
    this.writeParams(frameDt);
    const N = this.N;
    const K = this.kernels, B = this.bindGroups;
    const seg = (label, fn) => (timed ? timed(label, fn) : fn(pass));
    const run = (name, count, bg = B[name]) => seg(name, (p) => {
      const k = K[name];
      p.setPipeline(k.pipeline);
      p.setBindGroup(0, bg);
      p.dispatchWorkgroups(Math.ceil(count / k.wg));
    });
    const fluid = this.cfg.kind === 'fluid';
    const iters = Math.max(1, this.cfg.clothIters | 0);
    run('normals', this.nCloth);
    for (let s = 0; s < this.cfg.substeps; s++) {
      run('integrate', N);
      seg('scan', (p) => {
        p.setPipeline(K.scanBlocks.pipeline);
        p.setBindGroup(0, B.scanBlocks);
        p.dispatchWorkgroups(this.numBlocks);
        p.setPipeline(K.scanTop.pipeline);
        p.setBindGroup(0, B.scanTop);
        p.dispatchWorkgroups(1);
        p.setPipeline(K.scanAdd.pipeline);
        p.setBindGroup(0, B.scanAdd);
        p.dispatchWorkgroups(this.H / 256);
      });
      run('scatter', N);
      if (fluid) run('lambda', N);
      run('particles', N);
      seg('stretch', (p) => {
        const sk = K.stretch;
        p.setPipeline(sk.pipeline);
        for (let it = 0; it < iters; it++) {
          for (const c of this.stretchColors) {
            p.setBindGroup(0, c.bg);
            p.dispatchWorkgroups(Math.ceil(c.count / sk.wg));
          }
        }
      });
      run('finalize', N);
    }
    run('tear', this.buf.cons.size / 16);
    run('normals', this.nCloth);
  }

  // ---------------------------------------------------------------- cloth readback for picking
  encodeReadback(encoder) {
    const sb = this.staging.find((s) => s.state === 'idle');
    if (!sb) return;
    encoder.copyBufferToBuffer(this.buf.pos, 0, sb.buf, 0, this.nCloth * 16);
    sb.state = 'copied';
    this.pendingMap = sb;
  }

  afterSubmit() {
    const sb = this.pendingMap;
    if (!sb) return;
    this.pendingMap = null;
    sb.state = 'mapping';
    sb.buf
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        if (this.destroyed) return;
        this.clothPos = new Float32Array(sb.buf.getMappedRange().slice(0));
        sb.buf.unmap();
        sb.state = 'idle';
      })
      .catch(() => {});
  }

  // Screen-space broadphase for a batch of picks/cuts: the live cloth vertices whose projection
  // lies inside an NDC rectangle, and the triangles touching them. One O(V+T) pass per mouse
  // event instead of one per ray / per cut segment.
  broadphase(viewProj, x0, y0, x1, y1) {
    const P = this.clothPos;
    if (!P) return null;
    const m = viewProj, fl = this.clothFlags, V = this.nCloth;
    const inside = new Uint8Array(V);
    const verts = [];
    for (let v = 0; v < V; v++) {
      if (fl[v] & DEAD) continue;
      const x = P[v * 4], y = P[v * 4 + 1], z = P[v * 4 + 2];
      const w = m[3] * x + m[7] * y + m[11] * z + m[15];
      if (w <= 0) continue;
      const nx = (m[0] * x + m[4] * y + m[8] * z + m[12]) / w;
      const ny = (m[1] * x + m[5] * y + m[9] * z + m[13]) / w;
      if (nx >= x0 && nx <= x1 && ny >= y0 && ny <= y1) {
        inside[v] = 1;
        verts.push(v);
      }
    }
    const T = this.mesh.tris, tris = [];
    for (let t = 0; t < T.length; t += 3) {
      if (inside[T[t]] | inside[T[t + 1]] | inside[T[t + 2]]) tris.push(t);
    }
    return { verts, tris };
  }

  // Ray vs. cloth triangles using the latest read-back positions (optionally only the triangles of
  // a broadphase). Returns the nearest hit point or null.
  raycast(ro, rd, bp = null) {
    const P = this.clothPos;
    if (!P) return null;
    const T = this.mesh.tris, fl = this.clothFlags;
    const n = bp ? bp.tris.length : T.length / 3;
    let best = Infinity;
    for (let k = 0; k < n; k++) {
      const t = bp ? bp.tris[k] : k * 3;
      const a = T[t], b = T[t + 1], c = T[t + 2];
      if ((fl[a] | fl[b] | fl[c]) & DEAD) continue;
      const ax = P[a * 4], ay = P[a * 4 + 1], az = P[a * 4 + 2];
      const e1x = P[b * 4] - ax, e1y = P[b * 4 + 1] - ay, e1z = P[b * 4 + 2] - az;
      const e2x = P[c * 4] - ax, e2y = P[c * 4 + 1] - ay, e2z = P[c * 4 + 2] - az;
      const px = rd[1] * e2z - rd[2] * e2y, py = rd[2] * e2x - rd[0] * e2z, pz = rd[0] * e2y - rd[1] * e2x;
      const det = e1x * px + e1y * py + e1z * pz;
      if (Math.abs(det) < 1e-12) continue;
      const inv = 1 / det;
      const tx = ro[0] - ax, ty = ro[1] - ay, tz = ro[2] - az;
      const u = (tx * px + ty * py + tz * pz) * inv;
      if (u < 0 || u > 1) continue;
      const qx = ty * e1z - tz * e1y, qy = tz * e1x - tx * e1z, qz = tx * e1y - ty * e1x;
      const v = (rd[0] * qx + rd[1] * qy + rd[2] * qz) * inv;
      if (v < 0 || u + v > 1) continue;
      const dist = (e2x * qx + e2y * qy + e2z * qz) * inv;
      if (dist > 1e-4 && dist < best) best = dist;
    }
    return best < Infinity ? [ro[0] + rd[0] * best, ro[1] + rd[1] * best, ro[2] + rd[2] * best] : null;
  }

  // Remove every cloth particle within `radius` of any of the segments [[p0, p1], ...]
  // (optionally only among a broadphase's vertices). One upload per call. Returns number removed.
  cut(segments, radius, bp = null) {
    const P = this.clothPos;
    if (!P || !segments.length) return 0;
    const fl = this.clothFlags;
    const r2 = radius * radius;
    const killed = [];
    const n = bp ? bp.verts.length : this.nCloth;
    for (let k = 0; k < n; k++) {
      const v = bp ? bp.verts[k] : k;
      if (fl[v] & DEAD) continue;
      const x = P[v * 4], y = P[v * 4 + 1], z = P[v * 4 + 2];
      for (const [p0, p1] of segments) {
        const dx = p1[0] - p0[0], dy = p1[1] - p0[1], dz = p1[2] - p0[2];
        const ll = dx * dx + dy * dy + dz * dz;
        const px = x - p0[0], py = y - p0[1], pz = z - p0[2];
        const t = ll > 0 ? Math.min(Math.max((px * dx + py * dy + pz * dz) / ll, 0), 1) : 0;
        const ex = px - dx * t, ey = py - dy * t, ez = pz - dz * t;
        if (ex * ex + ey * ey + ez * ez < r2) {
          fl[v] |= DEAD;
          killed.push(v);
          break;
        }
      }
    }
    if (!killed.length) return 0;
    // fray the cloth around the wound (in rest space, so it stays attached to the material):
    // only frayed edges can tear, so a cut can rip open further but intact cloth never does
    const R = this.mesh.positions, fr = this.cfg.frayRadius ?? 0.035, fr2 = fr * fr;
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const k of killed) {
      for (let a = 0; a < 3; a++) {
        lo[a] = Math.min(lo[a], R[k * 3 + a] - fr);
        hi[a] = Math.max(hi[a], R[k * 3 + a] + fr);
      }
    }
    for (let v = 0; v < this.nCloth; v++) {
      if (fl[v] & (DEAD | FRAYED)) continue;
      const x = R[v * 3], y = R[v * 3 + 1], z = R[v * 3 + 2];
      if (x < lo[0] || x > hi[0] || y < lo[1] || y > hi[1] || z < lo[2] || z > hi[2]) continue;
      for (const k of killed) {
        const ax = x - R[k * 3], ay = y - R[k * 3 + 1], az = z - R[k * 3 + 2];
        if (ax * ax + ay * ay + az * az < fr2) { fl[v] |= FRAYED; break; }
      }
    }
    this.device.queue.writeBuffer(this.buf.info, 0, fl);
    this.wakeFrames = 3; // whatever rested on the removed cloth must feel gravity again
    this.attachmentsDirty = true;
    return killed.length;
  }

  // Let go of the rope: unpin everything.
  release() {
    const fl = this.clothFlags;
    for (let v = 0; v < this.nCloth; v++) fl[v] &= ~PINNED;
    this.device.queue.writeBuffer(this.buf.info, 0, fl);
    this.ropeTop = null;
    this.wakeFrames = 3;
  }

  // every particle starts awake and unsupported (63 = no skin hops to support)
  initialSleepState(N) {
    const s = new Uint32Array(N * 4);
    for (let i = 0; i < N; i++) s[i * 4 + 3] = 63 << 18;
    return s;
  }

  // Cloth connectivity (stretch edges) as CSR, for re-routing the long range attachments.
  buildClothGraph(mesh) {
    const C = mesh.stretch, cf = new Float32Array(C.buffer), V = mesh.numVerts;
    const deg = new Uint32Array(V + 1);
    for (let k = 0; k < C.length; k += 4) if (C[k] !== 0xffffffff) { deg[C[k]]++; deg[C[k + 1]]++; }
    const start = new Uint32Array(V + 1);
    for (let v = 0; v < V; v++) start[v + 1] = start[v] + deg[v];
    const nbr = new Uint32Array(start[V]), len = new Float32Array(start[V]), edge = new Uint32Array(start[V]);
    const fill = start.slice(0, V);
    for (let k = 0; k < C.length; k += 4) {
      const a = C[k], b = C[k + 1];
      if (a === 0xffffffff) continue;
      for (const [i, j] of [[a, b], [b, a]]) {
        const s = fill[i]++;
        nbr[s] = j;
        len[s] = cf[k + 2];
        edge[s] = C[k + 3];
      }
    }
    this.graph = { start, nbr, len, edge };
  }

  // Recompute the long range attachments over the cloth that is still connected (cuts and GPU
  // tears included): a piece that has been cut loose from the rope loses them and falls.
  // `force`: after a cut. Otherwise (periodic check) only re-route when new tears appeared.
  async refreshAttachments(force = true) {
    if (this.refreshing) { this.refreshAgain ||= force; return; }
    this.refreshing = true;
    this.attachmentsDirty = false;
    const d = this.device, E = this.mesh.numEdges;
    const st = d.createBuffer({ size: E * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(this.buf.edgeAlive, 0, st, 0, E * 4);
    d.queue.submit([enc.finish()]);
    try {
      await st.mapAsync(GPUMapMode.READ);
    } catch {
      this.refreshing = false;
      return;
    }
    const alive = new Uint32Array(st.getMappedRange().slice(0));
    st.destroy();
    if (this.destroyed) return;
    const torn = alive.reduce((n, a) => n + (a ? 0 : 1), 0);
    if (!force && torn === this.lastTorn) {
      this.refreshing = false;
      return;
    }
    this.lastTorn = torn;

    const V = this.nCloth, fl = this.clothFlags, { start, nbr, len, edge } = this.graph;
    const dist = new Float64Array(V).fill(Infinity);
    const pin = new Int32Array(V).fill(-1);
    const heap = new MinHeap();
    for (let v = 0; v < V; v++) {
      if ((fl[v] & (PINNED | DEAD)) === PINNED) { dist[v] = 0; pin[v] = v; heap.push(0, v); }
    }
    while (heap.size) {
      const [dv, v] = heap.pop();
      if (dv > dist[v]) continue;
      for (let s = start[v]; s < start[v + 1]; s++) {
        const u = nbr[s];
        if (fl[u] & DEAD || !alive[edge[s]]) continue;
        const nd = dv + len[s];
        if (nd < dist[u]) { dist[u] = nd; pin[u] = pin[v]; heap.push(nd, u); }
      }
    }
    const R = this.restLraData, RU = new Uint32Array(R.buffer), M = this.mesh.positions;
    let detached = 0;
    for (let v = 0; v < V; v++) {
      const p = pin[v];
      if (p >= 0 && dist[v] > 0) {
        RU[v * 8 + 3] = p;
        R[v * 8 + 4] = M[p * 3];
        R[v * 8 + 5] = M[p * 3 + 1];
        R[v * 8 + 6] = M[p * 3 + 2];
        R[v * 8 + 7] = dist[v];
      } else {
        if (RU[v * 8 + 3] !== 0xffffffff && !(fl[v] & DEAD)) detached++;
        RU[v * 8 + 3] = 0xffffffff;
      }
    }
    d.queue.writeBuffer(this.buf.restLra, 0, R);
    if (detached) this.wakeFrames = 3;
    this.refreshing = false;
    if (this.refreshAgain) {
      this.refreshAgain = false;
      this.refreshAttachments();
    }
  }

  get numPinned() {
    let n = 0;
    for (let v = 0; v < this.nCloth; v++) if ((this.clothFlags[v] & (PINNED | DEAD)) === PINNED) n++;
    return n;
  }

  destroy() {
    this.destroyed = true;
    for (const b of Object.values(this.buf)) b.destroy();
    for (const s of this.staging) s.buf.destroy();
  }
}

function computeFluidConstants(r, cfg) {
  const d = 2 * r;
  const h = cfg.hRatio * r;
  const poly6 = 315 / (64 * Math.PI * h ** 3);
  const spiky = 45 / (Math.PI * h ** 4);
  let rho0 = 0, g2 = 0;
  const R = Math.ceil(h / d);
  for (let z = -R; z <= R; z++) {
    for (let y = -R; y <= R; y++) {
      for (let x = -R; x <= R; x++) {
        const r2 = (x * x + y * y + z * z) * d * d;
        if (r2 >= h * h) continue;
        const q = 1 - r2 / (h * h);
        rho0 += poly6 * q * q * q;
        if (r2 > 0) {
          const t = 1 - Math.sqrt(r2) / h;
          const g = spiky * t * t;
          g2 += g * g;
        }
      }
    }
  }
  const invRho0 = 1 / rho0;
  const S0 = g2 * invRho0 * invRho0;
  const dq = 0.3;
  return {
    h,
    poly6,
    spiky,
    invRho0,
    eps: cfg.relax * S0,
    scorrK: cfg.scorr / S0,
    invWdq: 1 / (1 - dq * dq) ** 3,
  };
}
