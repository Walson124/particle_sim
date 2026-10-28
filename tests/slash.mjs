// Mouse-like slashes across the front of the hanging bag: no cloth may fly upwards, and the cloth
// must come to rest. Prints, per second, how many cloth particles rose by > 2 cm / > 10 cm, the
// largest rise, and how many still visibly jitter (> 0.5 mm in one frame). At the end, pieces of
// cloth cut off from the rope must not be left floating.
//   node tests/slash.mjs [scene] [extra query]
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';
const scene = process.argv[2] || 'sand';
const server = await createServer({ server: { port: 5193, strictPort: true }, logLevel: 'error' });
await server.listen();
const browser = await puppeteer.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: 'new', args: ['--enable-unsafe-webgpu'], defaultViewport: { width: 1000, height: 700 } });
const page = await browser.newPage();
page.on('pageerror', (e) => console.log('pageerror', e.message));
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

const read = () => page.evaluate(async () => {
  const sim = window.__sim, d = sim.device, N = sim.nCloth;
  const grab = async () => {
    const s = d.createBuffer({ size: N * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const e = d.createCommandEncoder(); e.copyBufferToBuffer(sim.buf.pos, 0, s, 0, N * 16); d.queue.submit([e.finish()]);
    await s.mapAsync(GPUMapMode.READ); const a = new Float32Array(s.getMappedRange().slice(0)); s.destroy(); return a;
  };
  const A = await grab();
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  const B = await grab();
  const last = window.__last, fl = sim.clothFlags;
  let rose2 = 0, rose10 = 0, maxRise = 0, jitter = 0, live = 0;
  for (let v = 0; v < N; v++) {
    if (fl[v] & 4) continue;
    live++;
    if (Math.hypot(B[v * 4] - A[v * 4], B[v * 4 + 1] - A[v * 4 + 1], B[v * 4 + 2] - A[v * 4 + 2]) > 5e-4) jitter++;
    if (last) {
      const dy = B[v * 4 + 1] - last[v * 4 + 1];
      if (dy > 0.02) rose2++;
      if (dy > 0.1) rose10++;
      maxRise = Math.max(maxRise, dy);
    }
  }
  window.__last = B;
  return { live, rose2, rose10, maxRise: +maxRise.toFixed(3), jitter };
});

// slash horizontally across the front of the bag at a fraction of its height
const slash = (frac) => page.evaluate((frac) => {
  const sim = window.__sim, cam = window.__cam, P = sim.clothPos, fl = sim.clothFlags;
  let lo = 1e9, hi = -1e9, w = 0;
  for (let v = 0; v < sim.nCloth; v++) {
    if (fl[v] & 4) continue;
    lo = Math.min(lo, P[v * 4 + 1]); hi = Math.max(hi, P[v * 4 + 1]);
    w = Math.max(w, Math.abs(P[v * 4]));
  }
  const y = lo + frac * (Math.min(hi, sim.cfg.shape.centerY + sim.cfg.shape.ay) - lo);
  const hits = [];
  for (let k = 0; k <= 40; k++) {
    const tgt = [-1.2 * w + (2.4 * w * k) / 40, y, 0], e = cam.eye;
    const d = tgt.map((x, a) => x - e[a]), l = Math.hypot(...d);
    const hit = sim.raycast(e, d.map((x) => x / l));
    if (hit) hits.push(hit);
  }
  const segs = [];
  for (let k = 1; k < hits.length; k++) segs.push([hits[k - 1], hits[k]]);
  return sim.cut(segs, sim.clothSpacing * (sim.cfg.slashWidth ?? 0.62));
}, frac);

// slash vertically down one side of the bag (at a fraction of its width, seen from the camera)
const slashV = (fx, f0, f1) => page.evaluate((fx, f0, f1) => {
  const sim = window.__sim, cam = window.__cam, P = sim.clothPos, fl = sim.clothFlags;
  let lo = 1e9, hi = -1e9, w = 0;
  for (let v = 0; v < sim.nCloth; v++) {
    if (fl[v] & 4) continue;
    lo = Math.min(lo, P[v * 4 + 1]); hi = Math.max(hi, P[v * 4 + 1]); w = Math.max(w, Math.abs(P[v * 4]));
  }
  hi = Math.min(hi, sim.cfg.shape.centerY + sim.cfg.shape.ay);
  const hits = [];
  for (let k = 0; k <= 40; k++) {
    const tgt = [fx * w, lo + (f0 + ((f1 - f0) * k) / 40) * (hi - lo), 0], e = cam.eye;
    const d = tgt.map((x, a) => x - e[a]), l = Math.hypot(...d);
    const hit = sim.raycast(e, d.map((x) => x / l));
    if (hit) hits.push(hit);
  }
  const segs = [];
  for (let k = 1; k < hits.length; k++) segs.push([hits[k - 1], hits[k]]);
  return sim.cut(segs, sim.clothSpacing * (sim.cfg.slashWidth ?? 0.62));
}, fx, f0, f1);

// pieces of cloth (connected through intact triangles) that have no path to the rope but are
// still up in the air
const floating = () => page.evaluate(async () => {
  const sim = window.__sim, d = sim.device, fl = sim.clothFlags, m = sim.mesh, E = m.numEdges, V = sim.nCloth;
  const sb = d.createBuffer({ size: E * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const e3 = d.createCommandEncoder(); e3.copyBufferToBuffer(sim.buf.edgeAlive, 0, sb, 0, E * 4); d.queue.submit([e3.finish()]);
  await sb.mapAsync(GPUMapMode.READ); const alive = new Uint32Array(sb.getMappedRange().slice(0)); sb.destroy();
  const parent = new Int32Array(V).map((_, i) => i);
  const find = (x) => { while (parent[x] !== x) x = parent[x] = parent[parent[x]]; return x; };
  const T = m.tris, TE = m.triEdges;
  for (let t = 0; t < m.numTris; t++) {
    const a = T[t * 3], b = T[t * 3 + 1], c = T[t * 3 + 2];
    if ((fl[a] | fl[b] | fl[c]) & 4 || !alive[TE[t * 4]] || !alive[TE[t * 4 + 1]] || !alive[TE[t * 4 + 2]]) continue;
    parent[find(a)] = find(b); parent[find(c)] = find(b);
  }
  const P = sim.clothPos, comp = new Map();
  for (let v = 0; v < V; v++) {
    if (fl[v] & 4) continue;
    const r = find(v);
    const c = comp.get(r) || { n: 0, pinned: false, minY: 1e9 };
    c.n++; if (fl[v] & 8) c.pinned = true; c.minY = Math.min(c.minY, P[v * 4 + 1]);
    comp.set(r, c);
  }
  const pieces = [...comp.values()];
  const float = pieces.filter((c) => !c.pinned && c.n >= 3 && c.minY > 0.05);
  return { pieces: pieces.length, floating: float.length, floatingParticles: float.reduce((s, c) => s + c.n, 0), worst: float.sort((a, b) => b.n - a.n).slice(0, 3).map((c) => [c.n, +c.minY.toFixed(2)]) };
});

await page.goto(`http://localhost:5193/?scene=${scene}&${process.argv[3] || ''}`, { waitUntil: 'load' });
await wait(3000);
await read();
let worst = 0, worstJ = 0;
for (let t = 1; t <= 11; t++) {
  if (t === 1) console.log('slash at 50%: removed', await slash(0.5));
  if (t === 5) console.log('slash at 25%: removed', await slash(0.25));
  if (t === 7) console.log('vertical slashes: removed', await slashV(0.45, 0.1, 0.8), await slashV(-0.3, 0.05, 0.6));
  await wait(1000);
  const s = await read();
  worst = Math.max(worst, s.rose2);
  if (t >= 8) worstJ = Math.max(worstJ, s.jitter);
  console.log(`t=${t}s`, JSON.stringify(s));
}
const fp = await floating();
console.log('floating pieces:', JSON.stringify(fp));
console.log(`summary: most cloth particles rising > 2 cm in a second: ${worst}; jitter at the end: ${worstJ}`);
await page.screenshot({ path: `tests/out/slash-${scene}.png` });
await browser.close(); await server.close();
