// Slash the bag open, let it pour, and track whether the spilled pile comes to rest and how much
// each frame costs while it does. Every few seconds prints, for the grains: the share asleep, how
// many move visibly (> 1 cm/s and > 5 cm/s), the 99th percentile speed, and the GPU ms per step.
//   node tests/settle.mjs [scene] [n] [secs] [extra query]
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';
const [scene = 'sand', n = '64000', secs = '30', extra = ''] = process.argv.slice(2);
const server = await createServer({ server: { port: 5194, strictPort: true }, logLevel: 'error' });
await server.listen();
const browser = await puppeteer.launch({
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  headless: 'new',
  args: ['--enable-unsafe-webgpu', '--enable-gpu', '--ignore-gpu-blocklist'],
  defaultViewport: { width: 1000, height: 700 },
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.log('pageerror', e.message));
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

const stats = () => page.evaluate(async () => {
  const sim = window.__sim, d = sim.device, N = sim.N;
  const enc = d.createCommandEncoder();
  const bufs = [sim.buf.pos, sim.buf.prev, sim.buf.sleep, sim.buf.vfix].map((b) => {
    const s = d.createBuffer({ size: N * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyBufferToBuffer(b, 0, s, 0, N * 16);
    return s;
  });
  d.queue.submit([enc.finish()]);
  const a = [];
  for (const s of bufs) { await s.mapAsync(GPUMapMode.READ); a.push(new Float32Array(s.getMappedRange().slice(0))); s.destroy(); }
  const [P, Q, SF, VF] = a, SL = new Uint32Array(SF.buffer), dt = 1 / 60 / sim.cfg.substeps;
  const sp = [];
  // where the moving grains are: still in the bag (above the pile), in the pile, or touching skin
  const r = sim.radius, g = 4 * r, grid = new Set();
  for (let v = 0; v < sim.nCloth; v++) if (!(sim.clothFlags[v] & 4)) grid.add(`${Math.floor(P[v * 4] / g)},${Math.floor(P[v * 4 + 1] / g)},${Math.floor(P[v * 4 + 2] / g)}`);
  const nearSkin = (i) => {
    const cx = Math.floor(P[i * 4] / g), cy = Math.floor(P[i * 4 + 1] / g), cz = Math.floor(P[i * 4 + 2] / g);
    for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let c = -1; c <= 1; c++) if (grid.has(`${cx + a},${cy + b},${cz + c}`)) return true;
    return false;
  };
  let asleep = 0, v1 = 0, v5 = 0, out = 0, inBag = 0, bySkin = 0;
  for (let i = sim.nCloth; i < N; i++) {
    const v = Math.hypot((P[i * 4] - Q[i * 4]) / dt - VF[i * 4], (P[i * 4 + 1] - Q[i * 4 + 1]) / dt - VF[i * 4 + 1], (P[i * 4 + 2] - Q[i * 4 + 2]) / dt - VF[i * 4 + 2]);
    sp.push(v);
    if (SL[i * 4 + 3] & 0x20000) asleep++;
    if (v > 0.01) {
      v1++;
      if (P[i * 4 + 1] > 0.3) inBag++;
      else if (nearSkin(i)) bySkin++;
    }
    if (v > 0.05) v5++;
    if (P[i * 4 + 1] < 0.3) out++;
  }
  sp.sort((x, y) => x - y);
  const G = N - sim.nCloth;
  return { spilled: out, asleep: +(asleep / G).toFixed(3), 'v>1cm/s': v1, inBag, bySkin, 'v>5cm/s': v5, v99: +sp[Math.floor(0.99 * (G - 1))].toFixed(4), ...(await window.__bench(20)) };
});

await page.goto(`http://localhost:5194/?scene=${scene}&n=${n}&${extra}`, { waitUntil: 'load' });
await wait(3000);
console.log('intact, 3 s after load', JSON.stringify(await stats()));
await wait(5000);
console.log('intact, 8 s after load', JSON.stringify(await stats()));
// a long horizontal slash across the lower front of the bag
console.log('slash removed', await page.evaluate(() => {
  const sim = window.__sim, cam = window.__cam, P = sim.clothPos;
  let lo = 1e9, hi = -1e9, w = 0;
  for (let v = 0; v < sim.nCloth; v++) { lo = Math.min(lo, P[v * 4 + 1]); hi = Math.max(hi, P[v * 4 + 1]); w = Math.max(w, Math.abs(P[v * 4])); }
  const y = lo + 0.3 * (Math.min(hi, sim.cfg.shape.centerY + sim.cfg.shape.ay) - lo);
  const hits = [];
  for (let k = 0; k <= 40; k++) {
    const tgt = [-1.2 * w + (2.4 * w * k) / 40, y, 0], e = cam.eye;
    const dd = tgt.map((x, ax) => x - e[ax]), l = Math.hypot(...dd);
    const hit = sim.raycast(e, dd.map((x) => x / l));
    if (hit) hits.push(hit);
  }
  const segs = [];
  for (let k = 1; k < hits.length; k++) segs.push([hits[k - 1], hits[k]]);
  return sim.cut(segs, sim.clothSpacing * (sim.cfg.slashWidth ?? 0.62));
}));
for (let t = 3; t <= Number(secs); t += 3) {
  await wait(3000);
  console.log(`t=${t}s`, JSON.stringify(await stats()));
}
if (process.env.KERNELS) console.log('kernels (ms per frame)', JSON.stringify(await page.evaluate(() => window.__sim.benchKernels())));
// VARY="key:a/b/c,..." re-measures the settled scene with live config changes (restored after)
for (const kv of (process.env.VARY || '').split(',').filter(Boolean)) {
  const [k, vals] = kv.split(':');
  for (const v of vals.split('/')) {
    console.log(`  ${k}=${v}`, JSON.stringify(await page.evaluate(async (k, v) => {
      const c = window.__sim.cfg, old = c[k];
      c[k] = Number(v);
      const b = await window.__bench(30);
      c[k] = old;
      return b;
    }, k, v)));
  }
}
await page.screenshot({ path: `tests/out/settle-${scene}.png` });
await browser.close(); await server.close();
