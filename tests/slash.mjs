// Mouse-like slashes across the front of the hanging bag: no cloth may fly upwards, and the cloth
// must come to rest. Prints, per second, how many cloth particles rose by > 2 cm / > 10 cm, the
// largest rise, and how many still visibly jitter (> 0.5 mm in one frame).
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

await page.goto(`http://localhost:5193/?scene=${scene}&${process.argv[3] || ''}`, { waitUntil: 'load' });
await wait(3000);
await read();
let worst = 0, worstJ = 0;
for (let t = 1; t <= 10; t++) {
  if (t === 1) console.log('slash at 50%: removed', await slash(0.5));
  if (t === 5) console.log('slash at 25%: removed', await slash(0.25));
  await wait(1000);
  const s = await read();
  worst = Math.max(worst, s.rose2);
  if (t >= 8) worstJ = Math.max(worstJ, s.jitter);
  console.log(`t=${t}s`, JSON.stringify(s));
}
console.log(`summary: most cloth particles rising > 2 cm in a second: ${worst}; jitter at the end: ${worstJ}`);
await page.screenshot({ path: `tests/out/slash-${scene}.png` });
await browser.close(); await server.close();
