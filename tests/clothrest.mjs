// Cut-off cloth landing on sand must come to rest and never shoot upwards.
// jitter: particles that moved > 0.5 mm within ~one frame; crawl: > 3 mm over the last second;
// rose: > 1 cm upwards over the last second. moving/fast/upward use the last substep's velocity.
// Punctures the bottom of the sandbag, lets it pour onto a pile, then cuts a thin ring through the
// middle so the lower piece drops onto the sand. Prints cloth motion statistics over time.
//   node tests/clothrest.mjs [scene] [extra query] [pour seconds]
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';
const scene = process.argv[2] || 'sand';
const pourMs = (+process.argv[4] || 3) * 1000;
const server = await createServer({ server: { port: 5190, strictPort: true }, logLevel: 'error' });
await server.listen();
const browser = await puppeteer.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: 'new', args: ['--enable-unsafe-webgpu'], defaultViewport: { width: 1000, height: 700 } });
const page = await browser.newPage();
page.on('pageerror', (e) => console.log('pageerror', e.message));
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

const probe = () => page.evaluate(async () => {
  const sim = window.__sim, d = sim.device, N = sim.nCloth;
  const enc = d.createCommandEncoder();
  const bufs = [sim.buf.pos, sim.buf.prev, sim.buf.sleep].map((b) => {
    const s = d.createBuffer({ size: N * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyBufferToBuffer(b, 0, s, 0, N * 16);
    return s;
  });
  d.queue.submit([enc.finish()]);
  const arr = [];
  for (const s of bufs) { await s.mapAsync(GPUMapMode.READ); arr.push(new Float32Array(s.getMappedRange().slice(0))); s.destroy(); }
  const SL = new Uint32Array(arr[2].buffer), dt = 1 / 60 / sim.cfg.substeps;
  let live = 0, moving = 0, fast = 0, up = 0, maxUp = 0, asleep = 0, high = 0;
  for (const v of window.__below) {
    if (sim.clothFlags[v] & 4) continue;
    live++;
    const vx = (arr[0][v * 4] - arr[1][v * 4]) / dt, vy = (arr[0][v * 4 + 1] - arr[1][v * 4 + 1]) / dt, vz = (arr[0][v * 4 + 2] - arr[1][v * 4 + 2]) / dt;
    const sp = Math.hypot(vx, vy, vz);
    if (sp > 0.01) moving++;
    if (sp > 0.2) fast++;
    if (vy > 0.3) up++;
    maxUp = Math.max(maxUp, vy);
    if (SL[v * 4 + 3] & 0x20000) asleep++;
    if (arr[0][v * 4 + 1] > 0.35) high++;
  }
  // visible motion: displacement over one frame and since the previous probe (1 s ago)
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  const s2 = d.createBuffer({ size: N * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const e2 = d.createCommandEncoder(); e2.copyBufferToBuffer(sim.buf.pos, 0, s2, 0, N * 16); d.queue.submit([e2.finish()]);
  await s2.mapAsync(GPUMapMode.READ); const P2 = new Float32Array(s2.getMappedRange().slice(0)); s2.destroy();
  let jitter = 0, crawl = 0, rose = 0;
  const last = window.__lastPos;
  for (const v of window.__below) {
    if (sim.clothFlags[v] & 4) continue;
    if (Math.hypot(P2[v * 4] - arr[0][v * 4], P2[v * 4 + 1] - arr[0][v * 4 + 1], P2[v * 4 + 2] - arr[0][v * 4 + 2]) > 5e-4) jitter++;
    if (last) {
      if (Math.hypot(P2[v * 4] - last[v * 4], P2[v * 4 + 1] - last[v * 4 + 1], P2[v * 4 + 2] - last[v * 4 + 2]) > 3e-3) crawl++;
      if (P2[v * 4 + 1] - last[v * 4 + 1] > 0.01) rose++;
    }
  }
  window.__lastPos = P2;
  return { live, moving, fast, upward: up, maxUp: +maxUp.toFixed(2), asleep, high, jitter, crawl, rose };
});

await page.goto(`http://localhost:5190/?scene=${scene}&${process.argv[3] || ''}`, { waitUntil: 'load' });
await wait(3000);
if (pourMs > 0) {
  await page.evaluate(() => {
    const sim = window.__sim, cam = window.__cam, P = sim.clothPos;
    let lo = 1e9;
    for (let v = 0; v < sim.nCloth; v++) lo = Math.min(lo, P[v * 4 + 1]);
    const e = cam.eye, tgt = [0, lo + 0.06, 0], d = tgt.map((x, a) => x - e[a]), l = Math.hypot(...d);
    const hit = sim.raycast(e, d.map((x) => x / l));
    if (hit) sim.cut([[hit, hit]], 0.03);
  });
  await wait(pourMs);
}
await page.evaluate(() => {
  const sim = window.__sim, P = sim.clothPos;
  let lo = 1e9, hi = -1e9;
  for (let v = 0; v < sim.nCloth; v++) {
    if (sim.clothFlags[v] & 4) continue;
    lo = Math.min(lo, P[v * 4 + 1]); hi = Math.max(hi, P[v * 4 + 1]);
  }
  const y0 = lo + 0.5 * (hi - lo);
  const segs = [];
  for (let v = 0; v < sim.nCloth; v++) {
    if (Math.abs(P[v * 4 + 1] - y0) < sim.clothSpacing * 0.5) segs.push([[P[v * 4], P[v * 4 + 1], P[v * 4 + 2]], [P[v * 4], P[v * 4 + 1], P[v * 4 + 2]]]);
  }
  window.__below = [];
  for (let v = 0; v < sim.nCloth; v++) if (P[v * 4 + 1] < y0 - sim.clothSpacing) window.__below.push(v);
  sim.cut(segs, sim.clothSpacing * 0.3);
});
let worst = 0;
for (let t = 0; t <= 10; t++) {
  if (t) await wait(1000);
  const s = await probe();
  if (t >= 2) worst = Math.max(worst, s.maxUp);
  console.log(`t=${t}s`, JSON.stringify(s));
}
console.log('worst upward cloth speed after landing:', worst.toFixed(2), 'm/s');
await page.screenshot({ path: `tests/out/clothrest-${scene}.png` });
await browser.close(); await server.close();
