// 1) Cut a full ring around the hanging bag: the lower part must fall.
// 2) Puncture it and let it pour: the pile on the floor must come to rest.
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';
import path from 'node:path';
const outDir = process.argv[2] || 'tests/out';
const server = await createServer({ server: { port: 5188, strictPort: true }, logLevel: 'error' });
await server.listen();
const browser = await puppeteer.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: 'new', args: ['--enable-unsafe-webgpu'], defaultViewport: { width: 1000, height: 700 } });
const page = await browser.newPage();
page.on('pageerror', (e) => console.log('pageerror', e.message));
page.on('console', (m) => { if (m.type() === 'error' && !m.text().includes('404')) console.log(m.text()); });
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

// cloth/grain heights + how many floor grains still move
const probe = () => page.evaluate(async () => {
  const sim = window.__sim, d = sim.device, N = sim.N;
  const enc = d.createCommandEncoder();
  const bufs = [[sim.buf.pos, 16], [sim.buf.prev, 16], [sim.buf.sleep, 4]].map(([b, bytes]) => {
    const s = d.createBuffer({ size: N * bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyBufferToBuffer(b, 0, s, 0, N * bytes);
    return s;
  });
  d.queue.submit([enc.finish()]);
  const arr = [];
  for (const s of bufs) { await s.mapAsync(GPUMapMode.READ); arr.push(new Float32Array(s.getMappedRange().slice(0))); s.destroy(); }
  const dt = 1 / 60 / sim.cfg.substeps;
  let lowCloth = 0, live = 0, floorGrains = 0, moving = 0, asleep = 0, grounded = 0;
  const SL = new Uint32Array(arr[2].buffer);
  for (let v = 0; v < sim.nCloth; v++) {
    if (sim.clothFlags[v] & 4) continue;
    live++;
    if (arr[0][v * 4 + 1] < 0.1) lowCloth++;
  }
  for (let i = sim.nCloth; i < N; i++) {
    if (arr[0][i * 4 + 1] > 0.06) continue;
    floorGrains++;
    if (SL[i] & 0x20000) asleep++;
    if (SL[i] & 0x10000) grounded++;
    const v = Math.hypot(arr[0][i * 4] - arr[1][i * 4], arr[0][i * 4 + 1] - arr[1][i * 4 + 1], arr[0][i * 4 + 2] - arr[1][i * 4 + 2]) / dt;
    if (v > 0.01) moving++;
  }
  // awake floor grains: speed histogram, counters, where they are
  const sp = [], cnt = [], ys = [], rs = [];
  for (let i = sim.nCloth; i < N; i++) {
    if (arr[0][i * 4 + 1] > 0.06 || SL[i] & 0x20000) continue;
    sp.push(Math.hypot(arr[0][i * 4] - arr[1][i * 4], arr[0][i * 4 + 1] - arr[1][i * 4 + 1], arr[0][i * 4 + 2] - arr[1][i * 4 + 2]) / dt);
    cnt.push(SL[i] & 0xffff);
    ys.push(arr[0][i * 4 + 1]);
    rs.push(Math.hypot(arr[0][i * 4], arr[0][i * 4 + 2]));
  }
  const q = (a, f) => { const b = [...a].sort((x, y) => x - y); return b.length ? +b[Math.floor(f * (b.length - 1))].toFixed(4) : null; };
  const awake = { n: sp.length, speed: [q(sp, 0.1), q(sp, 0.5), q(sp, 0.9)], counter: [q(cnt, 0.5), q(cnt, 0.9)], y: [q(ys, 0.1), q(ys, 0.5), q(ys, 0.9)], radius: [q(rs, 0.1), q(rs, 0.5), q(rs, 0.9)], sleepSteps: Math.round(0.25 * 60 * sim.cfg.substeps) };
  return { floorGrains, movingOnFloor: moving, asleep, grounded, awake };
});

await page.goto('http://localhost:5188/?scene=sand&' + (process.argv[3] || ''), { waitUntil: 'load' });
await wait(3000);
await page.evaluate(() => {
  const sim = window.__sim, P = sim.clothPos;
  let lo = 1e9, hi = -1e9;
  for (let v = 0; v < sim.nCloth; v++) { lo = Math.min(lo, P[v * 4 + 1]); hi = Math.max(hi, P[v * 4 + 1]); }
  const y0 = lo + 0.45 * (Math.min(hi, sim.cfg.shape.centerY + sim.cfg.shape.ay) - lo);
  const segs = [];
  for (let v = 0; v < sim.nCloth; v++) {
    if (Math.abs(P[v * 4 + 1] - y0) < sim.clothSpacing * 0.7) segs.push([[P[v * 4], P[v * 4 + 1], P[v * 4 + 2]], [P[v * 4], P[v * 4 + 1], P[v * 4 + 2]]]);
  }
  sim.cut(segs, sim.clothSpacing * 0.3);
});
console.log('ring cut, t=0  ', JSON.stringify(await probe()));
for (const t of [1.5, 3, 5, 8]) {
  await wait(t === 1.5 ? 1500 : t === 3 ? 1500 : t === 5 ? 2000 : 3000);
  console.log(`ring cut, t=${t}`, JSON.stringify(await probe()));
}
await page.screenshot({ path: path.join(outDir, 'sever.png') });

await page.goto('http://localhost:5188/?scene=sand&' + (process.argv[3] || ''), { waitUntil: 'load' });
await wait(3000);
await page.evaluate(() => {
  const sim = window.__sim, cam = window.__cam, P = sim.clothPos;
  let lo = 1e9;
  for (let v = 0; v < sim.nCloth; v++) lo = Math.min(lo, P[v * 4 + 1]);
  const e = cam.eye, tgt = [0, lo + 0.06, 0], d = tgt.map((x, a) => x - e[a]), l = Math.hypot(...d);
  const hit = sim.raycast(e, d.map((x) => x / l));
  if (hit) sim.cut([[hit, hit]], 0.03);
});
for (const t of [2, 4, 6, 8]) {
  await wait(2000);
  console.log(`pour t=${t}`, JSON.stringify(await probe()));
}
await browser.close(); await server.close();
