// Slash a scene's bag low on its side and count what spills out:  node tests/sidecut.mjs scene widthScale "set=..."
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';
const [scene = 'rice', width = '1', set = ''] = process.argv.slice(2);
const server = await createServer({ server: { port: 5182, strictPort: true }, logLevel: 'error' });
await server.listen();
const browser = await puppeteer.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: 'new', args: ['--enable-unsafe-webgpu'], defaultViewport: { width: 1280, height: 800 } });
const page = await browser.newPage();
page.on('pageerror', (e) => console.log('pageerror', e.message));
await page.goto(`http://localhost:5182/?scene=${scene}&${set}`, { waitUntil: 'load' });
await new Promise((r) => setTimeout(r, 3500));
// wait until the hidden settling phase is over
while (await page.evaluate(() => window.__sim.warmup > 0)) await new Promise((r) => setTimeout(r, 200));
console.log('  settled after', await page.evaluate(() => (performance.now() / 1000).toFixed(1)), 's');
const outside = () => page.evaluate(async () => {
  const sim = window.__sim, d = sim.device, N = sim.N;
  const st = d.createBuffer({ size: N * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = d.createCommandEncoder();
  enc.copyBufferToBuffer(sim.buf.pos, 0, st, 0, N * 16);
  d.queue.submit([enc.finish()]);
  await st.mapAsync(GPUMapMode.READ);
  const P = new Float32Array(st.getMappedRange().slice(0));
  st.destroy();
  // spilled = grains beyond the sack's live cloth footprint (bounding ellipse in xz, plus margin)
  let ax = 0, az = 0;
  for (let v = 0; v < sim.nCloth; v++) { if (sim.clothFlags[v] & 4) continue; ax = Math.max(ax, Math.abs(P[v * 4])); az = Math.max(az, Math.abs(P[v * 4 + 2])); }
  let n = 0;
  for (let i = sim.nCloth; i < N; i++) {
    const x = P[i * 4] / (ax + 0.01), z = P[i * 4 + 2] / (az + 0.01);
    if (x * x + z * z > 1) n++;
  }
  return n;
});
const r = await page.evaluate((w) => {
  const sim = window.__sim, P = sim.clothPos;
  let lo = 1e9, hi = -1e9, front = -1e9;
  for (let v = 0; v < sim.nCloth; v++) { lo = Math.min(lo, P[v * 4 + 1]); hi = Math.max(hi, P[v * 4 + 1]); }
  const y = lo + 0.3 * (Math.min(hi, sim.cfg.shape.centerY + sim.cfg.shape.ay) - lo);
  // front-facing side wall (+z), from directly in front at that height
  const segs = [];
  let last = null;
  for (let k = -4; k <= 4; k++) {
    const ro = [k * 0.025, y, 2], h = sim.raycast(ro, [0, 0, -1]);
    if (h && last) segs.push([last, h]);
    last = h;
  }
  return { y: +y.toFixed(3), segs: segs.length, removed: sim.cut(segs, sim.clothSpacing * (sim.cfg.slashWidth ?? 0.62) * w) };
}, Number(width));
console.log(scene, 'width x' + width, set || '', JSON.stringify(r));
const asleepFrac = () => page.evaluate(async () => {
  const sim = window.__sim, d = sim.device, N = sim.N;
  const st = d.createBuffer({ size: N * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = d.createCommandEncoder();
  enc.copyBufferToBuffer(sim.buf.sleep, 0, st, 0, N * 16);
  d.queue.submit([enc.finish()]);
  await st.mapAsync(GPUMapMode.READ);
  const W = new Uint32Array(st.getMappedRange().slice(0));
  st.destroy();
  let a = 0;
  for (let i = sim.nCloth; i < N; i++) if (W[i * 4 + 3] & 0x20000) a++;
  return +(a / (N - sim.nCloth)).toFixed(3);
});
for (const t of [0.1, 1, 3, 6]) {
  await new Promise((res) => setTimeout(res, t === 0.1 ? 100 : t === 1 ? 900 : t === 3 ? 2000 : 3000));
  console.log(`  t=${t}s spilled grains:`, await outside(), ' grains asleep:', await asleepFrac());
}
await browser.close(); await server.close();
