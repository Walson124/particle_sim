// A one-row-wide ring cut (the thinnest slash possible): the piece below it must fall to the floor
// instead of hanging across the gap. Prints the lower piece's height over time.
//   node tests/thincut.mjs [scene] [extra query]
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';
const scene = process.argv[2] || 'sand';
const server = await createServer({ server: { port: 5189, strictPort: true }, logLevel: 'error' });
await server.listen();
const browser = await puppeteer.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: 'new', args: ['--enable-unsafe-webgpu'], defaultViewport: { width: 1000, height: 700 } });
const page = await browser.newPage();
page.on('pageerror', (e) => console.log('pageerror', e.message));
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

await page.goto(`http://localhost:5189/?scene=${scene}&${process.argv[3] || ''}`, { waitUntil: 'load' });
await wait(3000);
const y0 = await page.evaluate(() => {
  const sim = window.__sim, P = sim.clothPos;
  let lo = 1e9, hi = -1e9;
  for (let v = 0; v < sim.nCloth; v++) { lo = Math.min(lo, P[v * 4 + 1]); hi = Math.max(hi, P[v * 4 + 1]); }
  const y0 = lo + 0.45 * (Math.min(hi, sim.cfg.shape.centerY + sim.cfg.shape.ay) - lo);
  const segs = [];
  for (let v = 0; v < sim.nCloth; v++) {
    if (Math.abs(P[v * 4 + 1] - y0) < sim.clothSpacing * 0.5) segs.push([[P[v * 4], P[v * 4 + 1], P[v * 4 + 2]], [P[v * 4], P[v * 4 + 1], P[v * 4 + 2]]]);
  }
  // remember which cloth is below the cut, then cut
  window.__below = [];
  for (let v = 0; v < sim.nCloth; v++) if (P[v * 4 + 1] < y0 - sim.clothSpacing) window.__below.push(v);
  sim.cut(segs, sim.clothSpacing * 0.3);
  return y0;
});
console.log(`${scene}: ring cut at y=${y0.toFixed(3)}`);
for (let t = 0; t <= 6; t += 1) {
  if (t) await wait(1000);
  const s = await page.evaluate(() => {
    const sim = window.__sim, P = sim.clothPos, fy = sim.cfg.floorY ?? 0;
    let top = -1e9, sum = 0, n = 0;
    for (const v of window.__below) {
      if (sim.clothFlags[v] & 4) continue;
      top = Math.max(top, P[v * 4 + 1]); sum += P[v * 4 + 1]; n++;
    }
    return { top: +(top - fy).toFixed(3), mean: +(sum / n - fy).toFixed(3) };
  });
  console.log(`t=${t}s lower piece above floor: top ${s.top} m, mean ${s.mean} m`);
}
await page.screenshot({ path: `tests/out/thincut-${scene}.png` });
await browser.close(); await server.close();
