// Cut the bag low and track upward "fliers" over time:  node tests/fliers.mjs scene "set=..."
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';
const [scene = 'sand', set = ''] = process.argv.slice(2);
const server = await createServer({ server: { port: 5190, strictPort: true }, logLevel: 'error' });
await server.listen();
const browser = await puppeteer.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: 'new', args: ['--enable-unsafe-webgpu'], defaultViewport: { width: 1000, height: 700 } });
const page = await browser.newPage();
page.on('pageerror', (e) => console.log('pageerror', e.message));
page.on('console', (m) => { if (m.type() === 'error' && !m.text().includes('404')) console.log(m.text()); });
await page.goto(`http://localhost:5190/?scene=${scene}&${set}`, { waitUntil: 'load' });
await new Promise((r) => setTimeout(r, 3000));
console.log('before cut', JSON.stringify(await page.evaluate(() => window.__sim.flierStats())));
await page.evaluate(() => {
  const sim = window.__sim, P = sim.clothPos, cam = window.__cam;
  let lo = 1e9, hi = -1e9;
  for (let v = 0; v < sim.nCloth; v++) { lo = Math.min(lo, P[v * 4 + 1]); hi = Math.max(hi, P[v * 4 + 1]); }
  const top = Math.min(hi, sim.cfg.shape.centerY + sim.cfg.shape.ay);
  // a horizontal slash across the front plus a puncture
  const hits = [];
  for (let k = -4; k <= 4; k++) {
    const tgt = [k * 0.02, lo + 0.25 * (top - lo), 0], e = cam.eye;
    const d = tgt.map((x, a) => x - e[a]), l = Math.hypot(...d);
    const h = sim.raycast(e, d.map((x) => x / l));
    if (h) hits.push(h);
  }
  const segs = [];
  for (let k = 1; k < hits.length; k++) segs.push([hits[k - 1], hits[k]]);
  sim.cut(segs, sim.clothSpacing * (sim.cfg.slashWidth ?? 0.62));
});
let worst = 0;
for (let k = 0; k < 16; k++) {
  await new Promise((r) => setTimeout(r, 250));
  const f = await page.evaluate(() => window.__sim.flierStats());
  worst = Math.max(worst, f.maxVy);
  console.log(`t+${((k + 1) * 0.25).toFixed(2)}`, JSON.stringify(f));
}
console.log('worst upward speed', worst);
await browser.close(); await server.close();
