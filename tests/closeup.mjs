// Close-up screenshots after a cut:  node tests/closeup.mjs scene outDir
// Cuts the bag low, lets it pour, then shoots the pile, the wound and a wide view.
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';
import path from 'node:path';
const [scene = 'sand', outDir = 'tests/out'] = process.argv.slice(2);
const server = await createServer({ server: { port: 5191, strictPort: true }, logLevel: 'error' });
await server.listen();
const browser = await puppeteer.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: 'new', args: ['--enable-unsafe-webgpu'], defaultViewport: { width: 1280, height: 800 } });
const page = await browser.newPage();
page.on('pageerror', (e) => console.log('pageerror', e.message));
page.on('console', (m) => { if (m.type() === 'error' && !m.text().includes('404')) console.log(m.text()); });
await page.goto(`http://localhost:5191/?scene=${scene}`, { waitUntil: 'load' });
await new Promise((r) => setTimeout(r, 3000));
await page.evaluate(() => {
  const sim = window.__sim, P = sim.clothPos;
  let lo = 1e9, hi = -1e9;
  for (let v = 0; v < sim.nCloth; v++) { lo = Math.min(lo, P[v * 4 + 1]); hi = Math.max(hi, P[v * 4 + 1]); }
  const y = lo + 0.2 * (Math.min(hi, sim.cfg.shape.centerY + sim.cfg.shape.ay) - lo);
  // puncture the front-facing side low on the bag
  const cam = window.__cam, e = cam.eye;
  const ro = e, tgt = [0, y, 0], d = [tgt[0] - ro[0], tgt[1] - ro[1], tgt[2] - ro[2]];
  const l = Math.hypot(...d);
  const hit = sim.raycast(ro, d.map((x) => x / l));
  if (hit) sim.cut([[hit, hit]], 0.025);
});
await new Promise((r) => setTimeout(r, 5000));
const shot = async (name, target, dist, yaw, pitch) => {
  await page.evaluate((t, d, y, p) => Object.assign(window.__cam, { target: t, dist: d, yaw: y, pitch: p }), target, dist, yaw, pitch);
  await new Promise((r) => setTimeout(r, 400));
  await page.screenshot({ path: path.join(outDir, `${scene}-${name}.png`) });
};
const S = await page.evaluate(() => window.__sim.cfg.shape);
await shot('pile', [0, 0.02, 0.1], 0.45, 0.55, 0.45);
await shot('wound', [0, S.centerY - 0.2 * S.ay, 0], 0.4, 0.55, 0.1);
await shot('wide', [0, 0.3, 0], 1.3, 0.55, 0.25);
await browser.close(); await server.close();
