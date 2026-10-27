// Headless smoke test: boots the dev server, opens each scene in Chrome with WebGPU,
// reports console errors + fps, performs a slash, and saves screenshots.
//   node tests/smoke.mjs [scene] [outDir] [n]
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
import path from 'node:path';

const scenes = process.argv[2] ? process.argv[2].split(',') : ['sand', 'rice', 'blood'];
const outDir = process.argv[3] || 'tests/out';
const n = process.argv[4] || '64000';
const settle = Number(process.env.SETTLE || 2500);
const after = Number(process.env.AFTER || 3000);
fs.mkdirSync(outDir, { recursive: true });

const chromePaths = [
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
];
const executablePath = chromePaths.find((p) => fs.existsSync(p));

const server = await createServer({ server: { port: 5199, strictPort: true }, logLevel: 'error' });
await server.listen();

const browser = await puppeteer.launch({
  executablePath,
  headless: 'new',
  args: ['--enable-unsafe-webgpu', '--enable-gpu', '--ignore-gpu-blocklist', '--window-size=1280,800'],
  defaultViewport: { width: 1280, height: 800 },
});

try {
  for (const scene of scenes) {
    const page = await browser.newPage();
    const logs = [];
    page.on('console', (m) => logs.push(`[${m.type()}] ${m.text()}`));
    page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));
    await page.goto(`http://localhost:5199/?scene=${scene}&n=${n}`, { waitUntil: 'load' });
    await new Promise((r) => setTimeout(r, settle));
    const info = await page.evaluate(async () => {
      const a = await navigator.gpu?.requestAdapter();
      return { adapter: a ? (a.info ? `${a.info.vendor} ${a.info.architecture} ${a.info.description}` : 'yes') : 'none', stats: document.getElementById('stats').textContent };
    });
    await page.screenshot({ path: path.join(outDir, `${scene}-0.png`) });

    // slash horizontally across the lower (filled) part of the bag, then a puncture higher up
    const pts = await page.evaluate(() => {
      // aim at the live cloth: a horizontal slash across the lower third of the bag
      const sim = window.__sim, P = sim.clothPos;
      let lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9];
      for (let v = 0; v < sim.nCloth; v++) for (let a = 0; a < 3; a++) {
        lo[a] = Math.min(lo[a], P[v * 4 + a]);
        hi[a] = Math.max(hi[a], P[v * 4 + a]);
      }
      const h = Math.min(hi[1], sim.cfg.shape.centerY + sim.cfg.shape.ay) - lo[1];
      const y = lo[1] + 0.3 * h;
      const cx = (lo[0] + hi[0]) / 2, cz = (lo[2] + hi[2]) / 2, wx = (hi[0] - lo[0]) / 2;
      const out = [];
      for (let k = 0; k <= 8; k++) out.push(window.__project([cx + (-0.6 + (1.2 * k) / 8) * wx, y, cz]));
      out.push(window.__project([cx + 0.3 * wx, lo[1] + 0.12 * h, cz]));
      return out;
    });
    const punct = pts.pop();
    await page.mouse.move(pts[0][0], pts[0][1]);
    await page.mouse.down();
    for (const [x, y] of pts.slice(1)) {
      await page.mouse.move(x, y, { steps: 4 });
      await new Promise((r) => setTimeout(r, 25));
    }
    await page.mouse.up();
    await page.mouse.click(punct[0], punct[1]);
    await new Promise((r) => setTimeout(r, after / 3));
    await page.screenshot({ path: path.join(outDir, `${scene}-1.png`) });
    await new Promise((r) => setTimeout(r, (2 * after) / 3));
    await page.screenshot({ path: path.join(outDir, `${scene}-2.png`) });
    const stats2 = await page.evaluate(() => {
      const s = window.__sim;
      let dead = 0;
      if (s) for (const f of s.clothFlags) if (f & 4) dead++;
      return { stats: document.getElementById('stats').textContent, dead, r: s?.radius, spacing: s?.clothSpacing };
    });
    console.log(`\n=== ${scene} ===`);
    console.log('adapter:', info.adapter);
    console.log('stats@start:', info.stats);
    console.log('after:', JSON.stringify(stats2));
    const uniq = [...new Set(logs)];
    console.log(uniq.slice(0, 40).join('\n') || '(no console output)');
    await page.close();
  }
} finally {
  await browser.close();
  await server.close();
}
