// Numeric probe: load a scene with overrides, sample debugStats over time (no interaction).
//   node tests/probe.mjs scene "set=a:1,b:2" [n] [seconds]
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';

const [scene = 'sand', set = '', n = '64000', secs = '4'] = process.argv.slice(2);
const server = await createServer({ server: { port: 5198, strictPort: true }, logLevel: 'error' });
await server.listen();
const browser = await puppeteer.launch({
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  headless: 'new',
  args: ['--enable-unsafe-webgpu', '--enable-gpu', '--ignore-gpu-blocklist'],
  defaultViewport: { width: 960, height: 600 },
});
try {
  const page = await browser.newPage();
  const errs = [];
  page.on('console', (m) => { if (m.type() === 'error') errs.push(m.text()); });
  page.on('pageerror', (e) => errs.push(e.message));
  await page.goto(`http://localhost:5198/?scene=${scene}&n=${n}&${set}`, { waitUntil: 'load' });
  const t0 = Date.now();
  for (let t = 0.5; t <= Number(secs); t += 0.5) {
    await new Promise((r) => setTimeout(r, t * 1000 - (Date.now() - t0)));
    const s = await page.evaluate(() => window.__sim?.debugStats());
    console.log(`t=${t.toFixed(1)}`, JSON.stringify(s));
  }
  if (process.env.SHOT) await page.screenshot({ path: process.env.SHOT });
  const uniq = [...new Set(errs)].filter((e) => !e.includes('404'));
  if (uniq.length) console.log(uniq.slice(0, 5).join('\n'));
} finally {
  await browser.close();
  await server.close();
}
