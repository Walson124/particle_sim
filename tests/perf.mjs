// fps for a list of override sets:  node tests/perf.mjs scene n "set1" "set2" ...
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';
const [scene = 'sand', n = '64000', ...sets] = process.argv.slice(2);
const server = await createServer({ server: { port: 5197, strictPort: true }, logLevel: 'error' });
await server.listen();
const browser = await puppeteer.launch({
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  headless: 'new',
  args: ['--enable-unsafe-webgpu', '--enable-gpu', '--ignore-gpu-blocklist', '--disable-gpu-vsync', '--disable-frame-rate-limit'],
  defaultViewport: { width: 1280, height: 800 },
});
try {
  for (const set of sets.length ? sets : ['']) {
    const page = await browser.newPage();
    await page.goto(`http://localhost:5197/?scene=${scene}&n=${n}&set=${set}`, { waitUntil: 'load' });
    await new Promise((r) => setTimeout(r, 5000));
    const b = await page.evaluate(() => window.__bench(60));
    if (process.env.KERNELS) console.log(JSON.stringify(await page.evaluate(() => window.__sim.benchKernels())));
    if (process.env.KERNELS) console.log('render', JSON.stringify(await page.evaluate(() => window.__profileRender())));
    const st = await page.evaluate(() => document.getElementById('stats').textContent);
    console.log(set.padEnd(36), JSON.stringify(b), st.split('·').slice(1, 3).join('·'));
    await page.close();
  }
} finally {
  await browser.close();
  await server.close();
}
