// Visible motion of an untouched scene: displacement of every particle over one second.
//   node tests/drift.mjs scene [waitSeconds] ["set=..."]
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';
const [scene = 'rice', wait = '8', set = ''] = process.argv.slice(2);
const server = await createServer({ server: { port: 5184, strictPort: true }, logLevel: 'error' });
await server.listen();
const browser = await puppeteer.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: 'new', args: ['--enable-unsafe-webgpu'] });
const page = await browser.newPage();
page.on('pageerror', (e) => console.log('pageerror', e.message));
await page.goto(`http://localhost:5184/?scene=${scene}&${set}`, { waitUntil: 'load' });
await new Promise((r) => setTimeout(r, Number(wait) * 1000));
const snap = () => page.evaluate(async () => {
  const sim = window.__sim, d = sim.device;
  const st = d.createBuffer({ size: sim.N * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = d.createCommandEncoder();
  enc.copyBufferToBuffer(sim.buf.pos, 0, st, 0, sim.N * 16);
  d.queue.submit([enc.finish()]);
  await st.mapAsync(GPUMapMode.READ);
  const a = Array.from(new Float32Array(st.getMappedRange()));
  st.destroy();
  return { a, nCloth: sim.nCloth, dead: Array.from(sim.clothFlags, (f) => f & 4) };
});
const A = await snap();
await new Promise((r) => setTimeout(r, 1000));
const B = await snap();
const stats = (lo, hi) => {
  const d = [];
  for (let i = lo; i < hi; i++) {
    if (i < A.nCloth && A.dead[i]) continue;
    d.push(Math.hypot(B.a[i * 4] - A.a[i * 4], B.a[i * 4 + 1] - A.a[i * 4 + 1], B.a[i * 4 + 2] - A.a[i * 4 + 2]) * 1000);
  }
  d.sort((x, y) => x - y);
  const q = (f) => +d[Math.floor(f * (d.length - 1))].toFixed(2);
  return `p50 ${q(0.5)} mm  p90 ${q(0.9)} mm  p99 ${q(0.99)} mm  max ${q(1)} mm  (>1mm: ${d.filter((x) => x > 1).length} of ${d.length})`;
};
console.log(`${scene} drift over 1 s after ${wait} s:`);
console.log('  skin  ', stats(0, A.nCloth));
console.log('  grains', stats(A.nCloth, A.a.length / 4));
await browser.close(); await server.close();
