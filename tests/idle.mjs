// Leave a scene untouched and track whether it comes to rest:  node tests/idle.mjs scene secs "set=..."
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';
const [scene = 'rice', secs = '15', set = ''] = process.argv.slice(2);
const server = await createServer({ server: { port: 5186, strictPort: true }, logLevel: 'error' });
await server.listen();
const browser = await puppeteer.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: 'new', args: ['--enable-unsafe-webgpu'] });
const page = await browser.newPage();
page.on('pageerror', (e) => console.log('pageerror', e.message));
await page.goto(`http://localhost:5186/?scene=${scene}&${set}`, { waitUntil: 'load' });
for (let t = 1; t <= Number(secs); t += t < 4 ? 1 : 2) {
  await new Promise((r) => setTimeout(r, t < 4 ? 1000 : 2000));
  const s = await page.evaluate(async () => {
    const sim = window.__sim, d = sim.device, N = sim.N;
    const enc = d.createCommandEncoder();
    const bufs = [[sim.buf.pos, 16], [sim.buf.prev, 16], [sim.buf.sleep, 16], [sim.buf.vfix, 16]].map(([b, n]) => {
      const st = d.createBuffer({ size: N * n, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      enc.copyBufferToBuffer(b, 0, st, 0, N * n);
      return st;
    });
    d.queue.submit([enc.finish()]);
    const a = [];
    for (const st of bufs) { await st.mapAsync(GPUMapMode.READ); a.push(new Float32Array(st.getMappedRange().slice(0))); st.destroy(); }
    const [P, Q, SF, VF] = a, SL = new Uint32Array(SF.buffer), dt = 1 / 60 / sim.cfg.substeps;
    const stat = (lo, hi) => {
      const sp = [];
      let asleep = 0, grounded = 0, ymax = -1e9;
      for (let i = lo; i < hi; i++) {
        if (i < sim.nCloth && sim.clothFlags[i] & 4) continue;
        sp.push(Math.hypot((P[i * 4] - Q[i * 4]) / dt - VF[i * 4], (P[i * 4 + 1] - Q[i * 4 + 1]) / dt - VF[i * 4 + 1], (P[i * 4 + 2] - Q[i * 4 + 2]) / dt - VF[i * 4 + 2]));
        if (SL[i * 4 + 3] & 0x20000) asleep++;
        if (SL[i * 4 + 3] & 0x10000) grounded++;
        ymax = Math.max(ymax, P[i * 4 + 1]);
      }
      sp.sort((x, y) => x - y);
      const q = (f) => +sp[Math.floor(f * (sp.length - 1))].toFixed(4);
      return { asleep: +(asleep / sp.length).toFixed(3), grounded: +(grounded / sp.length).toFixed(3), v50: q(0.5), v90: q(0.9), v99: q(0.99), top: +ymax.toFixed(4) };
    };
    return { warmup: sim.warmup, cloth: stat(0, sim.nCloth), grains: stat(sim.nCloth, N) };
  });
  console.log(`t=${t}s`, JSON.stringify(s));
}
await browser.close(); await server.close();
