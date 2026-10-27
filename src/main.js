import GUI from 'lil-gui';
import { mat4 } from './gpu/math.js';
import { Renderer } from './render/renderer.js';
import { PRESETS, QUALITY } from './sim/presets.js';
import { benchKernels, debugStats, flierStats } from './sim/debug.js';
import { Simulation } from './sim/simulation.js';

const canvas = document.getElementById('gpu');
const overlay = document.getElementById('overlay');
const octx = overlay.getContext('2d');
const statsEl = document.getElementById('stats');

function fail(msg) {
  const el = document.getElementById('error');
  el.hidden = false;
  el.innerHTML = msg;
  console.error(msg);
}

class OrbitCamera {
  constructor() {
    this.target = [0, 0.4, 0];
    this.dist = 1.6;
    this.yaw = 0.5;
    this.pitch = 0.2;
    this.fov = (38 * Math.PI) / 180;
  }
  set(c) {
    this.target = [...c.target];
    this.dist = c.dist;
    this.yaw = c.yaw;
    this.pitch = c.pitch;
  }
  get eye() {
    const cp = Math.cos(this.pitch);
    return [
      this.target[0] + this.dist * cp * Math.sin(this.yaw),
      this.target[1] + this.dist * Math.sin(this.pitch),
      this.target[2] + this.dist * cp * Math.cos(this.yaw),
    ];
  }
  matrices(aspect) {
    const eye = this.eye;
    const view = mat4.lookAt(eye, this.target, [0, 1, 0]);
    const proj = mat4.perspective(this.fov, aspect, 0.02, 30);
    const viewProj = mat4.multiply(proj, view);
    return { view, proj, viewProj, invViewProj: mat4.invert(viewProj), eye };
  }
  ray(ndcX, ndcY, aspect) {
    const m = this.matrices(aspect).invViewProj;
    const a = mat4.transformPoint(m, [ndcX, ndcY, 0]);
    const b = mat4.transformPoint(m, [ndcX, ndcY, 1]);
    const d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const l = Math.hypot(...d);
    return { ro: a, rd: [d[0] / l, d[1] / l, d[2] / l] };
  }
}

async function main() {
  if (!navigator.gpu) {
    fail('This demo needs <b>WebGPU</b>. Use a recent Chrome, Edge, or Safari 26+ (Firefox: enable <code>dom.webgpu.enabled</code>).');
    return;
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    fail('No WebGPU adapter available (GPU blocklisted or disabled).');
    return;
  }
  const want = ['maxStorageBuffersPerShaderStage', 'maxStorageBufferBindingSize', 'maxBufferSize'];
  const requiredLimits = {};
  for (const k of want) requiredLimits[k] = adapter.limits[k];
  const requiredFeatures = adapter.features.has('timestamp-query') ? ['timestamp-query'] : [];
  const device = await adapter.requestDevice({ requiredLimits, requiredFeatures });
  device.lost.then((info) => fail(`GPU device lost: ${info.message}`));
  device.addEventListener('uncapturederror', (e) => console.error('WebGPU error:', e.error.message));

  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  const renderer = new Renderer(device, format);
  const cam = new OrbitCamera();

  const params = new URLSearchParams(location.search);
  const state = {
    scenario: params.get('scene') in PRESETS ? params.get('scene') : 'sand',
    quality:
      Object.keys(QUALITY).find((k) => QUALITY[k] === Number(params.get('n'))) ||
      (matchMedia('(pointer: coarse)').matches ? 'Low (32k)' : 'Medium (64k)'),
    tool: 'Slash',
    toolSize: 1.0,
    paused: false,
    renderScale: Math.min(window.devicePixelRatio || 1, 1.5),
  };

  let sim = null;
  function rebuild(resetCamera = true) {
    if (sim) sim.destroy();
    const cfg = structuredClone(PRESETS[state.scenario]);
    // deeper granular piles need smaller steps to stay incompressible. Sand renders as a surface, so
    // mild interior compression is invisible: grow gently (~N^0.23, 64k: 24, 256k: 33 substeps)
    if (cfg.kind === 'grain') cfg.substeps = Math.round(cfg.substeps * (QUALITY[state.quality] / 64000) ** 0.23);
    // ?set=key:value,key:value overrides preset values (handy for tuning / tests)
    for (const kv of (params.get('set') || '').split(',').filter(Boolean)) {
      const [k, v] = kv.split(':');
      cfg[k] = Number(v);
    }
    sim = new Simulation(device, cfg, QUALITY[state.quality]);
    renderer.setSim(sim);
    if (resetCamera) cam.set(cfg.camera);
    buildTuning();
    // test / tuning hooks (tests/*.mjs)
    window.__sim = sim;
    window.__cam = cam;
    sim.debugStats = () => debugStats(sim);
    sim.benchKernels = () => benchKernels(sim);
    sim.flierStats = () => flierStats(sim);
    // test hook: world point -> CSS pixel
    window.__project = (p) => {
      const m = cam.matrices(aspect()).viewProj;
      const q = mat4.transformPoint(m, p);
      return [((q[0] + 1) / 2) * canvas.clientWidth, ((1 - q[1]) / 2) * canvas.clientHeight];
    };
  }

  // ------------------------------------------------------------------ UI
  const gui = new GUI({ title: 'Particle Breach' });
  gui.add(state, 'scenario', Object.fromEntries(Object.entries(PRESETS).map(([k, p]) => [p.label, k]))).name('Scenario').onChange(() => rebuild());
  gui.add(state, 'quality', Object.keys(QUALITY)).name('Particles').onChange(() => rebuild(false));
  gui.add(state, 'tool', ['Slash', 'Puncture']).name('Tool (click = puncture)');
  gui.add(state, 'toolSize', 0.5, 4, 0.1).name('Tool size');
  const actions = {
    reset: () => rebuild(false),
    release: () => sim.release(),
    pause: () => { state.paused = !state.paused; },
  };
  gui.add(actions, 'release').name('Cut the rope / release');
  gui.add(actions, 'reset').name('Reset  [R]');
  gui.add(state, 'paused').name('Pause  [Space]').listen();
  gui.add(state, 'renderScale', 0.5, 2, 0.25).name('Render scale');
  let tune = null;
  function buildTuning() {
    if (tune) tune.destroy();
    tune = gui.addFolder('Material');
    const c = sim.cfg;
    tune.add(c, 'substeps', 2, 48, 1).name('Substeps');
    tune.add(c, 'gravity', -20, 0, 0.1).name('Gravity');
    if (c.kind === 'fluid') {
      tune.add(c, 'pressure', 0, 3, 0.05).name('Blood pressure');
      tune.add(c, 'heartRate', 0, 3, 0.05).name('Heart rate (Hz)');
      tune.add(c, 'viscosity', 0, 0.15, 0.005).name('Viscosity');
      tune.add(c, 'fluidCohesion', 0, 0.02, 0.0005).name('Cohesion');
    } else {
      tune.add(c, 'muS', 0, 1.2, 0.01).name('Static friction');
      tune.add(c, 'muK', 0, 1.2, 0.01).name('Kinetic friction');
      tune.add(c, 'cohesion', 0, 0.5, 0.01).name('Wetness (cohesion)');
    }
    tune.add(c, 'floorMu', 0, 1.2, 0.01).name('Floor friction');
    tune.add(c, 'tear', 1.05, 3, 0.01).name('Skin tear limit');
    tune.add(c, 'bendK', 0, 1, 0.01).name('Skin stiffness');
    tune.close();
  }

  // ------------------------------------------------------------------ input
  const pointers = new Map();
  let tool = null; // { startX, startY, slashing, last }
  let orbit = null;
  const trail = [];
  const aspect = () => canvas.clientWidth / canvas.clientHeight;
  const toNdc = (x, y) => [(x / canvas.clientWidth) * 2 - 1, 1 - (y / canvas.clientHeight) * 2];
  const pick = (x, y) => {
    const [nx, ny] = toNdc(x, y);
    const { ro, rd } = cam.ray(nx, ny, aspect());
    return sim.raycast(ro, rd);
  };
  const slashRadius = () => sim.clothSpacing * (sim.cfg.slashWidth ?? 0.62) * state.toolSize;
  const punctureRadius = () => 0.012 * state.toolSize;

  // Cut along the stroke from the last sample to (x, y): sample rays every few pixels, joining
  // consecutive surface hits into 3D cut segments. All rays and cuts of one pointer event share a
  // single screen-space broadphase.
  function slashTo(x, y) {
    const last = tool.lastXY;
    const steps = Math.max(1, Math.ceil(Math.hypot(x - last[0], y - last[1]) / 4));
    const [ax, ay] = toNdc(last[0], last[1]);
    const [bx, by] = toNdc(x, y);
    const margin = 0.12 * state.toolSize;
    const m = cam.matrices(aspect());
    const bp = sim.broadphase(m.viewProj, Math.min(ax, bx) - margin, Math.min(ay, by) - margin, Math.max(ax, bx) + margin, Math.max(ay, by) + margin);
    if (!bp) return;
    const segments = [];
    for (let s = 1; s <= steps; s++) {
      const px = last[0] + ((x - last[0]) * s) / steps;
      const py = last[1] + ((y - last[1]) * s) / steps;
      const [nx, ny] = toNdc(px, py);
      const { ro, rd } = cam.ray(nx, ny, aspect());
      const hit = sim.raycast(ro, rd, bp);
      if (hit && tool.lastHit) {
        const gap = Math.hypot(hit[0] - tool.lastHit[0], hit[1] - tool.lastHit[1], hit[2] - tool.lastHit[2]);
        segments.push(gap < 0.06 ? [tool.lastHit, hit] : [hit, hit]);
      } else if (hit) {
        segments.push([hit, hit]);
      }
      tool.lastHit = hit;
    }
    sim.cut(segments, slashRadius(), bp);
    tool.lastXY = [x, y];
  }

  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, [e.offsetX, e.offsetY]);
    if (pointers.size >= 2) {
      tool = null;
      orbit = { pinch: pinchDist() };
      return;
    }
    if (e.button === 2 || e.button === 1 || e.altKey || e.ctrlKey || e.metaKey) {
      orbit = { pan: e.button === 1 || e.shiftKey };
    } else if (e.button === 0) {
      tool = { start: [e.offsetX, e.offsetY], lastXY: [e.offsetX, e.offsetY], lastHit: pick(e.offsetX, e.offsetY), slashing: false };
      trail.length = 0;
      trail.push({ x: e.offsetX, y: e.offsetY, t: performance.now() });
    }
  });
  const pinchDist = () => {
    const [a, b] = [...pointers.values()];
    return Math.hypot(a[0] - b[0], a[1] - b[1]);
  };
  canvas.addEventListener('pointermove', (e) => {
    const prev = pointers.get(e.pointerId);
    if (!prev) return;
    const dx = e.offsetX - prev[0], dy = e.offsetY - prev[1];
    pointers.set(e.pointerId, [e.offsetX, e.offsetY]);
    if (orbit) {
      if (pointers.size >= 2) {
        const d = pinchDist();
        cam.dist = Math.min(Math.max(cam.dist * (orbit.pinch / d), 0.3), 6);
        orbit.pinch = d;
        cam.yaw -= (dx * 0.5) / canvas.clientWidth * 3;
      } else if (orbit.pan) {
        const s = cam.dist * 0.0015;
        const right = [Math.cos(cam.yaw), 0, -Math.sin(cam.yaw)];
        cam.target[0] -= right[0] * dx * s;
        cam.target[2] -= right[2] * dx * s;
        cam.target[1] += dy * s;
      } else {
        cam.yaw -= dx * 0.006;
        cam.pitch = Math.min(Math.max(cam.pitch + dy * 0.005, -0.15), 1.45);
      }
      return;
    }
    if (tool) {
      if (!tool.slashing && Math.hypot(e.offsetX - tool.start[0], e.offsetY - tool.start[1]) > 5 && state.tool === 'Slash') tool.slashing = true;
      if (tool.slashing) slashTo(e.offsetX, e.offsetY);
      trail.push({ x: e.offsetX, y: e.offsetY, t: performance.now() });
    }
  });
  const end = (e) => {
    pointers.delete(e.pointerId);
    if (tool && !tool.slashing) {
      const hit = pick(tool.start[0], tool.start[1]);
      if (hit) sim.cut([[hit, hit]], punctureRadius());
    }
    tool = null;
    if (pointers.size === 0) orbit = null;
  };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    cam.dist = Math.min(Math.max(cam.dist * Math.exp(e.deltaY * 0.001), 0.3), 6);
  }, { passive: false });
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT') return;
    if (e.code === 'Space') { state.paused = !state.paused; e.preventDefault(); }
    if (e.code === 'KeyR') rebuild(false);
  });

  function drawTrail(now) {
    octx.clearRect(0, 0, overlay.width, overlay.height);
    while (trail.length && now - trail[0].t > 220) trail.shift();
    if (trail.length < 2) return;
    const dpr = overlay.width / overlay.clientWidth;
    octx.lineCap = 'round';
    for (let i = 1; i < trail.length; i++) {
      const a = trail[i - 1], b = trail[i];
      const k = 1 - (now - b.t) / 220;
      octx.strokeStyle = `rgba(255,255,255,${0.55 * k})`;
      octx.lineWidth = (1 + 2.5 * k) * dpr;
      octx.beginPath();
      octx.moveTo(a.x * dpr, a.y * dpr);
      octx.lineTo(b.x * dpr, b.y * dpr);
      octx.stroke();
    }
  }

  // test hook: GPU time per simulation step and per rendered frame (ms)
  window.__bench = async (steps = 60) => {
    const was = state.paused;
    state.paused = true;
    await device.queue.onSubmittedWorkDone();
    let t = performance.now();
    for (let k = 0; k < steps; k++) {
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      sim.step(pass, STEP);
      pass.end();
      device.queue.submit([enc.finish()]);
    }
    await device.queue.onSubmittedWorkDone();
    const simMs = (performance.now() - t) / steps;
    t = performance.now();
    for (let k = 0; k < 30; k++) {
      const enc = device.createCommandEncoder();
      renderer.render(enc, context.getCurrentTexture().createView(), sim, cam, 0);
      device.queue.submit([enc.finish()]);
    }
    await device.queue.onSubmittedWorkDone();
    const renderMs = (performance.now() - t) / 30;
    state.paused = was;
    return { simMs: +simMs.toFixed(2), renderMs: +renderMs.toFixed(2) };
  };

  window.__profileRender = async () => {
    const qs = device.createQuerySet({ type: 'timestamp', count: 32 });
    const res = device.createBuffer({ size: 256, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const rd = device.createBuffer({ size: 256, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    renderer.profile = { qs, labels: [] };
    const enc = device.createCommandEncoder();
    renderer.render(enc, context.getCurrentTexture().createView(), sim, cam, 0);
    const n = renderer.profile.labels.length * 2;
    enc.resolveQuerySet(qs, 0, n, res, 0);
    enc.copyBufferToBuffer(res, 0, rd, 0, n * 8);
    device.queue.submit([enc.finish()]);
    const labels = renderer.profile.labels;
    renderer.profile = null;
    await rd.mapAsync(GPUMapMode.READ);
    const t = new BigUint64Array(rd.getMappedRange());
    const out = {};
    labels.forEach((l, k) => { out[l] = +((out[l] || 0) + Number(t[2 * k + 1] - t[2 * k]) / 1e6).toFixed(3); });
    return { ...out, px: [canvas.width, canvas.height] };
  };

  // ------------------------------------------------------------------ loop
  rebuild();
  const STEP = 1 / 60;
  let acc = 0;
  let last = performance.now();
  let fpsT = last, fpsN = 0, fps = 0;
  let lastAttach = 0;
  let frameAvg = 16; // smoothed frame interval (ms)
  function frame(now) {
    const dt = Math.min((now - last) / 1000, 0.1);
    frameAvg += (Math.min(now - last, 100) - frameAvg) * 0.05;
    last = now;
    const w = Math.max(1, Math.floor(canvas.clientWidth * state.renderScale));
    const h = Math.max(1, Math.floor(canvas.clientHeight * state.renderScale));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    const ow = Math.floor(canvas.clientWidth * (window.devicePixelRatio || 1));
    const oh = Math.floor(canvas.clientHeight * (window.devicePixelRatio || 1));
    if (overlay.width !== ow || overlay.height !== oh) {
      overlay.width = ow;
      overlay.height = oh;
    }
    renderer.resize(w, h);

    const enc = device.createCommandEncoder();
    if (sim.warmup > 0) {
      const pass = enc.beginComputePass();
      for (let k = 0; k < 6 && sim.warmup > 0; k++) sim.step(pass, STEP);
      pass.end();
      acc = 0;
    } else if (!state.paused) {
      acc += dt;
      let steps = 0;
      const pass = enc.beginComputePass();
      // when the GPU cannot sustain 60 Hz, never catch up with extra steps (that only makes the next
      // frame slower): run one step per frame, i.e. smooth slight slow motion instead of stutter
      const maxSteps = frameAvg > STEP * 1000 * 1.3 ? 1 : 2;
      while (acc >= STEP && steps < maxSteps) {
        sim.step(pass, STEP);
        acc -= STEP;
        steps++;
      }
      pass.end();
      if (steps === maxSteps) acc = 0;
    }
    sim.encodeReadback(enc);
    renderer.render(enc, context.getCurrentTexture().createView(), sim, cam, now / 1000);
    device.queue.submit([enc.finish()]);
    sim.afterSubmit();
    drawTrail(now);
    // re-route rope attachments after cuts (throttled), and now and then while torn cloth may rip on
    if (now - lastAttach > 250 && (sim.attachmentsDirty || (now - lastAttach > 1000 && sim.cfg.pinAbove > 0 && sim.lastTorn !== undefined))) {
      lastAttach = now;
      sim.refreshAttachments(sim.attachmentsDirty);
    }

    fpsN++;
    if (now - fpsT > 500) {
      fps = (fpsN * 1000) / (now - fpsT);
      fpsN = 0;
      fpsT = now;
      const fmt = (n) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${n}`);
      statsEl.textContent = sim.warmup > 0 ? 'settling…' : `${fps.toFixed(0)} fps · ${fmt(sim.nGrain)} ${sim.cfg.kind === 'fluid' ? 'fluid' : 'grain'} particles · ${fmt(sim.nCloth)} skin particles · ${sim.cfg.substeps} substeps`;
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main().catch((e) => fail(`Failed to start: ${e.message}`));
