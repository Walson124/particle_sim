import { mat4, v3 } from '../gpu/math.js';
import { BLUR, CLOTH, COMPOSITE, FLOOR, GRAIN, TONEMAP } from './shaders.js';

const TU = GPUTextureUsage;
const VF = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
const FR = GPUShaderStage.FRAGMENT;
const SHADOW_RES = 2048;
const FRAME_BYTES = 560;

export class Renderer {
  constructor(device, format) {
    this.device = device;
    this.format = format;
    const d = device;

    this.frameBuf = d.createBuffer({ size: FRAME_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.shadowFrameBuf = d.createBuffer({ size: FRAME_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.frameData = new ArrayBuffer(FRAME_BYTES);
    this.frameF = new Float32Array(this.frameData);
    this.frameU = new Uint32Array(this.frameData);

    this.cmpSampler = d.createSampler({ compare: 'less', magFilter: 'linear', minFilter: 'linear' });
    this.linSampler = d.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    this.shadowTex = d.createTexture({ size: [SHADOW_RES, SHADOW_RES], format: 'depth32float', usage: TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING });
    this.shadowView = this.shadowTex.createView();

    const ubo = (binding) => ({ binding, visibility: VF, buffer: { type: 'uniform' } });
    const ssbo = (binding) => ({ binding, visibility: VF, buffer: { type: 'read-only-storage' } });
    const tex = (binding, sampleType) => ({ binding, visibility: FR, texture: { sampleType } });
    this.bgl = {
      grain: d.createBindGroupLayout({ entries: [ubo(0), ssbo(1), ssbo(2)] }),
      cloth: d.createBindGroupLayout({ entries: [ubo(0), ssbo(1), ssbo(2), ssbo(3), ssbo(4), ssbo(5), ssbo(6), ssbo(7)] }),
      frame: d.createBindGroupLayout({ entries: [ubo(0)] }),
      shadowTex: d.createBindGroupLayout({ entries: [tex(0, 'depth'), { binding: 1, visibility: FR, sampler: { type: 'comparison' } }] }),
      sceneDepth: d.createBindGroupLayout({ entries: [tex(2, 'depth')] }),
      blur: d.createBindGroupLayout({ entries: [ubo(0), tex(1, 'unfilterable-float'), tex(2, 'unfilterable-float')] }),
      comp: d.createBindGroupLayout({
        entries: [
          ubo(0), tex(1, 'float'), tex(2, 'unfilterable-float'), tex(3, 'depth'),
          { binding: 4, visibility: FR, sampler: { type: 'comparison' } },
          { binding: 5, visibility: FR, sampler: { type: 'filtering' } },
          tex(6, 'unfilterable-float'),
          { binding: 7, visibility: FR, buffer: { type: 'read-only-storage' } },
        ],
      }),
      tone: d.createBindGroupLayout({ entries: [ubo(0), tex(1, 'unfilterable-float')] }),
    };

    const mod = (label, code) => d.createShaderModule({ label, code });
    const grainM = mod('grain', GRAIN), clothM = mod('cloth', CLOTH), floorM = mod('floor', FLOOR);
    const blurM = mod('blur', BLUR), compM = mod('composite', COMPOSITE), toneM = mod('tone', TONEMAP);
    const layout = (...g) => d.createPipelineLayout({ bindGroupLayouts: g });
    const depth = (write = true, compare = 'less') => ({ format: 'depth32float', depthWriteEnabled: write, depthCompare: compare });
    const hdr = [{ format: 'rgba16float' }];
    const L = this.bgl;
    const P = (desc) => d.createRenderPipeline({ primitive: { topology: 'triangle-list', cullMode: 'none' }, ...desc });

    this.pipe = {
      bg: P({ layout: layout(L.frame, L.shadowTex), vertex: { module: floorM, entryPoint: 'vsFull' }, fragment: { module: floorM, entryPoint: 'fsBg', targets: hdr }, depthStencil: depth(false, 'always') }),
      floor: P({ layout: layout(L.frame, L.shadowTex), vertex: { module: floorM, entryPoint: 'vsFloor' }, fragment: { module: floorM, entryPoint: 'fsFloor', targets: hdr }, depthStencil: depth() }),
      cloth: P({ layout: layout(L.cloth, L.shadowTex), vertex: { module: clothM, entryPoint: 'vsCloth' }, fragment: { module: clothM, entryPoint: 'fsCloth', targets: hdr }, depthStencil: depth() }),
      clothShadow: P({ layout: layout(L.cloth), vertex: { module: clothM, entryPoint: 'vsCloth' }, depthStencil: depth() }),
      grain: P({ layout: layout(L.grain, L.shadowTex), vertex: { module: grainM, entryPoint: 'vsGrain' }, fragment: { module: grainM, entryPoint: 'fsGrain', targets: hdr }, depthStencil: depth() }),
      grainShadow: P({ layout: layout(L.grain), vertex: { module: grainM, entryPoint: 'vsGrain' }, fragment: { module: grainM, entryPoint: 'fsGrainDepth', targets: [] }, depthStencil: depth() }),
      spray: P({ layout: layout(L.grain, L.shadowTex), vertex: { module: grainM, entryPoint: 'vsSandSpray' }, fragment: { module: grainM, entryPoint: 'fsGrain', targets: hdr }, depthStencil: depth() }),
      rope: P({ layout: layout(L.grain, L.shadowTex), vertex: { module: grainM, entryPoint: 'vsRope' }, fragment: { module: grainM, entryPoint: 'fsGrain', targets: hdr }, depthStencil: depth() }),
      ropeShadow: P({ layout: layout(L.grain), vertex: { module: grainM, entryPoint: 'vsRope' }, fragment: { module: grainM, entryPoint: 'fsGrainDepth', targets: [] }, depthStencil: depth() }),
      fluidDepth: P({ layout: layout(L.grain, L.sceneDepth), vertex: { module: grainM, entryPoint: 'vsGrain' }, fragment: { module: grainM, entryPoint: 'fsFluidDepth', targets: [{ format: 'rgba32float' }] }, depthStencil: depth() }),
      sandSplat: P({ layout: layout(L.grain, L.sceneDepth), vertex: { module: grainM, entryPoint: 'vsGrain' }, fragment: { module: grainM, entryPoint: 'fsSandSplat', targets: [{ format: 'rgba32float' }] }, depthStencil: depth() }),
      fluidThick: P({
        layout: layout(L.grain, L.sceneDepth),
        vertex: { module: grainM, entryPoint: 'vsGrain' },
        fragment: {
          module: grainM, entryPoint: 'fsFluidThick',
          targets: [{ format: 'r16float', blend: { color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' } } }],
        },
      }),
      blurH: P({ layout: layout(L.blur), vertex: { module: blurM, entryPoint: 'vsFull' }, fragment: { module: blurM, entryPoint: 'fsBlurH', targets: [{ format: 'rg32float' }] } }),
      blurDisk: P({ layout: layout(L.blur), vertex: { module: blurM, entryPoint: 'vsFull' }, fragment: { module: blurM, entryPoint: 'fsBlurDisk', targets: [{ format: 'rg32float' }] } }),
      blurV: P({ layout: layout(L.blur), vertex: { module: blurM, entryPoint: 'vsFull' }, fragment: { module: blurM, entryPoint: 'fsBlurV', targets: [{ format: 'rg32float' }] } }),
      composite: P({ layout: layout(L.comp), vertex: { module: compM, entryPoint: 'vsFull' }, fragment: { module: compM, entryPoint: 'fsComposite', targets: hdr } }),
      tone: P({ layout: layout(L.tone), vertex: { module: toneM, entryPoint: 'vsFull' }, fragment: { module: toneM, entryPoint: 'fsTone', targets: [{ format }] } }),
    };

    this.bgShadowTex = d.createBindGroup({ layout: L.shadowTex, entries: [{ binding: 0, resource: this.shadowView }, { binding: 1, resource: this.cmpSampler }] });
    this.bgFrame = d.createBindGroup({ layout: L.frame, entries: [{ binding: 0, resource: { buffer: this.frameBuf } }] });
  }

  resize(w, h) {
    if (this.w === w && this.h === h) return;
    this.w = w;
    this.h = h;
    const d = this.device;
    for (const t of this.targets || []) t.destroy();
    const mk = (format, usage = TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING) => d.createTexture({ size: [w, h], format, usage });
    this.sceneColor = mk('rgba16float');
    this.sceneDepth = mk('depth32float');
    this.fluidDepth = mk('rgba32float'); // surface splat: depth (+ grain id, local normal for sand)
    this.fluidZ = mk('depth32float', TU.RENDER_ATTACHMENT);
    this.fluidThick = mk('r16float');
    this.blurA = mk('rg32float');
    this.blurB = mk('rg32float');
    this.composited = mk('rgba16float');
    this.targets = [this.sceneColor, this.sceneDepth, this.fluidDepth, this.fluidZ, this.fluidThick, this.blurA, this.blurB, this.composited];
    const L = this.bgl;
    const fb = { binding: 0, resource: { buffer: this.frameBuf } };
    this.bgSceneDepth = d.createBindGroup({ layout: L.sceneDepth, entries: [{ binding: 2, resource: this.sceneDepth.createView() }] });
    this.bgBlurH = d.createBindGroup({ layout: L.blur, entries: [fb, { binding: 1, resource: this.fluidDepth.createView() }, { binding: 2, resource: this.fluidThick.createView() }] });
    this.bgBlurV = d.createBindGroup({ layout: L.blur, entries: [fb, { binding: 1, resource: this.blurA.createView() }, { binding: 2, resource: this.blurA.createView() }] });
    this.makeCompositeBindGroup();
    this.bgTone = d.createBindGroup({ layout: L.tone, entries: [fb, { binding: 1, resource: this.composited.createView() }] });
  }

  // depends on both the render targets and the simulation's particle buffer
  makeCompositeBindGroup() {
    if (!this.sim || !this.sceneColor) return;
    this.bgComp = this.device.createBindGroup({
      layout: this.bgl.comp,
      entries: [
        { binding: 0, resource: { buffer: this.frameBuf } },
        { binding: 1, resource: this.sceneColor.createView() },
        { binding: 2, resource: this.blurB.createView() },
        { binding: 3, resource: this.shadowView },
        { binding: 4, resource: this.cmpSampler },
        { binding: 5, resource: this.linSampler },
        { binding: 6, resource: this.fluidDepth.createView() },
        { binding: 7, resource: { buffer: this.sim.buf.pos } },
      ],
    });
  }

  setSim(sim) {
    this.sim = sim;
    this.makeCompositeBindGroup();
    const d = this.device, L = this.bgl, B = sim.buf;
    const grain = (fbuf) => d.createBindGroup({
      layout: L.grain,
      entries: [fbuf, B.pos, B.info].map((b, i) => ({ binding: i, resource: { buffer: b } })),
    });
    const cloth = (fbuf) => d.createBindGroup({
      layout: L.cloth,
      entries: [fbuf, B.pos, B.info, B.tris, B.triEdges, B.edgeAlive, B.normals, B.restLra].map((b, i) => ({ binding: i, resource: { buffer: b } })),
    });
    this.bgGrain = grain(this.frameBuf);
    this.bgGrainShadow = grain(this.shadowFrameBuf);
    this.bgCloth = cloth(this.frameBuf);
    this.bgClothShadow = cloth(this.shadowFrameBuf);
  }

  writeFrame(buf, m, sim, time) {
    const f = this.frameF, u = this.frameU, cfg = sim.cfg;
    f.set(m.view, 0);
    f.set(m.proj, 16);
    f.set(m.viewProj, 32);
    f.set(mat4.invert(m.view), 48);
    f.set(this.lightVP, 64);
    f.set(mat4.invert(m.proj), 80);
    f.set([...m.eye, 1], 96);
    f.set([...this.lightDir, 1], 100);
    f.set([this.w, this.h, 1 / this.w, 1 / this.h], 104);
    f.set([sim.radius, m.ortho ? 1 : 0, time, cfg.clothMat], 108);
    u[112] = sim.nCloth;
    u[113] = cfg.pairs ? 2 : 1;
    u[114] = this.grainInstances(sim);
    u[115] = cfg.grainMat;
    const rope = sim.ropeTop;
    f.set(rope ? [...rope, 1] : [0, 0, 0, 0], 116);
    f.set(rope ? [rope[0], 3.5, rope[2], 0.0075] : [0, 0, 0, 0], 120);
    const a = cfg.absorb || [0, 0, 0], s = cfg.scatter || [0, 0, 0];
    // w: screen-space smoothing radius in particle radii (liquid: smooth, sand: keep it lumpy)
    f.set([...a, cfg.grainMat === 0 ? 2.6 : 2.2], 124);
    // w: bilateral depth tolerance of the surface smoothing, in particle radii
    f.set([...s, cfg.grainMat === 0 ? 7 : 3], 128);
    const S = cfg.shape;
    f.set([S.ax, S.ay, S.az, S.exp], 132);
    f.set([S.centerY, S.neckStart, S.neckScale, 0], 136);
    this.device.queue.writeBuffer(buf, 0, this.frameData);
  }

  // Debug: when `this.profile` is set, every pass records begin/end timestamps under a label.
  passDesc(desc, label) {
    const pr = this.profile;
    if (!pr) return desc;
    const i = pr.labels.length * 2;
    pr.labels.push(label);
    return { ...desc, timestampWrites: { querySet: pr.qs, beginningOfPassWriteIndex: i, endOfPassWriteIndex: i + 1 } };
  }

  // sand/fluid: one instance per particle; rice: RICE_PER_PAIR kernels per simulated pair
  grainInstances(sim) {
    return sim.cfg.pairs ? (sim.nGrain / 2) * 8 : sim.nGrain;
  }

  render(encoder, swapView, sim, cam, time) {
    const aspect = this.w / this.h;
    const cm = cam.matrices(aspect);
    // key light: orthographic shadow frustum around the scene
    this.lightDir = v3.norm([0.45, 0.82, 0.36]);
    const target = [0, 0.35, 0];
    const lightView = mat4.lookAt(v3.add(target, v3.scale(this.lightDir, 3)), target, [0, 1, 0]);
    const lightProj = mat4.ortho(-1.15, 1.15, -1.15, 1.15, 0.5, 6.5);
    this.lightVP = mat4.multiply(lightProj, lightView);
    this.writeFrame(this.frameBuf, { ...cm, ortho: false }, sim, time);
    this.writeFrame(this.shadowFrameBuf, { view: lightView, proj: lightProj, viewProj: this.lightVP, eye: v3.add(target, v3.scale(this.lightDir, 3)), ortho: true }, sim, time);

    const fluid = sim.cfg.kind === 'fluid';
    const sand = sim.cfg.grainMat === 0;
    const surface = fluid || sand; // rendered as a screen-space surface instead of particles
    const nInst = this.grainInstances(sim);
    const clothVerts = sim.mesh.numTris * 3;

    // shadow map
    let p = encoder.beginRenderPass(this.passDesc({
      colorAttachments: [],
      depthStencilAttachment: { view: this.shadowView, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
    }, 'shadow'));
    p.setPipeline(this.pipe.clothShadow);
    p.setBindGroup(0, this.bgClothShadow);
    p.draw(clothVerts);
    p.setPipeline(this.pipe.grainShadow);
    p.setBindGroup(0, this.bgGrainShadow);
    p.draw(6, nInst);
    if (sim.ropeTop) {
      p.setPipeline(this.pipe.ropeShadow);
      p.draw(6);
    }
    p.end();

    // main scene
    p = encoder.beginRenderPass(this.passDesc({
      colorAttachments: [{ view: this.sceneColor.createView(), clearValue: [0, 0, 0, 1], loadOp: 'clear', storeOp: 'store' }],
      depthStencilAttachment: { view: this.sceneDepth.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
    }, 'scene'));
    p.setBindGroup(1, this.bgShadowTex);
    p.setPipeline(this.pipe.bg);
    p.setBindGroup(0, this.bgFrame);
    p.draw(3);
    p.setPipeline(this.pipe.floor);
    p.draw(6);
    p.setPipeline(this.pipe.cloth);
    p.setBindGroup(0, this.bgCloth);
    p.draw(clothVerts);
    p.setBindGroup(0, this.bgGrain);
    if (!surface) {
      p.setPipeline(this.pipe.grain);
      p.draw(6, nInst);
    }
    if (sand) {
      p.setPipeline(this.pipe.spray);
      p.draw(6, sim.nGrain * 6);
    }
    if (sim.ropeTop) {
      p.setPipeline(this.pipe.rope);
      p.draw(6);
    }
    p.end();

    if (surface) {
      p = encoder.beginRenderPass(this.passDesc({
        colorAttachments: [{ view: this.fluidDepth.createView(), clearValue: [0, 0, 0, 0], loadOp: 'clear', storeOp: 'store' }],
        depthStencilAttachment: { view: this.fluidZ.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' },
      }, 'fluidDepth'));
      p.setPipeline(sand ? this.pipe.sandSplat : this.pipe.fluidDepth);
      p.setBindGroup(0, this.bgGrain);
      p.setBindGroup(1, this.bgSceneDepth);
      p.draw(6, nInst);
      p.end();

      p = encoder.beginRenderPass(this.passDesc({
        colorAttachments: [{ view: this.fluidThick.createView(), clearValue: [0, 0, 0, 0], loadOp: 'clear', storeOp: 'store' }],
      }, 'fluidThick'));
      if (fluid) {
        p.setPipeline(this.pipe.fluidThick);
        p.setBindGroup(0, this.bgGrain);
        p.setBindGroup(1, this.bgSceneDepth);
        p.draw(6, nInst);
      }
      p.end();

      const blurs = sand
        ? [[this.pipe.blurDisk, this.bgBlurH, this.blurB]]
        : [[this.pipe.blurH, this.bgBlurH, this.blurA], [this.pipe.blurV, this.bgBlurV, this.blurB]];
      for (const [pipe, bg, target] of blurs) {
        p = encoder.beginRenderPass(this.passDesc({ colorAttachments: [{ view: target.createView(), clearValue: [0, 0, 0, 0], loadOp: 'clear', storeOp: 'store' }] }, 'blur'));
        p.setPipeline(pipe);
        p.setBindGroup(0, bg);
        p.draw(3);
        p.end();
      }
    }

    p = encoder.beginRenderPass(this.passDesc({ colorAttachments: [{ view: this.composited.createView(), clearValue: [0, 0, 0, 1], loadOp: 'clear', storeOp: 'store' }] }, 'composite'));
    p.setPipeline(this.pipe.composite);
    p.setBindGroup(0, this.bgComp);
    p.draw(3);
    p.end();

    p = encoder.beginRenderPass(this.passDesc({ colorAttachments: [{ view: swapView, clearValue: [0, 0, 0, 1], loadOp: 'clear', storeOp: 'store' }] }, 'tone'));
    p.setPipeline(this.pipe.tone);
    p.setBindGroup(0, this.bgTone);
    p.draw(3);
    p.end();
  }
}
