# Particle Breach

Real-time GPU particle simulation in the browser (WebGPU). A bag full of particles hangs, sits,
or stands in a studio. Slash or stab it with the mouse and the contents pour out: sand from a
burlap sandbag, rice from a cotton sack, or blood from a body.

## Run

```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # static site in dist/
npm run preview    # serve dist/ locally
```

Needs a WebGPU browser: Chrome or Edge 113+ (desktop and Android), or Safari 26+. Firefox needs
`dom.webgpu.enabled`.

## Controls

| Input | Action |
| --- | --- |
| Left-drag | Slash (cuts a line through the skin) |
| Left-click | Puncture (round hole, size = *Tool size*) |
| Right-drag / Alt-drag / two fingers | Orbit |
| Middle-drag | Pan |
| Wheel / pinch | Zoom |
| `R` / `Space` | Reset / pause |

The panel switches scenario and particle count (32k–256k). It can also cut the rope, and the
*Material* folder holds live physics parameters. URL parameters: `?scene=sand|rice|blood&n=128000`.
For tuning you can add `&set=key:value,...`, which overrides any preset field (see
`src/sim/presets.js`).

## Deploying

`npm run build` writes a self-contained static site to `dist/` (about 40 KB gzipped, no runtime
dependencies, relative asset paths). You can upload it as-is:

- **Cloudflare Workers**: `wrangler.jsonc` defines a static-assets-only Worker serving `dist/`.
  Run `npx wrangler login` once, then `npm run deploy` (builds, then deploys) and open
  `https://particle-sim.<your-subdomain>.workers.dev` (live: https://particle-sim.walson124.workers.dev).
- **GitHub Pages**: push `dist/` to a `gh-pages` branch, or use the official Pages action with
  `path: dist`.
- **Netlify / Cloudflare Pages / Vercel**: build command `npm run build`, output directory `dist`.
- **Any web server or S3 bucket**: copy the contents of `dist/`.

The site needs HTTPS (or localhost), because WebGPU only runs in secure contexts.

## How it works

Everything runs on the GPU in WGSL compute shaders. The CPU only builds the scene and handles cuts.

**Unified particle solver** (position based dynamics, in the spirit of NVIDIA Flex). The bag's skin
is itself made of particles, so one collision system handles grain–grain, grain–skin and
fluid–skin contacts.

Each substep runs these kernels:

1. Integrate: velocity from the previous substep, gravity plus air damping, then predict
   positions.
2. Spatial hash: count → two-level prefix scan → scatter into cell-sorted arrays. The hash is
   x-major, so each row of three neighboring cells is one contiguous run of sorted particles:
   neighbor search is 9 row lookups instead of 27 cell lookups, and rows outside the search
   radius are skipped.
3. *Fluid only:* PBF density constraint. Computes λ with a poly6/spiky kernel; the skin counts as
   boundary particles.
4. Contacts: averaged Jacobi with Coulomb friction (static/kinetic) and inelastic normal damping.
   Fluid particles get a PBF pressure correction, XSPH viscosity and cohesion instead.
5. Skin stretch: graph-coloured Gauss–Seidel, with exact projection per colour.
6. Finalize: soft bending, long-range attachments to the rope, rest-shape anchoring for the body,
   and rigid rice-kernel pairs (each grain is a two-sphere capsule).

Once per frame, a tear pass rips over-strained edges and skin normals are rebuilt.

Robustness details that make it hold together:

- **Oriented sheet contacts.** A skin particle acts as a small disk. Penetration is measured along
  the skin normal toward the side the other particle came from. Grains can't squeeze through
  intact cloth. They leave only through holes.
- **Cuts** remove skin particles along the knife's path (ray-cast on read-back skin positions).
  The material around a wound is marked *frayed*, and only frayed edges can tear. Slits rip
  open under load, but intact cloth never ruptures by itself.
- **Substeps scale with particle count** (about N^0.23), so deep piles stay nearly incompressible.
- **No flinging.** Contacts always resolve overlap completely, but the outward speed they may add
  is capped (0.25 m/s along the push). Grains packed under load no longer spring out of a fresh
  cut or bounce off the pile, while stopping on impact and riding along with moving cloth work
  as before. The excess velocity is removed in the next substep, so positions stay correct.
- **Sleeping.** A supported particle (on the floor, on grounded grains, or, for skin, connected
  through the fabric to skin that is) that stays within about 1 mm of one spot for a while falls
  asleep: it freezes, skips the neighbour search and acts as immovable support. It wakes on a
  cut or release, when a fast neighbour hits it, or at a periodic audit if nothing holds it up any
  more or a touching grain is really travelling, which is how a draining pile stays awake.
  Grains must be still for 0.25 s (sand) or 2 s (rice, which drains slowly through a slit).
  Skin counts as held up through the fabric too (on a slope or in a fold nothing is directly
  below it), and its stillness is judged on its solved position, so cloth lying on the pile
  falls asleep instead of twitching forever.
- **Cut-off pieces fall.** The skin's long-range attachments to the rope are re-routed over the
  cloth that is still connected (cuts and tears included) after every cut, so a piece severed
  from the rope drops instead of hanging from nothing. Cutting a particle also kills its edges,
  and a bending spring only acts while both of its triangles are intact, so no spring can bridge
  even the thinnest cut or rip.
- **Cloth comes to rest.** The skin resists compression as stiffly as stretching (it still
  buckles into folds via the soft bending), and sheet contacts fade out towards the rim of each
  cloth particle's disk. Crumpled cloth on the floor or the pile then settles instead of
  twitching or being kicked upwards.
- **Heartbeat**: blood inside the body sees a pulsing lower rest density, so wounds spurt.

**Rendering**:

- **Sand** is drawn as a material, not as particles: the grains are splatted into a depth
  buffer, smoothed into one surface with an isotropic bilateral filter, and shaded with
  sub-millimetre grain detail (mineral colour speckle, per-grain facet normals, crevice
  darkening, quartz glints). The detail is anchored to each particle's local surface, so it moves
  with the sand. Loose grains (few neighbours) are drawn as dispersed ~1 mm grains instead.
- **Rice**: each simulated two-sphere pair is drawn as 8 real-size kernels (ray-traced
  spheroids, about 7 × 2.3 mm) packed in the pair's volume and carried rigidly with it.
- Everything is lit with 3×3 PCF shadow maps and AO from the contact count.
- The skin is procedural (woven burlap, printed cotton, or skin with subsurface wrap lighting),
  with bump mapping from screen-space derivatives.
- Blood uses screen-space fluid rendering: sphere depth + thickness → bilateral depth smoothing →
  normals → refraction, Beer–Lambert absorption and Fresnel reflection.
- ACES tonemapping.

## Layout

```
src/main.js            app shell: device, camera, input (slash/puncture), UI, frame loop
src/sim/presets.js     scenario + material parameters
src/sim/mesh.js        ring-meshed superellipsoid bag, constraint topology + colouring, particle fill
src/sim/kernels.js     all simulation WGSL
src/sim/simulation.js  GPU buffers, pipelines, step(), picking, cutting
src/sim/debug.js       readback statistics + per-kernel GPU timing (used by tests)
src/render/            render passes and their WGSL
tests/                 headless-Chrome harness (needs Chrome installed; uses your real GPU)
```

## Tests and tuning

```bash
npm run smoke                                   # load each scene, slash it, save screenshots to tests/out
node tests/probe.mjs sand "set=substeps:16" 64000 4   # physics stats over time (leaks, tears, compression, jitter)
KERNELS=1 node tests/perf.mjs sand 64000 ""     # GPU ms per kernel and per render pass
node tests/thincut.mjs sand                     # one-row ring cut: the lower piece must reach the floor
node tests/clothrest.mjs sand                   # cut-off cloth landing on the pile must settle, not twitch or fly
node tests/slash.mjs sand                       # mouse-like slashes: no cloth flies upwards
```

GPU time per frame on an AMD RDNA2 GPU (simulation + about 1 ms of rendering):

| Particles | Sandbag | Rice sack | Body (blood) |
| --- | --- | --- | --- |
| 64k | 6.4 ms | 6.0 ms | 6.7 ms |
| 128k | 11.3 ms | – | – |
| 256k | 24.5 ms (16 ms once poured sand has settled) | – | – |

If the GPU cannot sustain 60 Hz, the simulation runs one step per frame (smooth, slightly slow
motion) instead of trying to catch up and stuttering.

Picking uses a per-event screen-space broadphase, so a fast slash costs a few ms of CPU.
