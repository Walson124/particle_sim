// WGSL compute kernels for the unified particle solver.
//
// Per substep:  integrate -> count -> scanBlocks -> scanTop -> scanAdd -> scatter
//               -> [lambda] -> particles (contacts / PBF) -> stretch (per colour) -> finalize
// Neighbour kernels run one thread per *sorted* slot for memory coherence.

export const COMMON = /* wgsl */ `
struct Params {
  dt: f32, radius: f32, cellSize: f32, gravity: f32,
  numParticles: u32, numCloth: u32, hashMask: u32, numBlocks: u32,
  muS: f32, muK: f32, cohesion: f32, damping: f32,
  clothInvMass: f32, floorY: f32, bounds: f32, sor: f32,
  h: f32, invRho0: f32, lambdaEps: f32, scorrK: f32,
  invWdq: f32, poly6: f32, spikyGrad: f32, viscosity: f32,
  stretchK: f32, bendK: f32, compressK: f32, tearRatio: f32,
  anchorK: f32, lraSlack: f32, pairRest: f32, maxSpeed: f32,
  shapeA: vec4f,
  shapeB: vec4f,
  pulse: f32, time: f32, fluidCohesion: f32, floorMu: f32,
  clothSor: f32, boundaryPsi: f32, fluidOmega: f32, fluidMaxStep: f32,
  clothDisk: f32, contactDamp: f32, maxCorr: f32, wakeAll: f32,
  sleepTol: f32, vWake: f32, sleepSteps: f32, vSepMax: f32,
  pad4: f32, pad5: f32, pad6: f32, pad7: f32,
};

const KIND_GRAIN: u32 = 0u;
const KIND_FLUID: u32 = 1u;
const KIND_CLOTH: u32 = 2u;
const DEAD: u32 = 4u;
const PINNED: u32 = 8u;
const PAIRED: u32 = 16u;
const FRAYED: u32 = 32u; // near a cut: only these cloth particles' edges may rip
// dynamic state, copied into the sorted flags by scatter so neighbours can see it
const GROUNDED: u32 = 64u; // rests (through a chain of contacts) on the floor
const ASLEEP: u32 = 128u;  // at rest: frozen in place, acts as immovable support
const MOVING: u32 = 256u;  // really travelling (keeps leaving its still-period anchor), not just buzzing

// per-particle sleep state: xyz = where the current still period began, w = bitcast word with
// [0..15] stillness counter, 16 grounded, 17 asleep, [24..31] contacts, [18..23]: skin: hops to
// support; grains: countdown since the grain last really travelled (left its still-period anchor)
const S_GROUNDED: u32 = 0x10000u;
const S_ASLEEP: u32 = 0x20000u;
// skin is also supported through its own fabric: hops (along cloth links) to a skin particle that
// rests on the floor or on grounded grains. 63 = unsupported. Recomputed every substep, so a piece
// that is cut loose loses its support within a few frames.
const HOP_SHIFT: u32 = 18u;
const HOP_MASK: u32 = 0xfc0000u;
const HOP_NONE: u32 = 63u;
fn hopsOf(w: u32) -> u32 { return (w & HOP_MASK) >> HOP_SHIFT; }
const EMPTY: u32 = 0xffffffffu;

fn invMassOf(f: u32) -> f32 {
  if ((f & PINNED) != 0u) { return 0.0; }
  if ((f & 3u) == KIND_CLOTH) { return P.clothInvMass; }
  return 1.0;
}

fn cellOf(p: vec3f) -> vec3i { return vec3i(floor(p / P.cellSize)); }

// x-major hash: cells adjacent in x are adjacent in the table, so after the counting sort a row
// of three neighbouring cells is one contiguous run of particles (9 lookups instead of 27)
fn hashCell(c: vec3i) -> u32 {
  return (u32(c.x) + u32(c.y) * 2048u + u32(c.z) * 2654435761u) & P.hashMask;
}

fn shapeF(pw: vec3f) -> f32 {
  let p = pw - vec3f(0.0, P.shapeB.x, 0.0);
  let yn = p.y / P.shapeA.y;
  let s = mix(1.0, P.shapeB.z, smoothstep(P.shapeB.y, 1.0, yn));
  let e = P.shapeA.w;
  let q = max(abs(vec3f(p.x / (P.shapeA.x * s), yn, p.z / (P.shapeA.z * s))), vec3f(1e-6));
  return pow(pow(q.x, e) + pow(q.y, e) + pow(q.z, e), 1.0 / e);
}
`;

const TYPES = {
  pos: 'array<vec4f>', prev: 'array<vec4f>', posNew: 'array<vec4f>',
  src: 'array<vec4f>', dst: 'array<vec4f>', sortedPos: 'array<vec4f>', sortedDisp: 'array<vec4f>',
  restLra: 'array<vec4f>', normals: 'array<vec4f>',
  info: 'array<u32>', cellCount: 'array<u32>', cellStart: 'array<u32>', blockSums: 'array<u32>',
  edgeAlive: 'array<u32>', vtList: 'array<u32>',
  lambdaS: 'array<f32>', vfix: 'array<vec4f>', sleep: 'array<vec4f>', wake: 'array<u32>',
  pCellOff: 'array<vec2u>', adjStart: 'array<vec2u>', vtStart: 'array<vec2u>',
  adj: 'array<vec4u>', cons: 'array<vec4u>', tris: 'array<vec4u>', triEdges: 'array<vec4u>',
};

// binding spec: "name" (read-only), "name:rw", "name:rwa" (read_write, u32 -> atomic<u32>).
export function parseBinds(binds) {
  return binds.map((b) => {
    const [name, mode = 'r'] = b.split(':');
    return { name, mode };
  });
}

export function bindingHeader(binds) {
  return parseBinds(binds)
    .map(({ name, mode }, i) => {
      if (name === 'P') return `@group(0) @binding(${i}) var<uniform> P: Params;`;
      let t = TYPES[name];
      if (!t) throw new Error(`unknown buffer ${name}`);
      if (mode === 'rwa') t = t.replace('u32', 'atomic<u32>');
      const access = mode === 'r' ? 'read' : 'read_write';
      return `@group(0) @binding(${i}) var<storage, ${access}> ${name}: ${t};`;
    })
    .join('\n');
}

// Iterates candidate neighbours k (sorted slots) of xi within sqrt(R2) <= one cell. Rows of three
// cells whose y/z slab lies entirely outside the radius are skipped; a row that wraps the table end
// is split.
const neighborLoopOpen = (R2) => /* wgsl */ `
  let c = cellOf(xi);
  let fc = xi / P.cellSize - vec3f(c);
  let wrap = P.hashMask + 1u;
  for (var oz = -1; oz <= 1; oz++) {
    var ez = 0.0;
    if (oz < 0) { ez = fc.z; } else if (oz > 0) { ez = 1.0 - fc.z; }
    for (var oy = -1; oy <= 1; oy++) {
      var ey = 0.0;
      if (oy < 0) { ey = fc.y; } else if (oy > 0) { ey = 1.0 - fc.y; }
      if ((ey * ey + ez * ez) * P.cellSize * P.cellSize >= ${R2}) { continue; }
      let h0 = hashCell(c + vec3i(-1, oy, oz));
      let e = h0 + 3u;
      let a0 = cellStart[h0];
      let b0 = cellStart[min(e, wrap)];
      var a1 = 0u;
      var extra = 0u;
      if (e > wrap) { a1 = cellStart[0]; extra = cellStart[e - wrap] - a1; }
      let n = (b0 - a0) + extra;
      for (var q = 0u; q < n; q++) {
        var k = a0 + q;
        if (k >= b0) { k = a1 + (k - b0); }
`;
const NEIGHBOR_LOOP_CLOSE = /* wgsl */ `
      }
    }
  }
`;

export const KERNELS = {
  // Predict positions and bin them into the spatial hash (count pass of the counting sort).
  integrate: {
    wg: 256,
    binds: ['P', 'pos:rw', 'prev:rw', 'info:r', 'sleep:r', 'cellCount:rwa', 'pCellOff:rw', 'vfix:r'],
    code: /* wgsl */ `
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.numParticles) { return; }
  let f = info[i];
  if ((f & DEAD) != 0u) { pCellOff[i] = vec2u(EMPTY, 0u); return; }
  var p = pos[i];
  // velocity of the substep that just finished (PBD: v = (x - x_prev) / dt)
  var v = (p.xyz - prev[i].xyz) / P.dt - vfix[i].xyz;
  if (any(v != v)) { v = vec3f(0.0); }
  prev[i] = p;
  if (invMassOf(f) != 0.0 && (bitcast<u32>(sleep[i].w) & S_ASLEEP) == 0u) {
    v.y += P.gravity * P.dt;
    v *= P.damping;
    let sp = length(v);
    if (sp > P.maxSpeed) { v *= P.maxSpeed / sp; }
    p = vec4f(p.xyz + v * P.dt, p.w);
    pos[i] = p;
  }
  let h = hashCell(cellOf(p.xyz));
  pCellOff[i] = vec2u(h, atomicAdd(&cellCount[h], 1u));
}`,
  },

  scanBlocks: {
    wg: 256,
    binds: ['P', 'cellCount:rw', 'cellStart:rw', 'blockSums:rw'],
    code: /* wgsl */ `
var<workgroup> tmp: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let t = lid.x;
  let base = wid.x * 1024u + t * 4u;
  let a0 = cellCount[base];
  let a1 = cellCount[base + 1u];
  let a2 = cellCount[base + 2u];
  let a3 = cellCount[base + 3u];
  let s = a0 + a1 + a2 + a3;
  tmp[t] = s;
  workgroupBarrier();
  for (var off = 1u; off < 256u; off = off << 1u) {
    var x = 0u;
    if (t >= off) { x = tmp[t - off]; }
    workgroupBarrier();
    tmp[t] = tmp[t] + x;
    workgroupBarrier();
  }
  let ex = tmp[t] - s;
  cellStart[base] = ex;
  cellStart[base + 1u] = ex + a0;
  cellStart[base + 2u] = ex + a0 + a1;
  cellStart[base + 3u] = ex + a0 + a1 + a2;
  // counts are consumed: zero them here instead of in a separate clear pass
  cellCount[base] = 0u;
  cellCount[base + 1u] = 0u;
  cellCount[base + 2u] = 0u;
  cellCount[base + 3u] = 0u;
  if (t == 255u) { blockSums[wid.x] = tmp[255]; }
}`,
  },

  scanTop: {
    wg: 256,
    binds: ['P', 'blockSums:rw'],
    code: /* wgsl */ `
var<workgroup> tmp: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3u) {
  let t = lid.x;
  let nb = P.numBlocks;
  let base = t * 4u;
  var v = array<u32, 4>(0u, 0u, 0u, 0u);
  for (var k = 0u; k < 4u; k++) {
    if (base + k < nb) { v[k] = blockSums[base + k]; }
  }
  let s = v[0] + v[1] + v[2] + v[3];
  tmp[t] = s;
  workgroupBarrier();
  for (var off = 1u; off < 256u; off = off << 1u) {
    var x = 0u;
    if (t >= off) { x = tmp[t - off]; }
    workgroupBarrier();
    tmp[t] = tmp[t] + x;
    workgroupBarrier();
  }
  var run = tmp[t] - s;
  for (var k = 0u; k < 4u; k++) {
    if (base + k < nb) { blockSums[base + k] = run; }
    run += v[k];
  }
  // total number of inserted particles lives just past the block sums
  if (t == 255u) {
    blockSums[nb] = tmp[255];
    blockSums[nb + 1u] += 1u; // substep counter (staggers the sleep audit)
  }
}`,
  },

  scanAdd: {
    wg: 256,
    binds: ['P', 'cellStart:rw', 'blockSums:r'],
    code: /* wgsl */ `
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let h = gid.x;
  if (h <= P.hashMask) { cellStart[h] += blockSums[h >> 10u]; }
  if (h == 0u) {
    cellStart[P.hashMask + 1u] = blockSums[P.numBlocks];
    cellStart[P.hashMask + 2u] = blockSums[P.numBlocks + 1u];
  }
}`,
  },

  scatter: {
    wg: 256,
    binds: ['P', 'pos:r', 'prev:r', 'info:r', 'pCellOff:r', 'cellStart:r', 'sortedPos:rw', 'sortedDisp:rw', 'sleep:r'],
    code: /* wgsl */ `
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.numParticles) { return; }
  let co = pCellOff[i];
  if (co.x == EMPTY) { return; }
  let s = cellStart[co.x] + co.y;
  let p = pos[i].xyz;
  sortedPos[s] = vec4f(p, bitcast<f32>(i));
  let st = bitcast<u32>(sleep[i].w);
  var f = info[i];
  if ((st & S_GROUNDED) != 0u) { f |= GROUNDED; }
  if ((st & S_ASLEEP) != 0u) { f |= ASLEEP; }
  else if ((f & 3u) == KIND_GRAIN && (st & HOP_MASK) != 0u) { f |= MOVING; }
  sortedDisp[s] = vec4f(p - prev[i].xyz, bitcast<f32>(f));
}`,
  },

  lambda: {
    wg: 128,
    binds: ['P', 'sortedPos:r', 'sortedDisp:r', 'cellStart:r', 'lambdaS:rw'],
    code: /* wgsl */ `
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let total = cellStart[P.hashMask + 1u];
  let s = gid.x;
  if (s >= total) { return; }
  let fi = bitcast<u32>(sortedDisp[s].w);
  if ((fi & 3u) != KIND_FLUID) { lambdaS[s] = 0.0; return; }
  let xi = sortedPos[s].xyz;
  let h = P.h;
  let h2 = h * h;
  var rho = 0.0;
  var gsum = vec3f(0.0);
  var g2 = 0.0;
  ${neighborLoopOpen('h2')}
          let b = sortedPos[k];
          let rv = xi - b.xyz;
          let r2 = dot(rv, rv);
          if (r2 >= h2) { continue; }
          let fj = bitcast<u32>(sortedDisp[k].w);
          let fluidJ = (fj & 3u) == KIND_FLUID;
          let m = select(P.boundaryPsi, 1.0, fluidJ);
          let q = 1.0 - r2 / h2;
          rho += m * P.poly6 * q * q * q;
          if (k != s && r2 > 1e-14) {
            let dist = sqrt(r2);
            let t = 1.0 - dist / h;
            let gr = (P.spikyGrad * t * t * P.invRho0 * m) * (rv / dist);
            gsum += gr;
            if (fluidJ) { g2 += dot(gr, gr); }
          }
  ${NEIGHBOR_LOOP_CLOSE}
  var invR = P.invRho0;
  // heartbeat: fluid still inside the body wants to occupy more volume -> it spurts out of wounds
  if (P.shapeB.w > 0.0 && shapeF(xi) < 0.97) { invR *= 1.0 + P.shapeB.w * P.pulse; }
  let C = max(rho * invR - 1.0, 0.0);
  lambdaS[s] = -C / (g2 + dot(gsum, gsum) + P.lambdaEps);
}`,
  },

  particles: {
    wg: 64,
    wgVar: true,
    binds: ['P', 'sortedPos:r', 'sortedDisp:r', 'cellStart:r', 'lambdaS:r', 'posNew:rw', 'normals:r', 'sleep:rw', 'wake:rwa'],
    bindsGrain: ['P', 'sortedPos:r', 'sortedDisp:r', 'cellStart:r', 'vfix:rw', 'posNew:rw', 'normals:r', 'sleep:rw', 'wake:rwa'],
    code: /* wgsl */ `
@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let total = cellStart[P.hashMask + 1u];
  let s = gid.x;
  if (s >= total) { return; }
  let a = sortedPos[s];
  let i = bitcast<u32>(a.w);
  let xi = a.xyz;
  let da = sortedDisp[s];
  let fi = bitcast<u32>(da.w);
  let wi = invMassOf(fi);
  let ki = fi & 3u;
  vfix[i] = vec4f(0.0); // @grain
  if (wi == 0.0) { posNew[i] = vec4f(xi, 0.0); sleep[i] = vec4f(0.0, 0.0, 0.0, bitcast<f32>(HOP_NONE << HOP_SHIFT)); return; }

  // a sleeping grain stays frozen (and skips the neighbour search) until something wakes it
  let sv = sleep[i];
  let st = bitcast<u32>(sv.w);
  let woken = atomicExchange(&wake[i], 0u) != 0u || P.wakeAll > 0.5;
  let sleeping = (fi & ASLEEP) != 0u && !woken;
  // every 8th substep (staggered per particle) a sleeping particle re-checks its surroundings: it
  // wakes if nothing holds it up any more, or if a neighbour is moving (e.g. the pile is draining)
  let audit = sleeping && ((i + cellStart[P.hashMask + 2u]) & 7u) == 0u;
  if (sleeping && !audit) {
    posNew[i] = vec4f(xi, -1.0 - f32(st >> 24u));
    return;
  }
  var supported = xi.y <= P.floorY + 1.05 * P.radius;
  var stirred = false;

  let r = P.radius;
  let fast = length(da.xyz) > P.vWake * P.dt;
  var groundedN = false;
  let d0 = 2.0 * r;
  let h = P.h;
  let h2 = h * h;
  let reach = select(P.cellSize, d0 * 1.4, ki == KIND_FLUID);
  let rng = select(reach, max(h, reach), ki == KIND_FLUID);
  let rng2 = rng * rng;
  let lamI = lambdaS[s]; // @fluid
  let prevI = xi - da.xyz;
  var partner = EMPTY;
  if ((fi & PAIRED) != 0u) { partner = P.numCloth + ((i - P.numCloth) ^ 1u); }
  var nI = vec3f(0.0);
  if (ki == KIND_CLOTH) { nI = normals[i].xyz; }

  var dFluid = vec3f(0.0);
  var dCont = vec3f(0.0);
  var dFixed = vec3f(0.0); // contacts with immovable (asleep / pinned) neighbours: no over-relaxation
  var nCont = 0.0;
  var nb = 0.0;

  ${neighborLoopOpen('rng2')}
          if (k == s) { continue; }
          let b = sortedPos[k];
          let rv = xi - b.xyz;
          let r2 = dot(rv, rv);
          if (r2 >= rng2 || r2 < 1e-14) { continue; }
          let j = bitcast<u32>(b.w);
          if (j == partner) { continue; }
          let db = sortedDisp[k];
          let fj = bitcast<u32>(db.w);
          let kj = fj & 3u;
          if (ki == KIND_CLOTH && kj == KIND_CLOTH) { continue; }
          let dist = sqrt(r2);
          if (dist < d0 * 1.2) { nb += 1.0; }
          if (FLUID_SCENE && ki == KIND_FLUID && kj == KIND_FLUID) { // @fluid
            let nf = rv / dist; // @fluid
            let q = 1.0 - r2 / h2; // @fluid
            let ww = q * q * q; // @fluid
            let x = ww * P.invWdq; // @fluid
            let scorr = -P.scorrK * x * x * x * x; // @fluid
            let t = 1.0 - dist / h; // @fluid
            let grad = (-P.spikyGrad * t * t) * nf; // @fluid
            dFluid += ((lamI + lambdaS[k] + scorr) * P.invRho0 * P.fluidOmega) * grad; // @fluid
            dFluid += P.viscosity * ww * (db.xyz - da.xyz); // @fluid
            dFluid -= P.fluidCohesion * h * (dist / h) * t * nf; // @fluid
            continue; // @fluid
          } // @fluid
          if (FLUID_SCENE && ki == KIND_FLUID && dist < h) { // @fluid
            let t = 1.0 - dist / h; // @fluid
            let grad = (-P.spikyGrad * t * t) * (rv / dist); // @fluid
            dFluid += (lamI * P.boundaryPsi * P.invRho0 * P.fluidOmega) * grad; // @fluid
          } // @fluid
          var nrm = rv / dist;
          var pen = d0 - dist;
          if (ki == KIND_CLOTH || kj == KIND_CLOTH) {
            // Oriented sheet contact: the cloth particle acts as a small disk. Penetration is
            // measured along the sheet normal towards whichever side the other particle came
            // from, so a squeezed particle can never slip between cloth particles.
            var cn = nI;
            var rel = -rv;          // other particle relative to the cloth particle
            var prevRel = (b.xyz - db.xyz) - prevI;
            var sgn = -1.0;         // this thread moves along -n when it is the cloth
            if (kj == KIND_CLOTH) {
              cn = normals[j].xyz;
              rel = rv;
              prevRel = prevI - (b.xyz - db.xyz);
              sgn = 1.0;
            }
            if (dot(cn, cn) < 0.5) { continue; }
            // A particle that came from (nearly) within the sheet's plane has no meaningful side:
            // keep the one it is on now. Otherwise a sign flip of that noise would shove it through
            // the whole sheet thickness in one substep, kicking the cloth (it then rings, or flies).
            let pd = dot(prevRel, cn);
            let side = select(select(-1.0, 1.0, pd >= 0.0), select(-1.0, 1.0, dot(rel, cn) >= 0.0), abs(pd) < 0.25 * r);
            let dn = dot(rel, cn) * side;
            let tl = length(rel - dot(rel, cn) * cn);
            if (tl > P.clothDisk) { continue; }
            nrm = cn * (side * sgn);
            // the disk's rim is rounded off (thickness fades to 0), so a particle sliding in
            // sideways, e.g. at a fold, is pushed gradually instead of by the full sheet depth at once
            let th = d0 * (1.0 - smoothstep(0.75 * P.clothDisk, P.clothDisk, tl));
            pen = min(th - dn, th + d0);
          }
          if (pen > 0.0) {
            if ((fj & GROUNDED) != 0u && kj != KIND_FLUID) { groundedN = true; }
            if ((fj & ASLEEP) != 0u && fast) { atomicStore(&wake[j], 1u); }
            if (b.y < xi.y - 0.25 * r || kj == KIND_CLOTH) { supported = true; }
            if ((fj & MOVING) != 0u && kj != KIND_CLOTH) { stirred = true; }
            // sleeping neighbours are immovable support
            let wj = select(invMassOf(fj), 0.0, (fj & ASLEEP) != 0u);
            let fr = wi / (wi + wj);
            var corr = nrm * (pen * fr);
            // Coulomb friction on the relative tangential displacement of this substep
            let relD = da.xyz - db.xyz;
            let reln = dot(relD, nrm);
            let tang = relD - reln * nrm;
            // inelastic contact: damp relative normal motion (granular contacts dissipate energy)
            corr -= nrm * (reln * P.contactDamp * fr);
            let lt = length(tang);
            if (lt < P.muS * pen) {
              corr -= tang * fr;
            } else if (lt > 1e-12) {
              corr -= tang * (fr * min(P.muK * pen / lt, 1.0));
            }
            if (wj == 0.0) { dFixed += corr; } else { dCont += corr; }
            nCont += 1.0;
          } else if (P.cohesion > 0.0 && ki == KIND_GRAIN && kj == KIND_GRAIN) {
            dCont -= nrm * ((dist - d0) * P.cohesion * 0.5);
          }
  ${NEIGHBOR_LOOP_CLOSE}
  let fl2 = dot(dFluid, dFluid);
  let maxStep = P.fluidMaxStep * r;
  if (fl2 > maxStep * maxStep) { dFluid *= maxStep / sqrt(fl2); }
  // cap the contact push per substep: releasing compressed grains (e.g. at a fresh cut) then
  // spills them at a plausible speed instead of spraying them across the room
  var dc = (dCont * P.sor + dFixed) / max(nCont, 1.0);
  let dcl = length(dc);
  let maxCorr = P.maxCorr * r;
  if (dcl > maxCorr) { dc *= maxCorr / dcl; }
  if (audit && supported && !stirred) {
    posNew[i] = vec4f(xi, -1.0 - f32(st >> 24u)); // still resting: stay asleep
    return;
  }
  // anything that was asleep and is processed now has just been woken: restart its still period
  let wokeNow = (fi & ASLEEP) != 0u;
  var x = xi + dFluid + dc;

  // floor with friction
  let fl = P.floorY + r;
  if (x.y < fl) {
    let pen = fl - x.y;
    x.y = fl;
    let t = vec2f(x.x - prevI.x, x.z - prevI.z);
    let lt = length(t);
    if (lt < P.floorMu * pen) {
      x.x = prevI.x;
      x.z = prevI.z;
    } else if (lt > 1e-12) {
      let f = min(P.floorMu * pen / lt, 1.0);
      x.x -= t.x * f;
      x.z -= t.y * f;
    }
  }
  x.x = clamp(x.x, -P.bounds, P.bounds);
  x.z = clamp(x.z, -P.bounds, P.bounds);
  x.y = min(x.y, 6.0);
  // sleep bookkeeping: a supported particle that has stayed within sleepTol of where its still
  // period began (however much it buzzed in between) for sleepSteps substeps falls asleep
  let direct = x.y <= fl + 0.05 * r || groundedN;
  var hops = HOP_NONE;
  // 0 only while directly supported; otherwise keep the fabric value finalize() propagates
  if (ki == KIND_CLOTH) { hops = select(clamp(hopsOf(st), 1u, HOP_NONE), 0u, direct); }
  let grounded = direct || hops < HOP_NONE;
  var counter = 0u;
  var anchor = x;
  // grains: "really moving" countdown; being woken does not count, only travelling does
  var moved = max(hopsOf(st), 1u) - 1u;
  if (ki != KIND_FLUID && grounded && P.wakeAll < 0.5) {
    let c0 = select(st & 0xffffu, 0u, wokeNow);
    if (c0 > 0u && length(x - sv.xyz) < P.sleepTol) {
      counter = min(c0 + 1u, 0xffffu);
      anchor = sv.xyz;
    } else {
      if (c0 > 0u) { moved = HOP_NONE; } // left its anchor: it is travelling
      counter = 1u;
    }
  }
  if (ki != KIND_CLOTH) { hops = moved; }
  let asleep = f32(counter) >= P.sleepSteps;
  if (asleep) { x = prevI; }
  let word = counter | select(0u, S_GROUNDED, grounded) | select(0u, S_ASLEEP, asleep) | (hops << HOP_SHIFT) | (u32(min(nb, 255.0)) << 24u);
  sleep[i] = vec4f(anchor, bitcast<f32>(word));
  // w carries the neighbour count through to rendering (ambient occlusion); negative = asleep
  posNew[i] = vec4f(x, select(nb, -1.0 - nb, asleep));
  // Contacts may resolve overlap completely, but must not fling grains: the outward speed they add
  // along the push direction is capped (stopping an impact or riding along with moving cloth is
  // unaffected). integrate() removes the excess next substep, so positions stay fully resolved.
  let dcn = length(dc); // @grain
  if (dcn > 1e-9) { // @grain
    let n = dc / dcn; // @grain
    let vout = dot(x - prevI, n) / P.dt; // @grain
    let cap = max(P.vSepMax, dot(da.xyz, n) / P.dt); // @grain
    if (vout > cap) { vfix[i] = vec4f(n * (vout - cap), 0.0); } // @grain
  } // @grain
}`,
  },

  // Stretch constraints of one colour: exact projection, no two constraints share a vertex.
  stretch: {
    wg: 64,
    binds: ['P', 'posNew:rw', 'info:r', 'cons:r', 'edgeAlive:rwa'],
    code: /* wgsl */ `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let k = gid.x;
  if (k >= arrayLength(&cons)) { return; }
  let c = cons[k];
  if (c.x == EMPTY) { return; }
  let fa = info[c.x];
  let fb = info[c.y];
  if (((fa | fb) & DEAD) != 0u) { return; }
  if (atomicLoad(&edgeAlive[c.w]) == 0u) { return; }
  let qa = posNew[c.x];
  let qb = posNew[c.y];
  // sleeping particles (marked by a negative w) are immovable
  let wa = select(invMassOf(fa), 0.0, qa.w < 0.0);
  let wb = select(invMassOf(fb), 0.0, qb.w < 0.0);
  let ws = wa + wb;
  if (ws == 0.0) { return; }
  let pa = qa.xyz;
  let pb = qb.xyz;
  let d = pa - pb;
  let L = length(d);
  if (L < 1e-9) { return; }
  let rl = bitcast<f32>(c.z);
  let kk = select(P.compressK, P.stretchK, L > rl);
  let corr = (kk * (L - rl) / (ws * L)) * d;
  posNew[c.x] = vec4f(pa - wa * corr, qa.w);
  posNew[c.y] = vec4f(pb + wb * corr, qb.w);
}`,
  },

  // Once per frame on converged positions: edges strained past the tear limit rip.
  tear: {
    wg: 256,
    binds: ['P', 'pos:r', 'info:r', 'cons:r', 'edgeAlive:rw'],
    code: /* wgsl */ `
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let k = gid.x;
  if (k >= arrayLength(&cons)) { return; }
  let c = cons[k];
  if (c.x == EMPTY) { return; }
  let f = info[c.x] | info[c.y];
  if ((f & DEAD) != 0u || (f & FRAYED) == 0u) { return; }
  let L = length(pos[c.x].xyz - pos[c.y].xyz);
  if (L > bitcast<f32>(c.z) * P.tearRatio) { edgeAlive[c.w] = 0u; }
}`,
  },

  // posNew -> pos: soft bending (Jacobi), long range attachments, rest-shape anchoring, rice pairs.
  finalize: {
    wg: 256,
    binds: ['P', 'posNew:r', 'pos:rw', 'info:r', 'adjStart:r', 'adj:r', 'edgeAlive:r', 'restLra:r', 'sleep:rw'],
    code: /* wgsl */ `
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.numParticles) { return; }
  let p = posNew[i];
  let f = info[i];
  // asleep: frozen this substep (negative w carries the contact count)
  if (p.w < 0.0) { pos[i] = vec4f(p.xyz, -p.w - 1.0); return; }
  if (i >= P.numCloth) {
    var x = p.xyz;
    if ((f & PAIRED) != 0u) {
      let j = P.numCloth + ((i - P.numCloth) ^ 1u);
      let d = x - posNew[j].xyz;
      let L = length(d);
      // a sleeping partner does not move, so this end takes the whole correction
      let share = select(0.5, 1.0, posNew[j].w < 0.0);
      if (L > 1e-9) { x -= (share * (L - P.pairRest) / L) * d; }
    }
    x.y = max(x.y, P.floorY + P.radius);
    pos[i] = vec4f(x, p.w);
    return;
  }
  let wi = invMassOf(f);
  if ((f & DEAD) != 0u || wi == 0.0) { pos[i] = p; return; }

  var acc = vec3f(0.0);
  var cnt = 0.0;
  var hopN = HOP_NONE;
  let rg = adjStart[i];
  for (var k = rg.x; k < rg.x + rg.y; k++) {
    let e = adj[2u * k];
    let w = adj[2u * k + 1u];
    let fj = info[e.x];
    if ((fj & DEAD) != 0u || edgeAlive[e.y] == 0u) { continue; }
    // both triangles must be intact, or the spring would bridge a cut or a rip
    if (edgeAlive[w.x] == 0u || edgeAlive[w.y] == 0u || edgeAlive[w.z] == 0u || edgeAlive[w.w] == 0u) { continue; }
    hopN = min(hopN, hopsOf(bitcast<u32>(sleep[e.x].w)) + 1u);
    let d = p.xyz - posNew[e.x].xyz;
    let L = length(d);
    if (L < 1e-9) { continue; }
    let rl = bitcast<f32>(e.z);
    let wj = invMassOf(fj);
    acc -= (P.bendK * (L - rl) * (wi / (wi + wj)) / L) * d;
    cnt += 1.0;
  }
  var x = p.xyz + acc * (P.clothSor / max(cnt, 1.0));
  // support through the fabric: one hop further than the best-supported neighbour
  let sw = sleep[i];
  let word = bitcast<u32>(sw.w);
  if (hopsOf(word) != 0u) {
    let hops = min(hopN, HOP_NONE);
    sleep[i] = vec4f(sw.xyz, bitcast<f32>((word & ~HOP_MASK) | (hops << HOP_SHIFT)));
  }

  let rp = restLra[2u * i];
  let pin = bitcast<u32>(rp.w);
  if (pin != EMPTY) {
    let fp = info[pin];
    if ((fp & PINNED) != 0u && (fp & DEAD) == 0u) {
      let l = restLra[2u * i + 1u];
      let v = x - l.xyz;
      let L = length(v);
      let maxL = l.w * P.lraSlack;
      if (L > maxL) { x = l.xyz + v * (maxL / L); }
    }
  }
  if (P.anchorK > 0.0) { x = mix(x, rp.xyz, P.anchorK); }
  // constraints solved after the contact pass must not push cloth back through the floor
  x.y = max(x.y, P.floorY + P.radius);
  pos[i] = vec4f(x, p.w);
}`,
  },

  normals: {
    wg: 256,
    binds: ['P', 'pos:r', 'info:r', 'vtStart:r', 'vtList:r', 'tris:r', 'triEdges:r', 'edgeAlive:r', 'normals:rw'],
    code: /* wgsl */ `
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.numCloth) { return; }
  let rg = vtStart[i];
  var n = vec3f(0.0);
  for (var k = rg.x; k < rg.x + rg.y; k++) {
    let t = vtList[k];
    let tri = tris[t];
    let te = triEdges[t];
    if (((info[tri.x] | info[tri.y] | info[tri.z]) & DEAD) != 0u) { continue; }
    if (edgeAlive[te.x] == 0u || edgeAlive[te.y] == 0u || edgeAlive[te.z] == 0u) { continue; }
    let a = pos[tri.x].xyz;
    n += cross(pos[tri.y].xyz - a, pos[tri.z].xyz - a);
  }
  let L = length(n);
  normals[i] = vec4f(select(vec3f(0.0, 1.0, 0.0), n / L, L > 1e-14), 0.0);
}`,
  },
};
