// WGSL for all render passes. Each export is a self-contained module (common code prepended).

export const RCOMMON = /* wgsl */ `
struct Frame {
  view: mat4x4f, proj: mat4x4f, viewProj: mat4x4f, invView: mat4x4f,
  lightVP: mat4x4f, invProj: mat4x4f,
  camPos: vec4f, lightDir: vec4f, screen: vec4f,
  misc: vec4f,   // particle radius, isOrtho, time, cloth material
  misc2: vec4u,  // grain base index, stride, instance count, grain material
  ropeA: vec4f, ropeB: vec4f,
  colA: vec4f, colB: vec4f,
  shapeA: vec4f, shapeB: vec4f,
};
@group(0) @binding(0) var<uniform> F: Frame;

const PI: f32 = 3.14159265;
const SUN: vec3f = vec3f(1.0, 0.93, 0.84) * 2.5;

fn hashu(n: u32) -> u32 {
  var x = n * 747796405u + 2891336453u;
  x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  return (x >> 22u) ^ x;
}
fn hashf(n: u32) -> f32 { return f32(hashu(n) >> 8u) * (1.0 / 16777216.0); }
fn hash2(p: vec2f) -> f32 {
  let q = bitcast<vec2u>(vec2i(floor(p)));
  return hashf((q.x * 1597334677u) ^ (q.y * 3812015801u + 0x9e3779b9u));
}
fn hash3i(c: vec3i, seed: u32) -> u32 {
  let q = bitcast<vec3u>(c);
  return hashu((q.x * 73856093u) ^ (q.y * 19349663u) ^ (q.z * 83492791u) ^ (seed * 2654435761u));
}

// octahedral unit-vector encoding (2 floats)
fn octEncode(n: vec3f) -> vec2f {
  var p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
  if (n.z < 0.0) { p = (1.0 - abs(p.yx)) * select(vec2f(-1.0), vec2f(1.0), p >= vec2f(0.0)); }
  return p;
}
fn octDecode(e: vec2f) -> vec3f {
  var n = vec3f(e, 1.0 - abs(e.x) - abs(e.y));
  let t = max(-n.z, 0.0);
  n.x += select(t, -t, n.x >= 0.0);
  n.y += select(t, -t, n.y >= 0.0);
  return normalize(n);
}

// size of one visible sand grain (metres)
const SAND_GRAIN: f32 = 0.00045;

fn vnoise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = hash2(i);
  let b = hash2(i + vec2f(1.0, 0.0));
  let c = hash2(i + vec2f(0.0, 1.0));
  let d = hash2(i + vec2f(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
fn fbm(p: vec2f) -> f32 {
  var s = 0.0;
  var a = 0.5;
  var q = p;
  for (var i = 0; i < 5; i++) {
    s += a * vnoise(q);
    q = q * 2.03 + vec2f(1.7, 9.2);
    a *= 0.5;
  }
  return s;
}

fn skyColor(d: vec3f) -> vec3f {
  let t = clamp(d.y, -1.0, 1.0);
  let horizon = vec3f(0.34, 0.35, 0.37);
  let zenith = vec3f(0.11, 0.13, 0.17);
  let ground = vec3f(0.13, 0.12, 0.11);
  var c = mix(horizon, zenith, smoothstep(0.0, 0.7, t));
  c = mix(ground, c, smoothstep(-0.08, 0.0, t));
  // soft key light glow (large studio softbox)
  let g = max(dot(d, F.lightDir.xyz), 0.0);
  c += vec3f(1.0, 0.95, 0.88) * (pow(g, 24.0) * 0.8 + pow(g, 400.0) * 6.0);
  return c;
}

fn ambient(n: vec3f) -> vec3f {
  let t = n.y * 0.5 + 0.5;
  return mix(vec3f(0.15, 0.13, 0.12), vec3f(0.36, 0.40, 0.48), t) * 0.75;
}

fn shadowAt(sm: texture_depth_2d, ss: sampler_comparison, wp: vec3f, n: vec3f, bias: f32) -> f32 {
  let p = F.lightVP * vec4f(wp + n * bias, 1.0);
  let uv = vec2f(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) || p.z > 1.0) { return 1.0; }
  let texel = 1.0 / 2048.0;
  var s = 0.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      s += textureSampleCompareLevel(sm, ss, uv + vec2f(f32(x), f32(y)) * texel * 1.25, p.z - 0.0002);
    }
  }
  return s / 9.0;
}

@vertex fn vsFull(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}
`;

// ---------------------------------------------------------------------------------------------
// Grains (sand spheres / rice capsules / fluid spheres) as ray-traced impostors.
export const GRAIN = RCOMMON + /* wgsl */ `
@group(0) @binding(1) var<storage, read> pos: array<vec4f>;
@group(0) @binding(2) var<storage, read> info: array<u32>;
@group(1) @binding(0) var shadowMap: texture_depth_2d;
@group(1) @binding(1) var shadowSamp: sampler_comparison;
@group(1) @binding(2) var sceneDepth: texture_depth_2d;

struct GOut {
  @builtin(position) clip: vec4f,
  @location(0) vp: vec3f,
  @location(1) @interpolate(flat) pa: vec3f,
  @location(2) @interpolate(flat) pb: vec3f,
  @location(3) @interpolate(flat) rad: f32,
  @location(4) @interpolate(flat) id: u32,
  @location(5) @interpolate(flat) ao: f32,
  @location(6) @interpolate(flat) kind: u32,
};

fn impostor(wa: vec3f, wb: vec3f, rad: f32, vi: u32) -> GOut {
  var quad = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
                             vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
  let corner = quad[vi];
  var o: GOut;
  let pa = (F.view * vec4f(wa, 1.0)).xyz;
  let pb = (F.view * vec4f(wb, 1.0)).xyz;
  let ortho = F.misc.y > 0.5;
  let c = 0.5 * (pa + pb);
  let zf = max(pa.z, pb.z) + rad * 1.05;
  o.pa = pa;
  o.pb = pb;
  o.rad = rad;
  if (!ortho && zf > -0.01) {
    o.clip = vec4f(2.0, 2.0, 2.0, 1.0);
    return o;
  }
  var k = 1.0;
  if (!ortho) { k = zf / c.z; }
  let ax = pb.xy - pa.xy;
  let la = length(ax);
  var u = vec2f(1.0, 0.0);
  if (la > 1e-7) { u = ax / la; }
  let v = vec2f(-u.y, u.x);
  let infl = select(1.3, 1.02, ortho);
  let off = (u * (corner.x * (0.5 * la + rad)) + v * (corner.y * rad)) * infl;
  let q = vec3f((c.xy + off) * k, zf);
  o.vp = q;
  o.clip = F.proj * vec4f(q, 1.0);
  return o;
}

// Rice: every simulated two-sphere pair is drawn as RICE_PER_PAIR real-size kernels (spheroids
// about 7 x 2.3 mm) packed inside the pair's volume and carried rigidly in the pair's frame.
const RICE_PER_PAIR: u32 = 8u;

fn riceGrain(vi: u32, ii: u32) -> GOut {
  let pair = ii / RICE_PER_PAIR;
  let i0 = F.misc2.x + pair * 2u;
  let qa = pos[i0];
  let qb = pos[i0 + 1u];
  let r = F.misc.x;
  let c = 0.5 * (qa.xyz + qb.xyz);
  let ax = qb.xyz - qa.xyz;
  let L = length(ax);
  var U = vec3f(1.0, 0.0, 0.0);
  if (L > 1e-7) { U = ax / L; }
  // stable frame around the pair axis from a per-pair reference direction
  let hp = pair * 9781u + 17u;
  let refDir = vec3f(hashf(hp) - 0.5, hashf(hp + 1u) - 0.5, hashf(hp + 2u) - 0.5);
  var V = cross(U, refDir);
  if (dot(V, V) < 1e-6) { V = cross(U, vec3f(0.31, 0.83, 0.46)); }
  V = normalize(V);
  let W = cross(U, V);
  let hs = ii * 7919u + 3u;
  let t = (hashf(hs) - 0.5) * (L + 1.3 * r);
  let rr = sqrt(hashf(hs + 1u)) * 0.8 * r;
  let ang = hashf(hs + 2u) * 6.2831853;
  let cw = c + U * t + V * (rr * cos(ang)) + W * (rr * sin(ang));
  // kernels lean along the pair axis, with plenty of spread so the pile looks random
  let rnd = vec3f(hashf(hs + 3u) - 0.5, hashf(hs + 4u) - 0.5, hashf(hs + 5u) - 0.5) * 2.0;
  let dir = normalize(U * select(-1.0, 1.0, hashf(hs + 6u) > 0.5) * 1.2 + rnd);
  let sz = 0.9 + 0.2 * hashf(hs + 7u);
  let a = 1.08 * r * sz;   // half length
  let b = 0.36 * r * sz;   // radius
  var o = impostor(cw - dir * a, cw + dir * a, b, vi);
  o.id = ii;
  o.ao = 0.5 * (qa.w + qb.w);
  o.kind = 2u;
  return o;
}

@vertex fn vsGrain(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> GOut {
  let mat = F.misc2.w;
  if (mat == 1u) { return riceGrain(vi, ii); }
  let i0 = F.misc2.x + ii;
  let q = pos[i0];
  var rs = 1.0;
  // fluid spheres are inflated so the screen-space surface reads as one continuous liquid
  if (mat == 2u) { rs = 1.6; }
  // sand is rendered as a surface: packed grains are inflated so they fuse into one mass, while
  // stray grains (few neighbours) stay small and read as fine spray
  if (mat == 0u) {
    if (q.w < 2.5) {
      var o: GOut;
      o.clip = vec4f(2.0, 2.0, 2.0, 1.0);
      return o;
    }
    rs = mix(1.05, 1.35, smoothstep(2.5, 6.0, q.w));
  }
  var o = impostor(q.xyz, q.xyz, F.misc.x * rs, vi);
  o.id = i0;
  o.ao = q.w;
  o.kind = 0u;
  return o;
}

// Sand spray: a simulated grain with few neighbours (in the air, rolling loose) is drawn as a
// puff of SPRAY_PER_PARTICLE real-size grains scattered through its volume.
const SPRAY_PER_PARTICLE: u32 = 6u;

@vertex fn vsSandSpray(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> GOut {
  let p = ii / SPRAY_PER_PARTICLE;
  let q = pos[F.misc2.x + p];
  var o: GOut;
  if (q.w >= 3.5) {
    o.clip = vec4f(2.0, 2.0, 2.0, 1.0);
    return o;
  }
  let r = F.misc.x;
  let hs = ii * 7919u + 11u;
  let dir = normalize(vec3f(hashf(hs) - 0.5, hashf(hs + 1u) - 0.5, hashf(hs + 2u) - 0.5) + vec3f(1e-4));
  let c = q.xyz + dir * (pow(hashf(hs + 3u), 0.33) * 1.8 * r);
  let rad = SAND_GRAIN * (0.6 + 0.8 * hashf(hs + 4u));
  o = impostor(c, c, rad, vi);
  o.id = ii;
  o.ao = 0.0;
  o.kind = 3u;
  return o;
}

@vertex fn vsRope(@builtin(vertex_index) vi: u32) -> GOut {
  var o = impostor(F.ropeA.xyz, F.ropeB.xyz, F.ropeB.w, vi);
  o.id = 0u;
  o.ao = 0.0;
  o.kind = 1u;
  return o;
}

fn sphereHit(ro: vec3f, rd: vec3f, ce: vec3f, r: f32) -> f32 {
  let oc = ro - ce;
  let b = dot(oc, rd);
  let c = dot(oc, oc) - r * r;
  let h = b * b - c;
  if (h < 0.0) { return -1.0; }
  return -b - sqrt(h);
}

fn capHit(ro: vec3f, rd: vec3f, pa: vec3f, pb: vec3f, r: f32) -> f32 {
  let ba = pb - pa;
  let baba = dot(ba, ba);
  if (baba < 1e-12) { return sphereHit(ro, rd, pa, r); }
  let oa = ro - pa;
  let bard = dot(ba, rd);
  let baoa = dot(ba, oa);
  let rdoa = dot(rd, oa);
  let oaoa = dot(oa, oa);
  let a = baba - bard * bard;
  if (a < 1e-9 * baba) {
    // looking straight down the axis: nearest end cap
    let t0 = sphereHit(ro, rd, pa, r);
    let t1 = sphereHit(ro, rd, pb, r);
    if (t0 < 0.0) { return t1; }
    if (t1 < 0.0) { return t0; }
    return min(t0, t1);
  }
  let b = baba * rdoa - baoa * bard;
  let c = baba * oaoa - baoa * baoa - r * r * baba;
  let h = b * b - a * c;
  if (h < 0.0) { return -1.0; }
  let t = (-b - sqrt(h)) / a;
  let y = baoa + t * bard;
  if (y > 0.0 && y < baba) { return t; }
  var oc = ro - pb;
  if (y <= 0.0) { oc = oa; }
  let b2 = dot(rd, oc);
  let c2 = dot(oc, oc) - r * r;
  let h2 = b2 * b2 - c2;
  if (h2 > 0.0) { return -b2 - sqrt(h2); }
  return -1.0;
}

fn capNormal(p: vec3f, a: vec3f, b: vec3f) -> vec3f {
  let ba = b - a;
  let pa = p - a;
  let baba = dot(ba, ba);
  var h = 0.0;
  if (baba > 1e-12) { h = clamp(dot(pa, ba) / baba, 0.0, 1.0); }
  return normalize(pa - h * ba);
}

// prolate spheroid: centre c, unit axis u, half length a, radius b
fn spheroidHit(ro: vec3f, rd: vec3f, c: vec3f, u: vec3f, a: f32, b: f32) -> f32 {
  let s = b / a;
  let oc = ro - c;
  let o2 = oc + u * (dot(oc, u) * (s - 1.0));
  let d2 = rd + u * (dot(rd, u) * (s - 1.0));
  let A = dot(d2, d2);
  let B = dot(o2, d2);
  let C = dot(o2, o2) - b * b;
  let h = B * B - A * C;
  if (h < 0.0) { return -1.0; }
  return (-B - sqrt(h)) / A;
}

fn spheroidNormal(p: vec3f, c: vec3f, u: vec3f, a: f32, b: f32) -> vec3f {
  let q = p - c;
  let along = dot(q, u);
  return normalize(q - u * along + u * (along * (b * b) / (a * a)));
}

// ray hit for any impostor kind: view-space hit point and normal (ok = false on a miss)
struct Hit { p: vec3f, n: vec3f, ok: bool };
fn traceImpostor(i: GOut) -> Hit {
  let rd = rayDir(i.vp);
  var h: Hit;
  if (i.kind == 2u) {
    let c = 0.5 * (i.pa + i.pb);
    let a = 0.5 * length(i.pb - i.pa);
    let u = (i.pb - i.pa) / (2.0 * a);
    let t = spheroidHit(i.vp, rd, c, u, a, i.rad);
    h.ok = t >= 0.0;
    h.p = i.vp + rd * t;
    h.n = spheroidNormal(h.p, c, u, a, i.rad);
  } else {
    let t = capHit(i.vp, rd, i.pa, i.pb, i.rad);
    h.ok = t >= 0.0;
    h.p = i.vp + rd * t;
    h.n = capNormal(h.p, i.pa, i.pb);
  }
  return h;
}

fn rayDir(vp: vec3f) -> vec3f {
  if (F.misc.y > 0.5) { return vec3f(0.0, 0.0, -1.0); }
  return normalize(vp);
}

struct GFrag {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
};

fn shadeGrain(i: GOut, wp: vec3f, n: vec3f, v: vec3f, nv: vec3f) -> vec3f {
  let L = F.lightDir.xyz;
  let sh = shadowAt(shadowMap, shadowSamp, wp, n, i.rad * 0.6);
  var ao = mix(1.0, 0.42, smoothstep(2.0, 9.0, i.ao));
  ao *= mix(0.6, 1.0, smoothstep(0.0, i.rad * 4.0, wp.y));
  var albedo = vec3f(0.8);
  var wrap = 0.0;
  var specAmt = 0.04;
  var shin = 16.0;
  if (i.kind == 1u) {
    let along = dot((F.invView * vec4f(i.vp, 1.0)).xyz - F.ropeA.xyz, normalize(F.ropeB.xyz - F.ropeA.xyz));
    let tw = 0.5 + 0.5 * sin(along * 900.0 + atan2(nv.y, nv.x) * 3.0);
    albedo = vec3f(0.42, 0.33, 0.21) * (0.65 + 0.35 * tw);
    ao = 1.0;
  } else if (i.kind == 3u) {
    // loose sand grain
    let h1 = hashf(i.id * 3u + 11u);
    let h2 = hashf(i.id * 7u + 5u);
    albedo = mix(vec3f(0.50, 0.38, 0.24), vec3f(0.38, 0.28, 0.17), h1);
    if (h2 > 0.95) { albedo = vec3f(0.16, 0.14, 0.12); }
    else if (h2 > 0.9) { albedo = vec3f(0.78, 0.75, 0.68); }
    ao = 1.0;
  } else {
    // rice: translucent, slightly pearly white kernels with the odd chalky or yellowed one
    let h1 = hashf(i.id * 3u + 1u);
    let h2 = hashf(i.id * 5u + 2u);
    albedo = mix(vec3f(0.80, 0.79, 0.75), vec3f(0.72, 0.69, 0.61), h1 * h1);
    if (h2 > 0.93) { albedo = vec3f(0.88, 0.88, 0.86); }
    else if (h2 > 0.9) { albedo = vec3f(0.86, 0.78, 0.58); }
    wrap = 0.5;
    specAmt = 0.45;
    shin = 90.0;
  }
  let ndl = dot(n, L);
  let diff = max((ndl + wrap) / (1.0 + wrap), 0.0);
  let hv = normalize(L + v);
  let sp = pow(max(dot(n, hv), 0.0), shin) * specAmt * (shin + 8.0) / 25.0;
  var col = albedo * (SUN * diff * sh + ambient(n) * ao) + SUN * sp * sh;
  if (wrap > 0.0) {
    // subsurface glow when back-lit
    let back = pow(max(dot(-v, L), 0.0), 3.0);
    col += albedo * vec3f(1.0, 0.95, 0.85) * back * 0.6 * sh;
  }
  return col;
}

@fragment fn fsGrain(i: GOut) -> GFrag {
  let hit = traceImpostor(i);
  if (!hit.ok) { discard; }
  let hv = hit.p;
  let nv = hit.n;
  let clip = F.proj * vec4f(hv, 1.0);
  var o: GFrag;
  o.depth = clip.z / clip.w;
  let wp = (F.invView * vec4f(hv, 1.0)).xyz;
  let n = normalize((F.invView * vec4f(nv, 0.0)).xyz);
  let v = normalize(F.camPos.xyz - wp);
  o.color = vec4f(shadeGrain(i, wp, n, v, nv), 1.0);
  return o;
}

@fragment fn fsGrainDepth(i: GOut) -> @builtin(frag_depth) f32 {
  let hit = traceImpostor(i);
  if (!hit.ok) { discard; }
  let clip = F.proj * vec4f(hit.p, 1.0);
  return clip.z / clip.w;
}

// --- screen space fluid: linear depth of the front-most sphere
struct FDepth {
  @location(0) d: vec4f,
  @builtin(frag_depth) depth: f32,
};
@fragment fn fsFluidDepth(i: GOut) -> FDepth {
  let rd = rayDir(i.vp);
  let t = sphereHit(i.vp, rd, i.pa, i.rad);
  if (t < 0.0) { discard; }
  let hv = i.vp + rd * t;
  let clip = F.proj * vec4f(hv, 1.0);
  let z = clip.z / clip.w;
  if (z >= textureLoad(sceneDepth, vec2i(i.clip.xy), 0)) { discard; }
  var o: FDepth;
  o.d = vec4f(-hv.z, 0.0, 0.0, 1.0);
  o.depth = z;
  return o;
}

// --- sand surface splat: front-most grain per pixel -> (linear depth, grain id, local normal).
// The rim is dissolved with per-grain noise so silhouettes look grainy instead of round.
@fragment fn fsSandSplat(i: GOut) -> FDepth {
  let rd = rayDir(i.vp);
  let t = sphereHit(i.vp, rd, i.pa, i.rad);
  if (t < 0.0) { discard; }
  let hv = i.vp + rd * t;
  let nv = (hv - i.pa) / i.rad;
  let rim = 1.0 - max(dot(nv, -rd), 0.0);
  let nw = normalize((F.invView * vec4f(nv, 0.0)).xyz);
  let g = hashf(hash3i(vec3i(floor(nw * (i.rad / (SAND_GRAIN * 2.5)))), i.id));
  if (i.ao < 6.0 && rim > 0.5 && g < (rim - 0.5) * 2.4) { discard; }
  let clip = F.proj * vec4f(hv, 1.0);
  let z = clip.z / clip.w;
  if (z >= textureLoad(sceneDepth, vec2i(i.clip.xy), 0)) { discard; }
  var o: FDepth;
  o.d = vec4f(-hv.z, bitcast<f32>(i.id), octEncode(nw));
  o.depth = z;
  return o;
}

// --- additive thickness along the view ray
@fragment fn fsFluidThick(i: GOut) -> @location(0) vec4f {
  let rd = rayDir(i.vp);
  let oc = i.vp - i.pa;
  let b = dot(oc, rd);
  let c = dot(oc, oc) - i.rad * i.rad;
  let h = b * b - c;
  if (h < 0.0) { discard; }
  let hv = i.vp + rd * (-b - sqrt(h));
  let clip = F.proj * vec4f(hv, 1.0);
  if (clip.z / clip.w >= textureLoad(sceneDepth, vec2i(i.clip.xy), 0)) { discard; }
  return vec4f(2.0 * sqrt(h), 0.0, 0.0, 0.0);
}
`;

// ---------------------------------------------------------------------------------------------
// Cloth / skin.
export const CLOTH = RCOMMON + /* wgsl */ `
@group(0) @binding(1) var<storage, read> pos: array<vec4f>;
@group(0) @binding(2) var<storage, read> info: array<u32>;
@group(0) @binding(3) var<storage, read> tris: array<vec4u>;
@group(0) @binding(4) var<storage, read> triEdges: array<vec4u>;
@group(0) @binding(5) var<storage, read> edgeAlive: array<u32>;
@group(0) @binding(6) var<storage, read> normals: array<vec4f>;
@group(0) @binding(7) var<storage, read> rest: array<vec4f>;
@group(1) @binding(0) var shadowMap: texture_depth_2d;
@group(1) @binding(1) var shadowSamp: sampler_comparison;

struct COut {
  @builtin(position) clip: vec4f,
  @location(0) wp: vec3f,
  @location(1) n: vec3f,
  @location(2) rp: vec3f,
};

@vertex fn vsCloth(@builtin(vertex_index) vi: u32) -> COut {
  let t = vi / 3u;
  let tri = tris[t];
  let te = triEdges[t];
  var o: COut;
  let dead = ((info[tri.x] | info[tri.y] | info[tri.z]) & 4u) != 0u;
  let torn = edgeAlive[te.x] == 0u || edgeAlive[te.y] == 0u || edgeAlive[te.z] == 0u;
  if (dead || torn) {
    o.clip = vec4f(2.0, 2.0, 2.0, 1.0);
    return o;
  }
  var corners = array<u32, 3>(tri.x, tri.y, tri.z);
  let v = corners[vi % 3u];
  let n = normals[v].xyz;
  let wp = pos[v].xyz + n * (F.misc.x * 0.55);
  o.wp = wp;
  o.n = n;
  o.rp = rest[2u * v].xyz;
  o.clip = F.viewProj * vec4f(wp, 1.0);
  return o;
}

fn restNormal(rel: vec3f) -> vec3f {
  let e = F.shapeA.w;
  let yn = rel.y / F.shapeA.y;
  let s = mix(1.0, F.shapeB.z, smoothstep(F.shapeB.y, 1.0, yn));
  let a = vec3f(F.shapeA.x * s, F.shapeA.y, F.shapeA.z * s);
  let q = rel / a;
  let g = sign(q) * pow(max(abs(q), vec3f(1e-5)), vec3f(e - 1.0)) / a;
  return normalize(g);
}

// Plain weave: returns (height, thread id, 1 if warp on top)
fn weave(uv: vec2f, gap: f32) -> vec3f {
  let cell = floor(uv);
  let f = fract(uv);
  let over = ((i32(cell.x) + i32(cell.y)) & 1) == 0;
  // hand-spun yarn: every thread wanders a little, varies in thickness and has slubs along it
  let jx = hashf(u32(i32(cell.x) + 40000) * 7u);
  let jy = hashf(u32(i32(cell.y) + 40000) * 13u + 5u);
  let slubW = 0.8 + 0.4 * vnoise(vec2f(uv.y * 0.45, cell.x * 1.7));
  let slubF = 0.8 + 0.4 * vnoise(vec2f(uv.x * 0.45, cell.y * 1.3 + 9.0));
  let twW = (1.0 - gap) * (0.82 + 0.3 * jx) * slubW;
  let twF = (1.0 - gap) * (0.82 + 0.3 * jy) * slubF;
  let wu = abs(f.x - 0.5 - (jy - 0.5) * 0.12) * 2.0 / twW;
  let wv = abs(f.y - 0.5 - (jx - 0.5) * 0.12) * 2.0 / twF;
  let pw = sqrt(max(1.0 - wu * wu, 0.0));
  let pf = sqrt(max(1.0 - wv * wv, 0.0));
  let hW = pw * select(0.45, 1.0, over) * (0.7 + 0.3 * sin(PI * f.y));
  let hF = pf * select(1.0, 0.45, over) * (0.7 + 0.3 * sin(PI * f.x));
  if (hW > hF) { return vec3f(hW, cell.x, 1.0); }
  return vec3f(hF, cell.y + 7919.0, 0.0);
}

struct Surf { h: f32, albedo: vec3f };

fn fabric(rel: vec3f, pitch: f32, gap: f32, base: vec3f, fuzz: f32, fw: f32) -> Surf {
  let rn = restNormal(rel);
  let R = 0.5 * (F.shapeA.x + F.shapeA.z);
  let around = round(2.0 * PI * R / pitch);
  let ang = atan2(rel.z, rel.x);
  let uvSide = vec2f(ang / (2.0 * PI) * around, rel.y / pitch);
  let uvCap = rel.xz / pitch;
  let capW = smoothstep(0.55, 0.8, abs(rn.y));
  let ws = weave(uvSide, gap);
  let wc = weave(uvCap, gap);
  let w = mix(ws, wc, step(0.5, capW));
  var s: Surf;
  let detail = 1.0 - smoothstep(0.2, 0.55, fw / pitch);
  s.h = mix(0.6, w.x, detail);
  let th = mix(0.5, hashf(u32(abs(w.y)) * 13u + 7u), detail);
  let along = select(uvSide.x, uvSide.y, w.z > 0.5);
  let fib = vnoise(vec2f(along * 6.0, w.y * 3.1));
  // fine hairs along each thread and soft colour drift between threads
  let hair = vnoise(vec2f(along * 14.0, w.y * 5.3)) * vnoise(vec2f(along * 3.0, w.y * 11.0));
  var c = base * (0.86 + 0.22 * th) * (0.8 + 0.3 * fib * fuzz + 0.2 * (1.0 - fuzz)) * (0.92 + 0.16 * hair * fuzz);
  c *= mix(0.82, 0.55 + 0.45 * smoothstep(0.0, 0.5, w.x), detail);
  s.albedo = c;
  return s;
}

@fragment fn fsCloth(i: COut, @builtin(front_facing) ff: bool) -> @location(0) vec4f {
  let mat = u32(F.misc.w);
  let rel = i.rp - vec3f(0.0, F.shapeB.x, 0.0);
  let fwr = length(fwidth(i.rp));
  var s: Surf;
  var bump = 0.0;
  if (mat == 0u) {
    s = fabric(rel, 0.0026, 0.2, vec3f(0.53, 0.41, 0.26), 1.0, fwr);
    bump = 0.0006;
  } else if (mat == 1u) {
    s = fabric(rel, 0.0016, 0.05, vec3f(0.90, 0.88, 0.82), 0.4, fwr);
    bump = 0.0005;
    // printed bands
    let y = rel.y;
    let band = 1.0 - smoothstep(0.024, 0.026, abs(y - 0.02));
    let red1 = 1.0 - smoothstep(0.005, 0.0062, abs(y - 0.065));
    let red2 = 1.0 - smoothstep(0.005, 0.0062, abs(y + 0.025));
    let ink = s.albedo / vec3f(0.90, 0.88, 0.82);
    s.albedo = mix(s.albedo, vec3f(0.10, 0.20, 0.48) * ink, band * 0.92);
    s.albedo = mix(s.albedo, vec3f(0.62, 0.08, 0.06) * ink, max(red1, red2) * 0.92);
  } else {
    // skin: mottled with fine pores
    let m = vnoise(rel.xy * 22.0 + rel.zz * 13.0) * 0.6 + vnoise(rel.zy * 60.0) * 0.4;
    s.albedo = mix(vec3f(0.80, 0.56, 0.46), vec3f(0.70, 0.45, 0.38), m);
    s.h = vnoise(rel.xy * 900.0 + rel.z * 300.0) * 0.5 + vnoise(rel.zy * 400.0) * 0.5;
    bump = 0.00015;
  }

  // bump mapping via screen-space surface gradient
  let dpx = dpdx(i.wp);
  let dpy = dpdy(i.wp);
  let dhx = dpdx(s.h) * bump;
  let dhy = dpdy(s.h) * bump;
  var n = normalize(i.n);
  if (!ff) { n = -n; }
  let r1 = cross(dpy, n);
  let r2 = cross(n, dpx);
  let det = dot(dpx, r1);
  let detail = 1.0 - smoothstep(0.2, 0.55, fwr / 0.0034);
  if (abs(det) > 1e-20) {
    let grad = sign(det) * (dhx * r1 + dhy * r2);
    n = normalize(abs(det) * n - grad * detail);
  }

  var albedo = s.albedo;
  if (!ff) { albedo *= 0.45; } // inside of the bag

  let L = F.lightDir.xyz;
  let v = normalize(F.camPos.xyz - i.wp);
  let sh = shadowAt(shadowMap, shadowSamp, i.wp, n, 0.004);
  var col = vec3f(0.0);
  if (mat == 2u) {
    let ndl = dot(n, L);
    let wrapD = max((ndl + 0.5) / 1.5, 0.0);
    let lam = max(ndl, 0.0);
    let sss = vec3f(0.95, 0.35, 0.22) * (wrapD - lam);
    col = albedo * (SUN * (lam * sh + sss * mix(0.6, 1.0, sh)) + ambient(n));
    let hv = normalize(L + v);
    let fres = 0.03 + 0.97 * pow(1.0 - max(dot(n, v), 0.0), 5.0);
    col += SUN * sh * (pow(max(dot(n, hv), 0.0), 60.0) * 0.22 + fres * 0.05);
  } else {
    let ndl = max(dot(n, L), 0.0);
    // fabric: soft wrap plus fuzzy sheen at grazing angles
    let diff = max((dot(n, L) + 0.2) / 1.2, 0.0);
    let sheen = pow(1.0 - max(dot(n, v), 0.0), 4.0) * 0.25;
    col = albedo * (SUN * diff * sh + ambient(n)) + SUN * albedo * sheen * ndl * sh;
  }
  return vec4f(col, 1.0);
}
`;

// ---------------------------------------------------------------------------------------------
// Floor + background.
export const FLOOR = RCOMMON + /* wgsl */ `
@group(1) @binding(0) var shadowMap: texture_depth_2d;
@group(1) @binding(1) var shadowSamp: sampler_comparison;

struct FOut {
  @builtin(position) clip: vec4f,
  @location(0) wp: vec3f,
};

@vertex fn vsFloor(@builtin(vertex_index) vi: u32) -> FOut {
  var c = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
                          vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
  let p = vec3f(c[vi].x * 12.0, 0.0, c[vi].y * 12.0);
  var o: FOut;
  o.wp = p;
  o.clip = F.viewProj * vec4f(p, 1.0);
  return o;
}

@fragment fn fsFloor(i: FOut) -> @location(0) vec4f {
  let wp = i.wp;
  let n = vec3f(0.0, 1.0, 0.0);
  let large = fbm(wp.xz * 3.0);
  let mid = fbm(wp.xz * 18.0 + 3.1);
  let fine = vnoise(wp.xz * 220.0);
  let speck = step(0.93, vnoise(wp.xz * 140.0 + 11.0));
  var albedo = vec3f(0.25, 0.245, 0.235) * (0.78 + 0.32 * large + 0.12 * mid + 0.06 * fine);
  albedo = mix(albedo, vec3f(0.18, 0.17, 0.16), speck * 0.4);
  let L = F.lightDir.xyz;
  let sh = shadowAt(shadowMap, shadowSamp, wp, n, 0.002);
  let v = normalize(F.camPos.xyz - wp);
  let hv = normalize(L + v);
  let spec = pow(max(dot(n, hv), 0.0), 40.0) * 0.08 * (0.6 + 0.8 * mid);
  var col = albedo * (SUN * max(dot(n, L), 0.0) * sh + ambient(n)) + SUN * spec * sh;
  let dist = length(wp.xz);
  col = mix(col, skyColor(normalize(wp - F.camPos.xyz)), smoothstep(2.5, 9.0, dist));
  return vec4f(col, 1.0);
}

@fragment fn fsBg(@builtin(position) fc: vec4f) -> @location(0) vec4f {
  let ndc = vec2f(fc.x * F.screen.z * 2.0 - 1.0, 1.0 - fc.y * F.screen.w * 2.0);
  let vp = F.invProj * vec4f(ndc, 1.0, 1.0);
  let dir = normalize((F.invView * vec4f(vp.xyz / vp.w, 0.0)).xyz);
  return vec4f(skyColor(dir), 1.0);
}
`;

// ---------------------------------------------------------------------------------------------
// Screen space fluid: bilateral smoothing of depth + gaussian thickness.
export const BLUR = RCOMMON + /* wgsl */ `
@group(0) @binding(1) var tDepth: texture_2d<f32>;
@group(0) @binding(2) var tThick: texture_2d<f32>;

fn blur(fc: vec4f, dir: vec2i, second: bool) -> vec4f {
  let size = vec2i(textureDimensions(tDepth));
  let px = vec2i(fc.xy);
  let d0 = textureLoad(tDepth, px, 0).r;
  let rad = F.misc.x;
  var R = 6;
  if (d0 > 0.0) {
    let pxPerM = F.proj[1][1] * F.screen.y * 0.5 / d0;
    R = i32(clamp(rad * F.colA.w * pxPerM, 2.0, 22.0));
  }
  let sigma = f32(R) * 0.5;
  var sd = 0.0;
  var wd = 0.0;
  var st = 0.0;
  var wt = 0.0;
  for (var k = -R; k <= R; k++) {
    let p = clamp(px + dir * k, vec2i(0), size - 1);
    let g = exp(-f32(k * k) / (2.0 * sigma * sigma));
    var th = textureLoad(tThick, p, 0).r;
    if (second) { th = textureLoad(tThick, p, 0).g; }
    st += g * th;
    wt += g;
    if (d0 > 0.0) {
      let dk = textureLoad(tDepth, p, 0).r;
      if (dk > 0.0) {
        let dz = (dk - d0) / (rad * F.colB.w);
        let w = g * exp(-dz * dz);
        sd += w * dk;
        wd += w;
      }
    }
  }
  var dOut = 0.0;
  if (d0 > 0.0 && wd > 0.0) { dOut = sd / wd; }
  return vec4f(dOut, st / max(wt, 1e-6), 0.0, 1.0);
}

// Isotropic single-pass bilateral smoothing (golden-angle spiral of taps) for granular surfaces,
// where the separable filter leaves axis-aligned streaks in the depth.
@fragment fn fsBlurDisk(@builtin(position) fc: vec4f) -> @location(0) vec4f {
  let size = vec2i(textureDimensions(tDepth));
  let px = vec2i(fc.xy);
  let d0 = textureLoad(tDepth, px, 0).r;
  if (d0 <= 0.0) { return vec4f(0.0); }
  let rad = F.misc.x;
  let pxPerM = F.proj[1][1] * F.screen.y * 0.5 / d0;
  let R = clamp(rad * F.colA.w * pxPerM, 1.5, 28.0);
  var sd = d0;
  var wd = 1.0;
  const TAPS: i32 = 40;
  for (var k = 1; k <= TAPS; k++) {
    let fr = sqrt(f32(k) / f32(TAPS));
    let a = f32(k) * 2.39996323;
    let o = vec2f(cos(a), sin(a)) * (fr * R);
    let p = clamp(px + vec2i(round(o)), vec2i(0), size - 1);
    let dk = textureLoad(tDepth, p, 0).r;
    if (dk <= 0.0) { continue; }
    let dz = (dk - d0) / (rad * F.colB.w);
    let w = exp(-2.0 * fr * fr - dz * dz);
    sd += w * dk;
    wd += w;
  }
  return vec4f(sd / wd, 0.0, 0.0, 1.0);
}

@fragment fn fsBlurH(@builtin(position) fc: vec4f) -> @location(0) vec4f { return blur(fc, vec2i(1, 0), false); }
@fragment fn fsBlurV(@builtin(position) fc: vec4f) -> @location(0) vec4f { return blur(fc, vec2i(0, 1), true); }
`;

// ---------------------------------------------------------------------------------------------
// Composite the fluid surface over the scene.
export const COMPOSITE = RCOMMON + /* wgsl */ `
@group(0) @binding(1) var tScene: texture_2d<f32>;
@group(0) @binding(2) var tFluid: texture_2d<f32>;
@group(0) @binding(3) var shadowMap: texture_depth_2d;
@group(0) @binding(4) var shadowSamp: sampler_comparison;
@group(0) @binding(5) var linSamp: sampler;
@group(0) @binding(6) var tSplat: texture_2d<f32>;
@group(0) @binding(7) var<storage, read> pos: array<vec4f>;

fn viewPos(px: vec2i, d: f32) -> vec3f {
  let ndc = vec2f((f32(px.x) + 0.5) * F.screen.z * 2.0 - 1.0, 1.0 - (f32(px.y) + 0.5) * F.screen.w * 2.0);
  return vec3f(ndc.x * d / F.proj[0][0], ndc.y * d / F.proj[1][1], -d);
}

fn depthAt(px: vec2i) -> f32 {
  let size = vec2i(textureDimensions(tFluid));
  return textureLoad(tFluid, clamp(px, vec2i(0), size - 1), 0).r;
}

// view-space normal of the smoothed surface (picks the smaller depth step on each axis)
fn surfaceNormal(px: vec2i, p: vec3f) -> vec3f {
  let dR = depthAt(px + vec2i(1, 0));
  let dL = depthAt(px - vec2i(1, 0));
  let dU = depthAt(px - vec2i(0, 1));
  let dD = depthAt(px + vec2i(0, 1));
  var ddx = viewPos(px + vec2i(1, 0), dR) - p;
  let ddx2 = p - viewPos(px - vec2i(1, 0), dL);
  if (dR <= 0.0 || (dL > 0.0 && abs(ddx2.z) < abs(ddx.z))) { ddx = ddx2; }
  var ddy = viewPos(px - vec2i(0, 1), dU) - p;
  let ddy2 = p - viewPos(px + vec2i(0, 1), dD);
  if (dU <= 0.0 || (dD > 0.0 && abs(ddy2.z) < abs(ddy.z))) { ddy = ddy2; }
  var nv = normalize(cross(ddx, ddy));
  if (nv.z < 0.0) { nv = -nv; }
  return nv;
}

// Sand as a material: the smoothed splat surface gives the shape of the mass, the grain id and
// local sphere normal (both anchored to the moving particle) give sub-millimetre grain detail.
fn shadeSand(px: vec2i, fd: vec4f, sp: vec4f) -> vec3f {
  let p = viewPos(px, fd.r);
  let nv = surfaceNormal(px, p);
  let wp = (F.invView * vec4f(p, 1.0)).xyz;
  var n = normalize((F.invView * vec4f(nv, 0.0)).xyz);
  let v = normalize(F.camPos.xyz - wp);
  let L = F.lightDir.xyz;
  let id = bitcast<u32>(sp.g);
  let nl = octDecode(sp.ba);
  let r = F.misc.x;

  // grain cells on the particle's surface, never smaller than about a pixel
  let pixel = 2.0 * fd.r / (F.proj[1][1] * F.screen.y);
  let cs = max(SAND_GRAIN, pixel * 0.85);
  let cell = vec3i(floor(nl * (1.3 * r) / cs));
  let h = hash3i(cell, id);
  let h1 = hashf(h);
  let h2 = hashf(h ^ 0x68bc21ebu);
  let h3 = hashf(h ^ 0x02e5be93u);
  let detail = 1.0 - smoothstep(SAND_GRAIN, SAND_GRAIN * 4.0, pixel);

  // mineral mix: tan quartz/feldspar, with light, dark and iron-stained grains
  let tint = hashf(id * 2654435761u) * 0.08;
  var albedo = mix(vec3f(0.50, 0.38, 0.24), vec3f(0.38, 0.28, 0.17), h1) * (0.96 + tint);
  if (h2 > 0.965) { albedo = vec3f(0.15, 0.13, 0.11); }
  else if (h2 > 0.925) { albedo = vec3f(0.68, 0.65, 0.58); }
  else if (h2 > 0.88) { albedo = vec3f(0.52, 0.29, 0.16); }
  albedo = mix(vec3f(0.44, 0.33, 0.21), albedo, 0.55 + 0.45 * detail);

  // every grain is a tiny facet: tilt the normal per cell
  let tilt = vec3f(h1 - 0.5, h3 - 0.5, hashf(h ^ 0x9e3779b9u) - 0.5);
  n = normalize(n + tilt * (0.35 + 0.5 * detail));

  // crevices between grains: where the raw grain lies behind the smoothed surface
  let cavity = smoothstep(0.3 * r, 1.5 * r, sp.r - fd.r);
  var ao = mix(1.0, 0.7, smoothstep(4.0, 10.0, pos[id].w)) * (1.0 - 0.2 * cavity);
  ao *= mix(0.65, 1.0, smoothstep(0.0, r * 4.0, wp.y));

  let sh = shadowAt(shadowMap, shadowSamp, wp, n, r * 0.8);
  let ndl = max(dot(n, L), 0.0);
  var col = albedo * (SUN * ndl * sh * (1.0 - 0.12 * cavity) + ambient(n) * ao);
  // occasional quartz glint
  if (h3 > 0.985) {
    let hv = normalize(L + v);
    col += SUN * sh * pow(max(dot(n, hv), 0.0), 120.0) * 2.5 * detail;
  }
  return col;
}

@fragment fn fsComposite(@builtin(position) fc: vec4f) -> @location(0) vec4f {
  let px = vec2i(fc.xy);
  let sc = textureLoad(tScene, px, 0).rgb;
  let mat = F.misc2.w;
  if (mat == 1u) { return vec4f(sc, 1.0); }
  let fd = textureLoad(tFluid, px, 0);
  if (fd.r <= 0.0) { return vec4f(sc, 1.0); }
  if (mat == 0u) {
    let sp = textureLoad(tSplat, px, 0);
    if (sp.r <= 0.0) { return vec4f(sc, 1.0); }
    return vec4f(shadeSand(px, fd, sp), 1.0);
  }
  let thick = fd.g;
  let p = viewPos(px, fd.r);
  let nv = surfaceNormal(px, p);

  let wp = (F.invView * vec4f(p, 1.0)).xyz;
  let n = normalize((F.invView * vec4f(nv, 0.0)).xyz);
  let v = normalize(F.camPos.xyz - wp);
  let L = F.lightDir.xyz;
  let sh = shadowAt(shadowMap, shadowSamp, wp, n, 0.006);

  let uv = fc.xy * F.screen.zw;
  let refr = nv.xy * vec2f(1.0, -1.0) * 0.025 * clamp(thick * 40.0, 0.0, 1.0);
  let bg = textureSampleLevel(tScene, linSamp, uv - refr, 0.0).rgb;
  let T = exp(-F.colA.rgb * thick);
  let inscatter = F.colB.rgb * (ambient(n) * 0.9 + SUN * max(dot(n, L), 0.0) * sh * 0.6);
  var col = mix(inscatter, bg, T);
  let fres = 0.02 + 0.98 * pow(1.0 - max(dot(n, v), 0.0), 5.0);
  let refl = skyColor(reflect(-v, n));
  col = mix(col, refl, fres);
  let hv = normalize(L + v);
  col += SUN * sh * (pow(max(dot(n, hv), 0.0), 600.0) * 5.0 + pow(max(dot(n, hv), 0.0), 60.0) * 0.08);
  // fade the edge of the surface to avoid hard silhouettes on thin films
  let edge = smoothstep(0.0, 0.004, thick);
  return vec4f(mix(sc, col, edge), 1.0);
}
`;

// ---------------------------------------------------------------------------------------------
export const TONEMAP = RCOMMON + /* wgsl */ `
@group(0) @binding(1) var tHdr: texture_2d<f32>;

fn aces(x: vec3f) -> vec3f {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}

@fragment fn fsTone(@builtin(position) fc: vec4f) -> @location(0) vec4f {
  var c = textureLoad(tHdr, vec2i(fc.xy), 0).rgb * 0.95;
  c = aces(c);
  c = pow(c, vec3f(1.0 / 2.2));
  let uv = fc.xy * F.screen.zw;
  let vig = 1.0 - 0.35 * pow(length((uv - 0.5) * vec2f(1.2, 1.0)), 2.2);
  c *= vig;
  let dither = (hashf(u32(fc.x) * 1973u + u32(fc.y) * 9277u + u32(F.misc.z * 60.0) * 26699u) - 0.5) / 255.0;
  return vec4f(c + dither, 1.0);
}
`;
