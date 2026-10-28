// Scenario presets. Distances in metres, time in seconds.
//
// shape: superellipsoid "bag" (half-extents ax/ay/az, exponent exp) centred at (0, centerY, 0),
//        with an optional gathered neck that pinches the cross-section above neckStart (in y/ay units).
// kind:  'grain' (frictional granular contact) or 'fluid' (position based fluids).

export const QUALITY = {
  'Low (32k)': 32_000,
  'Medium (64k)': 64_000,
  'High (128k)': 128_000,
  'Ultra (256k)': 256_000,
};

export const PRESETS = {
  sand: {
    label: 'Sandbag',
    kind: 'grain',
    pairs: false,
    shape: { ax: 0.16, ay: 0.27, az: 0.12, exp: 2.4, centerY: 0.74, neckStart: 0.5, neckScale: 0.16 },
    pinAbove: 0.86, // pin (hang) every cloth vertex above this normalised height
    fillTop: 0.45,
    rope: true,
    clothSpacing: 2.3, // cloth vertex spacing in particle radii (must stay < ~2.6 to be leak-proof)
    clothMat: 0, // burlap
    grainMat: 0, // sand
    muS: 0.7,
    muK: 0.5,
    cohesion: 0.0,
    floorMu: 0.7,
    contactDamp: 0.7, // inelastic grain contacts
    stretchK: 1.0,
    bendK: 0.08,
    // as stiff in compression as in stretch: soft compression lets crumpled cloth (lying on the
    // floor or the pile) run a stretch/compress relaxation cycle and never stop twitching
    compressK: 1.0,
    tear: 1.65,
    clothInvMass: 0.3,
    clothIters: 4,
    clothSor: 1.5,
    anchorK: 0.0,
    lraSlack: 1.04,
    substeps: 24, // at 64k particles; scaled with the particle count
    damping: 0.5,
    gravity: -9.81,
    camera: { target: [0, 0.5, 0], dist: 1.85, yaw: 0.55, pitch: 0.14 },
  },

  rice: {
    label: 'Rice sack',
    kind: 'grain',
    pairs: true, // each grain is a rigid two-sphere "dumbbell" -> elongated rice kernels
    // a full sack lying on the floor: built close to the low pillow it settles into, so it does not
    // spend its first seconds slumping
    shape: { ax: 0.21, ay: 0.12, az: 0.15, exp: 3.2, centerY: 0.122, neckStart: 0.72, neckScale: 0.3 },
    warmup: 360, // hidden settling steps
    // a sack draining through a cut moves slowly inside: grains must be still for longer before they
    // sleep, or the pile behind the wound freezes and the flow stops
    sleepTime: 2,
    // the slow drain through a slit needs the contact pushes to carry kernels out of it
    vSepMax: 0.25,
    pinAbove: 0,
    fillTop: 0.78,
    rope: false,
    clothSpacing: 2.3,
    clothMat: 1, // printed cotton
    grainMat: 1, // rice
    muS: 0.42,
    muK: 0.3,
    cohesion: 0.0,
    floorMu: 0.45,
    contactDamp: 0.3,
    stretchK: 1.0,
    bendK: 0.1,
    compressK: 1.0,
    tear: 1.5,
    clothInvMass: 0.15, // heavy, stiff sack: the rim of a slumped sack carries all the hoop stress
    clothIters: 7,
    clothSor: 1.5,
    anchorK: 0.0,
    lraSlack: 1.0,
    substeps: 20,
    damping: 0.3,
    gravity: -9.81,
    camera: { target: [0, 0.22, 0], dist: 1.5, yaw: 0.6, pitch: 0.28 },
  },

  blood: {
    label: 'Body (blood)',
    kind: 'fluid',
    pairs: false,
    shape: { ax: 0.15, ay: 0.3, az: 0.1, exp: 2.3, centerY: 0.305, neckStart: 0.99, neckScale: 1.0 },
    pinAbove: 0,
    fillTop: 0.92,
    rope: false,
    clothSpacing: 2.0,
    clothMat: 2, // skin
    grainMat: 2, // fluid
    muS: 0.05,
    muK: 0.05,
    cohesion: 0.0,
    floorMu: 0.6,
    stretchK: 1.0,
    bendK: 0.2,
    compressK: 0.5,
    tear: 3.0,
    clothInvMass: 0.25,
    clothIters: 2,
    clothSor: 1.5,
    anchorK: 0.35, // skin is held to the body's rest shape (it has flesh and bone behind it)
    lraSlack: 1.0,
    substeps: 6,
    damping: 0.4,
    gravity: -9.81,
    // position based fluids
    hRatio: 3.5, // SPH kernel radius in particle radii
    relax: 0.05,
    scorr: 0.02,
    fluidOmega: 0.3, // under-relaxed Jacobi pressure solve (stable at few substeps)
    viscosity: 0.08,
    fluidCohesion: 0.012,
    boundaryPsi: 0.9,
    pressure: 0.25, // heartbeat-driven internal pressure
    fluidMaxStep: 0.5, // max PBF displacement per substep, in particle radii
    heartRate: 1.2,
    absorb: [55, 420, 480], // Beer-Lambert absorption per metre
    scatter: [0.16, 0.006, 0.006],
    slashWidth: 1.1, // wound width in skin-particle spacings (skin is anchored, so it must gape)
    camera: { target: [0, 0.28, 0], dist: 1.5, yaw: 0.5, pitch: 0.2 },
  },
};
