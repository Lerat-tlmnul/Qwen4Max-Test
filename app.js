/* Pélican à Vélo — île voxel 3D interactive */
(function () {
'use strict';

/* ============================== utilitaires ============================== */
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const clamp01 = v => clamp(v, 0, 1);
const lerp = (a, b, t) => a + (b - a) * t;
const TAU = Math.PI * 2;

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

let SEED = 1337;
function hash2(ix, iz) {
  const s = Math.sin(ix * 127.1 + iz * 311.7 + SEED * 0.0173) * 43758.5453;
  return s - Math.floor(s);
}
function vnoise(x, z) {
  const ix = Math.floor(x), iz = Math.floor(z);
  const fx = x - ix, fz = z - iz;
  const ux = fx * fx * (3 - 2 * fx), uz = fz * fz * (3 - 2 * fz);
  const a = hash2(ix, iz), b = hash2(ix + 1, iz), c = hash2(ix, iz + 1), d = hash2(ix + 1, iz + 1);
  return lerp(lerp(a, b, ux), lerp(c, d, ux), uz);
}

const ISLAND_R = 54, PATH_R = 34, PATH_H = 3, CELL = 2;
function heightAt(x, z) {
  const r = Math.hypot(x, z);
  if (r > ISLAND_R) return -1;
  if (Math.abs(r - PATH_R) <= 3.4) return PATH_H;
  if (r < 8) return 8;
  const base = 9.5 * (1 - (r / ISLAND_R) * (r / ISLAND_R));
  const n = vnoise(x * 0.045, z * 0.045) * 4 + vnoise(x * 0.11 + 40, z * 0.11 + 40) * 1.6;
  let h = Math.floor(base + n - 1.4);
  if (r >= 45) h = 1 + (vnoise(x * 0.2, z * 0.2) > 0.62 ? 1 : 0);
  return clamp(h, 1, 12);
}

/* ============================== rendu / scène ============================== */
const canvas = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.shadowMap.autoUpdate = false;

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0xbfe8f5, 150, 640);

const camera = new THREE.PerspectiveCamera(52, innerWidth / innerHeight, 0.5, 2200);

const MATS = [];
function lambert(opts) { const m = new THREE.MeshLambertMaterial(opts); MATS.push(m); return m; }

const hemi = new THREE.HemisphereLight(0xbfe8f5, 0x3a5a40, 0.6);
scene.add(hemi);
const ambient = new THREE.AmbientLight(0xffffff, 0.12);
scene.add(ambient);
const sunLight = new THREE.DirectionalLight(0xfff2d8, 1.0);
sunLight.castShadow = true;
sunLight.shadow.mapSize.set(2048, 2048);
sunLight.shadow.camera.left = -90; sunLight.shadow.camera.right = 90;
sunLight.shadow.camera.top = 90; sunLight.shadow.camera.bottom = -90;
sunLight.shadow.camera.near = 40; sunLight.shadow.camera.far = 700;
sunLight.shadow.bias = -0.0006;
scene.add(sunLight); scene.add(sunLight.target);
let shadowDirty = true;

/* dôme ciel dégradé */
const skyUniforms = {
  topColor: { value: new THREE.Color(0x3f9fe0) },
  bottomColor: { value: new THREE.Color(0xbfe8f5) },
  offset: { value: 30 }, exponent: { value: 0.85 }
};
const sky = new THREE.Mesh(
  new THREE.SphereGeometry(950, 24, 14),
  new THREE.ShaderMaterial({
    uniforms: skyUniforms, side: THREE.BackSide, fog: false, depthWrite: false,
    vertexShader: 'varying vec3 vP; void main(){ vP = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
    fragmentShader: 'uniform vec3 topColor; uniform vec3 bottomColor; uniform float offset; uniform float exponent; varying vec3 vP; void main(){ float h = normalize(vP + vec3(0.0, offset, 0.0)).y; gl_FragColor = vec4(mix(bottomColor, topColor, pow(max(h, 0.0), exponent)), 1.0); }'
  })
);
scene.add(sky);

/* halo lumineux (sprite radial) */
function glowTexture() {
  const c = document.createElement('canvas'); c.width = c.height = 128;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(64, 64, 4, 64, 64, 64);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.35, 'rgba(255,255,255,0.45)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grad; g.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
}
const GLOW_TEX = glowTexture();
function makeGlow(color, scale, opacity) {
  const m = new THREE.SpriteMaterial({ map: GLOW_TEX, color, transparent: true, opacity, depthWrite: false, fog: false });
  const s = new THREE.Sprite(m); s.scale.setScalar(scale); return s;
}

/* ============================== batch voxel ============================== */
const boxTpl = new THREE.BoxGeometry(1, 1, 1).toNonIndexed();
const TPL_P = boxTpl.attributes.position.array;
const TPL_N = boxTpl.attributes.normal.array;
const VCOUNT = TPL_P.length / 3;

class VoxelBatch {
  constructor() { this.p = []; this.n = []; this.c = []; }
  add(cx, cy, cz, sx, sy, sz, color, rx = 0, ry = 0, rz = 0, jitter = 0.05) {
    const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(rx, ry, rz));
    const m = new THREE.Matrix4().compose(new THREE.Vector3(cx, cy, cz), q, new THREE.Vector3(sx, sy, sz));
    const nm = new THREE.Matrix3().getNormalMatrix(m);
    const v = new THREE.Vector3(), nv = new THREE.Vector3();
    const j = 1 + (Math.random() * 2 - 1) * jitter;
    const r = clamp01(color.r * j), g = clamp01(color.g * j), b = clamp01(color.b * j);
    for (let i = 0; i < VCOUNT; i++) {
      v.fromArray(TPL_P, i * 3).applyMatrix4(m);
      nv.fromArray(TPL_N, i * 3).applyMatrix3(nm).normalize();
      this.p.push(v.x, v.y, v.z); this.n.push(nv.x, nv.y, nv.z); this.c.push(r, g, b);
    }
  }
  build(material, cast = true, receive = true) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(this.p, 3));
    geo.setAttribute('normal', new THREE.Float32BufferAttribute(this.n, 3));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(this.c, 3));
    const mesh = new THREE.Mesh(geo, material || lambert({ vertexColors: true }));
    mesh.castShadow = cast; mesh.receiveShadow = receive;
    return mesh;
  }
}
const C = h => new THREE.Color(h);
const PAL = {
  grass: C(0x58b647), grass2: C(0x4aa53c), sand: C(0xe8d29a), path: C(0xb08d55),
  dirt: C(0x8a5a33), stone: C(0x7d7f8c), trunk: C(0x8a5a2b), leaf: C(0x2e9e4f), leaf2: C(0x37b45c),
  white: C(0xf6f2ea), coral: C(0xe8483f), amber: C(0xffb347), rock: C(0x8d8d99)
};

/* helper "membre" : boîte étirée entre deux points */
const Y_AXIS = new THREE.Vector3(0, 1, 0);
function limbMesh(thickness, material) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), material);
  m.userData.t = thickness; return m;
}
function setLimb(mesh, a, b) {
  const dir = new THREE.Vector3().subVectors(b, a);
  const len = dir.length();
  mesh.position.copy(a).addScaledVector(dir, 0.5);
  mesh.scale.set(mesh.userData.t, len, mesh.userData.t);
  mesh.quaternion.setFromUnitVectors(Y_AXIS, dir.normalize());
}
function box(w, h, d, mat, x, y, z, parent) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
  m.position.set(x, y, z); (parent || scene).add(m); return m;
}

/* ============================== océan ============================== */
const WATER_N = 160, WATER_TILE = 7, WATER_SPAN = WATER_N * WATER_TILE;
const waterGeo = new THREE.BoxGeometry(WATER_TILE * 0.985, 1.4, WATER_TILE * 0.985);
const waterMat = new THREE.MeshPhongMaterial({ color: 0xffffff, shininess: 110, specular: 0x9fd8e8 });
MATS.push(waterMat);
const water = new THREE.InstancedMesh(waterGeo, waterMat, WATER_N * WATER_N);
water.receiveShadow = true;
water.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
scene.add(water);

const waterBase = [];
const waterActive = [];
{
  const col = new THREE.Color();
  const m0 = new THREE.Matrix4();
  let idx = 0;
  for (let i = 0; i < WATER_N; i++) {
    for (let j = 0; j < WATER_N; j++) {
      const x = (i - WATER_N / 2 + 0.5) * WATER_TILE;
      const z = (j - WATER_N / 2 + 0.5) * WATER_TILE;
      const r = Math.hypot(x, z);
      const t1 = clamp01((r - 60) / 60), t2 = clamp01((r - 120) / 90);
      col.setHex(0x3fd6e0).lerp(C(0x18a3cf), t1 * t1 * (3 - 2 * t1)).lerp(C(0x0d6fae), t2 * t2 * (3 - 2 * t2));
      const j2 = 0.94 + hash2(i, j) * 0.12;
      const entry = { x, z, r, h: hash2(i + 99, j), col: col.clone().multiplyScalar(j2) };
      waterBase.push(entry);
      if (r < 210) waterActive.push(idx);
      m0.makeTranslation(x, -0.35, z);
      water.setMatrixAt(idx, m0);
      water.setColorAt(idx, entry.col);
      idx++;
    }
  }
  water.instanceColor.setUsage(THREE.DynamicDrawUsage);
}
const deepPlane = new THREE.Mesh(new THREE.PlaneGeometry(4200, 4200), lambert({ color: 0x0d6fae }));
deepPlane.rotation.x = -Math.PI / 2; deepPlane.position.y = -1.2;
scene.add(deepPlane);

const _wm = new THREE.Matrix4();
const _wc = new THREE.Color();
const FOAM = new THREE.Color(0xeafcff);
let waveAmp = 1;
function updateWater(t, frame) {
  const doColor = frame % 3 === 0;
  for (let n = 0; n < waterActive.length; n++) {
    const k = waterActive[n];
    const w = waterBase[k];
    const y = -0.35 + (Math.sin(w.x * 0.09 + t * 1.3) + Math.cos(w.z * 0.075 + t * 1.05) + Math.sin((w.x + w.z) * 0.05 + t * 0.7)) / 3 * 0.55 * waveAmp;
    _wm.makeTranslation(w.x, y, w.z);
    water.setMatrixAt(k, _wm);
    if (doColor) {
      _wc.copy(w.col);
      if (w.r > 49 && w.r < 64) {
        const f = (Math.sin(t * 2.1 + w.x * 0.3 + w.z * 0.22) + 1) * 0.5;
        _wc.lerp(FOAM, f * 0.75 * (1 - Math.abs(w.r - 56) / 8));
      } else if (Math.sin(t * 3 + w.h * 40) > 0.93) {
        _wc.multiplyScalar(1.35);
      }
      water.setColorAt(k, _wc);
    }
  }
  water.instanceMatrix.needsUpdate = true;
  if (doColor) water.instanceColor.needsUpdate = true;
}

/* ============================== terrain + décor (reconstruit au seed) ============================== */
let terrainMesh = null, decoMesh = null;
const terrainMat = lambert({ vertexColors: true });
const decoMat = lambert({ vertexColors: true });

function buildIsland() {
  if (terrainMesh) { scene.remove(terrainMesh); terrainMesh.geometry.dispose(); }
  if (decoMesh) { scene.remove(decoMesh); decoMesh.geometry.dispose(); }

  const tb = new VoxelBatch();
  const half = Math.ceil(ISLAND_R / CELL) + 1;
  for (let i = -half; i <= half; i++) {
    for (let j = -half; j <= half; j++) {
      const x = i * CELL, z = j * CELL;
      const r = Math.hypot(x, z);
      if (r > ISLAND_R) continue;
      const h = heightAt(x, z);
      let top;
      if (Math.abs(r - PATH_R) <= 3.4) top = PAL.path;
      else if (r >= 45) top = PAL.sand;
      else if (r < 8) top = PAL.stone;
      else top = (hash2(i, j) > 0.5 ? PAL.grass : PAL.grass2);
      tb.add(x, h - 0.5, z, CELL, 1, CELL, top, 0, 0, 0, 0.06);
      if (h - 1 > -2) {
        const coreH = h + 1;
        tb.add(x, (h - 3) / 2, z, CELL, coreH, CELL, h > 7 ? PAL.stone : PAL.dirt, 0, 0, 0, 0.04);
      }
    }
  }
  terrainMesh = tb.build(terrainMat);
  scene.add(terrainMesh);

  /* décor végétal & rocheux */
  const rng = mulberry32(SEED);
  const db = new VoxelBatch();
  const FLOWER_COLS = [C(0xff5a4e), C(0xffb347), C(0xe858c8), C(0xf6f2ea), C(0x5fe8f5)];

  function palm(x, z) {
    const g = heightAt(x, z);
    const leanA = rng() * TAU, lean = 0.12 + rng() * 0.14;
    const segs = 6 + Math.floor(rng() * 2);
    let px = x, pz = z, py = g;
    for (let s = 0; s < segs; s++) {
      db.add(px, py + 0.55, pz, 0.85 - s * 0.04, 1.15, 0.85 - s * 0.04, PAL.trunk, 0, 0, 0, 0.08);
      py += 1.05; px += Math.cos(leanA) * lean; pz += Math.sin(leanA) * lean;
    }
    const fronds = 7;
    for (let f = 0; f < fronds; f++) {
      const a = (f / fronds) * TAU + rng() * 0.4;
      let fx = px, fy = py + 0.4, fz = pz;
      for (let s = 0; s < 3; s++) {
        const w = 1.5 - s * 0.35;
        db.add(fx + Math.cos(a) * (0.8 + s * 0.9), fy - s * s * 0.22, fz + Math.sin(a) * (0.8 + s * 0.9),
          w, 0.35, w, s % 2 ? PAL.leaf : PAL.leaf2, 0, -a, 0.12 * s, 0.07);
      }
    }
    db.add(px + 0.4, py - 0.2, pz + 0.3, 0.45, 0.45, 0.45, C(0x6b4423));
    db.add(px - 0.35, py - 0.3, pz - 0.25, 0.4, 0.4, 0.4, C(0x6b4423));
  }

  let placed = 0, tries = 0;
  while (placed < 16 && tries++ < 300) {
    const a = rng() * TAU, r = 11 + rng() * 39;
    const x = Math.cos(a) * r, z = Math.sin(a) * r;
    if (Math.abs(r - PATH_R) < 5.5) continue;
    if (x > 38 && Math.abs(z) < 12) continue;
    palm(x, z); placed++;
  }
  for (let k = 0; k < 24; k++) {
    const a = rng() * TAU, r = 8 + rng() * 45;
    const x = Math.cos(a) * r, z = Math.sin(a) * r;
    if (Math.abs(r - PATH_R) < 4.5) continue;
    const g = heightAt(x, z), n = 2 + Math.floor(rng() * 3);
    for (let s = 0; s < n; s++) {
      const w = 0.7 + rng() * 1.3;
      db.add(x + (rng() - 0.5) * 1.6, g + w * 0.35, z + (rng() - 0.5) * 1.6, w, w * 0.8, w, PAL.rock, 0, rng() * TAU, 0, 0.09);
    }
  }
  for (let k = 0; k < 80; k++) {
    const a = rng() * TAU, r = 9 + rng() * 34;
    const x = Math.cos(a) * r, z = Math.sin(a) * r;
    if (Math.abs(r - PATH_R) < 4.2 || r >= 44) continue;
    const g = heightAt(x, z);
    db.add(x, g + 0.4, z, 0.16, 0.85, 0.16, C(0x2e7d32), 0, 0, (rng() - 0.5) * 0.3);
    db.add(x, g + 0.95, z, 0.42, 0.42, 0.42, FLOWER_COLS[Math.floor(rng() * FLOWER_COLS.length)], 0, rng() * TAU, 0, 0.04);
  }
  for (let k = 0; k < 110; k++) {
    const a = rng() * TAU, r = 9 + rng() * 40;
    const x = Math.cos(a) * r, z = Math.sin(a) * r;
    if (Math.abs(r - PATH_R) < 4 || r >= 45) continue;
    const g = heightAt(x, z);
    db.add(x, g + 0.22, z, 0.3, 0.5, 0.3, C(0x3d8f35), 0, rng() * TAU, 0, 0.12);
  }
  for (let k = 0; k < 26; k++) {
    const a = rng() * TAU, r = 46 + rng() * 7;
    const x = Math.cos(a) * r, z = Math.sin(a) * r;
    if (x > 38 && Math.abs(z) < 12) continue;
    const g = heightAt(x, z);
    db.add(x, g + 0.1, z, 0.3, 0.18, 0.3, rng() > 0.5 ? C(0xf3e9d8) : C(0xf0b8c8), 0, rng() * TAU, 0, 0.05);
  }
  decoMesh = db.build(decoMat);
  scene.add(decoMesh);
  shadowDirty = true;
}

/* ============================== phare, ponton, parasol ============================== */
const lampMat = new THREE.MeshLambertMaterial({ color: 0xfff3c0, emissive: 0xffd970, emissiveIntensity: 0.3 });
MATS.push(lampMat);
let beamGroup, lampGlow;
function buildStructures() {
  const sb = new VoxelBatch();
  const LH = 8;
  sb.add(0, LH + 1, 0, 7.5, 2, 7.5, PAL.stone, 0, 0, 0, 0.05);
  for (let s = 0; s < 5; s++) {
    sb.add(0, LH + 3.2 + s * 2.4, 0, 4.2 - s * 0.15, 2.4, 4.2 - s * 0.15, s % 2 ? PAL.coral : PAL.white, 0, 0, 0, 0.03);
  }
  sb.add(0, LH + 15.4, 0, 5.2, 0.6, 5.2, C(0x33424e));
  for (let p = 0; p < 8; p++) {
    const a = (p / 8) * TAU;
    sb.add(Math.cos(a) * 2.3, LH + 16.1, Math.sin(a) * 2.3, 0.18, 0.9, 0.18, C(0x33424e));
  }
  sb.add(0, LH + 18.4, 0, 3.4, 0.5, 3.4, PAL.coral);
  sb.add(0, LH + 18.9, 0, 2.4, 0.5, 2.4, PAL.coral);
  sb.add(0, LH + 19.4, 0, 1.3, 0.6, 1.3, PAL.coral);
  const lighthouse = sb.build();
  scene.add(lighthouse);
  box(2.8, 2.1, 2.8, lampMat, 0, LH + 16.9, 0);
  lampGlow = makeGlow(0xffd970, 16, 0);
  lampGlow.position.set(0, LH + 16.9, 0);
  scene.add(lampGlow);
  beamGroup = new THREE.Group();
  beamGroup.position.set(0, LH + 16.9, 0);
  const beamMat = new THREE.MeshBasicMaterial({ color: 0xffe9a8, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, fog: false });
  const b1 = new THREE.Mesh(new THREE.BoxGeometry(70, 1.6, 2.4), beamMat);
  b1.position.x = 36;
  const b2 = b1.clone(); b2.position.x = -36;
  beamGroup.add(b1, b2);
  beamGroup.userData.mat = beamMat;
  scene.add(beamGroup);

  /* ponton */
  const pb = new VoxelBatch();
  for (let k = 0; k < 10; k++) {
    pb.add(47 + k * 1.9, 1.5, 0, 1.75, 0.4, 4.2, k % 2 ? C(0x9a6a38) : C(0x8a5a2b), 0, 0, 0, 0.06);
    if (k % 3 === 0) {
      pb.add(47 + k * 1.9, -0.4, 1.8, 0.5, 3.6, 0.5, C(0x6b4423));
      pb.add(47 + k * 1.9, -0.4, -1.8, 0.5, 3.6, 0.5, C(0x6b4423));
    }
  }
  pb.add(64.5, 2.6, 1.8, 0.4, 2.2, 0.4, C(0x6b4423));
  pb.add(64.5, 2.6, -1.8, 0.4, 2.2, 0.4, C(0x6b4423));
  scene.add(pb.build());

  /* parasol + serviette + ballon */
  const ub = new VoxelBatch();
  const ux = 42, uz = 11, ug = heightAt(ux, uz);
  ub.add(ux, ug + 2, uz, 0.25, 4.2, 0.25, C(0xd8d2c4));
  for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) {
    ub.add(ux + i * 1.35, ug + 4.1 - (Math.abs(i) + Math.abs(j)) * 0.28, uz + j * 1.35,
      1.4, 0.35, 1.4, (i + j) % 2 === 0 ? PAL.coral : PAL.white, 0, 0, 0, 0.03);
  }
  ub.add(ux + 2.6, ug + 0.12, uz + 1.4, 3, 0.16, 1.7, C(0x35c4d8), 0, 0.3, 0, 0.02);
  ub.add(ux - 2.4, ug + 0.45, uz - 1.8, 0.7, 0.7, 0.7, PAL.coral, 0, 0.5, 0);
  ub.add(ux - 2.4, ug + 1.0, uz - 1.8, 0.5, 0.35, 0.5, PAL.white, 0, 0.5, 0);
  scene.add(ub.build());
}

/* ============================== pélican + vélo ============================== */
const M = {
  white: lambert({ color: 0xf6f2ea }), tip: lambert({ color: 0x2e2e38 }),
  beak: lambert({ color: 0xff9d2e }), pouch: lambert({ color: 0xffc46b }),
  feet: lambert({ color: 0xff8c1a }), eye: lambert({ color: 0x14141c }),
  frame: lambert({ color: 0xe8483f }), tire: lambert({ color: 0x23232e }),
  metal: lambert({ color: 0xcfd6dd }), seat: lambert({ color: 0x4a3226 }),
  dark: lambert({ color: 0x333333 }), helmet: lambert({ color: 0xff5a4e }),
  stripe: lambert({ color: 0xffb347 })
};
const pelicanRoot = new THREE.Group();   // position sur le chemin
const pelicanLean = new THREE.Group();   // inclinaison + rebond
pelicanRoot.add(pelicanLean);
scene.add(pelicanRoot);
const pelicanHit = [];

function buildWheel() {
  const g = new THREE.Group();
  const R = 1.55;
  for (let k = 0; k < 16; k++) {
    const a = (k / 16) * TAU;
    const seg = box(0.58, 0.58, 0.5, M.tire, Math.cos(a) * R, Math.sin(a) * R, 0, g);
    seg.rotation.z = a;
  }
  for (let k = 0; k < 3; k++) {
    const sp = box(0.13, R * 2 - 0.5, 0.13, M.metal, 0, 0, 0, g);
    sp.rotation.z = k * Math.PI / 3;
  }
  box(0.45, 0.45, 0.55, M.dark, 0, 0, 0, g);
  return g;
}
const wheelRear = buildWheel(), wheelFront = buildWheel();
wheelRear.position.set(-2.5, 1.84, 0);
wheelFront.position.set(2.5, 1.84, 0);
pelicanLean.add(wheelRear, wheelFront);

const CRANK = new THREE.Vector3(0, 1.5, 0);
const framePts = {
  C: CRANK, RH: new THREE.Vector3(-2.5, 1.84, 0), FH: new THREE.Vector3(2.5, 1.84, 0),
  S: new THREE.Vector3(-1.5, 3.6, 0), H: new THREE.Vector3(2.1, 3.5, 0)
};
[[framePts.C, framePts.S, 0.28], [framePts.C, new THREE.Vector3(2.0, 2.9, 0), 0.28],
 [framePts.S, framePts.H, 0.26], [framePts.S, framePts.RH, 0.2],
 [framePts.C, framePts.RH, 0.2], [framePts.H, framePts.FH, 0.24],
 [framePts.H, new THREE.Vector3(2.3, 3.95, 0), 0.22]].forEach(([a, b, t]) => {
  const l = limbMesh(t, M.frame); setLimb(l, a, b); pelicanLean.add(l);
});
box(0.26, 0.26, 2.3, M.metal, 2.35, 3.95, 0, pelicanLean);
box(0.3, 0.3, 0.5, M.frame, 2.35, 3.95, 1.05, pelicanLean);
box(0.3, 0.3, 0.5, M.frame, 2.35, 3.95, -1.05, pelicanLean);
box(1.35, 0.32, 0.75, M.seat, -1.55, 3.78, 0, pelicanLean);
box(0.95, 0.95, 0.16, M.dark, 0, 1.5, 0.28, pelicanLean);

const crankGroup = new THREE.Group();
crankGroup.position.copy(CRANK);
pelicanLean.add(crankGroup);
box(0.2, 1.15, 0.2, M.metal, 0, 0.575, 0.35, crankGroup);
box(0.2, 1.15, 0.2, M.metal, 0, -0.575, -0.35, crankGroup);
const pedalA = box(0.78, 0.17, 0.52, M.dark, 0, 1.15, 0.5, crankGroup);
const pedalB = box(0.78, 0.17, 0.52, M.dark, 0, -1.15, -0.5, crankGroup);

/* corps du pélican */
const bird = new THREE.Group();
pelicanLean.add(bird);
const bodyMain = box(2.6, 1.9, 1.5, M.white, -0.7, 5.1, 0, bird); bodyMain.rotation.z = -0.14;
box(1.4, 1.4, 1.3, M.white, 0.5, 5.35, 0, bird);
const tail = box(1.25, 0.5, 0.9, M.white, -2.2, 5.55, 0, bird); tail.rotation.z = 0.5;
const tailTip = box(0.55, 0.42, 0.82, M.tip, -2.8, 5.9, 0, bird); tailTip.rotation.z = 0.5;
const neck = box(0.82, 1.7, 0.82, M.white, 0.95, 6.35, 0, bird); neck.rotation.z = -0.22;

const headGroup = new THREE.Group();
headGroup.position.set(1.35, 7.3, 0);
bird.add(headGroup);
box(1.15, 1.05, 1.0, M.white, 0, 0, 0, headGroup);
box(1.28, 0.42, 1.12, M.helmet, 0, 0.68, 0, headGroup);
box(1.3, 0.16, 0.3, M.stripe, 0, 0.9, 0, headGroup);
box(0.5, 0.35, 0.3, M.white, -0.45, 0.5, 0, headGroup);
box(0.24, 0.24, 0.12, M.eye, 0.42, 0.18, 0.52, headGroup);
box(0.24, 0.24, 0.12, M.eye, 0.42, 0.18, -0.52, headGroup);
box(2.7, 0.32, 0.56, M.beak, 1.55, -0.05, 0, headGroup);
box(0.36, 0.42, 0.5, M.beak, 2.85, -0.2, 0, headGroup);
const pouch = box(2.3, 0.55, 0.5, M.pouch, 1.4, -0.38, 0, headGroup);

/* ailes vers le guidon */
[[1, 0.85], [-1, -0.85]].forEach(([s, z]) => {
  const SH = new THREE.Vector3(-0.4, 5.7, z);
  const E = new THREE.Vector3(0.9, 4.9, s * 1.0);
  const G = new THREE.Vector3(2.35, 3.98, s * 1.02);
  const w1 = limbMesh(0.52, M.white); setLimb(w1, SH, E); bird.add(w1);
  const w2 = limbMesh(0.44, M.white); setLimb(w2, E, G); bird.add(w2);
  box(0.46, 0.46, 0.46, M.tip, G.x, G.y, G.z, bird);
});

/* pattes dynamiques */
const legs = [], feet = [];
[0.55, -0.55].forEach(z => {
  const leg = limbMesh(0.28, M.feet); bird.add(leg); legs.push({ mesh: leg, hip: new THREE.Vector3(-0.9, 4.35, z), z });
  const foot = box(0.95, 0.24, 0.56, M.feet, 0, 0, z, bird); feet.push(foot);
});
pelicanLean.traverse(o => { if (o.isMesh) { pelicanHit.push(o); o.castShadow = false; } });

/* ombre portée du pélican */
const blobShadow = new THREE.Mesh(new THREE.BoxGeometry(5.6, 0.1, 2.7),
  new THREE.MeshBasicMaterial({ color: 0x06222e, transparent: true, opacity: 0.24, depthWrite: false }));
blobShadow.position.y = PATH_H + 0.07;
scene.add(blobShadow);

/* ============================== nuages, mouettes, étoiles, embruns, soleil ============================== */
const cloudMat = lambert({ color: 0xffffff, transparent: true, opacity: 0.94 });
const clouds = [];
{
  const rng = mulberry32(777);
  for (let k = 0; k < 9; k++) {
    const cb = new VoxelBatch();
    const n = 5 + Math.floor(rng() * 5);
    for (let s = 0; s < n; s++) {
      const w = 5 + rng() * 9;
      cb.add((rng() - 0.5) * 16, (rng() - 0.5) * 3, (rng() - 0.5) * 8, w, 2.4 + rng() * 2, 5 + rng() * 5, C(0xffffff), 0, 0, 0, 0.02);
    }
    const m = cb.build(cloudMat, false, false);
    m.position.set((rng() - 0.5) * 700, 58 + rng() * 34, (rng() - 0.5) * 600);
    m.userData.speed = 1 + rng() * 1.4;
    scene.add(m); clouds.push(m);
  }
}

const gullMat = lambert({ color: 0xf4f4f0 });
const gullTip = lambert({ color: 0x3a3a44 });
const gullBeak = lambert({ color: 0xff9d2e });
const gulls = [];
for (let k = 0; k < 5; k++) {
  const g = new THREE.Group();
  box(0.95, 0.36, 0.36, gullMat, 0, 0, 0, g);
  box(0.34, 0.3, 0.3, gullMat, 0.55, 0.14, 0, g);
  box(0.3, 0.12, 0.12, gullBeak, 0.82, 0.1, 0, g);
  box(0.4, 0.16, 0.3, gullTip, -0.6, 0.05, 0, g);
  const wings = [];
  [1, -1].forEach(s => {
    const pivot = new THREE.Group(); pivot.position.set(0, 0.1, s * 0.18); g.add(pivot);
    box(1.35, 0.12, 0.5, gullMat, -0.1, 0, s * 0.72, pivot);
    box(0.4, 0.12, 0.5, gullTip, -0.7, 0, s * 0.72, pivot);
    wings.push({ pivot, s });
  });
  g.userData = { r: 62 + k * 13, y: 26 + k * 3.5, a: k * 1.3, sp: 0.1 + k * 0.022, wings, ph: k * 2.1 };
  scene.add(g); gulls.push(g);
}

const stars = (function () {
  const n = 520, pos = new Float32Array(n * 3), rng = mulberry32(4242);
  for (let i = 0; i < n; i++) {
    const a = rng() * TAU, e = rng() * Math.PI * 0.46 + 0.06;
    const r = 820;
    pos[i * 3] = Math.cos(a) * Math.cos(e) * r;
    pos[i * 3 + 1] = Math.sin(e) * r;
    pos[i * 3 + 2] = Math.sin(a) * Math.cos(e) * r;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const mat = new THREE.PointsMaterial({ color: 0xcfe4ff, size: 2.2, sizeAttenuation: false, transparent: true, opacity: 0, fog: false, depthWrite: false });
  const p = new THREE.Points(geo, mat); scene.add(p); return p;
})();

const SPRAY_MAX = 600;
const spray = (function () {
  const pos = new Float32Array(SPRAY_MAX * 3);
  const data = [];
  const rng = mulberry32(99);
  for (let i = 0; i < SPRAY_MAX; i++) {
    data.push({ a: rng() * TAU, r: 49 + rng() * 13, y: rng() * 5, vy: 0.6 + rng() * 1.3, ph: rng() * TAU });
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const mat = new THREE.PointsMaterial({ color: 0xdff8ff, size: 0.85, transparent: true, opacity: 0.75, depthWrite: false });
  const p = new THREE.Points(geo, mat);
  p.userData = { data, pos };
  scene.add(p); return p;
})();
let sprayCount = 260;

const sunVoxel = new THREE.Group();
{
  const sm = new THREE.MeshBasicMaterial({ color: 0xffd75e, fog: false });
  for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) {
    box(13, 13, 13, sm, i * 13.5, j * 13.5, 0, sunVoxel);
  }
  const glow = makeGlow(0xffcf6e, 190, 0.85); sunVoxel.add(glow);
  scene.add(sunVoxel);
}
const moonVoxel = new THREE.Group();
{
  const mm = new THREE.MeshBasicMaterial({ color: 0xe8eef8, fog: false });
  box(16, 16, 16, mm, 0, 0, 0, moonVoxel);
  box(9, 9, 9, mm, 10, 8, 0, moonVoxel);
  const glow = makeGlow(0xbcd0f0, 110, 0.5); moonVoxel.add(glow);
  scene.add(moonVoxel);
}

/* ============================== caméra orbitale ============================== */
const cam = {
  az: 0.85, pol: 1.12, dist: 95,
  tAz: 0.85, tPol: 1.12, tDist: 95,
  target: new THREE.Vector3(0, 7, 0),
  minD: 10, maxD: 340
};
const DEFAULT_CAM = { az: 0.85, pol: 1.12, dist: 95 };
let autoOrbit = true, follow = false, lastInteract = -10, dragging = false;
const pointers = new Map();
let pinchDist = 0, downInfo = null;

function interact() { lastInteract = perfT; }
canvas.addEventListener('pointerdown', e => {
  canvas.setPointerCapture(e.pointerId);
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (pointers.size === 1) {
    dragging = true; canvas.classList.add('dragging');
    downInfo = { x: e.clientX, y: e.clientY, t: performance.now() };
  } else if (pointers.size === 2) {
    const [a, b] = [...pointers.values()];
    pinchDist = Math.hypot(a.x - b.x, a.y - b.y);
  }
  interact();
});
canvas.addEventListener('pointermove', e => {
  if (!pointers.has(e.pointerId)) { hoverCheck(e); return; }
  const prev = pointers.get(e.pointerId);
  const dx = e.clientX - prev.x, dy = e.clientY - prev.y;
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (pointers.size === 1) {
    cam.tAz -= dx * 0.0052;
    cam.tPol = clamp(cam.tPol - dy * 0.0042, 0.18, 1.46);
  } else if (pointers.size === 2) {
    const [a, b] = [...pointers.values()];
    const d = Math.hypot(a.x - b.x, a.y - b.y);
    if (pinchDist > 0) cam.tDist = clamp(cam.tDist * pinchDist / d, cam.minD, cam.maxD);
    pinchDist = d;
    syncZoomUI();
  }
  interact();
});
function endPointer(e) {
  if (pointers.has(e.pointerId)) {
    pointers.delete(e.pointerId);
    if (pointers.size === 0) {
      dragging = false; canvas.classList.remove('dragging');
      if (downInfo && Math.hypot(e.clientX - downInfo.x, e.clientY - downInfo.y) < 6 && performance.now() - downInfo.t < 450) {
        tryClickPelican(e);
      }
      downInfo = null;
    }
    pinchDist = 0;
  }
}
canvas.addEventListener('pointerup', endPointer);
canvas.addEventListener('pointercancel', endPointer);
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  cam.tDist = clamp(cam.tDist * Math.exp(e.deltaY * 0.0012), cam.minD, cam.maxD);
  syncZoomUI(); interact();
}, { passive: false });

const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();
function pickPelican(e) {
  ndc.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  return raycaster.intersectObjects(pelicanHit, false).length > 0;
}
let hoverThrottle = 0;
function hoverCheck(e) {
  const now = performance.now();
  if (now - hoverThrottle < 90) return;
  hoverThrottle = now;
  canvas.style.cursor = pickPelican(e) ? 'pointer' : 'grab';
}
let hopT = -1;
function tryClickPelican(e) {
  if (pickPelican(e)) {
    hopT = 0; ringBell(); toast('Ring ring ! Le pélican vous salue.');
  }
}

/* ============================== audio synthétique ============================== */
let audioCtx = null, masterGain = null, soundOn = true;
function initAudio() {
  if (audioCtx) return;
  const AC = window.AudioContext || window.webkitAudioContext;
  if (!AC) return;
  audioCtx = new AC();
  masterGain = audioCtx.createGain();
  masterGain.gain.value = soundOn ? 0.9 : 0;
  masterGain.connect(audioCtx.destination);

  /* ressac : bruit brun filtré + LFO */
  const len = audioCtx.sampleRate * 2;
  const buf = audioCtx.createBuffer(1, len, audioCtx.sampleRate);
  const d = buf.getChannelData(0);
  let last = 0;
  for (let i = 0; i < len; i++) {
    const w = Math.random() * 2 - 1;
    last = (last + 0.02 * w) / 1.02;
    d[i] = last * 3.2;
  }
  const src = audioCtx.createBufferSource(); src.buffer = buf; src.loop = true;
  const lp = audioCtx.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 480;
  const wg = audioCtx.createGain(); wg.gain.value = 0.16;
  const lfo = audioCtx.createOscillator(); lfo.frequency.value = 0.13;
  const lfoG = audioCtx.createGain(); lfoG.gain.value = 0.08;
  lfo.connect(lfoG); lfoG.connect(wg.gain);
  src.connect(lp); lp.connect(wg); wg.connect(masterGain);
  src.start(); lfo.start();

  const src2 = audioCtx.createBufferSource(); src2.buffer = buf; src2.loop = true; src2.playbackRate.value = 1.7;
  const bp = audioCtx.createBiquadFilter(); bp.type = 'bandpass'; bp.frequency.value = 1400; bp.Q.value = 0.7;
  const hg = audioCtx.createGain(); hg.gain.value = 0.022;
  const lfo2 = audioCtx.createOscillator(); lfo2.frequency.value = 0.09;
  const lfo2G = audioCtx.createGain(); lfo2G.gain.value = 0.014;
  lfo2.connect(lfo2G); lfo2G.connect(hg.gain);
  src2.connect(bp); bp.connect(hg); hg.connect(masterGain);
  src2.start(); lfo2.start();

  scheduleGull();
}
function scheduleGull() {
  setTimeout(() => {
    if (audioCtx && soundOn && !document.hidden) gullCry();
    scheduleGull();
  }, 7000 + Math.random() * 11000);
}
function gullCry() {
  const t0 = audioCtx.currentTime;
  for (let k = 0; k < 2; k++) {
    const t = t0 + k * 0.28;
    const o = audioCtx.createOscillator(); o.type = 'sine';
    o.frequency.setValueAtTime(1350 + Math.random() * 200, t);
    o.frequency.exponentialRampToValueAtTime(820, t + 0.22);
    const g = audioCtx.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.045, t + 0.04);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.26);
    o.connect(g); g.connect(masterGain);
    o.start(t); o.stop(t + 0.3);
  }
}
function ringBell() {
  if (!audioCtx || !soundOn) return;
  const t0 = audioCtx.currentTime;
  [1760, 2217, 2637].forEach((f, i) => {
    const o = audioCtx.createOscillator(); o.type = 'triangle'; o.frequency.value = f;
    const g = audioCtx.createGain();
    const t = t0 + i * 0.09;
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.09 / (i + 1), t + 0.012);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.55);
    o.connect(g); g.connect(masterGain);
    o.start(t); o.stop(t + 0.6);
  });
}
document.addEventListener('pointerdown', function firstGesture() {
  if (soundOn) { initAudio(); if (audioCtx && audioCtx.state === 'suspended') audioCtx.resume(); }
  document.removeEventListener('pointerdown', firstGesture);
}, { once: true });

const soundBtn = document.getElementById('soundBtn');
const soundLabel = document.getElementById('soundLabel');
soundBtn.addEventListener('click', () => {
  soundOn = !soundOn;
  if (soundOn) { initAudio(); if (audioCtx && audioCtx.state === 'suspended') audioCtx.resume(); }
  if (masterGain) masterGain.gain.setTargetAtTime(soundOn ? 0.9 : 0, audioCtx.currentTime, 0.05);
  soundBtn.classList.toggle('on', soundOn);
  soundBtn.classList.toggle('off', !soundOn);
  soundLabel.textContent = soundOn ? 'Son activé' : 'Son coupé';
});

/* ============================== interface ============================== */
const $ = id => document.getElementById(id);
const ui = {
  zoom: $('zoom'), zoomOut: $('zoomOut'), time: $('time'), timeOut: $('timeOut'),
  wave: $('wave'), waveOut: $('waveOut'), spray: $('spray'), sprayOut: $('sprayOut'),
  speed: $('speed'), speedOut: $('speedOut'), quality: $('quality'),
  autoOrbit: $('autoOrbit'), follow: $('follow'), wire: $('wire'),
  lapCount: $('lapCount')
};
let toastTimer = null;
function toast(msg) {
  const el = $('toast');
  el.textContent = msg; el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), 2400);
}
function distToSlider(d) { return 100 * Math.log(d / cam.minD) / Math.log(cam.maxD / cam.minD); }
function sliderToDist(v) { return cam.minD * Math.pow(cam.maxD / cam.minD, v / 100); }
function syncZoomUI() {
  ui.zoom.value = distToSlider(cam.tDist);
  ui.zoomOut.textContent = Math.round(cam.tDist) + ' m';
}
ui.zoom.addEventListener('input', () => { cam.tDist = sliderToDist(+ui.zoom.value); ui.zoomOut.textContent = Math.round(cam.tDist) + ' m'; interact(); });

let timeOfDay = 10.5;
function fmtTime(t) {
  const h = Math.floor(t) % 24, m = Math.floor((t % 1) * 60);
  return h + ' h ' + String(m).padStart(2, '0');
}
ui.time.addEventListener('input', () => { timeOfDay = +ui.time.value; ui.timeOut.textContent = fmtTime(timeOfDay); applyTime(); });
ui.wave.addEventListener('input', () => { waveAmp = +ui.wave.value; ui.waveOut.textContent = waveAmp.toFixed(2).replace(/0$/, '') + '×'; });
ui.spray.addEventListener('input', () => { sprayCount = +ui.spray.value; ui.sprayOut.textContent = sprayCount; });
let speedMul = 1;
ui.speed.addEventListener('input', () => { speedMul = +ui.speed.value; ui.speedOut.textContent = speedMul.toFixed(2).replace(/0$/, '') + '×'; });
ui.autoOrbit.addEventListener('change', () => { autoOrbit = ui.autoOrbit.checked; });
ui.follow.addEventListener('change', () => {
  follow = ui.follow.checked;
  if (follow) { cam.tDist = 24; cam.tPol = 0.95; syncZoomUI(); }
  toast(follow ? 'Caméra : poursuite du pélican' : 'Caméra : vue de l\'île');
});
ui.wire.addEventListener('change', () => {
  const w = ui.wire.checked;
  MATS.forEach(m => { m.wireframe = w; });
  toast(w ? 'Mode fil de fer activé' : 'Mode fil de fer désactivé');
});
ui.quality.addEventListener('change', applyQuality);
function applyQuality() {
  const q = ui.quality.value;
  const dpr = window.devicePixelRatio || 1;
  renderer.setPixelRatio(q === 'low' ? 1 : q === 'med' ? Math.min(dpr, 1.5) : Math.min(dpr, 2));
  renderer.shadowMap.enabled = q !== 'low';
  MATS.forEach(m => { m.needsUpdate = true; });
  shadowDirty = true;
  onResize();
}

$('resetBtn').addEventListener('click', () => {
  timeOfDay = 10.5; ui.time.value = 10.5; ui.timeOut.textContent = fmtTime(10.5);
  waveAmp = 1; ui.wave.value = 1; ui.waveOut.textContent = '1.0×';
  sprayCount = 260; ui.spray.value = 260; ui.sprayOut.textContent = '260';
  speedMul = 1; ui.speed.value = 1; ui.speedOut.textContent = '1.0×';
  ui.quality.value = 'med'; applyQuality();
  ui.wire.checked = false; MATS.forEach(m => { m.wireframe = false; });
  autoOrbit = true; ui.autoOrbit.checked = true;
  follow = false; ui.follow.checked = false;
  cam.tAz = DEFAULT_CAM.az; cam.tPol = DEFAULT_CAM.pol; cam.tDist = DEFAULT_CAM.dist;
  syncZoomUI(); applyTime();
  toast('Réglages réinitialisés');
});
$('randBtn').addEventListener('click', () => {
  SEED = (Math.random() * 1e6) | 0;
  buildIsland();
  toast('Nouvelle île générée !');
});
$('shotBtn').addEventListener('click', () => {
  renderer.render(scene, camera);
  const a = document.createElement('a');
  a.download = 'pelican-a-velo.png';
  a.href = renderer.domElement.toDataURL('image/png');
  a.click();
  toast('Vue exportée en PNG');
});

let paused = false;
document.addEventListener('keydown', e => {
  if (e.code === 'Space' && !/INPUT|SELECT|TEXTAREA|BUTTON/.test(document.activeElement.tagName)) {
    e.preventDefault();
    paused = !paused;
    toast(paused ? 'Pause — le pélican s\'arrête' : 'Reprise du pédalage');
  }
});

/* ============================== cycle jour / nuit ============================== */
const colNightTop = C(0x0a1030), colDayTop = C(0x3f9fe0), colDuskTop = C(0x4a3f7a);
const colNightHor = C(0x16213e), colDayHor = C(0xbfe8f5), colDuskHor = C(0xff9a5c);
const colSunDay = C(0xfff2d8), colSunDusk = C(0xff8a3d), colMoon = C(0x7d95c9);
const _skyTop = new THREE.Color(), _skyHor = new THREE.Color(), _lc = new THREE.Color();
let nightF = 0;
function applyTime() {
  const ang = (timeOfDay - 6) / 12 * Math.PI;
  const elev = Math.sin(ang);
  const dayF = clamp01((elev + 0.12) / 0.5);
  const duskF = clamp01(1 - Math.abs(elev) / 0.38);
  nightF = 1 - dayF;

  _skyTop.copy(colNightTop).lerp(colDayTop, dayF).lerp(colDuskTop, duskF * 0.65);
  _skyHor.copy(colNightHor).lerp(colDayHor, dayF).lerp(colDuskHor, duskF * 0.85);
  skyUniforms.topColor.value.copy(_skyTop);
  skyUniforms.bottomColor.value.copy(_skyHor);
  scene.fog.color.copy(_skyHor);
  scene.fog.near = 90 + cam.dist * 0.5;
  scene.fog.far = 430 + cam.dist * 1.6;

  const dir = new THREE.Vector3(Math.cos(ang), Math.max(elev, 0.06), Math.sin(ang) * 0.55).normalize();
  if (elev > -0.05) {
    sunLight.position.copy(dir).multiplyScalar(300);
    _lc.copy(colSunDay).lerp(colSunDusk, duskF);
    sunLight.color.copy(_lc);
    sunLight.intensity = 0.25 + 0.95 * dayF;
  } else {
    sunLight.position.set(-dir.x * 300, 180, -dir.z * 300);
    sunLight.color.copy(colMoon);
    sunLight.intensity = 0.34;
  }
  hemi.color.copy(_skyHor);
  hemi.intensity = 0.3 + 0.38 * dayF;
  ambient.intensity = 0.07 + 0.08 * dayF;

  sunVoxel.position.copy(dir).multiplyScalar(780);
  sunVoxel.visible = elev > -0.12;
  moonVoxel.position.set(-dir.x * 780, Math.max(-elev, 0.08) * 780, -dir.z * 780);
  moonVoxel.visible = elev < 0.1;
  stars.material.opacity = nightF * 0.95;

  lampMat.emissiveIntensity = 0.25 + 2.4 * nightF;
  lampGlow.material.opacity = 0.85 * nightF;
  beamGroup.userData.mat.opacity = 0.2 * nightF;
  cloudMat.color.setScalar(0.35 + 0.65 * dayF);
  waterMat.color.setScalar(0.5 + 0.5 * dayF);
  deepPlane.material.color.setHex(0x0d6fae).multiplyScalar(0.4 + 0.6 * dayF);
  spray.material.color.copy(C(0xdff8ff)).lerp(C(0xffd98a), nightF * 0.8);
  shadowDirty = true;
}

/* ============================== boucle ============================== */
buildIsland();
buildStructures();
applyQuality();
applyTime();
syncZoomUI();

let perfT = 0, prevNow = performance.now(), frame = 0;
let pathAngle = 0.6, wheelAngle = 0, crankAngle = 0, totalAngle = 0, laps = 0;
const PATH_SPEED = 0.16;
const pelicanPos = new THREE.Vector3();
const _footTarget = new THREE.Vector3();
const _desiredTarget = new THREE.Vector3();

function updatePelican(dt, t) {
  const v = paused ? 0 : PATH_SPEED * speedMul;
  pathAngle += v * dt;
  totalAngle += v * dt;
  const newLaps = Math.floor(totalAngle / TAU);
  if (newLaps !== laps) { laps = newLaps; ui.lapCount.textContent = laps; }

  const x = Math.cos(pathAngle) * PATH_R, z = Math.sin(pathAngle) * PATH_R;
  pelicanRoot.position.set(x, PATH_H, z);
  pelicanRoot.rotation.y = -pathAngle - Math.PI / 2;
  pelicanPos.set(x, PATH_H, z);

  const linSpeed = v * PATH_R;
  wheelAngle -= linSpeed / 1.55 * dt;
  wheelRear.rotation.z = wheelAngle;
  wheelFront.rotation.z = wheelAngle;
  crankAngle = wheelAngle * 0.55;
  crankGroup.rotation.z = crankAngle;

  /* pieds sur les pédales (positions locales du manivellon) */
  for (let i = 0; i < 2; i++) {
    const a = crankAngle + i * Math.PI;
    const px = -Math.sin(a) * 1.15;
    const py = 1.5 + Math.cos(a) * 1.15;
    const pz = i === 0 ? 0.5 : -0.5;
    feet[i].position.set(px, py + 0.16, pz);
    _footTarget.set(px, py + 0.2, pz);
    setLimb(legs[i].mesh, legs[i].hip, _footTarget);
  }

  /* rebond, poche qui ballotte, queue, tête */
  const bounce = Math.sin(crankAngle * 2) * 0.1 * clamp01(speedMul);
  let hop = 0;
  if (hopT >= 0) {
    hopT += dt / 0.6;
    if (hopT >= 1) hopT = -1;
    else hop = Math.sin(hopT * Math.PI) * 1.3;
  }
  pelicanLean.position.y = bounce + hop;
  pelicanLean.rotation.x = 0.05 + Math.sin(t * 1.7) * 0.015;
  pouch.scale.y = 1 + Math.sin(t * 5 + crankAngle) * 0.14;
  pouch.position.y = -0.38 - Math.sin(t * 5 + crankAngle) * 0.05;
  tail.rotation.y = Math.sin(t * 3.2) * 0.14;
  tailTip.rotation.y = tail.rotation.y;
  headGroup.rotation.z = Math.sin(t * 2.3) * 0.05 + (hopT >= 0 ? 0.18 : 0);

  blobShadow.position.set(x, PATH_H + 0.07, z);
  blobShadow.rotation.y = pelicanRoot.rotation.y;
  blobShadow.material.opacity = 0.24 * (1 - hop * 0.4);
}

function updateCamera(dt, t) {
  if (autoOrbit && !dragging && t - lastInteract > 2.5) cam.tAz += dt * 0.055;
  const k = 1 - Math.exp(-dt * 7);
  cam.az += (cam.tAz - cam.az) * k;
  cam.pol += (cam.tPol - cam.pol) * k;
  cam.dist += (cam.tDist - cam.dist) * k;
  scene.fog.near = 90 + cam.dist * 0.5;
  scene.fog.far = 430 + cam.dist * 1.6;
  if (follow) _desiredTarget.copy(pelicanPos).add(new THREE.Vector3(0, 4.5, 0));
  else _desiredTarget.set(0, 7, 0);
  cam.target.lerp(_desiredTarget, 1 - Math.exp(-dt * 4));
  const sp = Math.sin(cam.pol), cp = Math.cos(cam.pol);
  camera.position.set(
    cam.target.x + cam.dist * sp * Math.cos(cam.az),
    cam.target.y + cam.dist * cp,
    cam.target.z + cam.dist * sp * Math.sin(cam.az)
  );
  camera.lookAt(cam.target);
}

function updateAmbient(dt, t) {
  clouds.forEach(c => {
    c.position.x += c.userData.speed * dt;
    if (c.position.x > 420) c.position.x = -420;
  });
  gulls.forEach(g => {
    const u = g.userData;
    u.a += u.sp * dt;
    g.position.set(Math.cos(u.a) * u.r, u.y + Math.sin(t * 0.9 + u.ph) * 1.6, Math.sin(u.a) * u.r);
    g.rotation.y = -u.a - Math.PI / 2;
    u.wings.forEach(w => { w.pivot.rotation.x = Math.sin(t * 7 + u.ph) * 0.65 * w.s; });
  });
  const sd = spray.userData, pos = sd.pos;
  for (let i = 0; i < sprayCount; i++) {
    const p = sd.data[i];
    p.y += p.vy * dt;
    if (p.y > 5.2) { p.y = 0; p.a = Math.random() * TAU; }
    pos[i * 3] = Math.cos(p.a) * p.r + Math.sin(t * 1.3 + p.ph) * 0.5;
    pos[i * 3 + 1] = p.y + 0.3;
    pos[i * 3 + 2] = Math.sin(p.a) * p.r + Math.cos(t * 1.1 + p.ph) * 0.5;
  }
  spray.geometry.setDrawRange(0, sprayCount);
  spray.geometry.attributes.position.needsUpdate = true;
  beamGroup.rotation.y += dt * 0.55;
  sunVoxel.rotation.z += dt * 0.05;
}

function onResize() {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
}
window.addEventListener('resize', onResize);
onResize();

let loaderHidden = false;
function tick(dt, t) {
  frame++;
  updateWater(t, frame);
  updatePelican(dt, t);
  updateAmbient(dt, t);
  updateCamera(dt, t);
  if (shadowDirty && renderer.shadowMap.enabled) {
    renderer.shadowMap.needsUpdate = true;
    shadowDirty = false;
  }
  renderer.render(scene, camera);
}
function loop(now) {
  requestAnimationFrame(loop);
  const dt = Math.min((now - prevNow) / 1000, 0.05);
  prevNow = now;
  perfT = now / 1000;
  tick(dt, perfT);
  if (!loaderHidden) {
    loaderHidden = true;
    setTimeout(() => document.getElementById('loader').classList.add('hide'), 250);
  }
}
requestAnimationFrame(loop);

/* hook de test / inspection */
window.__debug = {
  renderer, scene, camera, cam, sunLight,
  advance(seconds) {
    const step = 1 / 30;
    for (let s = 0; s < seconds; s += step) { perfT += step; tick(step, perfT); }
  },
  capture() { renderer.render(scene, camera); return renderer.domElement.toDataURL('image/png'); },
  state() {
    return {
      laps, paused, speedMul, timeOfDay, waveAmp, sprayCount, nightF,
      pelican: pelicanPos.toArray(), camDist: cam.dist, follow, autoOrbit,
      drawCalls: renderer.info.render.calls, triangles: renderer.info.render.triangles
    };
  }
};

})();