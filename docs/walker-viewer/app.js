import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const JOINT_NAMES = [
  "鼻", "左眼", "右眼", "左耳", "右耳", "左肩", "右肩", "左肘", "右肘",
  "左腕", "右腕", "左髋", "右髋", "左膝", "右膝", "左踝", "右踝",
];
const STAGES = {
  stage1_walker_static_human_moving: ["Stage 1 · 人运动 / 助步器静止", "stage-one"],
  stage2_feet_static_walker_moving: ["Stage 2 · 双脚近似静止 / 助步器运动", "stage-two"],
  transition: ["过渡状态", "transition"],
  warming_up: ["预热", "warming"],
};
const ERROR_COLORS = [0x009e73, 0xe69f00, 0xd55e00, 0x777777];

const viewport = document.getElementById("viewport");
const loading = document.getElementById("loading");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf5f8fa);
const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 100);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
viewport.prepend(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 0.5;
controls.maxDistance = 12;

scene.add(new THREE.HemisphereLight(0xffffff, 0xa9b8c0, 2.0));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.6);
keyLight.position.set(3, 6, 4);
scene.add(keyLight);

const groups = {
  ground: new THREE.Group(), trails: new THREE.Group(), walker: new THREE.Group(),
  cameras: new THREE.Group(), human: new THREE.Group(),
};
Object.values(groups).forEach((group) => scene.add(group));

const humanLineGeometry = new THREE.BufferGeometry();
const humanLine = new THREE.LineSegments(
  humanLineGeometry,
  new THREE.LineBasicMaterial({ color: 0x20252a, depthTest: false, transparent: true, opacity: 0.95 }),
);
humanLine.renderOrder = 100;
groups.human.add(humanLine);

const jointGeometry = new THREE.SphereGeometry(0.025, 14, 10);
const jointMeshes = Array.from({ length: 17 }, (_, index) => {
  const mesh = new THREE.Mesh(jointGeometry, new THREE.MeshBasicMaterial({ color: ERROR_COLORS[3], depthTest: false }));
  mesh.userData.jointIndex = index;
  mesh.renderOrder = 110;
  groups.human.add(mesh);
  return mesh;
});

const walkerFrame = new THREE.LineSegments(
  new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x00a6d6, linewidth: 1 }),
);
const walkerHandles = new THREE.LineSegments(
  new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xcc79a7, linewidth: 1 }),
);
groups.walker.add(walkerFrame, walkerHandles);

const footGeometry = new THREE.SphereGeometry(0.032, 12, 8);
const footMeshes = [];
const cameraGeometry = new THREE.ConeGeometry(0.035, 0.08, 4);
const cameraMeshes = {
  left: new THREE.Mesh(cameraGeometry, new THREE.MeshBasicMaterial({ color: 0x2e8b57 })),
  right: new THREE.Mesh(cameraGeometry, new THREE.MeshBasicMaterial({ color: 0x7b61a8 })),
};
cameraMeshes.left.rotation.z = Math.PI;
cameraMeshes.right.rotation.z = Math.PI;
groups.cameras.add(cameraMeshes.left, cameraMeshes.right);
const cameraBaseline = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x5b6573 }));
groups.cameras.add(cameraBaseline);

const trailMaterial = new THREE.LineBasicMaterial({ color: 0x2e8b57, transparent: true, opacity: 0.68 });
const cameraTrail = new THREE.Line(new THREE.BufferGeometry(), trailMaterial);
const leftAnkleTrail = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x0072b2, transparent: true, opacity: 0.75 }));
const rightAnkleTrail = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xd55e00, transparent: true, opacity: 0.75 }));
groups.trails.add(cameraTrail, leftAnkleTrail, rightAnkleTrail);

let data;
let currentFrame = 0;
let selectedJoint = null;
let playing = false;
let speed = 1;
let lastTick = performance.now();
let accumulator = 0;
let center = new THREE.Vector3();
let span = 3;

function worldVector(pointMm) {
  return new THREE.Vector3(pointMm[0] / 1000, pointMm[2] / 1000, -pointMm[1] / 1000);
}

function transformModelPoint(pointMm, frame) {
  const r = frame.rotation_world_from_left;
  const t = frame.translation_world_from_left_mm;
  return worldVector([
    r[0][0] * pointMm[0] + r[0][1] * pointMm[1] + r[0][2] * pointMm[2] + t[0],
    r[1][0] * pointMm[0] + r[1][1] * pointMm[1] + r[1][2] * pointMm[2] + t[1],
    r[2][0] * pointMm[0] + r[2][1] * pointMm[1] + r[2][2] * pointMm[2] + t[2],
  ]);
}

function setSegments(line, segments) {
  const values = [];
  for (const [a, b] of segments) values.push(a.x, a.y, a.z, b.x, b.y, b.z);
  line.geometry.dispose();
  line.geometry = new THREE.BufferGeometry();
  line.geometry.setAttribute("position", new THREE.Float32BufferAttribute(values, 3));
  line.geometry.computeBoundingSphere();
}

function setTrail(line, points) {
  line.geometry.dispose();
  line.geometry = new THREE.BufferGeometry().setFromPoints(points);
}

function qualityColor(error) {
  if (error == null || !Number.isFinite(error)) return ERROR_COLORS[3];
  if (error <= 5) return ERROR_COLORS[0];
  if (error <= 10) return ERROR_COLORS[1];
  return ERROR_COLORS[2];
}

function buildGround() {
  const grid = new THREE.GridHelper(4.8, 24, 0x7d8991, 0xc5cdd2);
  grid.position.copy(center);
  grid.position.y = 0;
  groups.ground.add(grid);
  const axes = new THREE.AxesHelper(0.45);
  axes.position.set(center.x - span * 0.45, 0.002, center.z + span * 0.30);
  groups.ground.add(axes);
}

function configureScene() {
  const positions = [];
  for (const frame of data.frames) {
    for (const point of frame.points_mm) if (point) positions.push(worldVector(point));
    positions.push(worldVector(frame.translation_world_from_left_mm));
  }
  const box = new THREE.Box3().setFromPoints(positions);
  box.getCenter(center);
  center.y = Math.max(0.65, center.y * 0.75);
  const size = new THREE.Vector3();
  box.getSize(size);
  span = Math.max(size.x, size.y, size.z, 2.2);
  controls.target.copy(center);
  buildGround();

  const trail = data.frames.map((frame) => worldVector(frame.translation_world_from_left_mm));
  setTrail(cameraTrail, trail);
  for (const name of data.walker.foot_nodes) {
    const mesh = new THREE.Mesh(footGeometry, new THREE.MeshBasicMaterial({ color: 0x009e73 }));
    groups.walker.add(mesh);
    footMeshes.push([name, mesh]);
  }
  setView("recommended");
}

function updateJointDetail() {
  const detail = document.getElementById("joint-detail");
  if (selectedJoint == null) {
    detail.textContent = "点击人体关节点可查看三维坐标、重投影误差和来源。";
    return;
  }
  const frame = data.frames[currentFrame];
  const point = frame.points_mm[selectedJoint];
  if (!point) {
    detail.textContent = `${JOINT_NAMES[selectedJoint]}：当前帧无可显示三维点。`;
    return;
  }
  const error = frame.errors_px[selectedJoint];
  detail.textContent = `${JOINT_NAMES[selectedJoint]}：(${point.map((v) => v.toFixed(1)).join(", ")}) mm；重投影误差 ${error == null ? "未知" : `${error.toFixed(2)} px`}；${frame.strict[selectedJoint] ? "严格双目点" : "放宽显示点"}。`;
}

function updateFrame(index) {
  currentFrame = Math.max(0, Math.min(data.frames.length - 1, index));
  const frame = data.frames[currentFrame];
  const pointVectors = frame.points_mm.map((point) => point ? worldVector(point) : null);
  const humanSegments = [];
  for (const [a, b] of data.skeleton_edges) if (pointVectors[a] && pointVectors[b]) humanSegments.push([pointVectors[a], pointVectors[b]]);
  setSegments(humanLine, humanSegments);
  jointMeshes.forEach((mesh, index2) => {
    mesh.visible = Boolean(pointVectors[index2]);
    if (pointVectors[index2]) {
      mesh.position.copy(pointVectors[index2]);
      mesh.material.color.setHex(qualityColor(frame.errors_px[index2]));
      mesh.scale.setScalar(index2 === selectedJoint ? 1.45 : 1);
    }
  });

  const nodes = {};
  for (const [name, point] of Object.entries(data.walker.nodes_left_camera_mm)) nodes[name] = transformModelPoint(point, frame);
  const handles = new Set(Object.values(data.walker.handle_segments).map((edge) => edge.join("|")));
  const frameSegments = [], handleSegments = [];
  for (const [a, b] of data.walker.edges) {
    (handles.has(`${a}|${b}`) || handles.has(`${b}|${a}`) ? handleSegments : frameSegments).push([nodes[a], nodes[b]]);
  }
  setSegments(walkerFrame, frameSegments);
  setSegments(walkerHandles, handleSegments);
  for (const [name, mesh] of footMeshes) {
    mesh.position.copy(nodes[name]);
    mesh.material.color.setHex(Math.abs(nodes[name].y) <= 0.025 ? 0x009e73 : 0xd55e00);
  }

  const leftCamera = transformModelPoint(data.walker.camera_centers_left_camera_mm.left, frame);
  const rightCamera = transformModelPoint(data.walker.camera_centers_left_camera_mm.right, frame);
  cameraMeshes.left.position.copy(leftCamera);
  cameraMeshes.right.position.copy(rightCamera);
  setTrail(cameraBaseline, [leftCamera, rightCamera]);

  const start = Math.max(0, currentFrame - 119);
  const ankleTrail = (jointIndex) => data.frames.slice(start, currentFrame + 1)
    .map((item) => item.points_mm[jointIndex]).filter(Boolean)
    .map((point) => { const value = worldVector(point); value.y = 0.007; return value; });
  setTrail(leftAnkleTrail, ankleTrail(15));
  setTrail(rightAnkleTrail, ankleTrail(16));

  const stage = STAGES[frame.stage] || [frame.stage, "transition"];
  const badge = document.getElementById("stage-badge");
  badge.textContent = stage[0];
  badge.dataset.stage = stage[1];
  document.getElementById("frame-label").textContent = `${currentFrame + 1} / ${data.frame_count} · pair ${String(frame.pair).padStart(4, "0")}`;
  document.getElementById("time-value").textContent = `${(currentFrame / data.fps).toFixed(2)} s`;
  const metric = (value) => value == null ? "—" : `${value.toFixed(1)} mm`;
  document.getElementById("left-hand-value").textContent = metric(frame.wrist_handle_mm.left);
  document.getElementById("right-hand-value").textContent = metric(frame.wrist_handle_mm.right);
  document.getElementById("walker-foot-value").textContent = metric(frame.walker_feet_max_abs_z_mm);
  document.getElementById("frame-slider").value = currentFrame;
  updateJointDetail();
}

function setView(name) {
  const distance = span * 1.05;
  const targets = {
    recommended: new THREE.Vector3(center.x + distance * 0.72, center.y + distance * 0.45, center.z + distance * 0.9),
    front: new THREE.Vector3(center.x, center.y + distance * 0.15, center.z + distance),
    side: new THREE.Vector3(center.x + distance, center.y + distance * 0.15, center.z),
    top: new THREE.Vector3(center.x, center.y + distance * 1.25, center.z + 0.001),
  };
  camera.position.copy(targets[name] || targets.recommended);
  controls.target.copy(center);
  controls.update();
}

function resize() {
  const width = viewport.clientWidth;
  const height = viewport.clientHeight;
  renderer.setSize(width, height, false);
  camera.aspect = width / Math.max(height, 1);
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(viewport);

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
renderer.domElement.addEventListener("pointerdown", (event) => {
  if (event.button !== 0) return;
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObjects(jointMeshes.filter((mesh) => mesh.visible))[0];
  if (hit) {
    selectedJoint = hit.object.userData.jointIndex;
    updateFrame(currentFrame);
  }
});

document.getElementById("play-button").addEventListener("click", () => {
  playing = !playing;
  document.getElementById("play-button").textContent = playing ? "暂停" : "播放";
});
document.getElementById("frame-slider").addEventListener("input", (event) => { playing = false; document.getElementById("play-button").textContent = "播放"; updateFrame(Number(event.target.value)); });
document.getElementById("speed-select").addEventListener("change", (event) => { speed = Number(event.target.value); });
document.querySelectorAll("[data-view]").forEach((button) => button.addEventListener("click", () => setView(button.dataset.view)));
for (const [id, group] of [["show-human", "human"], ["show-walker", "walker"], ["show-cameras", "cameras"], ["show-trails", "trails"], ["show-ground", "ground"]]) {
  document.getElementById(id).addEventListener("change", (event) => { groups[group].visible = event.target.checked; });
}

function animate(now) {
  requestAnimationFrame(animate);
  const elapsed = (now - lastTick) / 1000;
  lastTick = now;
  if (playing) {
    accumulator += elapsed * data.fps * speed;
    if (accumulator >= 1) {
      const steps = Math.floor(accumulator);
      accumulator -= steps;
      updateFrame((currentFrame + steps) % data.frames.length);
    }
  }
  controls.update();
  renderer.render(scene, camera);
}

async function loadMotionData() {
  const embedded = document.getElementById("motion-data");
  if (embedded) return JSON.parse(embedded.textContent);
  const response = await fetch("./data/motion.json");
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

try {
  data = await loadMotionData();
  document.getElementById("frame-slider").max = data.frame_count - 1;
  configureScene();
  updateFrame(0);
  loading.remove();
  resize();
  requestAnimationFrame(animate);
} catch (error) {
  loading.textContent = `数据载入失败：${error.message}`;
  loading.setAttribute("role", "alert");
}
