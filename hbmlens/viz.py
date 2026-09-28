"""3D viewer of a virtual HBM cube with its failing cells (three.js, single HTML file).

Layout (illustrative, not a floorplan of any real product):

* each stack is drawn as a base (logic) die with one core die per channel on top
* on each core die, banks are tiles in a grid: columns = pseudo channel x bank group,
  rows = sid x bank
* inside a bank tile, x = column/word position in the row and depth = row number
* failing words are points, coloured by the fault kind when ground truth is known
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .faults import FaultSet
from .mapping import LinearMapper
from .records import FailLog

KIND_ORDER = ("dq-lane", "row", "column", "cell", "retention", "unknown")
KIND_COLOR = {"dq-lane": "#ff5d73", "row": "#ffb020", "column": "#3fc1ff",
              "cell": "#b6f36b", "retention": "#c792ff", "unknown": "#e8e8e8"}


def _kinds_for(indices: np.ndarray, faults: FaultSet | None) -> np.ndarray:
    kinds = np.full(len(indices), KIND_ORDER.index("unknown"), dtype=np.int8)
    if faults is None:
        return kinds
    # later entries win, so list broad faults first and specific ones last
    for f in sorted(faults.faults, key=lambda f: -len(f.indices)):
        kinds[np.isin(indices, f.indices)] = KIND_ORDER.index(f.kind)
    ret = faults.retention.indices
    if len(ret):
        kinds[np.isin(indices, ret) & (kinds == KIND_ORDER.index("unknown"))] = KIND_ORDER.index("retention")
    return kinds


def viewer_data(log: FailLog, mapper: LinearMapper, faults: FaultSet | None = None,
                max_points: int = 60_000, seed: int = 0) -> dict:
    g = mapper.geometry
    idx = log.failing_indices().astype(np.int64)
    kinds = _kinds_for(idx.astype(np.uint64), faults)
    shown = np.arange(len(idx))
    if len(idx) > max_points:  # stratified sample so small fault kinds stay visible
        rng = np.random.default_rng(seed)
        keep = []
        per_kind = max_points // max(len(np.unique(kinds)), 1)
        for k in np.unique(kinds):
            members = np.flatnonzero(kinds == k)
            keep.append(members if len(members) <= per_kind else rng.choice(members, per_kind, replace=False))
        shown = np.sort(np.concatenate(keep))
    f = mapper.decode(idx[shown])
    pos_in_row = f["column"] * g.words_per_column + f["word"]
    points = np.stack([f["stack"], f["channel"], f["pseudo_channel"], f["sid"], f["bank_group"],
                       f["bank"], f["row"], pos_in_row, kinds[shown]], axis=1).astype(np.int32)
    counts = {KIND_ORDER[k]: int((kinds == k).sum()) for k in np.unique(kinds)}
    return {
        "geometry": g.as_dict(),
        "describe": g.describe(),
        "mapper": mapper.describe(),
        "pattern": log.meta.pattern,
        "backend": log.meta.backend,
        "temperature_c": log.meta.temperature_c,
        "total_fails": log.meta.total_fails,
        "failing_words": int(len(idx)),
        "shown_words": int(len(shown)),
        "kinds": list(KIND_ORDER),
        "colors": KIND_COLOR,
        "counts": counts,
        "synthetic": faults is not None,
        "points": points.tolist(),
    }


def write_viewer(path: "str | Path", log: FailLog, mapper: LinearMapper, faults: FaultSet | None = None,
                 **kw) -> Path:
    data = viewer_data(log, mapper, faults, **kw)
    html = _TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>hbmlens viewer</title>
<style>
  html,body{margin:0;height:100%;background:#0b0e13;color:#dfe5ee;font:13px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;overflow:hidden}
  #panel{position:absolute;top:12px;left:12px;max-width:340px;background:rgba(16,21,29,.88);border:1px solid #243041;border-radius:8px;padding:12px 14px}
  h1{margin:0 0 4px;font-size:15px;font-weight:600}
  .muted{color:#8a96a8}.row{display:flex;align-items:center;gap:8px;margin:3px 0}
  .sw{width:10px;height:10px;border-radius:2px;display:inline-block}
  label{cursor:pointer}input[type=range]{width:150px}
  .num{font-variant-numeric:tabular-nums;margin-left:auto;color:#aeb8c6}
  #warn{margin-top:8px;padding:6px 8px;border-radius:6px;background:#2a2214;color:#f3cf7a;font-size:12px}
</style></head>
<body>
<div id="panel">
  <h1>hbmlens &middot; virtual HBM</h1>
  <div class="muted" id="meta"></div>
  <div id="legend" style="margin-top:8px"></div>
  <div class="row" style="margin-top:8px"><span>die spacing</span><input id="gap" type="range" min="0.15" max="2.5" step="0.05" value="0.9"></div>
  <div id="warn"></div>
</div>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.169.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.169.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
const D = __DATA__;
const g = D.geometry;
const cols = g.pseudo_channels * g.bank_groups, rowsT = g.sids * g.banks_per_group;
const T = 1.0, GAP = 0.18, dieW = cols * (T + GAP) + GAP, dieD = rowsT * (T + GAP) + GAP;
const stackPitch = dieW + 3.0;

document.getElementById("meta").innerHTML =
  `${D.describe}<br>pattern <b>${D.pattern}</b> on <b>${D.backend}</b>` +
  (D.temperature_c != null ? ` at ${D.temperature_c}&deg;C` : "") +
  `<br>${D.total_fails.toLocaleString()} failing reads, ${D.failing_words.toLocaleString()} words` +
  (D.shown_words < D.failing_words ? ` (${D.shown_words.toLocaleString()} drawn)` : "") +
  `<br><span class="muted">map: ${D.mapper}</span>`;
document.getElementById("warn").textContent = D.synthetic
  ? "Synthetic faults injected into a virtual device. Layout is illustrative, not a real floorplan."
  : "Layout is illustrative; GPU address map assumed, not known.";

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0e13);
const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.05, 500);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const sun = new THREE.DirectionalLight(0xffffff, 1.1); sun.position.set(6, 12, 8); scene.add(sun);

const dieMat = new THREE.MeshStandardMaterial({ color: 0x1b2533, transparent: true, opacity: 0.55, roughness: 0.8 });
const baseMat = new THREE.MeshStandardMaterial({ color: 0x3a3f4a, roughness: 0.6 });
const tileMat = new THREE.MeshStandardMaterial({ color: 0x24344a, transparent: true, opacity: 0.8 });
const tsvMat = new THREE.MeshStandardMaterial({ color: 0xb08d57, metalness: 0.6, roughness: 0.4 });
const layers = [];   // {mesh group, channel}
const groups = {};   // kind -> THREE.Points
let gap = 0.9;

function dieY(ch) { return 0.35 + (ch + 1) * gap; }
function build() {
  for (let s = 0; s < g.stacks; s++) {
    const x0 = s * stackPitch;
    const base = new THREE.Mesh(new THREE.BoxGeometry(dieW + 0.6, 0.3, dieD + 0.6), baseMat);
    base.position.set(x0 + dieW / 2, 0.15, dieD / 2); scene.add(base);
    for (let ch = 0; ch < g.channels; ch++) {
      const grp = new THREE.Group();
      const die = new THREE.Mesh(new THREE.BoxGeometry(dieW, 0.06, dieD), dieMat);
      die.position.set(dieW / 2, 0, dieD / 2); grp.add(die);
      for (let c = 0; c < cols; c++) for (let r = 0; r < rowsT; r++) {
        const t = new THREE.Mesh(new THREE.BoxGeometry(T, 0.02, T), tileMat);
        t.position.set(GAP + c * (T + GAP) + T / 2, 0.04, GAP + r * (T + GAP) + T / 2); grp.add(t);
      }
      grp.position.set(x0, dieY(ch), 0); grp.userData = { stack: s, ch }; scene.add(grp); layers.push(grp);
    }
    for (const [tx, tz] of [[0.35, 0.35], [dieW - 0.35, 0.35], [0.35, dieD - 0.35], [dieW - 0.35, dieD - 0.35]]) {
      const tsv = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 1, 8), tsvMat);
      tsv.userData = { x0, tx, tz }; scene.add(tsv); layers.push(tsv);
    }
  }
  const byKind = {};
  for (const p of D.points) {
    const [s, ch, pc, sid, bg, bank, row, x, k] = p;
    const c = pc * g.bank_groups + bg, r = sid * g.banks_per_group + bank;
    const px = s * stackPitch + GAP + c * (T + GAP) + (x + 0.5) / (g.columns * g.words_per_column) * T;
    const pz = GAP + r * (T + GAP) + (row + 0.5) / g.rows * T;
    (byKind[k] ||= []).push(px, ch, pz);
  }
  const legend = document.getElementById("legend");
  D.kinds.forEach((name, k) => {
    if (!byKind[k]) return;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(byKind[k], 3));
    geo.userData.raw = Float32Array.from(byKind[k]);
    const pts = new THREE.Points(geo, new THREE.PointsMaterial({ color: D.colors[name], size: 0.035, sizeAttenuation: true }));
    scene.add(pts); groups[name] = pts;
    const row = document.createElement("div"); row.className = "row";
    row.innerHTML = `<label><input type="checkbox" checked> <span class="sw" style="background:${D.colors[name]}"></span> ${name}</label><span class="num">${(D.counts[name] || 0).toLocaleString()}</span>`;
    row.querySelector("input").onchange = e => { pts.visible = e.target.checked; };
    legend.appendChild(row);
  });
  relayout();
  const cx = ((g.stacks - 1) * stackPitch + dieW) / 2;
  controls.target.set(cx, dieY(g.channels / 2), dieD / 2);
  camera.position.set(cx + dieW * 1.3, dieY(g.channels) + 3.5, dieD * 2.2);
}
function relayout() {
  const top = dieY(g.channels - 1);
  for (const o of layers) {
    if (o.isGroup) o.position.y = dieY(o.userData.ch);
    else { o.scale.y = top; o.position.set(o.userData.x0 + o.userData.tx, top / 2 + 0.15, o.userData.tz); }
  }
  for (const pts of Object.values(groups)) {
    const raw = pts.geometry.userData.raw, pos = pts.geometry.attributes.position.array;
    for (let i = 0; i < raw.length; i += 3) pos[i + 1] = dieY(raw[i + 1]) + 0.07;
    pts.geometry.attributes.position.needsUpdate = true;
  }
}
document.getElementById("gap").oninput = e => { gap = +e.target.value; relayout(); };
addEventListener("resize", () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });
build();
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
</script></body></html>
"""
