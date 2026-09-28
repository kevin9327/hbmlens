"""3D viewer of a virtual HBM cube with its failing cells (three.js, single HTML file).

Layout (illustrative, not a floorplan of any real product):

* each stack is drawn as a base (logic) die with one core die per channel on top
* on each core die, banks are tiles in a grid: columns = pseudo channel x bank group,
  rows = sid x bank
* inside a bank tile, x = column/word position in the row and depth = row number
* failing words are points, coloured by the fault kind when ground truth is known
* analyzer signatures are drawn as outlines; clicking a tile opens the bank's
  full-resolution fail bitmap; a slider replays the run element by element
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze.core import Signature
from .faults import FaultSet
from .mapping import LinearMapper
from .patterns.base import Pattern
from .records import FailLog

KIND_ORDER = ("dq-lane", "row", "column", "bank", "cell", "retention", "unknown")
KIND_COLOR = {"dq-lane": "#ff5d73", "row": "#ffb020", "column": "#3fc1ff", "bank": "#ff8a3d",
              "cell": "#b6f36b", "retention": "#c792ff", "unknown": "#e8e8e8"}


def _kinds_for(indices: np.ndarray, faults: FaultSet | None) -> np.ndarray:
    kinds = np.full(len(indices), KIND_ORDER.index("unknown"), dtype=np.int8)
    if faults is None:
        return kinds
    # broad faults first so that specific ones (cells, rows) stay visible on top
    for f in sorted(faults.faults, key=lambda f: -len(f.indices)):
        kinds[np.isin(indices, f.indices)] = KIND_ORDER.index(f.kind)
    ret = faults.retention.indices
    if len(ret):
        kinds[np.isin(indices, ret) & (kinds == KIND_ORDER.index("unknown"))] = KIND_ORDER.index("retention")
    return kinds


def _kinds_from_signatures(cells: pd.DataFrame, signatures: list[Signature]) -> dict[int, int]:
    """word index -> kind index, using analyzer signatures (for logs without ground truth)."""
    out: dict[int, int] = {}
    for s in sorted(signatures, key=lambda s: -s.words):
        sel = np.ones(len(cells), dtype=bool)
        for k, v in s.location.items():
            sel &= cells[k].to_numpy() == v
        sel &= cells["bit"].isin(s.bits).to_numpy()
        for idx in cells.loc[sel, "index"].unique():
            out[int(idx)] = KIND_ORDER.index(s.kind)
    return out


def viewer_data(log: FailLog, mapper: LinearMapper, cells: pd.DataFrame, *,
                faults: FaultSet | None = None, signatures: list[Signature] | None = None,
                bitmaps: dict[int, np.ndarray] | None = None, pattern: Pattern | None = None,
                max_points: int = 60_000, seed: int = 0) -> dict:
    g = mapper.geometry
    words = cells.groupby("index", as_index=False).agg(
        element=("element", "min"), **{lvl: (lvl, "first") for lvl in
                                        ("stack", "channel", "pseudo_channel", "sid", "bank_group", "bank", "row", "pos")})
    idx = words["index"].to_numpy(np.uint64)
    if faults is not None:
        kinds = _kinds_for(idx, faults)
    elif signatures:
        lut = _kinds_from_signatures(cells, signatures)
        kinds = np.array([lut.get(int(i), KIND_ORDER.index("unknown")) for i in idx], dtype=np.int8)
    else:
        kinds = np.full(len(idx), KIND_ORDER.index("unknown"), dtype=np.int8)
    shown = np.arange(len(idx))
    if len(idx) > max_points:  # stratified sample so small fault kinds stay visible
        rng = np.random.default_rng(seed)
        keep = []
        per_kind = max_points // max(len(np.unique(kinds)), 1)
        for k in np.unique(kinds):
            members = np.flatnonzero(kinds == k)
            keep.append(members if len(members) <= per_kind else rng.choice(members, per_kind, replace=False))
        shown = np.sort(np.concatenate(keep))
    w = words.iloc[shown]
    points = np.stack([w["stack"], w["channel"], w["pseudo_channel"], w["sid"], w["bank_group"], w["bank"],
                       w["row"], w["pos"], kinds[shown], w["element"]], axis=1).astype(np.int32)
    counts = {KIND_ORDER[k]: int((kinds == k).sum()) for k in np.unique(kinds)}
    bm_out = {}
    for bk, bm in (bitmaps or {}).items():
        packed = np.packbits(bm.ravel(), bitorder="little")
        bm_out[str(bk)] = {"n": int(bm.sum()), "b64": base64.b64encode(packed.tobytes()).decode()}
    sig_out = [{"kind": s.kind, "location": s.location, "bits": s.bits[:12], "words": s.words}
               for s in (signatures or []) if s.kind != "cell"]
    n_cells = sum(1 for s in (signatures or []) if s.kind == "cell")
    elements = [step.label() for step in pattern.steps] if pattern is not None else []
    return {
        "geometry": g.as_dict(), "describe": g.describe(), "mapper": mapper.describe(),
        "pattern": log.meta.pattern, "backend": log.meta.backend, "temperature_c": log.meta.temperature_c,
        "total_fails": log.meta.total_fails, "failing_words": int(len(idx)), "shown_words": int(len(shown)),
        "kinds": list(KIND_ORDER), "colors": KIND_COLOR, "counts": counts,
        "colour_source": "injected ground truth" if faults is not None else ("analyzer" if signatures else "none"),
        "synthetic": faults is not None, "signatures": sig_out, "isolated_cells": n_cells,
        "bitmaps": bm_out, "elements": elements, "points": points.tolist(),
    }


def write_viewer(path: "str | Path", data: dict) -> Path:
    html = _TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>hbmlens viewer</title>
<style>
  html,body{margin:0;height:100%;background:#0b0e13;color:#dfe5ee;font:12.5px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;overflow:hidden}
  .card{position:absolute;background:rgba(16,21,29,.9);border:1px solid #243041;border-radius:8px;padding:10px 12px}
  #panel{top:10px;left:10px;width:300px;max-height:calc(100% - 20px);overflow:auto}
  #zoom{top:10px;right:10px;width:300px;display:none}
  #tip{position:absolute;pointer-events:none;display:none;padding:5px 8px;border-radius:6px;background:#111822;border:1px solid #33445c;font-size:12px;white-space:nowrap}
  h1{margin:0 0 4px;font-size:14px;font-weight:600} h2{margin:10px 0 4px;font-size:12px;color:#aeb8c6;font-weight:600}
  .muted{color:#8a96a8}#meta{overflow-wrap:anywhere}.row{display:flex;align-items:center;gap:8px;margin:2px 0}
  .sw{width:10px;height:10px;border-radius:2px;display:inline-block}
  label{cursor:pointer} input[type=range]{width:100%}
  .num{font-variant-numeric:tabular-nums;margin-left:auto;color:#aeb8c6}
  .warn{margin-top:8px;padding:6px 8px;border-radius:6px;background:#2a2214;color:#f3cf7a;font-size:11.5px}
  canvas#bm{width:100%;image-rendering:pixelated;border:1px solid #243041;background:#05070a}
  button{background:#1d2a3a;color:#dfe5ee;border:1px solid #33445c;border-radius:5px;padding:2px 8px;cursor:pointer}
</style></head>
<body>
<div id="panel" class="card">
  <h1>hbmlens &middot; virtual HBM</h1>
  <div class="muted" id="meta"></div>
  <h2>failing words <span class="muted" id="src"></span></h2><div id="legend"></div>
  <div class="row"><label><input type="checkbox" id="sigs" checked> analyzer outlines</label></div>
  <h2>replay <span class="muted" id="elabel"></span></h2>
  <input id="elem" type="range" min="0" max="0" step="1" value="0">
  <h2>die spacing</h2><input id="gap" type="range" min="0.15" max="2.5" step="0.05" value="0.9">
  <h2>signatures</h2><div id="siglist" class="muted"></div>
  <div class="warn" id="warn"></div>
</div>
<div id="zoom" class="card"><div class="row"><b id="ztitle"></b><button id="zclose" style="margin-left:auto">close</button></div>
  <canvas id="bm"></canvas><div class="muted" id="znote"></div></div>
<div id="tip"></div>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.169.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.169.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
const D = __DATA__;
const g = D.geometry;
const cols = g.pseudo_channels * g.bank_groups, rowsT = g.sids * g.banks_per_group;
const T = 1.0, GAP = 0.18, dieW = cols * (T + GAP) + GAP, dieD = rowsT * (T + GAP) + GAP;
const stackPitch = dieW + 3.0, WPR = g.columns * g.words_per_column;
const $ = id => document.getElementById(id);

$("meta").innerHTML = `${D.describe}<br>pattern <b>${D.pattern}</b> on <b>${D.backend}</b>` +
  (D.temperature_c != null ? ` at ${D.temperature_c}&deg;C` : "") +
  `<br>${D.total_fails.toLocaleString()} failing reads, ${D.failing_words.toLocaleString()} words` +
  (D.shown_words < D.failing_words ? ` (${D.shown_words.toLocaleString()} drawn)` : "") +
  `<br><span class="muted">map: ${D.mapper}</span>`;
$("src").textContent = `(colour: ${D.colour_source})`;
$("warn").textContent = (D.synthetic ? "Synthetic faults injected into a virtual device. " : "") +
  "Layout is illustrative, not a real floorplan; the address map is an assumption.";

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x0b0e13);
const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.05, 500);
const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true;
scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const sun = new THREE.DirectionalLight(0xffffff, 1.1); sun.position.set(6, 12, 8); scene.add(sun);
const dieMat = new THREE.MeshStandardMaterial({ color: 0x1b2533, transparent: true, opacity: 0.55, roughness: 0.8 });
const baseMat = new THREE.MeshStandardMaterial({ color: 0x3a3f4a, roughness: 0.6 });
const tileMat = new THREE.MeshStandardMaterial({ color: 0x24344a, transparent: true, opacity: 0.8 });
const tsvMat = new THREE.MeshStandardMaterial({ color: 0xb08d57, metalness: 0.6, roughness: 0.4 });
let gap = 0.9;
const dieY = ch => 0.35 + (ch + 1) * gap;
const tileOrigin = (s, pc, sid, bg, bank) => {
  const c = pc * g.bank_groups + bg, r = sid * g.banks_per_group + bank;
  return [s * stackPitch + GAP + c * (T + GAP), GAP + r * (T + GAP)];
};
const bankKey = (s, ch, pc, sid, bg, bank) =>
  (((((s * g.channels + ch) * g.pseudo_channels + pc) * g.sids + sid) * g.bank_groups + bg) * g.banks_per_group + bank);

// ---- dies, tiles, TSVs -------------------------------------------------
const layered = [], tiles = [];
for (let s = 0; s < g.stacks; s++) {
  const x0 = s * stackPitch;
  const base = new THREE.Mesh(new THREE.BoxGeometry(dieW + 0.6, 0.3, dieD + 0.6), baseMat);
  base.position.set(x0 + dieW / 2, 0.15, dieD / 2); scene.add(base);
  for (let ch = 0; ch < g.channels; ch++) {
    const grp = new THREE.Group();
    const die = new THREE.Mesh(new THREE.BoxGeometry(dieW, 0.06, dieD), dieMat);
    die.position.set(dieW / 2, 0, dieD / 2); grp.add(die);
    for (let pc = 0; pc < g.pseudo_channels; pc++) for (let sid = 0; sid < g.sids; sid++)
      for (let bg = 0; bg < g.bank_groups; bg++) for (let bank = 0; bank < g.banks_per_group; bank++) {
        const [tx, tz] = tileOrigin(0, pc, sid, bg, bank);
        const t = new THREE.Mesh(new THREE.BoxGeometry(T, 0.02, T), tileMat);
        t.position.set(tx + T / 2, 0.04, tz + T / 2);
        t.userData = { s, ch, pc, sid, bg, bank, key: bankKey(s, ch, pc, sid, bg, bank) };
        grp.add(t); tiles.push(t);
      }
    grp.position.set(x0, dieY(ch), 0); grp.userData = { ch }; scene.add(grp); layered.push(grp);
  }
  for (const [tx, tz] of [[0.35, 0.35], [dieW - 0.35, 0.35], [0.35, dieD - 0.35], [dieW - 0.35, dieD - 0.35]]) {
    const tsv = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 1, 8), tsvMat);
    tsv.userData = { tsv: true, x0, tx, tz }; scene.add(tsv); layered.push(tsv);
  }
}

// ---- failing words as points, one Points object per kind, sorted by element --
const maxElem = D.points.reduce((m, p) => Math.max(m, p[9]), 0);
const groups = {};
const byKind = {};
for (const p of D.points) (byKind[p[8]] ||= []).push(p);
D.kinds.forEach((name, k) => {
  const list = byKind[k]; if (!list) return;
  list.sort((a, b) => a[9] - b[9]);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(list.length * 3), 3));
  const pts = new THREE.Points(geo, new THREE.PointsMaterial({ color: D.colors[name], size: 0.035 }));
  pts.userData = { list, name }; scene.add(pts); groups[name] = pts;
  const row = document.createElement("div"); row.className = "row";
  row.innerHTML = `<label><input type="checkbox" checked> <span class="sw" style="background:${D.colors[name]}"></span> ${name}</label><span class="num">${(D.counts[name] || 0).toLocaleString()}</span>`;
  row.querySelector("input").onchange = e => { pts.visible = e.target.checked; };
  $("legend").appendChild(row);
});

// ---- analyzer signature outlines ------------------------------------------
const sigGroup = new THREE.Group(); scene.add(sigGroup);
function lineBox(x0, z0, x1, z1, y, color) {
  const pts = [[x0, z0], [x1, z0], [x1, z1], [x0, z1], [x0, z0]].map(([x, z]) => new THREE.Vector3(x, y, z));
  return new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color }));
}
function buildSignatures() {
  sigGroup.clear();
  for (const s of D.signatures) {
    const L = s.location, col = D.colors[s.kind] || "#fff";
    const y = dieY(L.channel) + 0.12;
    if (s.kind === "dq-lane") {
      const [ax] = tileOrigin(L.stack, L.pseudo_channel, 0, 0, 0);
      const [bx] = tileOrigin(L.stack, L.pseudo_channel, 0, g.bank_groups - 1, 0);
      sigGroup.add(lineBox(ax - 0.06, GAP - 0.06, bx + T + 0.06, dieD - GAP + 0.06, y, col));
    } else {
      const [tx, tz] = tileOrigin(L.stack, L.pseudo_channel, L.sid, L.bank_group, L.bank);
      if (s.kind === "row") {
        const z = tz + (L.row + 0.5) / g.rows * T;
        sigGroup.add(lineBox(tx, z - 0.01, tx + T, z + 0.01, y, col));
      } else if (s.kind === "column") {
        const x = tx + (L.column * g.words_per_column) / WPR * T, w = g.words_per_column / WPR * T;
        sigGroup.add(lineBox(x - 0.01, tz, x + w + 0.01, tz + T, y, col));
      } else {
        sigGroup.add(lineBox(tx - 0.03, tz - 0.03, tx + T + 0.03, tz + T + 0.03, y, col));
      }
    }
  }
}
$("sigs").onchange = e => { sigGroup.visible = e.target.checked; };
$("siglist").innerHTML = D.signatures.length || D.isolated_cells
  ? D.signatures.map(s => `<div><span class="sw" style="background:${D.colors[s.kind]}"></span> ${s.kind} ` +
      Object.entries(s.location).map(([k, v]) => `${k.replace("pseudo_channel", "pc").replace("bank_group", "bg")}=${v}`).join(" ") +
      ` bit ${s.bits.join(",")} &middot; ${s.words.toLocaleString()} words</div>`).join("") +
    (D.isolated_cells ? `<div><span class="sw" style="background:${D.colors.cell}"></span> ${D.isolated_cells} isolated cells</div>` : "")
  : "run the analyzer to list signatures";

// ---- layout and replay -------------------------------------------------------
let upTo = maxElem;
function relayout() {
  const top = dieY(g.channels - 1);
  for (const o of layered) {
    if (o.userData.tsv) { o.scale.y = top; o.position.set(o.userData.x0 + o.userData.tx, top / 2 + 0.15, o.userData.tz); }
    else o.position.y = dieY(o.userData.ch);
  }
  for (const pts of Object.values(groups)) {
    const list = pts.userData.list, pos = pts.geometry.attributes.position.array;
    let n = 0;
    list.forEach((p, i) => {
      const [s, ch, pc, sid, bg, bank, row, x, , el] = p;
      const [tx, tz] = tileOrigin(s, pc, sid, bg, bank);
      pos[i * 3] = tx + (x + 0.5) / WPR * T; pos[i * 3 + 1] = dieY(ch) + 0.07; pos[i * 3 + 2] = tz + (row + 0.5) / g.rows * T;
      if (el <= upTo) n = i + 1;
    });
    pts.geometry.attributes.position.needsUpdate = true;
    pts.geometry.setDrawRange(0, n); pts.geometry.computeBoundingSphere();
  }
  buildSignatures();
}
$("gap").oninput = e => { gap = +e.target.value; relayout(); };
$("elem").max = maxElem; $("elem").value = maxElem;
const elemLabel = k => (D.elements[k] ? `step ${k}: ${D.elements[k]}` : `step ${k}`);
$("elabel").textContent = `up to ${elemLabel(maxElem)}`;
$("elem").oninput = e => { upTo = +e.target.value; $("elabel").textContent = `up to ${elemLabel(upTo)}`; relayout(); };

// ---- hover tooltip and bank zoom --------------------------------------------
const ray = new THREE.Raycaster(); ray.params.Points.threshold = 0.02;
const mouse = new THREE.Vector2();
function pick(ev, objects) {
  mouse.set(ev.clientX / innerWidth * 2 - 1, -(ev.clientY / innerHeight) * 2 + 1);
  ray.setFromCamera(mouse, camera);
  return ray.intersectObjects(objects, false);
}
renderer.domElement.addEventListener("pointermove", ev => {
  const hit = pick(ev, Object.values(groups).filter(p => p.visible))[0];
  const tip = $("tip");
  if (!hit) { tip.style.display = "none"; return; }
  const p = hit.object.userData.list[hit.index];
  const [s, ch, pc, sid, bg, bank, row, x, , el] = p;
  tip.innerHTML = `${hit.object.userData.name} &middot; stack ${s} ch ${ch} pc ${pc} sid ${sid} bg ${bg} bank ${bank}<br>` +
    `row ${row}, column ${Math.floor(x / g.words_per_column)}, word ${x % g.words_per_column} &middot; first seen ${elemLabel(el)}`;
  tip.style.left = ev.clientX + 14 + "px"; tip.style.top = ev.clientY + 12 + "px"; tip.style.display = "block";
});
renderer.domElement.addEventListener("click", ev => {
  const hits = pick(ev, tiles); if (!hits.length) return;
  // dies are translucent: prefer the first tile along the ray that has failures
  const hit = hits.find(h => D.bitmaps[String(h.object.userData.key)]) || hits[0];
  const u = hit.object.userData, bm = D.bitmaps[String(u.key)];
  $("zoom").style.display = "block";
  $("ztitle").textContent = `stack ${u.s} ch ${u.ch} pc ${u.pc} sid ${u.sid} bg ${u.bg} bank ${u.bank}`;
  const cv = $("bm"); cv.width = WPR; cv.height = g.rows;
  const ctx = cv.getContext("2d"), img = ctx.createImageData(WPR, g.rows);
  if (bm) {
    const bytes = Uint8Array.from(atob(bm.b64), c => c.charCodeAt(0));
    for (let i = 0; i < WPR * g.rows; i++) {
      const on = (bytes[i >> 3] >> (i & 7)) & 1, o = i * 4;
      img.data[o] = on ? 255 : 12; img.data[o + 1] = on ? 93 : 16; img.data[o + 2] = on ? 115 : 22; img.data[o + 3] = 255;
    }
    $("znote").textContent = `${bm.n.toLocaleString()} failing words; x = word position in row (${WPR}), y = row (${g.rows})`;
  } else {
    for (let i = 0; i < WPR * g.rows; i++) { const o = i * 4; img.data[o] = 12; img.data[o + 1] = 16; img.data[o + 2] = 22; img.data[o + 3] = 255; }
    $("znote").textContent = "no failing words recorded in this bank (or not among the banks exported)";
  }
  ctx.putImageData(img, 0, 0);
});
$("zclose").onclick = () => { $("zoom").style.display = "none"; };

addEventListener("resize", () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });
relayout();
const cx = ((g.stacks - 1) * stackPitch + dieW) / 2;
controls.target.set(cx, dieY(g.channels / 2), dieD / 2);
camera.position.set(cx + dieW * 1.4, dieY(g.channels) + 4.0, dieD * 2.6);
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
</script></body></html>
"""
