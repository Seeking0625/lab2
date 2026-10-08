/* SAM 遥感标注修正 UI —— 前端交互
 * 模式：select 选择/改类 | merge 多选合并 | resegment 点选调SAM重分割 | box 框选调SAM新增
 */
"use strict";

const $ = (s) => document.querySelector(s);
const cv = $("#cv"), ctx = cv.getContext("2d");
const UNLABELED_COLOR = "#9e9e9e";
async function api(url, options) {
  const r = await fetch(url, options);
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || `请求失败 (${r.status})`);
  return data;
}
const post = (url, data) => api(url, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(data)});
window.addEventListener("unhandledrejection", e => { hideBusy(); alert(e.reason?.message || String(e.reason)); e.preventDefault(); });
const escapeHtml = s => String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

const state = {
  catalog: [], prompts: null,
  images: [], imgIdx: -1, imageName: "",
  img: null, ann: null,
  classes: [], activeCls: 0,
  view: { scale: 1, ox: 0, oy: 0 },
  mode: "select",
  selected: new Set(),   // 实例 id 集合（merge 多选 / select 单选）
  hoverId: null,
  pending: null,         // {cands, replaceId} SAM 候选待确认
  undoStack: [],
  dirty: false,
  drag: null,            // {type:'pan'|'box', sx, sy, ...}
  spaceDown: false,
};

/* ---------------- 工具 ---------------- */
const img2scr = (x, y) => [x * state.view.scale + state.view.ox, y * state.view.scale + state.view.oy];
const scr2img = (x, y) => [(x - state.view.ox) / state.view.scale, (y - state.view.oy) / state.view.scale];

function clsColor(name) {
  const c = state.classes.find((c) => c.name === name);
  return c ? c.color : UNLABELED_COLOR;
}

function pushUndo() {
  state.undoStack.push(JSON.stringify(state.ann));
  if (state.undoStack.length > 30) state.undoStack.shift();
}

function markDirty() {
  state.dirty = true;
  $("#dirty").style.visibility = "visible";
  renumber();
  render(); refreshInstList(); refreshSelInfo(); refreshStats();
}

function renumber() {
  state.ann.instances.forEach((inst, i) => { inst.id = i; });
}

function pointInRing(px, py, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i][0], yi = ring[i][1], xj = ring[j][0], yj = ring[j][1];
    if ((yi > py) !== (yj > py) && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi)
      inside = !inside;
  }
  return inside;
}

function visibleInstances() {
  if (!state.ann) return [];
  const filter = $("#instanceFilter").value;
  const minArea = Math.max(0, Number($("#minVisibleArea").value) || 0);
  return state.ann.instances.filter(i => i.area >= minArea &&
    (filter === "all" || (filter === "todo" ? i.class === "unlabeled" : i.class !== "unlabeled")));
}
function hitInstance(px, py) {
  const visible = visibleInstances();
  for (let i = visible.length - 1; i >= 0; i--) {
    const inst = visible[i];
    if ((inst.polygons || []).some((r) => pointInRing(px, py, r)) &&
        !(inst.holes || []).some(r => pointInRing(px, py, r))) return inst;
  }
  return null;
}

function bboxOf(inst) { return inst.bbox; } // [x,y,w,h]

/* ---------------- 渲染 ---------------- */
function resizeCanvas() {
  const wrap = $("#canvasWrap");
  cv.width = wrap.clientWidth * devicePixelRatio;
  cv.height = wrap.clientHeight * devicePixelRatio;
  render();
}

function render() {
  if (!state.ann) return;
  const { scale, ox, oy } = state.view;
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.fillStyle = "#141519";
  ctx.fillRect(0, 0, cv.width, cv.height);
  if (state.img) {
    ctx.setTransform(devicePixelRatio * scale, 0, 0, devicePixelRatio * scale,
                     devicePixelRatio * ox, devicePixelRatio * oy);
    ctx.imageSmoothingEnabled = scale < 2;
    ctx.drawImage(state.img, 0, 0);
  }
  const lw = 1.6 / scale;
  for (const inst of $("#showMasks").checked ? visibleInstances() : []) {
    const sel = state.selected.has(inst.id);
    const hov = state.hoverId === inst.id;
    const color = clsColor(inst.class);
    ctx.lineWidth = sel ? lw * 2.2 : lw;
    ctx.strokeStyle = sel ? "#ffffff" : hov ? "#ffdd57" : color;
    ctx.setLineDash(sel ? [6 / scale, 4 / scale] : []);
    ctx.beginPath();
    for (const ring of [...(inst.polygons || []), ...(inst.holes || [])]) {
      ring.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
      ctx.closePath();
    }
      ctx.globalAlpha = 0.30;
      ctx.fillStyle = color;
      ctx.fill("evenodd");
      ctx.globalAlpha = 1;
      ctx.stroke();
  }
  ctx.setLineDash([]);
  // SAM 候选预览
  if (state.pending) {
    state.pending.cands.forEach((c, i) => {
      ctx.lineWidth = lw * 2;
      ctx.strokeStyle = ["#00e5ff", "#ff9100", "#76ff03"][i % 3];
      for (const ring of c.polygons || []) {
        ctx.beginPath();
        ring.forEach((p, j) => j ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
        ctx.closePath();
        ctx.stroke();
        const [sx, sy] = img2scr(ring[0][0], ring[0][1]);
        ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
        ctx.fillStyle = ctx.strokeStyle;
        ctx.font = "bold 14px sans-serif";
        ctx.fillText(`${i + 1}`, sx, sy - 6);
        ctx.setTransform(devicePixelRatio * scale, 0, 0, devicePixelRatio * scale,
                         devicePixelRatio * ox, devicePixelRatio * oy);
      }
    });
    ctx.setLineDash([]);
  }
  // Positive/negative prompt markers remain visible while comparing candidates.
  if (state.prompts) {
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    for (const [x,y,label] of state.prompts.points) {
      const [sx,sy] = img2scr(x,y);
      ctx.beginPath(); ctx.arc(sx,sy,5,0,Math.PI*2);
      ctx.fillStyle = label ? "#76ff03" : "#ff5252"; ctx.fill();
    }
    ctx.setTransform(devicePixelRatio*scale,0,0,devicePixelRatio*scale,devicePixelRatio*ox,devicePixelRatio*oy);
  }
  // 框选拖拽预览
  if (state.drag && state.drag.type === "box") {
    const d = state.drag;
    ctx.lineWidth = lw * 1.5;
    ctx.strokeStyle = "#00e5ff";
    ctx.strokeRect(d.x0, d.y0, d.x1 - d.x0, d.y1 - d.y0);
  }
  $("#hud").textContent = state.imageName +
    `  |  缩放 ${(scale * 100).toFixed(0)}%  |  实例 ${state.ann.instances.length}`;
}

/* ---------------- 侧栏刷新 ---------------- */
function refreshClassList() {
  $("#clsCount").textContent = state.classes.length;
  $("#clsList").innerHTML = "";
  state.classes.forEach((c, i) => {
    const row = document.createElement("div");
    row.className = "cls-row" + (i === state.activeCls ? " active" : "");
    row.innerHTML = `<span class="dot" style="background:${c.color}"></span><span>${i + 1}. ${escapeHtml(c.name)}</span>`;
    row.onclick = () => { state.activeCls = i; applyActiveClass(); refreshClassList(); };
    $("#clsList").appendChild(row);
  });
}

function applyActiveClass() {
  if (!state.ann || state.selected.size === 0) return;
  pushUndo();
  const name = state.classes[state.activeCls].name;
  for (const id of state.selected) { state.ann.instances[id].class = name; state.ann.instances[id].label_source = "manual"; }
  markDirty();
}

function refreshSelInfo() {
  const box = $("#selInfo");
  if (state.selected.size === 0) {
    box.innerHTML = `<div class="helpline">未选中实例。点击画布中的彩色区域选择。</div>`;
    return;
  }
  if (state.mode === "merge" || state.selected.size > 1) {
    box.innerHTML = `<div class="helpline">已选 ${state.selected.size} 个实例。点击左侧类别可批量标注；Shift+点击可增减选择。</div>`;
    const btn = document.createElement("button");
    btn.className = "primary"; btn.style.margin = "8px 12px";
    btn.textContent = `合并所选 (${state.selected.size})`;
    btn.disabled = state.selected.size < 2;
    btn.onclick = mergeSelected;
    box.appendChild(btn);
    return;
  }
  const inst = state.ann.instances[[...state.selected][0]];
  if (!inst) { state.selected.clear(); return refreshSelInfo(); }
  box.innerHTML = `
    <div class="info-row"><span>ID</span><b>#${inst.id}</b></div>
    <div class="info-row"><span>面积(px)</span><b>${inst.area.toLocaleString()}</b></div>
    <div class="info-row"><span>SAM 分数</span><b>${inst.score}</b></div>
    <div class="info-row"><span>分类置信分</span><b>${inst.class_score == null ? "—" : (inst.class_score * 100).toFixed(1) + "%"}</b></div>
    <div style="padding:4px 12px">${(inst.class_candidates || []).map(c => escapeHtml(c.class) + " " + (c.score * 100).toFixed(0) + "%").join("；")}</div>
    <div class="info-row"><span>顶点</span><b>${(inst.polygons || []).reduce((s, r) => s + r.length, 0)}</b></div>
    <div style="padding:4px 12px">类别（下拉或按数字键）</div>
    <div style="padding:0 12px"><select id="instClass">
      <option value="">unlabeled</option>
      ${state.classes.map((c) => `<option value="${escapeHtml(c.name)}" ${c.name === inst.class ? "selected" : ""}>${escapeHtml(c.name)}</option>`).join("")}
    </select></div>
    <div style="padding:8px 12px"><button id="btnDel">删除实例 (Del)</button></div>`;
  $("#instClass").onchange = (e) => {
    pushUndo();
    inst.class = e.target.value || "unlabeled";
    inst.label_source = "manual";
    markDirty();
  };
  $("#btnDel").onclick = deleteSelected;
}

function refreshInstList() {
  const list = $("#instList");
  list.innerHTML = "";
  $("#instCount").textContent = state.ann ? `${visibleInstances().length}/${state.ann.instances.length}` : 0;
  if (!state.ann) return;
  const frag = document.createDocumentFragment();
  visibleInstances().sort((a,b) => b.area-a.area).forEach((inst) => {
    const row = document.createElement("div");
    row.className = "inst-row";
    const unl = inst.class === "unlabeled";
    row.innerHTML = `<span class="dot" style="background:${clsColor(inst.class)};display:inline-block;vertical-align:-2px;margin-right:6px"></span>#${inst.id} ${unl ? '<span style="color:#777">未标注</span>' : escapeHtml(inst.class) + (inst.label_source === "model" ? "（预测）" : "")}`;
    row.onclick = () => {
      state.selected = new Set([inst.id]);
      centerOn(inst);
      render(); refreshSelInfo(); refreshInstList();
    };
    frag.appendChild(row);
  });
  list.appendChild(frag);
}

function refreshStats() {
  const n = state.ann.instances.length;
  const labeled = state.ann.instances.filter((i) => i.class !== "unlabeled").length;
  const predicted = state.ann.instances.filter(i => i.class !== "unlabeled" && i.label_source === "model").length;
  const byCls = {};
  state.ann.instances.forEach((i) => {
    if (i.class !== "unlabeled") byCls[i.class] = (byCls[i.class] || 0) + 1;
  });
  $("#stats").innerHTML =
    `共 ${n} 实例 / 人工确认 ${labeled-predicted} / 预测待确认 ${predicted}<br>` +
    Object.entries(byCls).map(([k, v]) => `${escapeHtml(k)}: ${v}`).join("　");
}

function centerOn(inst) {
  const [x, y, w, h] = bboxOf(inst);
  const wrap = $("#canvasWrap");
  state.view.scale = Math.min(wrap.clientWidth / w, wrap.clientHeight / h, 8) * 0.9;
  state.view.ox = wrap.clientWidth / 2 - (x + w / 2) * state.view.scale;
  state.view.oy = wrap.clientHeight / 2 - (y + h / 2) * state.view.scale;
}

/* ---------------- 编辑操作 ---------------- */
function deleteSelected() {
  if (!state.selected.size) return;
  pushUndo();
  state.ann.instances = state.ann.instances.filter((i) => !state.selected.has(i.id));
  state.selected.clear();
  markDirty();
}

async function mergeSelected() {
  if (state.selected.size < 2) return;
  const picked = state.ann.instances.filter(i => state.selected.has(i.id));
  showBusy("合并掩码…");
  try {
    const merged = await post("/api/edit/merge", {image:state.imageName, instances:picked});
    pushUndo();
    state.ann.instances = state.ann.instances.filter(i => !state.selected.has(i.id));
    state.ann.instances.push(merged);
    state.selected.clear();
    markDirty(); setMode("select");
  } finally { hideBusy(); }
}

function undo() {
  if (!state.undoStack.length) return;
  state.ann = JSON.parse(state.undoStack.pop());
  state.selected.clear();
  state.dirty = true;
  $("#dirty").style.visibility = "visible";
  renumber(); render(); refreshInstList(); refreshSelInfo(); refreshStats();
}

/* ---------------- SAM 交互 ---------------- */
async function samPoint(imgPt, replaceId) {
  const inst = replaceId != null ? state.ann.instances[replaceId] : null;
  const bbox = inst
    ? inst.bbox
    : [imgPt[0] - 512, imgPt[1] - 512, 1024, 1024];
  showBusy("SAM 推理中…");
  try {
    const r = await fetch("/api/sam/point", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: state.imageName, points: [[Math.round(imgPt[0]), Math.round(imgPt[1]), 1]],
        bbox: bbox.map(Math.round),
      }),
    });
    const cands = await r.json();
    if (!r.ok) throw new Error(cands.error || "SAM 推理失败");
    if (!cands.length) return alert("SAM 未返回有效掩码，请换位置点击");
    pushUndo();
    state.pending = { cands, replaceId };
    showCands(cands);
  } finally { hideBusy(); }
}

async function splitAuto(imgPt, replaceId) {
  const old = state.ann.instances[replaceId];
  showBusy("正在拆分实例…");
  pushUndo();
  try {
    const r = await post("/api/edit/split-auto", {
      image: state.imageName,
      original: old,
      point: [Math.round(imgPt[0]), Math.round(imgPt[1])],
    });
    state.ann.instances.splice(replaceId, 1, ...r.parts);
    state.pending = null; state.prompts = null;
    state.dirty = true; markDirty();
    render(); refreshSelInfo(); refreshInstList();
  } catch (e) {
    state.undoStack.pop();
    alert(e.message || "拆分失败");
  } finally { hideBusy(); }
}

async function samBox(box) {
  showBusy("SAM 推理中…");
  try {
    const r = await fetch("/api/sam/box", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: state.imageName, box: box.map(Math.round) }),
    });
    const cands = await r.json();
    if (!r.ok) throw new Error(cands.error || "SAM 推理失败");
    if (!cands.length) return alert("SAM 未返回有效掩码，请框选更明显的目标");
    pushUndo();
    state.pending = { cands, replaceId: null };
    showCands(cands);
  } finally { hideBusy(); }
}

function showCands(cands) {
  render();
  const box = $("#selInfo");
  box.innerHTML = `<div class="helpline">SAM 返回 ${cands.length} 个候选掩码。按 <span class="kbd">1</span>/<span class="kbd">2</span>/<span class="kbd">3</span> 或点击卡片采用，<span class="kbd">Esc</span> 放弃。</div>`;
  cands.forEach((c, i) => {
    const card = document.createElement("div");
    card.className = "cand-card";
    card.innerHTML = `<b>候选 ${i + 1}</b>（分数 ${c.score}，面积 ${c.area.toLocaleString()}）`;
    card.onclick = () => applyCandidate(i);
    box.appendChild(card);
  });
}

async function applyCandidate(i) {
  if (!state.pending) return;
  const { cands, replaceId } = state.pending;
  const c = cands[i];
  if (state.pending.kind === "refine") {
    const old = state.ann.instances[replaceId];
    c.class = old.class;
    if (old.label_source) c.label_source = old.label_source;
    state.ann.instances[replaceId] = c;
  } else if (replaceId != null) {
    const old = state.ann.instances[replaceId];
    c.class = old && old.class !== "unlabeled" ? old.class : "unlabeled";
    showBusy("拆分实例并保留剩余区域…");
    try {
      const parts = await post("/api/edit/split", {image:state.imageName, original:old, candidate:c});
      state.ann.instances.splice(replaceId, 1, ...parts);
      if (parts.length === 1) console.info("候选覆盖了整个实例，已执行替换而非拆分");
    } finally { hideBusy(); }
  } else {
    c.class = state.classes[state.activeCls].name;
    c.label_source = "manual";
    state.ann.instances.push(c);
  }
  state.pending = null; state.prompts = null;
  state.dirty = true;
  markDirty();
  setMode(replaceId != null ? "select" : state.mode);
}

function cancelPending() {
  if (!state.pending) return;
  state.pending = null; state.prompts = null;
  state.undoStack.pop(); // 撤销时丢弃推送的快照
  render(); refreshSelInfo();
}

/* ---------------- 模式与提示 ---------------- */
const HINTS = {
  select: "左键点选实例；数字键 1-9 改类别；按住空格拖动平移，滚轮缩放",
  pan: "按住左键拖动平移图片，滚轮缩放；切回「选择/改类」编辑实例",
  merge: "依次点击需要合并的实例（可多点几个），右侧「合并所选」",
  resegment: "点击实例内部进行拆分，采用候选后保留剩余区域",
  refine: "先左键点选要修边的实例，再左键添加内部点、右键排除背景；最后采用候选",
  crop: "拖出一个目标清楚的局部区域，保存为独立图片后再标注",
  box: "按住左键拖出矩形框住新目标：SAM 分割后作为新实例（用当前激活类别）",
};
function setMode(m) {
  if (state.pending && m !== state.mode) cancelPending();
  state.prompts = null;
  state.mode = m;
  updateCanvasCursor();
  if (m !== "merge") state.selected.clear();
  document.querySelectorAll("button.mode").forEach((b) =>
    b.classList.toggle("active", b.dataset.mode === m));
  const hint = $("#hint");
  hint.textContent = HINTS[m];
  hint.style.display = "block";
  refreshSelInfo();
  render();
}

/* ---------------- 加载 ---------------- */
async function loadImages(preferred = null) {
  state.catalog = await api("/api/image-catalog");
  let group = $("#imageGroup").value;
  if (!state.catalog.some(i => group === "all" || i.kind === group)) {
    group = "original"; $("#imageGroup").value = group;
  }
  const requested = preferred || new URLSearchParams(location.search).get("img");
  const requestedItem = requested && state.catalog.find(i => i.name.replace(/\.[^.]+$/, "") === requested.replace(/\.[^.]+$/, ""));
  if (requestedItem && requestedItem.kind !== group && group !== "all") {
    group = requestedItem.kind; $("#imageGroup").value = group;
  }
  const items = state.catalog.filter(i => group === "all" || i.kind === group);
  state.images = items.map(i => i.name);
  $("#imgCount").textContent = `(${items.length}/${state.catalog.length})`;
  const list = $("#imgList"); list.innerHTML = "";
  items.forEach((item,i) => {
    const d = document.createElement("div");
    d.textContent = item.name;
    d.title = `${item.source ? "来源 "+item.source+"；" : ""}${item.width || "?"}×${item.height || "?"}；${item.count} 个候选`;
    d.onclick = () => loadImage(i); list.appendChild(d);
  });
  if (items.length) await loadImage(Math.max(state.images.findIndex(n => n === requestedItem?.name),0));
  else $("#selInfo").textContent = state.catalog.length ? "该分组暂无图片，请切换其他分组。" : "暂无图片，请把图片放入 dataset/images，或配置 DOTA 数据目录后重启服务。";
  window.__state = state;
}

async function loadImage(i) {
  if (i < 0 || i >= state.images.length) return;
  if (state.dirty && !confirm("当前修改尚未保存，确定切换图片并放弃修改？")) return;
  state.imgIdx = i;
  state.imageName = state.images[i].replace(/\.[^.]+$/, "");
  state.selected.clear(); state.pending = null; state.prompts = null; state.undoStack = [];
  state.dirty = false; $("#dirty").style.visibility = "hidden";
  document.querySelectorAll("#imgList div").forEach((d, k) =>
    d.classList.toggle("active", k === i));

  showBusy(`加载 ${state.imageName} …`);
  try {
    const img = new Image();
    await new Promise((res, rej) => {
      img.onload = res;
      img.onerror = () => rej(new Error("图像加载失败: " + img.src));
      img.src = `/api/image/${state.images[i]}`;
      if (img.complete && img.naturalWidth) res();   // 命中缓存时 onload 可能已错过
    });
    state.img = img;

    let ann = null;
    const r = await fetch(`/api/annotation/${state.imageName}`);
    if (r.ok) ann = await r.json();
    if (ann) normalizeAnn(ann);
    if (!ann) {
      ann = { image: state.images[i], width: img.naturalWidth, height: img.naturalHeight, instances: [] };
      $("#selInfo").innerHTML = `<div class="helpline">该图还没有分割结果。</div>`;
      const btn = document.createElement("button");
      btn.className = "primary"; btn.style.margin = "8px 12px";
      btn.textContent = "运行 SAM 自动分割（约1分钟）";
      btn.onclick = () => runAutoSegment();
      $("#selInfo").appendChild(btn);
    }
    state.ann = ann;
    fitView(); render(); refreshInstList(); refreshSelInfo(); refreshStats();
  } finally { hideBusy(); }
}

async function runAutoSegment() {
  showBusy("SAM 自动分割中，大图约需 0.5~2 分钟…");
  try {
    const r = await fetch(`/api/segment/${state.imageName}`, {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({profile:$("#segmentProfile").value})});
    const result = await r.json();
    if (!r.ok) throw new Error(result.error || "自动分割失败");
    pushUndo(); state.ann = result;
    markDirty();
    state.dirty = false; $("#dirty").style.visibility = "hidden";
  } finally { hideBusy(); }
}

function fitView() {
  if (!state.img) return;
  const wrap = $("#canvasWrap");
  const s = Math.min(wrap.clientWidth / state.img.naturalWidth,
                     wrap.clientHeight / state.img.naturalHeight);
  state.view.scale = s;
  state.view.ox = (wrap.clientWidth - state.img.naturalWidth * s) / 2;
  state.view.oy = (wrap.clientHeight - state.img.naturalHeight * s) / 2;
  render();
}

function showBusy(msg) { $("#busy").style.display = "flex"; $("#busyMsg").textContent = msg; }
function hideBusy() { $("#busy").style.display = "none"; }

function normalizeAnn(ann) {
  // 兼容阶段一生成的旧格式：单环 polygon -> polygons 多环
  for (const inst of ann.instances || []) {
    if (!inst.polygons && inst.polygon) inst.polygons = [inst.polygon];
  }
}

/* ---------------- 保存 ---------------- */
async function save() {
  if (!state.ann) return;
  showBusy("保存中…");
  try {
    const r = await fetch(`/api/annotation/${state.imageName}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state.ann),
    });
    if (!r.ok) throw new Error((await r.json()).error || "保存失败");
    state.dirty = false;
    $("#dirty").style.visibility = "hidden";
  } finally { hideBusy(); }
}

/* ---------------- 事件绑定 ---------------- */
function updateCanvasCursor() {
  cv.style.cursor = state.drag?.type === "pan" ? "grabbing" :
    (state.mode === "pan" || state.spaceDown ? "grab" : "default");
}
cv.addEventListener("wheel", (e) => {
  e.preventDefault();
  const f = e.deltaY < 0 ? 1.12 : 1 / 1.12;
  const [mx, my] = [e.offsetX, e.offsetY];
  const [ix, iy] = scr2img(mx, my);
  state.view.scale = Math.min(Math.max(state.view.scale * f, 0.02), 40);
  state.view.ox = mx - ix * state.view.scale;
  state.view.oy = my - iy * state.view.scale;
  render();
}, { passive: false });

cv.addEventListener("mousedown", (e) => {
  if (e.button === 1 || (e.button === 0 && (state.spaceDown || state.mode === "pan"))) {
    state.drag = { type: "pan", sx: e.offsetX, sy: e.offsetY, ox: state.view.ox, oy: state.view.oy };
    e.preventDefault();
    updateCanvasCursor();
    return;
  }
  if (!state.ann) return;
  const [ix, iy] = scr2img(e.offsetX, e.offsetY);

  if (state.mode === "refine" && (e.button === 0 || e.button === 2)) {
    e.preventDefault(); return refinePoint([ix,iy], e.button === 2 ? 0 : 1);
  }
  if (e.button !== 0) return;
  if (state.pending) {  // 候选确认期：点候选
    for (let k = 0; k < state.pending.cands.length; k++) {
      if ((state.pending.cands[k].polygons || []).some((r) => pointInRing(ix, iy, r)))
        return applyCandidate(k);
    }
    return cancelPending();
  }
  if (state.mode === "box" || state.mode === "crop") {
    const [x0, y0] = scr2img(e.offsetX, e.offsetY);
    state.drag = { type: "box", x0, y0, x1: x0, y1: y0 };
    return;
  }
  const inst = hitInstance(ix, iy);
  if (state.mode === "merge") {
    if (inst) {
      state.selected.has(inst.id) ? state.selected.delete(inst.id) : state.selected.add(inst.id);
      render(); refreshSelInfo();
    }
    return;
  }
  // select / resegment
  if (state.mode === "resegment" && inst) return splitAuto([ix, iy], inst.id);
  if (e.shiftKey && inst) {
    state.selected.has(inst.id) ? state.selected.delete(inst.id) : state.selected.add(inst.id);
  } else state.selected = new Set(inst ? [inst.id] : []);
  render(); refreshSelInfo(); refreshInstList();
});

window.addEventListener("mousemove", (e) => {
  if (!state.drag) return;
  const rect = cv.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  if (state.drag.type === "pan") {
    state.view.ox = state.drag.ox + (mx - state.drag.sx);
    state.view.oy = state.drag.oy + (my - state.drag.sy);
    render();
  } else if (state.drag.type === "box") {
    const [ix, iy] = scr2img(mx, my);
    state.drag.x1 = ix; state.drag.y1 = iy;
    render();
  }
});

window.addEventListener("mouseup", (e) => {
  if (!state.drag) return;
  const d = state.drag;
  state.drag = null;
  updateCanvasCursor();
  if (d.type === "box" && Math.abs(d.x1 - d.x0) > 8 && Math.abs(d.y1 - d.y0) > 8) {
    const box = [Math.min(d.x0,d.x1),Math.min(d.y0,d.y1),Math.abs(d.x1-d.x0),Math.abs(d.y1-d.y0)];
    if (state.mode === "crop") cropRegion(box); else samBox(box);
  }
});

cv.addEventListener("mousemove", (e) => {
  if (state.drag || !state.ann || state.pending || state.mode === "pan" || state.spaceDown) return;
  const [ix, iy] = scr2img(e.offsetX, e.offsetY);
  const inst = hitInstance(ix, iy);
  const h = inst ? inst.id : null;
  if (h !== state.hoverId) { state.hoverId = h; render(); }
});

cv.addEventListener("contextmenu", (e) => e.preventDefault());

window.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) {
    state.spaceDown = true; e.preventDefault(); updateCanvasCursor();
  }
  if ((e.ctrlKey || e.metaKey) && e.key === "s") { e.preventDefault(); return save(); }
  if ((e.ctrlKey || e.metaKey) && e.key === "z") { e.preventDefault(); return undo(); }
  if (e.key === "Escape") {
    if (state.pending) return cancelPending();
    state.selected.clear(); render(); refreshSelInfo();
    return;
  }
  if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
  if (e.key === "Delete" || e.key === "Backspace") {
    return deleteSelected();
  }
  if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
  if (e.key === "Enter" && state.mode === "merge") return mergeSelected();
  if (e.key === "Enter" && state.pending) return applyCandidate(0);
  if (e.key.toLowerCase() === "n") return nextUnlabeled();
  if (/^[1-9]$/.test(e.key)) {
    const k = +e.key - 1;
    if (state.pending) {  // 候选确认期：数字键选用候选掩码
      if (k < state.pending.cands.length) applyCandidate(k);
      return;
    }
    if (k < state.classes.length) { state.activeCls = k; applyActiveClass(); refreshClassList(); }
  }
});

window.addEventListener("keyup", (e) => {
  if (e.code === "Space") { state.spaceDown = false; updateCanvasCursor(); }
});
window.addEventListener("blur", () => {
  state.spaceDown = false; state.drag = null; updateCanvasCursor(); render();
});

document.querySelectorAll("button.mode").forEach((b) =>
  b.addEventListener("click", () => setMode(b.dataset.mode)));
$("#btnUndo").onclick = undo;
$("#btnSave").onclick = save;
$("#btnFit").onclick = fitView;
$("#btnPrev").onclick = () => loadImage(state.imgIdx - 1);
$("#btnNext").onclick = () => loadImage(state.imgIdx + 1);
$("#btnAddCls").onclick = addClass;
$("#newClsName").addEventListener("keydown", (e) => { if (e.key === "Enter") addClass(); });

async function addClass() {
  const name = $("#newClsName").value.trim();
  if (!name) return;
  const r = await fetch("/api/classes", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (r.ok) {
    state.classes = await r.json();
    $("#newClsName").value = "";
    refreshClassList();
  } else {
    alert((await r.json()).error || "添加失败");
  }
}

window.addEventListener("resize", resizeCanvas);

/* ---------------- 报告 ---------------- */
let lastReport = null;

function mdToHtml(md) {
  const lines = escapeHtml(md).split("\n");
  let html = "", inTable = false;
  const closeTable = () => { if (inTable) { html += "</tbody></table>"; inTable = false; } };
  for (const ln of lines) {
    if (/^\|/.test(ln)) {
      const cells = ln.split("|").slice(1, -1).map((s) => s.trim());
      if (/^[-\s|:]+$/.test(ln)) continue;  // 分隔行
      if (!inTable) { html += "<table><thead><tr>" +
        cells.map((c) => `<th>${c}</th>`).join("") + "</tr></thead><tbody>";
        inTable = true; continue; }
      html += "<tr>" + cells.map((c) => `<td>${c}</td>`).join("") + "</tr>";
      continue;
    }
    closeTable();
    if (/^# /.test(ln)) html += `<h1>${ln.slice(2)}</h1>`;
    else if (/^## /.test(ln)) html += `<h2>${ln.slice(3)}</h2>`;
    else if (/^- /.test(ln)) html += `<p>• ${ln.slice(2)}</p>`;
    else if (ln.trim()) html += `<p>${ln}</p>`;
  }
  closeTable();
  return html;
}

async function showReport() {
  if (!state.ann) return;
  if (state.dirty) {
    if (!confirm("有未保存修改，报告将基于磁盘上的标注生成。先保存？\n确定=保存并生成，取消=直接生成")) {
      // 不保存直接生成
    } else { await save(); }
  }
  showBusy("生成报告中…");
  try {
    const r = await fetch(`/api/report/${state.imageName}`);
    if (!r.ok) return alert("报告生成失败：标注不存在，请先运行分割");
    lastReport = await r.json();
    $("#reportBody").innerHTML = mdToHtml(lastReport.markdown);
    $("#reportModal").style.display = "block";
  } finally { hideBusy(); }
}

$("#btnReport").onclick = showReport;
$("#btnCloseReport").onclick = () => { $("#reportModal").style.display = "none"; };
$("#btnDlReport").onclick = () => {
  if (!lastReport) return;
  const blob = new Blob([lastReport.markdown], { type: "text/markdown" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `report_${state.imageName}.md`;
  a.click();
  URL.revokeObjectURL(a.href);
};

/* ---------------- 启动 ---------------- */
(async function init() {
  state.classes = await (await fetch("/api/classes")).json();
  refreshClassList();
  resizeCanvas();
  await loadImages();
  setMode("select");
})();

/* ---------------- 分类、尺度与视频实验 ---------------- */
$("#btnAuto").onclick = async () => {
  if (!state.ann) return;
  if (state.ann.instances.length && !confirm("重新分割会替换当前实例。是否继续？")) return;
  await runAutoSegment();
};
$("#btnConfirmLabels").onclick = () => {
  if (!state.ann || !state.selected.size) return;
  pushUndo();
  for (const id of state.selected) {
    const inst = state.ann.instances[id];
    if (inst.class !== "unlabeled") inst.label_source = "manual";
  }
  markDirty();
};
$("#btnClassify").onclick = async () => {
  if (!state.ann) return;
  showBusy("区域自动分类…");
  try {
    const ann = await post(`/api/classify/${state.imageName}`, state.ann);
    pushUndo(); state.ann = ann; markDirty();
  } finally { hideBusy(); }
};
async function experiment(kind) {
  if (state.dirty) await save();
  const data = {kind, image:state.imageName, video:$("#videoSelect").value, stride:Number($("#videoStride").value)};
  if (kind === "train") data.image_names = state.images;
  const task = await post("/api/tasks", data);
  const status = $("#taskStatus");
  for (const id of ["#btnTrain", "#btnScale", "#btnVideo"]) $(id).disabled = true;
  try {
    while (true) {
      const job = await api(`/api/tasks/${task.id}`);
      status.textContent = `${kind}：${job.status === "queued" ? "等待运行" : "正在运行，可继续查看图片"}`;
      if (job.status === "failed") throw new Error(job.error);
      if (job.status === "done") {
        status.textContent = "完成。";
        const a = document.createElement("a");
        a.href = "/api/artifacts/" + job.report;
        a.textContent = "查看实验报告"; a.target = "_blank"; a.style.color = "#8db9ff";
        status.appendChild(a);
        if (kind === "video") {
          if (state.dirty) await save();
          await loadImages();
        }
        return;
      }
      await new Promise(resolve => setTimeout(resolve, 1500));
    }
  } catch(e) { status.textContent = e.message; throw e; }
  finally { for (const id of ["#btnTrain", "#btnScale", "#btnVideo"]) $(id).disabled = false; }
}
$("#btnTrain").onclick = () => experiment("train");
$("#btnScale").onclick = () => experiment("scale");
$("#btnVideo").onclick = () => experiment("video");
api("/api/videos").then(names => {
  for (const name of names) {
    const option = document.createElement("option"); option.value = name; option.textContent = name;
    $("#videoSelect").appendChild(option);
  }
});
window.addEventListener("beforeunload", e => { if (state.dirty) { e.preventDefault(); e.returnValue = ""; } });

/* ---------------- Low-effort annotation workflow ---------------- */
async function refinePoint(point, label) {
  if (!state.prompts) {
    const inst = hitInstance(point[0],point[1]);
    if (!inst || !label) return;
    state.prompts = {replaceId:inst.id, bbox:inst.bbox.slice(), points:[]};
    pushUndo();
  }
  state.prompts.points.push([Math.round(point[0]),Math.round(point[1]),label]);
  const prompts = state.prompts;
  showBusy("根据正负点修正边界…");
  try {
    const cands = await post("/api/sam/point",{image:state.imageName,points:prompts.points,bbox:prompts.bbox});
    if (!cands.length) throw new Error("未得到候选，请调整提示点或框选目标。");
    state.pending = {cands,replaceId:prompts.replaceId,kind:"refine"};
    showCands(cands);
  } finally { hideBusy(); }
}
async function cropRegion(box) {
  if (state.dirty) await save();
  showBusy("保存局部图…");
  try {
    const ann = await post("/api/crop",{image:state.imageName,box:box.map(Math.round),annotation:state.ann});
    $("#imageGroup").value = "crop";
    await loadImages(ann.image);
    setMode("select");
  } finally { hideBusy(); }
}
function nextUnlabeled() {
  const todo = visibleInstances().filter(i => i.class === "unlabeled").sort((a,b) => b.area-a.area);
  if (!todo.length) { $("#taskStatus").textContent = "当前筛选范围没有待标注实例。"; return; }
  const current = [...state.selected][0];
  const idx = todo.findIndex(i => i.id === current);
  const next = todo[(idx+1)%todo.length];
  state.selected = new Set([next.id]); centerOn(next);
  render(); refreshSelInfo(); refreshInstList();
}
$("#btnNextInstance").onclick = nextUnlabeled;
for (const id of ["#instanceFilter","#minVisibleArea","#showMasks"]) {
  $(id).onchange = () => {state.selected.clear();render();refreshInstList();refreshSelInfo();};
}
$("#imageGroup").onchange = async () => {
  if (state.dirty) await save();
  const group = $("#imageGroup").value;
  const first = state.catalog.find(i => group === "all" || i.kind === group);
  if (!first) {
    $("#imageGroup").value = state.catalog.find(i => i.name.replace(/\.[^.]+$/, "") === state.imageName)?.kind || "all";
    $("#taskStatus").textContent = "该分组还没有图片，可用“裁成局部图”创建。"; return;
  }
  await loadImages(first.name);
};
