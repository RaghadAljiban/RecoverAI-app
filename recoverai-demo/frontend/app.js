// ── state ────────────────────────────────────────────────────
const state = {
  stage: "welcome",
  exercise: null,
  file: null,
  fileUrl: null,
  isAnalyzing: false,
};

const STAGE_ORDER = ["welcome", "choose", "upload", "analyzing", "result"];
const PROGRESS_STAGES = ["choose", "upload", "analyzing", "result"];

const ICONS = {
  check: '<svg viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"/></svg>',
  cross: '<svg viewBox="0 0 24 24"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>',
};

// ── helpers ──────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const qsa = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function setStage(name) {
  state.stage = name;
  qsa(".stage").forEach((el) => {
    const match = el.dataset.stage === name;
    el.hidden = !match;
    el.classList.toggle("is-active", match);
    if (match) {
      el.style.animation = "none";
      // force reflow then re-apply animation
      void el.offsetWidth;
      el.style.animation = "";
    }
  });

  const topbar = $("topbar");
  topbar.hidden = name === "welcome";

  qsa("#progress .step").forEach((step) => {
    const i = PROGRESS_STAGES.indexOf(step.dataset.step);
    const cur = PROGRESS_STAGES.indexOf(name);
    step.classList.toggle("is-active", i === cur);
    step.classList.toggle("is-done", i >= 0 && i < cur);
  });
}

function selectExercise(name) {
  state.exercise = name;
  qsa(".exercise-card").forEach((card) => {
    card.classList.toggle("is-selected", card.dataset.exercise === name);
  });
  $("chooseContinueBtn").disabled = !name;
}

function setFile(file) {
  if (state.fileUrl) URL.revokeObjectURL(state.fileUrl);
  state.file = file;
  state.fileUrl = file ? URL.createObjectURL(file) : null;

  const dz = $("dropzone");
  const empty = $("dropzoneEmpty");
  const preview = $("dropzonePreview");
  const errorBox = $("dropzoneError");

  errorBox.hidden = true;

  if (file) {
    empty.hidden = true;
    preview.hidden = false;
    dz.classList.add("has-file");
    $("previewVideo").src = state.fileUrl;
    $("previewName").textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MB`;
    $("analyzeBtn").disabled = false;
  } else {
    empty.hidden = false;
    preview.hidden = true;
    dz.classList.remove("has-file");
    $("previewVideo").removeAttribute("src");
    $("analyzeBtn").disabled = true;
  }
}

function showFileError(msg) {
  $("dropzoneError").hidden = false;
  $("dropzoneErrorMsg").textContent = msg;
}

function setAnalyzingProgress(pct, title, sub) {
  $("analyzingFill").style.width = `${Math.max(0, Math.min(100, pct))}%`;
  $("analyzingPct").textContent = `${Math.round(pct)}%`;
  if (title) $("analyzingTitle").textContent = title;
  if (sub !== undefined) $("analyzingSub").textContent = sub;
}

function fmt(value, digits = 3) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return value.toFixed(digits);
  return String(value);
}

// ── stage 2 · exercise grid ──────────────────────────────────
async function loadExercises() {
  const res = await fetch("/api/exercises");
  const data = await res.json();
  const grid = $("exerciseGrid");
  grid.innerHTML = "";
  data.exercises.forEach(({ id, name }) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "exercise-card";
    btn.dataset.exercise = name;
    const initials = name.split(/[^A-Za-z]/).filter(Boolean).map((w) => w[0]).slice(0, 2).join("").toUpperCase();
    btn.innerHTML = `
      <div class="exercise-card-icon">${initials}</div>
      <p class="exercise-card-name">${name}</p>
      <p class="exercise-card-hint">Exercise ${id}</p>
    `;
    btn.addEventListener("click", () => selectExercise(name));
    grid.appendChild(btn);
  });
}

// ── stage 3 · upload handlers ────────────────────────────────
function initDropzone() {
  const dz = $("dropzone");
  const input = $("fileInput");

  dz.addEventListener("dragover", (e) => {
    e.preventDefault();
    dz.classList.add("is-dragover");
  });
  dz.addEventListener("dragleave", () => dz.classList.remove("is-dragover"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("is-dragover");
    const file = e.dataTransfer.files?.[0];
    if (file) handleFileSelection(file);
  });

  input.addEventListener("change", () => {
    const file = input.files?.[0];
    if (file) handleFileSelection(file);
  });
}

function handleFileSelection(file) {
  if (!file.type.startsWith("video/") && !/\.(mp4|mov|avi|mkv|webm|m4v)$/i.test(file.name)) {
    showFileError("That doesn't look like a video file.");
    return;
  }
  if (file.size > 200 * 1024 * 1024) {
    showFileError("File is too large. Try a clip under 200 MB.");
    return;
  }
  setFile(file);
}

// ── stage 4 · analyze ────────────────────────────────────────
const ANALYZE_STAGES = [
  { from: 0,  to: 20, title: "Reading your video…",          sub: "Decoding frames." },
  { from: 20, to: 70, title: "Extracting pose…",             sub: "Running MediaPipe on each frame." },
  { from: 70, to: 88, title: "Computing features…",          sub: "Normalizing pose and building feature tensor." },
  { from: 88, to: 96, title: "Running classifier…",          sub: "Conditioned TCN inference." },
  { from: 96, to: 99, title: "Preparing your overlay…",      sub: "Drawing the skeleton on every frame." },
];

let progressTimer = null;
function startProgressAnimation() {
  let pct = 0;
  let stage = 0;
  setAnalyzingProgress(0, ANALYZE_STAGES[0].title, ANALYZE_STAGES[0].sub);
  progressTimer = setInterval(() => {
    const s = ANALYZE_STAGES[stage];
    const target = s.to;
    pct = Math.min(target, pct + (target - pct) * 0.04 + 0.3);
    setAnalyzingProgress(pct, s.title, s.sub);
    if (pct >= target - 0.5 && stage < ANALYZE_STAGES.length - 1) {
      stage += 1;
      setAnalyzingProgress(pct, ANALYZE_STAGES[stage].title, ANALYZE_STAGES[stage].sub);
    }
  }, 220);
}

function stopProgressAnimation() {
  if (progressTimer) clearInterval(progressTimer);
  progressTimer = null;
}

async function runAnalysis() {
  if (state.isAnalyzing) return;
  if (!state.exercise || !state.file) return;

  state.isAnalyzing = true;
  setStage("analyzing");
  startProgressAnimation();

  const form = new FormData();
  form.append("video", state.file);
  form.append("exercise_name", state.exercise);

  try {
    const res = await fetch("/api/classify", { method: "POST", body: form });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(detail.detail || `HTTP ${res.status}`);
    }
    const result = await res.json();
    setAnalyzingProgress(100, "Done.", "");
    await new Promise((r) => setTimeout(r, 320));
    renderResult(result);
    setStage("result");
  } catch (err) {
    setStage("result");
    renderError(err.message || String(err));
  } finally {
    stopProgressAnimation();
    state.isAnalyzing = false;
  }
}

// ── stage 5 · result rendering ───────────────────────────────
function renderResult(r) {
  $("resultError").hidden = true;
  $("resultCard").hidden = false;

  const isOk = r.prediction_idx === 1;
  $("resultCard").classList.toggle("is-bad", !isOk);
  $("resultIcon").innerHTML = isOk ? ICONS.check : ICONS.cross;
  $("resultVerdict").textContent = r.prediction_label;
  $("resultConfidence").textContent = `${(r.confidence * 100).toFixed(0)}%`;
  $("resultExercise").textContent = r.exercise_name;

  requestAnimationFrame(() => {
    $("confidenceFill").style.width = `${(r.confidence * 100).toFixed(1)}%`;
  });

  if (r.overlay_url) {
    $("overlayVideo").src = r.overlay_url;
    $("overlayVideo").hidden = false;
  } else {
    $("overlayVideo").hidden = true;
  }

  const tbody = $("metricsTable").querySelector("tbody");
  tbody.innerHTML = "";
  const rows = [
    ["P(correct)", fmt(r.prob_correct, 3)],
    ["P(incorrect)", fmt(r.prob_incorrect, 3)],
    ["Threshold", fmt(r.threshold, 2)],
    ["Frames processed", r.info?.n_frames ?? "—"],
    ["Clip duration", r.debug?.clip_duration_sec ? `${r.debug.clip_duration_sec.toFixed(2)} s` : "—"],
    ["Pose detection rate", r.info?.detection_rate != null ? `${(r.info.detection_rate * 100).toFixed(0)}%` : "—"],
    ["Used FPS", fmt(r.debug?.used_fps, 3)],
    ["Raw logit", fmt(r.logit, 4)],
  ];
  rows.forEach(([k, v]) => {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    const td = document.createElement("td");
    th.textContent = k;
    td.textContent = v;
    tr.append(th, td);
    tbody.appendChild(tr);
  });
}

function renderError(message) {
  $("resultCard").hidden = true;
  $("resultError").hidden = false;
  $("resultErrorMsg").textContent = message;
}

// ── event delegation for data-action buttons ─────────────────
document.addEventListener("click", (e) => {
  const btn = e.target.closest("[data-action]");
  if (!btn) return;
  const action = btn.dataset.action;
  switch (action) {
    case "goto-choose":
      setStage("choose");
      break;
    case "goto-upload":
      if (state.exercise) {
        $("uploadExerciseName").textContent = state.exercise;
        setStage("upload");
      }
      break;
    case "back-to-welcome":
      setStage("welcome");
      break;
    case "back-to-choose":
      setStage("choose");
      break;
    case "open-picker":
      $("fileInput").click();
      break;
    case "replace-file":
      setFile(null);
      $("fileInput").value = "";
      $("fileInput").click();
      break;
    case "analyze":
      runAnalysis();
      break;
    case "restart-choose":
      setFile(null);
      $("fileInput").value = "";
      setStage("choose");
      break;
    case "restart-upload":
      setFile(null);
      $("fileInput").value = "";
      $("uploadExerciseName").textContent = state.exercise || "your exercise";
      setStage("upload");
      break;
  }
});

// ── init ─────────────────────────────────────────────────────
loadExercises().catch((err) => console.error("Failed to load exercises", err));
initDropzone();
setStage("welcome");
