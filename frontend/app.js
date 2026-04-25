/**
 * Wildfire ICS — Frontend Application Logic
 * app.js  |  Vanilla JS, no external dependencies
 *
 * API contract (critical):
 *   POST /reset  → returns Observation directly
 *   POST /step   → returns StepResult { observation, reward, done, info }
 *   POST /auto_step → returns { steps: [StepSnapshot], done: bool }
 *   GET  /state/render → trimmed ground-truth snapshot (fog bypassed)
 */

"use strict";

// ── API field helpers (snake_case from Python; tolerate camelCase if ever used) ─
function pickStat(obj, ...keys) {
  if (!obj) return undefined;
  for (const k of keys) {
    if (Object.prototype.hasOwnProperty.call(obj, k) && obj[k] != null) {
      return obj[k];
    }
  }
  return undefined;
}

/**
 * Build display-ready episode metrics from the latest observation.
 * Falls back to grid-visible cells for land % only when server omits area_saved_pct.
 */
function normalizeEpisodeStats(obs) {
  const st = obs?.stats ?? {};
  const cellsBurned = pickStat(st, "cells_burned", "cellsBurned") ?? 0;
  const popLost = pickStat(st, "population_lost", "populationLost") ?? 0;
  const totalPop = pickStat(st, "total_population", "totalPopulation") ?? 0;

  let areaSaved = pickStat(st, "area_saved_pct", "areaSavedPct");
  let civSafe = pickStat(st, "civilians_saved_pct", "civiliansSavedPct");

  if (areaSaved == null && obs?.grid?.length) {
    let burnable = 0;
    let burnedVis = 0;
    for (const row of obs.grid) {
      for (const cell of row) {
        const f = cell.fuel_type;
        if (!f || f === "water" || f === "road") continue;
        if (cell.fire_state === "unknown") continue;
        burnable++;
        if (cell.fire_state === "burned_out") burnedVis++;
      }
    }
    if (burnable > 0) {
      areaSaved = Math.round(1000 * (burnable - burnedVis) / burnable) / 10;
    }
  }

  if (civSafe == null && totalPop > 0) {
    civSafe = Math.round(1000 * (totalPop - popLost) / totalPop) / 10;
  } else if (civSafe == null && popLost === 0) {
    civSafe = 100.0;
  }

  const containment = pickStat(st, "containment_pct", "containmentPct");
  if (areaSaved == null && containment != null) {
    areaSaved = containment;
  }

  return {
    areaSaved,
    civSafe,
    cellsBurned,
    popLost,
    totalPop,
    currentStep: pickStat(st, "current_step", "currentStep"),
    raw: st,
  };
}

// ── Simulation state ──────────────────────────────────────────────────────────
const sim = {
  obs: null,            // current Observation (agent's view)
  cumulativeReward: 0,
  lastStepReward: 0,
  done: false,
  groundTruthData: null, // from GET /state/render when toggle is on
  agentMode: "heuristic",
  tier: "easy",
  seed: 42,
  playing: false,
  speed: 600,           // ms between auto_step calls
  playTimer: null,
  cellSize: 0,          // computed per reset
};

// ── Canvas setup ──────────────────────────────────────────────────────────────
const canvas = document.getElementById("grid-canvas");
const ctx = canvas.getContext("2d");
const canvasWrap = document.getElementById("canvas-wrap");

// ── Cell colour function — mirrors env/rendering.py exactly ─────────────────
function cellColor(cell) {
  const fs = cell.fire_state;
  const intensity = cell.fire_intensity ?? 0;

  if (fs === "unknown") return "rgba(0,0,0,0.82)";

  if (fs === "burning") {
    const sat = 0.4 + 0.6 * intensity;
    const g = Math.round((1.0 - sat * 0.8) * 255);
    return `rgb(255,${g},0)`;
  }
  if (fs === "ember")       return "#e55c00";
  if (fs === "burned_out")  return "#3f3530";
  if (fs === "firebreak")   return "#8c5a28";
  if (fs === "suppressed")  return "#88cc88";

  // Unburned — shade by fuel type
  const fuel = cell.fuel_type ?? "grass";
  switch (fuel) {
    case "water":  return "#4d80e6";
    case "road":   return "#b0b0b0";
    case "timber": return "#1a7a1a";
    case "shrub":  return "#7fba33";
    case "urban":  return "#ccbfb2";
    default:       return "#a8d95e"; // grass
  }
}

// ── Canvas renderer ───────────────────────────────────────────────────────────
function renderCanvas(obs, groundTruth = null) {
  if (!obs || !obs.grid || obs.grid.length === 0) return;

  const rows = obs.grid.length;
  const cols = obs.grid[0].length;

  // Resize canvas if grid dimensions changed
  const panelW = canvasWrap.parentElement.clientWidth - 24;
  const panelH = canvasWrap.parentElement.clientHeight - 24;
  const cs = Math.max(4, Math.floor(Math.min(panelW / cols, panelH / rows)));
  sim.cellSize = cs;

  if (canvas.width !== cs * cols || canvas.height !== cs * rows) {
    canvas.width  = cs * cols;
    canvas.height = cs * rows;
  }

  // Build a lookup for ground-truth overlay (only unknown cells get overridden)
  const gtGrid = groundTruth?.grid ?? null;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      let cell = obs.grid[r][c];

      // Ground-truth overlay: if toggle is on and cell is unknown, show real state
      if (gtGrid && cell.fire_state === "unknown") {
        cell = { ...cell, ...gtGrid[r][c], _gt_overlay: true };
      }

      const color = cellColor(cell);
      ctx.fillStyle = color;
      ctx.fillRect(c * cs, r * cs, cs, cs);

      // Ground-truth overlay marker (slightly transparent to distinguish)
      if (cell._gt_overlay) {
        ctx.fillStyle = "rgba(255,200,0,0.08)";
        ctx.fillRect(c * cs, r * cs, cs, cs);
      }

      // Populated cell: blue border
      if (cell.is_populated) {
        ctx.strokeStyle = "#58a6ff";
        ctx.lineWidth = Math.max(1, cs * 0.1);
        ctx.strokeRect(c * cs + 0.5, r * cs + 0.5, cs - 1, cs - 1);
      }

      // Crew present: green dot
      if (cell.crew_present) {
        ctx.fillStyle = "#00ff88";
        const r2 = Math.max(2, cs * 0.22);
        ctx.beginPath();
        ctx.arc(c * cs + cs / 2, r * cs + cs / 2, r2, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  // Draw crew markers from resources (labelled)
  const crews = obs.resources?.crews ?? [];
  for (const crew of crews) {
    if (!crew.is_deployed || !crew.is_active) continue;
    const cx = crew.col * cs + cs / 2;
    const cy = crew.row * cs + cs / 2;
    const r2 = Math.max(3, cs * 0.28);

    ctx.beginPath();
    ctx.arc(cx, cy, r2, 0, Math.PI * 2);
    ctx.fillStyle = crew.is_active ? "lime" : "#f85149";
    ctx.fill();
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 1;
    ctx.stroke();

    if (cs >= 10) {
      const label = crew.crew_id.replace("crew_", "c");
      ctx.fillStyle = "#fff";
      ctx.font = `bold ${Math.max(7, cs * 0.4)}px 'Courier New', monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label, cx, cy);
    }
  }

  // Pulse canvas border if fire is active
  const burning = obs.stats?.cells_burning ?? 0;
  if (burning > 0) {
    canvasWrap.classList.add("fire-active");
  } else {
    canvasWrap.classList.remove("fire-active");
  }

  // Update step progress bar
  const cur = obs.stats?.current_step ?? 0;
  const max = obs.stats?.max_steps ?? 1;
  document.getElementById("step-progress-fill").style.width =
    `${Math.min(100, (cur / max) * 100)}%`;
}

// ── Stats panel ───────────────────────────────────────────────────────────────
function updateStats(obs, cumulativeReward, lastStepReward) {
  if (!obs?.stats) return;
  const stats = obs.stats;

  const cur = pickStat(stats, "current_step", "currentStep") ?? 0;
  const max = pickStat(stats, "max_steps", "maxSteps") ?? 1;

  setText("stat-step", `${cur} / ${max}`);

  const n = normalizeEpisodeStats(obs);
  setText(
    "stat-land-saved-val",
    n.areaSaved != null ? `${Number(n.areaSaved).toFixed(1)}%` : "—"
  );
  setText(
    "stat-civilians-safe-val",
    n.civSafe != null ? `${Number(n.civSafe).toFixed(1)}%` : "—"
  );
  setText("stat-cells-burned-val", n.cellsBurned);
  setText("stat-burning-val", pickStat(stats, "cells_burning", "cellsBurning") ?? 0);
  setText("stat-pop-threat-val", pickStat(stats, "population_threatened", "populationThreatened") ?? 0);
  setText("stat-pop-lost-val", n.popLost);

  // Cumulative reward
  setText("reward-total", cumulativeReward.toFixed(3));

  // Per-step delta with colour
  const deltaEl = document.getElementById("reward-delta");
  if (deltaEl) {
    const sign = lastStepReward >= 0 ? "+" : "";
    deltaEl.textContent = `${sign}${lastStepReward.toFixed(3)} this step`;
    deltaEl.className = "reward-delta " + (lastStepReward >= 0 ? "positive" : "negative");
  }
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

// ── Resources panel ───────────────────────────────────────────────────────────
function updateResources(resources) {
  if (!resources) return;

  const crewBody = document.getElementById("crew-tbody");
  if (crewBody) {
    crewBody.innerHTML = "";
    for (const crew of (resources.crews ?? [])) {
      const tr = document.createElement("tr");
      let cls = "crew-idle";
      let status = "STAGING";
      if (!crew.is_active) { cls = "crew-lost"; status = "LOST"; }
      else if (crew.is_deployed) { cls = "crew-deployed"; status = `${crew.row},${crew.col}`; }
      tr.className = cls;
      tr.innerHTML = `<td>${crew.crew_id.replace("crew_","C")}</td><td>${status}</td>`;
      crewBody.appendChild(tr);
    }
  }

  const tankerBody = document.getElementById("tanker-tbody");
  if (tankerBody) {
    tankerBody.innerHTML = "";
    for (const tanker of (resources.tankers ?? [])) {
      const tr = document.createElement("tr");
      tr.className = "tanker-row";
      const cd = tanker.cooldown_remaining ?? 0;
      const maxCd = 5; // matches TierConfig.tanker_cooldown default
      const pct = cd === 0 ? 0 : (cd / maxCd) * 100;
      const readyClass = cd === 0 ? "tanker-ready" : "tanker-charging";
      const readyLabel = cd === 0 ? "READY" : `CD:${cd}`;
      tr.innerHTML = `
        <td>${tanker.tanker_id.replace("tanker_","T")}</td>
        <td class="${readyClass}">${readyLabel}</td>
        <td>
          <div class="cooldown-bar-wrap">
            <div class="cooldown-bar-fill" style="width:${pct}%"></div>
          </div>
        </td>`;
      tankerBody.appendChild(tr);
    }
  }

  // Budgets
  const fb = resources.firebreak_budget ?? 0;
  const rb = resources.recon_budget ?? 0;
  setText("firebreak-budget", `FB: ${fb}`);
  setText("recon-budget",     `RC: ${rb}`);
}

// ── Weather panel ─────────────────────────────────────────────────────────────
function updateWeather(weather) {
  if (!weather) return;

  const speed = weather.wind_speed_kmh ?? 0;
  const dir   = weather.wind_direction_deg ?? 0;
  const hum   = weather.humidity_pct ?? 0;
  const rain  = weather.rain_active ?? false;

  setText("wind-speed-val", `${speed.toFixed(0)} km/h`);
  setText("wind-dir-val",   `${dir.toFixed(0)}°`);
  setText("humidity-val",   `${hum.toFixed(0)}%`);

  // Rotate needle: 0° = North (top of dial)
  const needle = document.getElementById("wind-needle");
  if (needle) needle.style.transform = `translateX(-50%) translateY(-100%) rotate(${dir}deg)`;

  const rainBadge = document.getElementById("rain-badge");
  if (rainBadge) rainBadge.classList.toggle("active", rain);
}

// ── Events log ────────────────────────────────────────────────────────────────
let _lastEventSet = [];

function updateEvents(events) {
  if (!events || events.length === 0) return;

  const newEvents = events.filter(e => !_lastEventSet.includes(e));
  if (newEvents.length === 0) return;
  _lastEventSet = events;

  const log = document.getElementById("events-log");
  if (!log) return;

  for (const evt of newEvents.slice().reverse()) {
    const div = document.createElement("div");
    div.className = "event-entry";
    div.textContent = evt;
    log.insertBefore(div, log.firstChild);
  }

  // Keep at most 30 entries
  while (log.children.length > 30) log.removeChild(log.lastChild);
}

// ── Action log ────────────────────────────────────────────────────────────────
function updateActionLog(action) {
  if (!action) return;
  setText("last-action-type", action.action_type?.toUpperCase() ?? "—");
  const params = { ...action };
  delete params.action_type;
  const paramStr = Object.entries(params)
    .filter(([, v]) => v !== null && v !== undefined)
    .map(([k, v]) => `${k}: ${v}`)
    .join(" | ") || "—";
  setText("last-action-params", paramStr);
}

// ── Terminal overlay ──────────────────────────────────────────────────────────
async function showTerminal() {
  const overlay = document.getElementById("terminal-overlay");
  if (!overlay) return;

  const card = document.getElementById("terminal-card");
  if (!card) return;

  const n = normalizeEpisodeStats(sim.obs);
  const title = card.querySelector("h2");

  if (n.popLost === 0) {
    title.textContent = "✅ EPISODE COMPLETE";
    title.className = "win";
  } else {
    title.textContent = "⚠ EPISODE ENDED";
    title.className = "loss";
  }

  const landStr = n.areaSaved != null ? `${Number(n.areaSaved).toFixed(1)}%` : "—";
  const civStr = n.civSafe != null ? `${Number(n.civSafe).toFixed(1)}%` : "—";
  setText("terminal-land-saved", landStr);
  setText("terminal-civilians-safe", civStr);
  setText("terminal-cells-burned", String(n.cellsBurned));
  setText("terminal-pop-lost", n.popLost);
  setText("terminal-reward", sim.cumulativeReward.toFixed(3));
  setText("terminal-step", n.currentStep ?? "—");

  overlay.classList.add("show");

  // Authoritative end-game numbers (ground truth — fixes blank UI if observation JSON differed)
  try {
    const st = await apiGet("/state");
    if (st.error) return;
    const tb = st.total_burnable ?? 0;
    const burned = st.cells_burned ?? 0;
    const landPct = tb > 0 ? Math.round(1000 * (tb - burned) / tb) / 10 : 100;
    const tp = st.total_population ?? 0;
    const lost = st.population_lost ?? 0;
    const civPct = tp > 0 ? Math.round(1000 * (tp - lost) / tp) / 10 : 100;
    setText("terminal-land-saved", `${landPct}%`);
    setText("terminal-civilians-safe", `${civPct}%`);
    setText("terminal-cells-burned", String(burned));
    setText("terminal-pop-lost", String(lost));
    setText("terminal-step", st.current_step ?? "—");
  } catch (e) {
    console.warn("Could not refresh end-game stats from /state", e);
  }
}

function hideTerminal() {
  document.getElementById("terminal-overlay")?.classList.remove("show");
}

// ── API helpers ───────────────────────────────────────────────────────────────
async function apiPost(path, body = null, params = {}) {
  const url = new URL(path, window.location.origin);
  for (const [k, v] of Object.entries(params)) url.searchParams.set(k, v);
  const opts = { method: "POST" };
  if (body) { opts.body = JSON.stringify(body); opts.headers = { "Content-Type": "application/json" }; }
  const res = await fetch(url, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? res.statusText);
  }
  return res.json();
}

async function apiGet(path) {
  const res = await fetch(path);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? res.statusText);
  }
  return res.json();
}

// ── Full UI update from obs ───────────────────────────────────────────────────
function applyObservation(obs) {
  sim.obs = obs;
  renderCanvas(obs, sim.groundTruthData);
  updateStats(obs, sim.cumulativeReward, sim.lastStepReward);
  updateResources(obs.resources);
  updateWeather(obs.weather);
  updateEvents(obs.recent_events ?? []);
}

// ── Reset flow ────────────────────────────────────────────────────────────────
async function doReset() {
  stopPlay();
  hideTerminal();
  setStatus("Resetting…");
  setControlsEnabled(false);

  sim.cumulativeReward = 0;
  sim.lastStepReward = 0;
  sim.done = false;
  sim.groundTruthData = null;
  _lastEventSet = [];
  document.getElementById("events-log").innerHTML = "";
  setText("last-action-type", "—");
  setText("last-action-params", "—");

  try {
    // POST /reset → returns Observation directly (not wrapped in StepResult)
    const obs = await apiPost("/reset", null, {
      task_id: sim.tier,
      seed: sim.seed,
    });
    applyObservation(obs);
    setStatus("Ready");
  } catch (e) {
    setStatus(`Error: ${e.message}`);
    console.error(e);
  } finally {
    setControlsEnabled(true);
  }
}

// ── Auto-step (agent drives the sim) ────────────────────────────────────────
async function doAutoStep() {
  if (sim.done) { stopPlay(); return; }
  if (!sim.obs) { stopPlay(); return; }

  try {
    // POST /auto_step → { steps: [StepSnapshot], done: bool }
    const data = await apiPost("/auto_step", null, {
      n: 1,
      agent: sim.agentMode,
    });

    for (const snap of data.steps) {
      // StepSnapshot: { observation, reward, done, info, action_taken }
      sim.lastStepReward = snap.reward;
      sim.cumulativeReward += snap.reward;
      sim.done = snap.done;

      applyObservation(snap.observation);
      updateActionLog(snap.action_taken);

      if (snap.done) {
        stopPlay();
        await showTerminal();
        break;
      }
    }

    // Refresh ground-truth overlay if active
    if (document.getElementById("gt-toggle")?.checked) {
      refreshGroundTruth();
    }
  } catch (e) {
    setStatus(`Step error: ${e.message}`);
    console.error(e);
    stopPlay();
  }
}

// ── Ground truth overlay ──────────────────────────────────────────────────────
async function refreshGroundTruth() {
  try {
    const gt = await apiGet("/state/render");
    sim.groundTruthData = gt;
    renderCanvas(sim.obs, gt);
  } catch (e) {
    console.warn("Ground truth fetch failed:", e.message);
  }
}

// ── Play / pause ──────────────────────────────────────────────────────────────
function startPlay() {
  if (sim.playing || sim.done || !sim.obs) return;
  sim.playing = true;
  updatePlayButton();
  doAutoStep();
  sim.playTimer = setInterval(doAutoStep, sim.speed);
}

function stopPlay() {
  if (sim.playTimer) { clearInterval(sim.playTimer); sim.playTimer = null; }
  sim.playing = false;
  updatePlayButton();
}

function togglePlay() {
  if (sim.playing) stopPlay(); else startPlay();
}

function updatePlayButton() {
  const btn = document.getElementById("btn-play");
  if (!btn) return;
  btn.textContent = sim.playing ? "⏸ Pause" : "▶ Play";
  btn.classList.toggle("playing", sim.playing);
}

// ── Status line ───────────────────────────────────────────────────────────────
function setStatus(msg) {
  const el = document.getElementById("status-text");
  if (el) el.textContent = msg;
}

function setControlsEnabled(enabled) {
  ["btn-reset", "btn-play", "btn-step"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.disabled = !enabled;
  });
}

// ── Canvas hover tooltip ──────────────────────────────────────────────────────
const tooltip = document.getElementById("cell-tooltip");

canvas.addEventListener("mousemove", (e) => {
  if (!sim.obs || sim.cellSize === 0) return;
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width  / rect.width;
  const scaleY = canvas.height / rect.height;
  const px = (e.clientX - rect.left) * scaleX;
  const py = (e.clientY - rect.top)  * scaleY;
  const col = Math.floor(px / sim.cellSize);
  const row = Math.floor(py / sim.cellSize);

  const grid = sim.obs.grid;
  if (row < 0 || row >= grid.length || col < 0 || col >= grid[0].length) {
    tooltip.style.display = "none";
    return;
  }
  const cell = grid[row][col];

  tooltip.textContent =
    `(${row},${col}) ${cell.fire_state}` +
    (cell.fuel_type ? ` · ${cell.fuel_type}` : "") +
    (cell.is_populated ? " · 🏘 pop" : "") +
    (cell.fire_intensity ? ` · int:${cell.fire_intensity.toFixed(2)}` : "");

  const wrapRect = canvasWrap.getBoundingClientRect();
  tooltip.style.left = `${e.clientX - wrapRect.left + 10}px`;
  tooltip.style.top  = `${e.clientY - wrapRect.top  + 10}px`;
  tooltip.style.display = "block";
});

canvas.addEventListener("mouseleave", () => { tooltip.style.display = "none"; });

// ── Controls wiring ───────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {

  document.getElementById("btn-reset")?.addEventListener("click", doReset);

  document.getElementById("btn-play")?.addEventListener("click", togglePlay);

  document.getElementById("btn-step")?.addEventListener("click", async () => {
    if (sim.done || !sim.obs) return;
    stopPlay();
    await doAutoStep();
  });

  // Tier selector
  document.getElementById("tier-select")?.addEventListener("change", (e) => {
    sim.tier = e.target.value;
  });

  // Seed input
  document.getElementById("seed-input")?.addEventListener("change", (e) => {
    sim.seed = parseInt(e.target.value, 10) || 42;
  });

  // Agent selector
  document.getElementById("agent-select")?.addEventListener("change", (e) => {
    sim.agentMode = e.target.value;
    // Reset active agent on next /reset or stop and let server re-create
    if (sim.playing) stopPlay();
  });

  // Speed slider
  document.getElementById("speed-slider")?.addEventListener("input", (e) => {
    sim.speed = parseInt(e.target.value, 10);
    setText("speed-label", `${sim.speed}ms`);
    if (sim.playing) {
      clearInterval(sim.playTimer);
      sim.playTimer = setInterval(doAutoStep, sim.speed);
    }
  });

  // Ground truth toggle
  document.getElementById("gt-toggle")?.addEventListener("change", async (e) => {
    if (e.target.checked) {
      await refreshGroundTruth();
    } else {
      sim.groundTruthData = null;
      renderCanvas(sim.obs, null);
    }
  });

  // Terminal "Play again" button
  document.getElementById("btn-play-again")?.addEventListener("click", doReset);

  // Auto-reset on load with easy tier
  doReset();
});

// ── Resize handler — redraw canvas when window resizes ───────────────────────
window.addEventListener("resize", () => {
  if (sim.obs) renderCanvas(sim.obs, sim.groundTruthData);
});
