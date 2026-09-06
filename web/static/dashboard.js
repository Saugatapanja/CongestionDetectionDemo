/**
 * Police Command Center - Frontend Logic
 * Handles real-time telemetry polling, video streaming controls,
 * mode transitions, and video uploads.
 */

let currentMode = "traffic";
let isPolling = true;
let lastMapRefresh = 0;

// Update Live Clock
function updateClock() {
  const now = new Date();
  const timeStr = now.toLocaleTimeString('en-US', { hour12: false });
  const clockEl = document.getElementById("clockDisplay");
  if (clockEl) {
    clockEl.innerText = `${timeStr} IST`;
  }
}
setInterval(updateClock, 1000);
updateClock();

// Switch Mode (Traffic vs Crowd)
async function switchMode(mode, force = false) {
  if (currentMode === mode && !force) return;
  currentMode = mode;

  // Update Buttons
  const btnTraffic = document.getElementById("btnModeTraffic");
  const btnCrowd = document.getElementById("btnModeCrowd");
  const trafficPanel = document.getElementById("trafficTelemetryPanel");
  const crowdPanel = document.getElementById("crowdTelemetryPanel");
  const trafficMap = document.getElementById("trafficMapContainer");
  const camLabel = document.getElementById("cameraLabel");

  if (mode === "traffic") {
    btnTraffic.className = "flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all bg-blue-600 text-white shadow-lg";
    btnCrowd.className = "flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all text-slate-400 hover:text-white";
    trafficPanel.classList.remove("hidden");
    crowdPanel.classList.add("hidden");
    trafficMap.classList.remove("hidden");
  } else {
    btnCrowd.className = "flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all bg-blue-600 text-white shadow-lg";
    btnTraffic.className = "flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all text-slate-400 hover:text-white";
    trafficPanel.classList.add("hidden");
    crowdPanel.classList.remove("hidden");
    trafficMap.classList.add("hidden");
    if (camLabel) camLabel.innerText = "CCTV-04: Overhead Durga Puja Pandal Sanctum";
  }

  // Notify backend
  try {
    await fetch("/api/set_mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: mode }),
    });
  } catch (err) {
    console.error("Error setting mode:", err);
  }
}

function toggleVideoReload() {
  const feed = document.getElementById("videoFeed");
  const loading = document.getElementById("loadingOverlay");
  if (loading) loading.classList.remove("hidden");

  // Detach previous stream to terminate stale connection immediately
  if (feed) {
    feed.src = "";
  }

  setTimeout(() => {
    if (feed) {
      feed.src = `/video_feed?t=${Date.now()}`;
    }
    if (loading) {
      setTimeout(() => loading.classList.add("hidden"), 800);
    }
  }, 100);
}

function refreshMap() {
  const mapFrame = document.getElementById("mapFrame");
  if (mapFrame) {
    mapFrame.src = `/api/map?t=${Date.now()}`;
  }
}

// Telemetry Polling Loop
async function pollTelemetry() {
  if (!isPolling) return;

  try {
    const res = await fetch("/api/telemetry");
    if (res.ok) {
      const data = await res.json();
      updateUI(data);
    }
  } catch (err) {
    // Backend momentarily restarting or disconnected
  } finally {
    setTimeout(pollTelemetry, 800);
  }
}

function updateUI(data) {
  // Update FPS
  const fpsEl = document.getElementById("fpsValue");
  if (fpsEl && data.fps) {
    fpsEl.innerText = `${data.fps} FPS`;
  }

  if (currentMode === "traffic" && data.traffic) {
    updateTrafficUI(data.traffic);
  } else if (currentMode === "crowd" && data.crowd) {
    updateCrowdUI(data.crowd);
  }
}

function updateTrafficUI(t) {
  // 0. Update Header Badges (Road name, Route banner, Full coverage)
  const camLabel = document.getElementById("cameraLabel");
  if (camLabel && t.monitored_road) {
    camLabel.innerText = t.monitored_road;
  }
  const routeBanner = document.getElementById("routeBanner");
  if (routeBanner && t.source_junction && t.dest_junction) {
    const formatName = (n) => n.replace(/_/g, " ");
    routeBanner.innerText = `${formatName(t.source_junction)} → ${formatName(t.dest_junction)}`;
  }
  const coverageBadge = document.getElementById("coverageBadge");
  if (coverageBadge) {
    if (t.roi_mode === "full") {
      coverageBadge.innerText = "100% Full Video Coverage";
      coverageBadge.className = "px-2 py-0.5 rounded bg-emerald-500/20 border border-emerald-500/30 text-emerald-300 text-xs font-medium";
    } else {
      coverageBadge.innerText = "Calibrated Lane Polygon";
      coverageBadge.className = "px-2 py-0.5 rounded bg-amber-500/20 border border-amber-500/30 text-amber-300 text-xs font-medium";
    }
  }

  // Periodic Map Refresh when diversion is active (throttled to every 8s)
  if (t.is_divert_recommended && (Date.now() - lastMapRefresh > 8000)) {
    lastMapRefresh = Date.now();
    refreshMap();
  }

  // 1. LOS Badge & Status
  const losLetter = document.getElementById("losLetter");
  const losBadge = document.getElementById("losBadge");
  const statusText = document.getElementById("trafficStatusText");
  const statusDesc = document.getElementById("trafficStatusDesc");

  losLetter.innerText = t.level_of_service || "A";
  statusText.innerText = t.status || "FREE FLOW";

  if (t.level_of_service === "A" || t.level_of_service === "B") {
    losBadge.className = "w-20 h-20 rounded-2xl flex flex-col items-center justify-center font-black text-3xl bg-emerald-500/20 text-emerald-400 border-2 border-emerald-500/50 shadow-lg";
    statusText.className = "text-lg font-bold text-emerald-400";
    statusDesc.innerText = "Vehicles moving smoothly at design speed. No queues or congestion.";
  } else if (t.level_of_service === "C" || t.level_of_service === "D") {
    losBadge.className = "w-20 h-20 rounded-2xl flex flex-col items-center justify-center font-black text-3xl bg-amber-500/20 text-amber-400 border-2 border-amber-500/50 shadow-lg";
    statusText.className = "text-lg font-bold text-amber-400";
    statusDesc.innerText = "Moderate vehicle density. Queue forming; monitor closely.";
  } else {
    // Congested or Gridlock (E or F)
    losBadge.className = "w-20 h-20 rounded-2xl flex flex-col items-center justify-center font-black text-3xl bg-red-500/20 text-red-400 border-2 border-red-500/50 shadow-lg pulse-alert";
    statusText.className = "text-lg font-bold text-red-400";
    statusDesc.innerText = "Severe congestion / gridlock! Alternate diversion dispatch recommended.";
  }

  // 2. Occupancy Bar
  const occ = t.occupancy_pct || 0.0;
  document.getElementById("occupancyText").innerText = `${occ.toFixed(1)}%`;
  const occBar = document.getElementById("occupancyBar");
  occBar.style.width = `${Math.min(100, occ)}%`;

  if (occ < 30) {
    occBar.className = "h-full bg-emerald-500 rounded-full transition-all duration-300";
  } else if (occ < 65) {
    occBar.className = "h-full bg-amber-500 rounded-full transition-all duration-300";
  } else {
    occBar.className = "h-full bg-red-500 rounded-full transition-all duration-300";
  }

  // 3. Breakdown Counts & Kinematic Features
  document.getElementById("totalVehiclesCount").innerText = `${t.vehicle_count || 0} Total`;
  const bd = t.vehicle_breakdown || {};
  const carCount = bd["Car"] || 0;
  const countCarsEl = document.getElementById("countCars");
  if (countCarsEl) {
    countCarsEl.innerText = carCount;
    if (carCount > 6 || t.car_threshold_exceeded) {
      countCarsEl.className = "text-lg font-bold text-red-400 mono animate-pulse";
    } else {
      countCarsEl.className = "text-lg font-bold text-white mono";
    }
  }

  document.getElementById("countBuses").innerText = bd["Bus"] || 0;
  document.getElementById("countTrucks").innerText = bd["Truck"] || 0;
  document.getElementById("countBikes").innerText = bd["Motorcycle"] || 0;
  document.getElementById("stationaryCount").innerText = t.stationary_vehicles || 0;

  const confEl = document.getElementById("aiConfidenceVal");
  if (confEl) confEl.innerText = `${t.avg_confidence_pct || 92.5}%`;
  const speedEl = document.getElementById("avgSpeedVal");
  if (speedEl) speedEl.innerText = `${t.avg_speed_kmh || 0.0} km/h`;
  const flowEl = document.getElementById("flowRateVal");
  if (flowEl) flowEl.innerText = `${t.flow_rate_vpm || 0} vpm`;

  // 4. Police Directive Banner
  const banner = document.getElementById("policeDirectiveBanner");
  const title = document.getElementById("directiveTitle");
  const msg = document.getElementById("directiveMsg");
  const icon = document.getElementById("directiveIcon");

  if (t.is_divert_recommended) {
    banner.className = "mt-4 p-3.5 rounded-xl border flex items-center space-x-4 transition-all bg-red-950/60 border-red-500/50 text-red-200 pulse-alert";
    icon.className = "w-10 h-10 rounded-lg flex items-center justify-center bg-red-600/30 text-red-400 text-lg flex-shrink-0";
    icon.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i>';

    if (t.car_threshold_exceeded || carCount > 6) {
      title.innerText = `POLICE DIRECTIVE: CRITICAL CAR DENSITY (${carCount} CARS > 6)`;
      title.className = "text-xs font-bold uppercase tracking-wider text-red-400";
      msg.innerText = t.police_message || t.recommendation || `Kolkata Police Alert: Over 6 cars (${carCount}) detected. Immediate diversion recommended.`;
    } else {
      title.innerText = "POLICE DIVERSION DIRECTIVE ACTIVATED";
      title.className = "text-xs font-bold uppercase tracking-wider text-red-400";
      msg.innerText = t.recommendation || "Divert vehicles via alternate secondary corridors.";
    }
  } else {
    banner.className = "mt-4 p-3.5 rounded-xl border flex items-center space-x-4 transition-all bg-emerald-950/40 border-emerald-500/40 text-emerald-300";
    icon.className = "w-10 h-10 rounded-lg flex items-center justify-center bg-emerald-600/30 text-emerald-400 text-lg flex-shrink-0";
    icon.innerHTML = '<i class="fa-solid fa-circle-check"></i>';
    title.innerText = "POLICE DIRECTIVE: FLOW NORMAL";
    title.className = "text-xs font-bold uppercase tracking-wider text-emerald-400";
    msg.innerText = "Traffic flowing within capacity. No police diversions required.";
  }

  // 5. Alternate Routes List with Multi-Factor Confidence Scoring
  const routesList = document.getElementById("routeOptionsList");
  if (t.alternate_routes && t.alternate_routes.length > 0) {
    routesList.innerHTML = t.alternate_routes
      .map((r) => {
        const conf = r.confidence_score || 85.0;
        const confColor = conf >= 85 ? "bg-emerald-500 text-emerald-400" : (conf >= 70 ? "bg-blue-500 text-blue-400" : "bg-amber-500 text-amber-400");
        const badgeBg = conf >= 85 ? "bg-emerald-500/20 border-emerald-500/40" : (conf >= 70 ? "bg-blue-500/20 border-blue-500/40" : "bg-amber-500/20 border-amber-500/40");
        const isRec = r.is_recommended;

        return `
          <div class="p-3.5 rounded-xl border ${isRec ? 'bg-slate-900 border-emerald-500/50 shadow-md' : 'bg-slate-900/60 border-slate-700/60'} space-y-2">
            <div class="flex items-center justify-between">
              <div class="flex items-center space-x-2">
                <span class="px-2 py-0.5 rounded text-[10px] font-bold border ${badgeBg} ${confColor}">
                  ★ ${r.confidence_score}% CONFIDENCE
                </span>
                <span class="text-xs font-bold text-white">${r.corridor_name || ('Route #' + r.rank)}</span>
              </div>
              <span class="text-xs mono font-bold text-emerald-400">Saves ${r.time_saved_min}m</span>
            </div>

            <!-- Confidence Progress Bar -->
            <div class="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
              <div class="h-full ${conf >= 85 ? 'bg-emerald-500' : 'bg-blue-500'} rounded-full" style="width: ${conf}%"></div>
            </div>

            <div class="flex items-center justify-between text-[11px] text-slate-400">
              <span>Time: <b class="text-slate-200">${r.total_time_min}m</b> (${r.total_dist_km} km)</span>
              <span>Capacity: <b class="text-slate-200">${r.min_capacity_vph} vph</b></span>
            </div>

            <div class="text-[11px] text-blue-300 bg-blue-950/40 p-1.5 rounded border border-blue-900/50">
              <i class="fa-solid fa-truck-ramp-box mr-1"></i> ${r.vehicle_suitability}
            </div>

            <div class="text-[11px] text-slate-300 bg-slate-950/60 p-2 rounded border border-slate-800 font-medium">
              ${r.police_action}
            </div>
          </div>
        `;
      })
      .join("");
  } else if (!t.is_divert_recommended) {
    routesList.innerHTML = `<div class="text-xs text-slate-400 italic">No diversion currently needed. Primary arterial route is operating within capacity.</div>`;
  }
}

function updateCrowdUI(c) {
  // Total devotees
  document.getElementById("totalHeadsCount").innerText = c.total_heads || 0;

  // Zones breakdown
  const zonesContainer = document.getElementById("pandalZonesList");
  if (c.zones && c.zones.length > 0) {
    zonesContainer.innerHTML = c.zones
      .map((z) => {
        let statusBadge = `<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">NORMAL</span>`;
        let barColor = "bg-emerald-500";
        if (z.level === 2) {
          statusBadge = `<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500/20 text-amber-400 border border-amber-500/30">ELEVATED</span>`;
          barColor = "bg-amber-500";
        } else if (z.level === 3) {
          statusBadge = `<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-red-500/20 text-red-400 border border-red-500/30 animate-pulse">SURGE ALERT</span>`;
          barColor = "bg-red-500";
        }

        const pct = Math.min(100, (z.density_per_m2 / 5.0) * 100);

        return `
          <div class="p-3 bg-slate-800/50 rounded-xl border border-slate-700/60 space-y-1.5">
            <div class="flex items-center justify-between text-xs">
              <span class="font-bold text-slate-200">${z.name}</span>
              ${statusBadge}
            </div>
            <div class="flex items-center justify-between text-xs mono text-slate-400">
              <span>Count: <b class="text-white">${z.head_count}</b></span>
              <span>Density: <b class="text-white">${z.density_per_m2}</b> p/m²</span>
            </div>
            <div class="w-full h-1.5 bg-slate-700 rounded-full overflow-hidden">
              <div class="h-full ${barColor} rounded-full" style="width: ${pct}%"></div>
            </div>
          </div>
        `;
      })
      .join("");
  }

  // Directives Banner & Alerts
  const banner = document.getElementById("policeDirectiveBanner");
  const title = document.getElementById("directiveTitle");
  const msg = document.getElementById("directiveMsg");
  const icon = document.getElementById("directiveIcon");
  const alertsContainer = document.getElementById("crowdAlertsList");

  if (c.active_alerts && c.active_alerts.length > 0) {
    const topAlert = c.active_alerts[0];
    banner.className = "mt-4 p-3.5 rounded-xl border flex items-center space-x-4 transition-all bg-red-950/60 border-red-500/50 text-red-200 pulse-alert";
    icon.className = "w-10 h-10 rounded-lg flex items-center justify-center bg-red-600/30 text-red-400 text-lg flex-shrink-0";
    icon.innerHTML = '<i class="fa-solid fa-bullhorn"></i>';
    title.innerText = `TACTICAL DIRECTIVE: ${topAlert.title}`;
    title.className = "text-xs font-bold uppercase tracking-wider text-red-400";
    msg.innerText = topAlert.police_action;

    alertsContainer.innerHTML = c.active_alerts
      .map(
        (a) => `
      <div class="p-2.5 bg-red-950/40 border border-red-500/40 rounded-lg text-xs space-y-1">
        <div class="flex justify-between items-center text-red-400 font-bold">
          <span>${a.title}</span>
          <span class="mono text-[10px] text-slate-400">${a.timestamp.split(" ")[1]}</span>
        </div>
        <div class="text-slate-300 font-medium">${a.police_action}</div>
        <div class="text-slate-400 text-[11px]">${a.details}</div>
      </div>
    `
      )
      .join("");
  } else {
    banner.className = "mt-4 p-3.5 rounded-xl border flex items-center space-x-4 transition-all bg-emerald-950/40 border-emerald-500/40 text-emerald-300";
    icon.className = "w-10 h-10 rounded-lg flex items-center justify-center bg-emerald-600/30 text-emerald-400 text-lg flex-shrink-0";
    icon.innerHTML = '<i class="fa-solid fa-shield-check"></i>';
    title.innerText = "PANDAL CROWD SAFETY: OPTIMAL";
    title.className = "text-xs font-bold uppercase tracking-wider text-emerald-400";
    msg.innerText = "All sanctum and exit corridors flowing smoothly. No bottlenecks.";

    alertsContainer.innerHTML = `
      <div class="text-xs text-emerald-400/80 italic flex items-center space-x-2">
        <i class="fa-solid fa-check"></i>
        <span>All pandal zones operating within safe crowd capacity limits.</span>
      </div>
    `;
  }
}

// Upload Modal Handlers
function openUploadModal() {
  document.getElementById("uploadModal").classList.remove("hidden");
  toggleUploadModeFields();
}

function closeUploadModal() {
  document.getElementById("uploadModal").classList.add("hidden");
}

function toggleUploadModeFields() {
  const modeSelect = document.getElementById("uploadModeSelect");
  const trafficOptions = document.getElementById("trafficUploadOptions");
  if (!modeSelect || !trafficOptions) return;
  if (modeSelect.value === "traffic") {
    trafficOptions.classList.remove("hidden");
  } else {
    trafficOptions.classList.add("hidden");
  }
}

async function loadKolkataNodes() {
  try {
    const res = await fetch("/api/kolkata_nodes");
    if (res.ok) {
      const data = await res.json();
      const srcSelect = document.getElementById("uploadSourceJunction");
      const destSelect = document.getElementById("uploadDestJunction");
      const roadDatalist = document.getElementById("kolkataRoadsList");

      if (data.nodes && srcSelect && destSelect) {
        const prevSrc = srcSelect.value;
        const prevDest = destSelect.value;

        const optionsHtml = data.nodes.map(n => 
          `<option value="${n.id}">${n.label} (${n.zone})</option>`
        ).join("");

        srcSelect.innerHTML = optionsHtml;
        destSelect.innerHTML = optionsHtml;

        if (prevSrc && data.nodes.some(n => n.id === prevSrc)) {
          srcSelect.value = prevSrc;
        } else {
          srcSelect.value = "Shyambazar_5Point";
        }

        if (prevDest && data.nodes.some(n => n.id === prevDest)) {
          destSelect.value = prevDest;
        } else {
          destSelect.value = "Park_Circus_7Point";
        }
      }

      if (data.popular_roads && roadDatalist) {
        roadDatalist.innerHTML = data.popular_roads.map(r => `<option value="${r}">`).join("");
      }
    }
  } catch (err) {
    console.error("Could not fetch Kolkata nodes:", err);
  }
}

async function handleVideoUpload(event) {
  event.preventDefault();
  const fileInput = document.getElementById("videoFileInput");
  const modeSelect = document.getElementById("uploadModeSelect");
  const progress = document.getElementById("uploadProgress");

  if (!fileInput.files || fileInput.files.length === 0) return;

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  formData.append("mode", modeSelect.value);

  if (modeSelect.value === "traffic") {
    const roadName = document.getElementById("uploadRoadName")?.value || "Central Avenue (CR Avenue)";
    const srcJunc = document.getElementById("uploadSourceJunction")?.value || "Shyambazar_5Point";
    const destJunc = document.getElementById("uploadDestJunction")?.value || "Park_Circus_7Point";
    const roiMode = document.getElementById("uploadRoiMode")?.value || "full";

    formData.append("road_name", roadName);
    formData.append("source_junction", srcJunc);
    formData.append("dest_junction", destJunc);
    formData.append("roi_mode", roiMode);
  }

  progress.classList.remove("hidden");

  try {
    const res = await fetch("/api/upload", {
      method: "POST",
      body: formData,
    });
    if (res.ok) {
      const data = await res.json();
      closeUploadModal();

      // Instantly update UI header labels and telemetry from upload response
      if (data.mode === "traffic") {
        const camLabel = document.getElementById("cameraLabel");
        if (camLabel && data.road_name) {
          camLabel.innerText = data.road_name;
        }

        const routeBanner = document.getElementById("routeBanner");
        if (routeBanner && data.source_junction && data.dest_junction) {
          const formatName = (n) => n.replace(/_/g, " ");
          routeBanner.innerText = `${formatName(data.source_junction)} → ${formatName(data.dest_junction)}`;
        }

        const coverageBadge = document.getElementById("coverageBadge");
        if (coverageBadge) {
          if (data.roi_mode === "full") {
            coverageBadge.innerText = "100% Full Video Coverage";
            coverageBadge.className = "px-2 py-0.5 rounded bg-emerald-500/20 border border-emerald-500/30 text-emerald-300 text-xs font-medium";
          } else {
            coverageBadge.innerText = "Calibrated Polygon";
            coverageBadge.className = "px-2 py-0.5 rounded bg-amber-500/20 border border-amber-500/30 text-amber-300 text-xs font-medium";
          }
        }

        // Reset telemetry card indicators immediately
        const countCars = document.getElementById("countCars");
        if (countCars) {
          countCars.innerText = "0";
          countCars.className = "text-lg font-bold text-white mono";
        }
        document.getElementById("totalVehiclesCount").innerText = "0 Total";
        document.getElementById("countBuses").innerText = "0";
        document.getElementById("countTrucks").innerText = "0";
        document.getElementById("countBikes").innerText = "0";
        document.getElementById("stationaryCount").innerText = "0";
        document.getElementById("occupancyText").innerText = "0.0%";
        document.getElementById("occupancyBar").style.width = "0%";
        document.getElementById("trafficStatusText").innerText = "ANALYZING VIDEO";
        document.getElementById("trafficStatusText").className = "text-lg font-bold text-blue-400";
        document.getElementById("trafficStatusDesc").innerText = `AI pipeline analyzing real-time vehicles on ${data.road_name}...`;
        document.getElementById("losLetter").innerText = "A";
        document.getElementById("losBadge").className = "w-20 h-20 rounded-2xl flex flex-col items-center justify-center font-black text-3xl bg-blue-500/20 text-blue-400 border-2 border-blue-500/50 shadow-lg";

        const routesList = document.getElementById("routeOptionsList");
        if (routesList) {
          routesList.innerHTML = `<div class="text-xs text-blue-400 italic flex items-center space-x-2"><i class="fa-solid fa-spinner fa-spin"></i><span>Analyzing road video and tracking vehicles...</span></div>`;
        }
      }

      await switchMode(data.mode, true);
      toggleVideoReload();
      refreshMap();
    } else {
      const err = await res.json();
      alert("Upload error: " + (err.detail || "Failed to upload video"));
    }
  } catch (err) {
    alert("Network error while uploading video");
  } finally {
    progress.classList.add("hidden");
  }
}

async function resetDemo() {
  try {
    await fetch("/api/reset_demo", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target: "all" }),
    });
    toggleVideoReload();
    refreshMap();
  } catch (err) {
    console.error("Error resetting demo:", err);
  }
}

// Initialize dynamic nodes and start polling
loadKolkataNodes();
pollTelemetry();
