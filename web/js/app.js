const state = {
  activeTab: "live",
  mobileCamIndex: 1,
  driveMode: "AUTO",
  carRunning: false,
  carConnected: false,
  serverOnline: true,
  cam1Src: "CSI",
  cam2Src: "USB-0",
  fovPct: 100,
  maxSpeed: 85,
  roi: { w_top: 123, h_top: 105, w_bot: 5, h_bot: 222 },
  blackboxAuto: true,
  blackboxManual: false,
  blackboxRecording: false,

  keysPressed: new Set(),

  // Smooth physics state
  manualSpeed: 0.0,
  manualSteer: 0.0,
  lastTickMs: 0,

  simTime: 0,
  speedPct: 0,
  steerErr: 0,
  curveVal: 0,
  cmd: "STOP",
  detectedSigns: [
    { name: "TocDo30", conf: 0.89, box: [0.22, 0.24, 58, 58] },
    { name: "RePhai", conf: 0.82, box: [0.62, 0.22, 62, 62] },
  ],
};

const el = {
  connState: document.getElementById("connState"),
  connText: document.getElementById("connText"),
  statusText: document.getElementById("statusText"),
  driveBtns: document.querySelectorAll("[data-drive]"),
  panelCam1: document.getElementById("panelCam1"),
  panelCam2: document.getElementById("panelCam2"),
  btnMobileCamToggle: document.getElementById("btnMobileCamToggle"),
  btnSwitchCam1: document.getElementById("btnSwitchCam1"),
  btnSwitchCam2: document.getElementById("btnSwitchCam2"),
  btnSwapCam: document.getElementById("btnSwapCam"),
  cam1Src: document.getElementById("cam1Src"),
  cam2Src: document.getElementById("cam2Src"),

  padBtns: document.querySelectorAll(".pad-btn[data-key]"),
  btnDpadStop: document.getElementById("btnDpadStop"),
  btnPowerToggle: document.getElementById("btnPowerToggle"),

  sliderMaxSpeed: document.getElementById("sliderMaxSpeed"),
  maxSpeedLabel: document.getElementById("maxSpeedLabel"),
  sliderFov: document.getElementById("sliderFov"),
  fovVal: document.getElementById("fovVal"),
  sliderRoiWTop: document.getElementById("sliderRoiWTop"),
  roiWTopVal: document.getElementById("roiWTopVal"),
  sliderRoiHTop: document.getElementById("sliderRoiHTop"),
  roiHTopVal: document.getElementById("roiHTopVal"),
  sliderRoiWBot: document.getElementById("sliderRoiWBot"),
  roiWBotVal: document.getElementById("roiWBotVal"),
  sliderRoiHBot: document.getElementById("sliderRoiHBot"),
  roiHBotVal: document.getElementById("roiHBotVal"),

  telSpeed: document.getElementById("telSpeed"),
  telSteer: document.getElementById("telSteer"),
  kvSign: document.getElementById("kvSign"),
  uartPacket: document.getElementById("uartPacket"),

  cCam1: document.getElementById("canvasCam1"),
  cCam2: document.getElementById("canvasCam2"),
  imgCam1: document.getElementById("imgCam1"),
  imgCam2: document.getElementById("imgCam2"),

  tabBtns: document.querySelectorAll("[data-tab]"),
  liveView: document.getElementById("liveView"),
  blackboxView: document.getElementById("blackboxView"),
  bbTopPill: document.getElementById("bbTopPill"),
  bbTopText: document.getElementById("bbTopText"),
  btnBbRec: document.getElementById("btnBbRec"),
  chkBbAuto: document.getElementById("chkBbAuto"),
  bbStatusMsg: document.getElementById("bbStatusMsg"),
  btnBbRefresh: document.getElementById("btnBbRefresh"),
  bbSessionCount: document.getElementById("bbSessionCount"),
  bbSessionList: document.getElementById("bbSessionList"),
  bbEmptyDetail: document.getElementById("bbEmptyDetail"),
  bbDetailWrap: document.getElementById("bbDetailWrap"),
  bbDetailTitle: document.getElementById("bbDetailTitle"),
  bbDetailMeta: document.getElementById("bbDetailMeta"),
  btnBbExport: document.getElementById("btnBbExport"),
  btnBbDelete: document.getElementById("btnBbDelete"),
  btnBbToggleRaw: document.getElementById("btnBbToggleRaw"),
  bbImgCam1: document.getElementById("bbImgCam1"),
  bbImgWarp: document.getElementById("bbImgWarp"),
  bbImgCam2: document.getElementById("bbImgCam2"),
  bbNoCam1: document.getElementById("bbNoCam1"),
  bbNoWarp: document.getElementById("bbNoWarp"),
  bbNoCam2: document.getElementById("bbNoCam2"),
  bbChart: document.getElementById("bbChart"),
  btnBbFirst: document.getElementById("btnBbFirst"),
  btnBbPrev: document.getElementById("btnBbPrev"),
  btnBbPlay: document.getElementById("btnBbPlay"),
  btnBbNext: document.getElementById("btnBbNext"),
  btnBbLast: document.getElementById("btnBbLast"),
  bbSpeedSelect: document.getElementById("bbSpeedSelect"),
  bbTimeLabel: document.getElementById("bbTimeLabel"),
  bbTickLabel: document.getElementById("bbTickLabel"),
  btnBbNextAnomaly: document.getElementById("btnBbNextAnomaly"),
  bbCurPacket: document.getElementById("bbCurPacket"),
  bbTelemetryGrid: document.getElementById("bbTelemetryGrid"),
  bbAnomalyCount: document.getElementById("bbAnomalyCount"),
  bbAnomalyList: document.getElementById("bbAnomalyList"),
  bbNotesInput: document.getElementById("bbNotesInput"),
  bbNoteSavedMsg: document.getElementById("bbNoteSavedMsg"),
  btnBbSaveNote: document.getElementById("btnBbSaveNote"),
};

function syncMobileCam() {
  el.panelCam1.classList.toggle("mobile-hidden", state.mobileCamIndex !== 1);
  el.panelCam2.classList.toggle("mobile-hidden", state.mobileCamIndex !== 2);
  el.btnMobileCamToggle.textContent = state.mobileCamIndex === 1 ? "Xem Cam 2" : "Xem Cam 1";
}

function setDriveMode(mode, send = true) {
  state.driveMode = mode;
  if (mode === "AUTO") {
    state.manualSpeed = 0;
    state.manualSteer = 0;
  }
  el.driveBtns.forEach((b) => b.classList.toggle("active", b.dataset.drive === mode));
  if (send) sendControl({ drive_mode: mode });
}

function setCarRunning(running, send = true) {
  state.carRunning = running;
  if (!running) {
    state.manualSpeed = 0;
    state.manualSteer = 0;
    state.keysPressed.clear();
    syncPads();
  }
  el.btnPowerToggle.textContent = running ? "Kết thúc" : "Bắt đầu";
  el.btnPowerToggle.classList.toggle("running", running);
  el.statusText.textContent = running ? "Sẵn sàng chạy" : "Đã dừng";
  if (send) {
    sendControl({
      car_running: running,
      manual_speed: 0,
      manual_steer: 0,
    });
  }
}

async function sendControl(payload) {
  try {
    await fetch("/api/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    // Bo qua khi server tam ngat
  }
}

function updateConnectionUI(serverOnline, carConnected) {
  const prevConnected = state.carConnected;
  state.serverOnline = serverOnline;
  state.carConnected = carConnected;
  el.connState.classList.remove("online", "offline", "server-down");
  if (!serverOnline) {
    el.connState.classList.add("server-down");
    el.connText.textContent = "Mất kết nối server";
  } else if (carConnected) {
    el.connState.classList.add("online");
    el.connText.textContent = "Đã kết nối xe";
  } else {
    el.connState.classList.add("offline");
    el.connText.textContent = "Chưa kết nối xe";
  }

  // Bat/tat luong MJPEG that khi xe ket noi/ngat ket noi
  if (carConnected && !prevConnected) {
    const ts = Date.now();
    el.imgCam1.src = `/stream/cam1.mjpg?t=${ts}`;
    el.imgCam2.src = `/stream/cam2.mjpg?t=${ts}`;
  } else if (!carConnected && prevConnected) {
    el.imgCam1.removeAttribute("src");
    el.imgCam2.removeAttribute("src");
  }
  el.imgCam1.classList.toggle("hidden", !carConnected);
  el.imgCam2.classList.toggle("hidden", !carConnected);
  el.cCam1.classList.toggle("hidden", carConnected);
  el.cCam2.classList.toggle("hidden", carConnected);
}

async function pollServerStatus() {
  try {
    const res = await fetch("/api/status", { cache: "no-store" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const data = await res.json();
    updateConnectionUI(Boolean(data.server_online), Boolean(data.car_connected));

    if (data.roi) {
      state.roi = { ...state.roi, ...data.roi };
      el.sliderRoiWTop.value = state.roi.w_top;
      el.roiWTopVal.textContent = state.roi.w_top;
      el.sliderRoiHTop.value = state.roi.h_top;
      el.roiHTopVal.textContent = state.roi.h_top;
      el.sliderRoiWBot.value = state.roi.w_bot;
      el.roiWBotVal.textContent = state.roi.w_bot;
      el.sliderRoiHBot.value = state.roi.h_bot;
      el.roiHBotVal.textContent = state.roi.h_bot;
    }

    if (data.car_connected) {
      if (typeof data.car_running === "boolean" && data.car_running !== state.carRunning) {
        setCarRunning(data.car_running, false);
      }
      if (data.drive_mode && data.drive_mode !== state.driveMode) {
        setDriveMode(data.drive_mode, false);
      }
      state.speedPct = data.speed ?? 0;
      state.steerErr = data.steer ?? 0;
      state.cmd = data.cmd ?? "STOP";
      el.cam1Src.textContent = data.cam1_src || "CSI";
      el.cam2Src.textContent = data.cam2_src || "USB-0";
      el.telSpeed.textContent = `${state.speedPct}%`;
      const errStr = `${state.steerErr >= 0 ? "+" : ""}${Number(state.steerErr).toFixed(2)}`;
      el.telSteer.textContent = `${errStr} (${data.servo_deg ?? 90}°)`;
      el.uartPacket.textContent = data.packet || "<0,+0.00,STOP>";
      renderDetectedSigns(data.signs || []);
    }

    if (typeof data.blackbox_auto === "boolean") {
      state.blackboxAuto = data.blackbox_auto;
      el.chkBbAuto.checked = data.blackbox_auto;
    }
    if (typeof data.blackbox_manual === "boolean") {
      state.blackboxManual = data.blackbox_manual;
      el.btnBbRec.classList.toggle("active", data.blackbox_manual);
      el.btnBbRec.textContent = data.blackbox_manual ? "Dừng ghi" : "Ghi Hộp đen";
    }
    const bb = data.blackbox || {};
    const prevRec = state.blackboxRecording;
    state.blackboxRecording = Boolean(data.car_connected && bb.recording);
    el.bbTopPill.classList.toggle("hidden", !state.blackboxRecording);
    if (state.blackboxRecording) {
      el.bbTopText.textContent = `REC ${Number(bb.duration_s || 0).toFixed(0)}s`;
    }
    if (prevRec && !state.blackboxRecording && state.activeTab === "blackbox") {
      loadBlackBoxSessions();
    }
  } catch {
    updateConnectionUI(false, false);
  }
}
pollServerStatus();
setInterval(pollServerStatus, 250);

el.driveBtns.forEach((b) => b.addEventListener("click", () => setDriveMode(b.dataset.drive)));
el.btnMobileCamToggle.addEventListener("click", () => {
  state.mobileCamIndex = state.mobileCamIndex === 1 ? 2 : 1;
  syncMobileCam();
});
syncMobileCam();

el.btnSwitchCam1.addEventListener("click", () => {
  sendControl({ action: "switch_cam1" });
});
el.btnSwitchCam2.addEventListener("click", () => {
  sendControl({ action: "switch_cam2" });
});
el.btnSwapCam.addEventListener("click", () => {
  sendControl({ action: "swap_cams" });
});

el.sliderMaxSpeed.addEventListener("input", (e) => {
  state.maxSpeed = Number(e.target.value);
  el.maxSpeedLabel.textContent = `${state.maxSpeed}%`;
  sendControl({ max_speed: state.maxSpeed });
});
el.sliderFov.addEventListener("input", (e) => {
  state.fovPct = Number(e.target.value);
  el.fovVal.textContent = `${state.fovPct}%`;
  sendControl({ fov_pct: state.fovPct });
});

function bindRoiSlider(slider, labelEl, key) {
  slider.addEventListener("input", (e) => {
    const val = Number(e.target.value);
    state.roi[key] = val;
    labelEl.textContent = val;
    sendControl({ roi: { [key]: val } });
  });
}
bindRoiSlider(el.sliderRoiWTop, el.roiWTopVal, "w_top");
bindRoiSlider(el.sliderRoiHTop, el.roiHTopVal, "h_top");
bindRoiSlider(el.sliderRoiWBot, el.roiWBotVal, "w_bot");
bindRoiSlider(el.sliderRoiHBot, el.roiHBotVal, "h_bot");

el.btnPowerToggle.addEventListener("click", () => setCarRunning(!state.carRunning));
el.btnDpadStop.addEventListener("click", () => setCarRunning(false));

// Keyboard & D-Pad
const keyMap = {
  w: "w", arrowup: "w",
  s: "s", arrowdown: "s",
  a: "a", arrowleft: "a",
  d: "d", arrowright: "d",
};

function syncPads() {
  el.padBtns.forEach((b) => b.classList.toggle("pressed", state.keysPressed.has(b.dataset.key)));
}

window.addEventListener("keydown", (e) => {
  const tag = (e.target && e.target.tagName) || "";
  if (tag === "TEXTAREA" || tag === "INPUT" || tag === "SELECT") {
    return;
  }
  const k = e.key.toLowerCase();

  if (state.activeTab === "blackbox") {
    if (k === " ") {
      e.preventDefault();
      toggleBbPlay();
    } else if (k === "arrowleft") {
      e.preventDefault();
      stepBbIndex(-1);
    } else if (k === "arrowright") {
      e.preventDefault();
      stepBbIndex(1);
    }
    return;
  }

  if (k === " ") {
    e.preventDefault();
    setCarRunning(false);
    return;
  }
  if (keyMap[k]) {
    e.preventDefault();
    if (!state.carRunning) setCarRunning(true);
    if (state.driveMode !== "MANUAL") setDriveMode("MANUAL");
    state.keysPressed.add(keyMap[k]);
    syncPads();
  }
});

window.addEventListener("keyup", (e) => {
  const k = e.key.toLowerCase();
  if (keyMap[k]) {
    state.keysPressed.delete(keyMap[k]);
    syncPads();
  }
});

el.padBtns.forEach((btn) => {
  const k = btn.dataset.key;
  btn.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    if (!state.carRunning) setCarRunning(true);
    if (state.driveMode !== "MANUAL") setDriveMode("MANUAL");
    state.keysPressed.add(k);
    syncPads();
  });
  const up = () => {
    state.keysPressed.delete(k);
    syncPads();
  };
  btn.addEventListener("pointerup", up);
  btn.addEventListener("pointerleave", up);
  btn.addEventListener("pointercancel", up);
});

// Canvas Preview
function drawCam1(ctx, w, h, curve) {
  ctx.fillStyle = "#101216";
  ctx.fillRect(0, 0, w, h);
  const shift = curve * 70;
  ctx.fillStyle = "#1a1e24";
  ctx.beginPath();
  ctx.moveTo(w * 0.35 + shift * 0.5, h * 0.32);
  ctx.lineTo(w * 0.65 + shift * 0.5, h * 0.32);
  ctx.lineTo(w * 0.9, h);
  ctx.lineTo(w * 0.1, h);
  ctx.closePath();
  ctx.fill();

  ctx.strokeStyle = "#cbd5e1";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(w * 0.15, h);
  ctx.lineTo(w * 0.37 + shift * 0.5, h * 0.32);
  ctx.moveTo(w * 0.85, h);
  ctx.lineTo(w * 0.63 + shift * 0.5, h * 0.32);
  ctx.stroke();

  ctx.strokeStyle = "#3b82f6";
  ctx.setLineDash([6, 6]);
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(w * 0.5, h);
  ctx.lineTo(w * 0.5 + shift * 0.5, h * 0.32);
  ctx.stroke();
  ctx.setLineDash([]);

  // Ve khung ROI vang (dong bo 480x240)
  const rxTop = (state.roi.w_top / 480) * w;
  const ryTop = (state.roi.h_top / 240) * h;
  const rxBot = (state.roi.w_bot / 480) * w;
  const ryBot = (state.roi.h_bot / 240) * h;
  const roiPts = [
    [rxTop, ryTop],
    [w - rxTop, ryTop],
    [w - rxBot, ryBot],
    [rxBot, ryBot],
  ];
  ctx.strokeStyle = "#facc15";
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  roiPts.forEach(([px, py], i) => (i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py)));
  ctx.closePath();
  ctx.stroke();
  ctx.fillStyle = "#ef4444";
  roiPts.forEach(([px, py]) => {
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.fill();
  });

  if (!state.carConnected) {
    ctx.fillStyle = "rgba(17, 18, 20, 0.72)";
    ctx.fillRect(8, 8, 175, 22);
    ctx.fillStyle = "#fcd34d";
    ctx.font = "11.5px sans-serif";
    ctx.fillText("Chưa kết nối xe (Mô phỏng)", 14, 23);
  }
}

function drawCam2(ctx, w, h, signs) {
  ctx.fillStyle = "#101216";
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = "#171a20";
  ctx.fillRect(0, h * 0.58, w, h * 0.42);

  ctx.strokeStyle = "#eab308";
  ctx.lineWidth = 2;
  ctx.font = "12px sans-serif";
  signs.forEach((s) => {
    const [rx, ry, bw, bh] = s.box;
    const x = w * rx;
    const y = h * ry;
    ctx.strokeRect(x, y, bw, bh);
    ctx.fillStyle = "#eab308";
    ctx.fillText(`${s.name} (${Math.round(s.conf * 100)}%)`, x, y - 6);
  });
}

let _lastSignsKey = "";
function renderDetectedSigns(signs) {
  const key = !signs || signs.length === 0
    ? ""
    : signs.map((s) => `${s.name}:${Math.round(s.conf * 100)}`).join("|");
  if (key === _lastSignsKey) return;
  _lastSignsKey = key;

  if (!key) {
    el.kvSign.innerHTML = '<span class="muted">Không có</span>';
    return;
  }
  el.kvSign.innerHTML = signs
    .map((s) => `<span class="sign-tag">${s.name} (${Math.round(s.conf * 100)}%)</span>`)
    .join("");
}

function stepManualControls(dt) {
  const gas = state.keysPressed.has("w");
  const brake = state.keysPressed.has("s");
  const left = state.keysPressed.has("a");
  const right = state.keysPressed.has("d");

  // Tăng tốc / Giảm tốc mượt kiểu đua xe
  const accelRate = 135; // %/s
  const brakeRate = 175; // %/s
  const coastRate = 85;  // %/s
  const revLimit = -Math.round(state.maxSpeed * 0.7);

  if (gas && !brake) {
    state.manualSpeed = Math.min(state.maxSpeed, state.manualSpeed + accelRate * dt);
  } else if (brake && !gas) {
    state.manualSpeed = Math.max(revLimit, state.manualSpeed - brakeRate * dt);
  } else {
    if (state.manualSpeed > 0) state.manualSpeed = Math.max(0, state.manualSpeed - coastRate * dt);
    else if (state.manualSpeed < 0) state.manualSpeed = Math.min(0, state.manualSpeed + coastRate * dt);
  }

  // Rẽ trái / phải mượt
  const steerRate = 4.2;
  const returnRate = 5.5;
  if (left && !right) {
    state.manualSteer = Math.max(-1.0, state.manualSteer - steerRate * dt);
  } else if (right && !left) {
    state.manualSteer = Math.min(1.0, state.manualSteer + steerRate * dt);
  } else {
    if (state.manualSteer > 0) state.manualSteer = Math.max(0, state.manualSteer - returnRate * dt);
    else if (state.manualSteer < 0) state.manualSteer = Math.min(0, state.manualSteer + returnRate * dt);
  }
}

let _lastManualSend = 0;
function tick(now) {
  const dt = state.lastTickMs ? Math.min(0.05, (now - state.lastTickMs) * 0.001) : 0.016;
  state.lastTickMs = now;
  state.simTime = now * 0.001;

  if (state.carRunning && state.driveMode === "MANUAL") {
    stepManualControls(dt);
    const mSpd = Math.round(state.manualSpeed);
    const mStr = Number(state.manualSteer.toFixed(2));
    if (now - _lastManualSend >= 50) {
      _lastManualSend = now;
      sendControl({ manual_speed: mSpd, manual_steer: mStr });
    }
  }

  // Khi chua ket noi xe, chay mo phong tren Canvas
  if (!state.carConnected) {
    if (!state.carRunning) {
      state.speedPct = 0;
      state.steerErr = 0;
      state.cmd = "STOP";
    } else if (state.driveMode === "MANUAL") {
      state.speedPct = Math.round(state.manualSpeed);
      state.steerErr = Number(state.manualSteer.toFixed(2));
      if (state.speedPct === 0 && Math.abs(state.steerErr) < 0.05) {
        state.cmd = "STOP";
      } else if (Math.abs(state.steerErr) >= 0.55) {
        state.cmd = state.steerErr > 0 ? "SHARP_R" : "SHARP_L";
      } else if (Math.abs(state.steerErr) >= 0.12) {
        state.cmd = "CURVE";
      } else {
        state.cmd = "NORM";
      }
    } else {
      state.curveVal = Math.sin(state.simTime * 0.8) * 0.32;
      state.speedPct = Math.min(state.maxSpeed, Math.round(75 - Math.abs(state.curveVal) * 20));
      state.steerErr = Number(Math.max(-1, Math.min(1, state.curveVal * 1.12)).toFixed(2));
      state.cmd = Math.abs(state.steerErr) >= 0.12 ? "CURVE" : "NORM";
    }

    const servoDeg = Math.round(Math.max(45, Math.min(135, 90 + state.steerErr * 60)));
    const errStr = `${state.steerErr >= 0 ? "+" : ""}${state.steerErr.toFixed(2)}`;
    el.telSpeed.textContent = `${state.speedPct}%`;
    el.telSteer.textContent = `${errStr} (${servoDeg}°)`;
    el.uartPacket.textContent = `<${state.speedPct},${errStr},${state.cmd}>`;

    drawCam1(el.cCam1.getContext("2d"), el.cCam1.width, el.cCam1.height, state.curveVal);
    drawCam2(el.cCam2.getContext("2d"), el.cCam2.width, el.cCam2.height, state.detectedSigns);
    renderDetectedSigns(state.detectedSigns);
  }

  requestAnimationFrame(tick);
}

requestAnimationFrame(tick);

// ============================================================================
// HOP DEN (BLACK BOX REPLAY & DIAGNOSTICS)
// ============================================================================
const bbState = {
  sessions: [],
  currentId: null,
  meta: null,
  telemetry: [],
  anomalies: [],
  index: 0,
  playing: false,
  speed: 1,
  showRawCam1: false,
  playTimer: null,
};

function escHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function switchTab(tab) {
  state.activeTab = tab;
  el.tabBtns.forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  el.liveView.classList.toggle("hidden", tab !== "live");
  el.blackboxView.classList.toggle("hidden", tab !== "blackbox");
  if (tab === "blackbox") {
    loadBlackBoxSessions();
  } else {
    stopBbPlay();
  }
}

el.tabBtns.forEach((b) => b.addEventListener("click", () => switchTab(b.dataset.tab)));

el.btnBbRec.addEventListener("click", () => {
  const next = !state.blackboxManual;
  state.blackboxManual = next;
  el.btnBbRec.classList.toggle("active", next);
  el.btnBbRec.textContent = next ? "Dừng ghi" : "Ghi Hộp đen";
  sendControl({ blackbox_manual: next });
});

el.chkBbAuto.addEventListener("change", (e) => {
  state.blackboxAuto = e.target.checked;
  sendControl({ blackbox_auto: state.blackboxAuto });
});

el.btnBbRefresh.addEventListener("click", () => loadBlackBoxSessions());

async function loadBlackBoxSessions() {
  try {
    el.bbStatusMsg.textContent = "Đang tải...";
    const res = await fetch("/api/blackbox/sessions", { cache: "no-store" });
    const data = await res.json();
    bbState.sessions = data.sessions || [];
    el.bbSessionCount.textContent = bbState.sessions.length;
    el.bbStatusMsg.textContent = "";
    renderSessionList();
    if (!bbState.currentId && bbState.sessions.length > 0) {
      selectSession(bbState.sessions[0].session_id);
    }
  } catch {
    el.bbStatusMsg.textContent = "Không tải được danh sách hộp đen";
  }
}

function renderSessionList() {
  if (!bbState.sessions.length) {
    el.bbSessionList.innerHTML = '<div class="bb-empty">Chưa có lần đi nào được ghi lại.</div>';
    return;
  }
  el.bbSessionList.innerHTML = bbState.sessions.map((s) => {
    const sm = s.summary || {};
    const anom = Number(sm.anomalies_count || 0);
    const badgeCls = anom === 0 ? "ok" : (anom <= 3 ? "warn" : "bad");
    const badgeTxt = anom === 0 ? "Ổn" : `${anom} sự cố`;
    const active = s.session_id === bbState.currentId ? " active" : "";
    const noteDot = s.notes ? " · Có ghi chú" : "";
    return `
      <div class="bb-session-item${active}" data-sid="${escHtml(s.session_id)}">
        <div class="bb-sess-top">
          <span>${escHtml(s.session_id)}</span>
          <span class="bb-badge ${badgeCls}">${badgeTxt}</span>
        </div>
        <div class="bb-sess-sub">
          <span>${Number(s.duration_s || 0).toFixed(1)}s · ${sm.frames || 0} ảnh</span>
          <span>${escHtml((s.config && s.config.drive_mode_initial) || "AUTO")}${noteDot}</span>
        </div>
      </div>`;
  }).join("");

  el.bbSessionList.querySelectorAll(".bb-session-item").forEach((item) => {
    item.addEventListener("click", () => selectSession(item.dataset.sid));
  });
}

async function selectSession(sid) {
  stopBbPlay();
  bbState.currentId = sid;
  renderSessionList();
  try {
    el.bbStatusMsg.textContent = `Đang mở ${sid}...`;
    const res = await fetch(`/api/blackbox/session/${encodeURIComponent(sid)}`, { cache: "no-store" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const data = await res.json();
    bbState.meta = data.meta || {};
    bbState.telemetry = data.telemetry || [];
    bbState.anomalies = data.anomalies || [];
    bbState.index = 0;

    el.bbEmptyDetail.classList.add("hidden");
    el.bbDetailWrap.classList.remove("hidden");
    el.bbDetailTitle.textContent = sid;

    const cfg = bbState.meta.config || {};
    const roi = cfg.roi || {};
    el.bbDetailMeta.textContent =
      ` · ${Number(bbState.meta.duration_s || 0).toFixed(1)}s · ${bbState.telemetry.length} ticks` +
      (cfg.fov_pct ? ` · FOV ${cfg.fov_pct}%` : "") +
      (roi.w_top !== undefined ? ` · ROI(${roi.w_top},${roi.h_top},${roi.w_bot},${roi.h_bot})` : "");

    el.bbNotesInput.value = bbState.meta.notes || "";
    el.bbNoteSavedMsg.textContent = "";
    renderAnomalies();
    renderBbAtIndex(0);
    el.bbStatusMsg.textContent = "";
  } catch {
    el.bbStatusMsg.textContent = "Lỗi đọc dữ liệu phiên";
  }
}

function renderAnomalies() {
  const list = bbState.anomalies;
  el.bbAnomalyCount.textContent = list.length;
  if (!list.length) {
    el.bbAnomalyList.innerHTML = '<div class="bb-empty">Không phát hiện sự cố bất thường.</div>';
    return;
  }
  el.bbAnomalyList.innerHTML = list.map((a, i) => `
    <div class="bb-anomaly-row" data-seq="${Number(a.seq || 0)}" data-idx="${i}">
      <span class="bb-anomaly-t">${Number(a.t || 0).toFixed(2)}s</span>
      <span class="bb-badge warn">${escHtml(a.type)}</span>
      <span>${escHtml(a.desc)}</span>
    </div>`).join("");

  el.bbAnomalyList.querySelectorAll(".bb-anomaly-row").forEach((rowEl) => {
    rowEl.addEventListener("click", () => {
      const seq = Number(rowEl.dataset.seq);
      const idx = bbState.telemetry.findIndex((r) => Number(r.seq) >= seq);
      if (idx >= 0) {
        stopBbPlay();
        renderBbAtIndex(idx);
      }
    });
  });
}

function setBbFrameImg(imgEl, noEl, fname) {
  if (fname && bbState.currentId) {
    const url = `/api/blackbox/session/${encodeURIComponent(bbState.currentId)}/frame/${encodeURIComponent(fname)}`;
    if (imgEl.dataset.curUrl !== url) {
      imgEl.dataset.curUrl = url;
      imgEl.src = url;
    }
    imgEl.classList.remove("hidden");
    noEl.classList.add("hidden");
  } else {
    imgEl.removeAttribute("src");
    imgEl.dataset.curUrl = "";
    imgEl.classList.add("hidden");
    noEl.classList.remove("hidden");
  }
}

function renderBbAtIndex(idx) {
  const tel = bbState.telemetry;
  if (!tel.length) {
    el.bbTimeLabel.textContent = "0.00s / 0.00s";
    el.bbTickLabel.textContent = "Tick 0 / 0";
    el.bbTelemetryGrid.innerHTML = "";
    drawBbChart();
    return;
  }
  bbState.index = Math.max(0, Math.min(tel.length - 1, idx));
  const row = tel[bbState.index];
  const totalT = Number(tel[tel.length - 1].t || 0);

  el.bbTimeLabel.textContent = `${Number(row.t || 0).toFixed(2)}s / ${totalT.toFixed(2)}s`;
  el.bbTickLabel.textContent = `Tick ${bbState.index + 1} / ${tel.length}`;
  el.bbCurPacket.textContent = row.packet || "<0,+0.00,STOP>";

  const imgs = row.resolved_imgs || {};
  const c1File = bbState.showRawCam1 ? (imgs.r1 || imgs.c1) : (imgs.c1 || imgs.r1);
  setBbFrameImg(el.bbImgCam1, el.bbNoCam1, c1File);
  setBbFrameImg(el.bbImgWarp, el.bbNoWarp, imgs.wp);
  setBbFrameImg(el.bbImgCam2, el.bbNoCam2, imgs.c2);

  const lane = row.lane || {};
  const corner = lane.corner || {};
  const junc = lane.junction || {};
  const ctrl = lane.ctrl || {};
  const signsTxt = (row.signs && row.signs.length)
    ? row.signs.map((s) => `${s.name}(${Math.round((s.conf || 0) * 100)}%)`).join(", ")
    : "Không";

  const pairs = [
    ["Chế độ / Lệnh", `${row.mode || "-"} / ${row.cmd || "-"}`],
    ["Tốc độ / Servo", `${row.speed ?? 0}% / ${row.servo ?? 90}°`],
    ["Err / Curve", `${Number(row.err || 0).toFixed(2)} / ${Number(row.curve || 0).toFixed(2)}`],
    ["Trạng thái làn", `${lane.state || "UNKNOWN"} (${Math.round((lane.conf || 0) * 100)}%)`],
    ["Observed / Mất", `${lane.observed ? "Có" : "MẤT"} (miss=${lane.missing_frames ?? 0})`],
    ["Lệch tâm / Góc", `${lane.e_px ?? 0}px / ${lane.heading_deg ?? 0}°`],
    ["Góc cua", `${corner.state || "NONE"} (${corner.dir || "-"} | ${corner.zone || "-"})`],
    ["Ngã rẽ / Ý định", `${junc.type || "NONE"} / ${corner.intent || "-"}`],
    ["PP / Stanley", `${ctrl.pp_deg ?? "-"}° / ${ctrl.stanley_deg ?? "-"}°`],
    ["Ánh sáng / Trễ", `${lane.lighting || "-"} / ${row.lane_age_ms ?? 0}ms`],
    ["Biển báo", signsTxt],
    ["Hành động", row.sign_action || (row.events && row.events.join(", ")) || "-"],
  ];

  el.bbTelemetryGrid.innerHTML = pairs.map(([k, v]) =>
    `<div class="bb-kv"><span>${escHtml(k)}</span><strong>${escHtml(v)}</strong></div>`
  ).join("");

  drawBbChart();
}

function drawBbChart() {
  const canvas = el.bbChart;
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.fillStyle = "#101215";
  ctx.fillRect(0, 0, w, h);

  const tel = bbState.telemetry;
  if (tel.length < 2) return;

  const padL = 8, padR = 8, padT = 16, padB = 18;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;
  const midY = padT + plotH * 0.5;

  // Duong giua (err = 0)
  ctx.strokeStyle = "#262b33";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, midY);
  ctx.lineTo(w - padR, midY);
  ctx.stroke();

  const n = tel.length;
  const xAt = (i) => padL + (i / (n - 1)) * plotW;

  // Dai mau trang thai lan duong o day
  const barY = h - 12;
  const stepW = Math.max(1, plotW / n + 0.5);
  for (let i = 0; i < n; i++) {
    const lane = tel[i].lane || {};
    let col = "#ef4444";
    if (lane.observed && lane.state === "TWO_EDGES") col = "#22c55e";
    else if (lane.observed) col = "#f59e0b";
    ctx.fillStyle = col;
    ctx.fillRect(xAt(i), barY, stepW, 7);
  }

  // Duong Toc do (%) mau xanh ngoc
  ctx.strokeStyle = "#34d399";
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const spd = Math.max(-100, Math.min(100, Number(tel[i].speed || 0)));
    const y = padT + plotH - ((spd + 100) / 200) * plotH;
    if (i === 0) ctx.moveTo(xAt(i), y);
    else ctx.lineTo(xAt(i), y);
  }
  ctx.stroke();

  // Duong Goc lai (err -1..+1) mau xanh duong
  ctx.strokeStyle = "#60a5fa";
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const err = Math.max(-1, Math.min(1, Number(tel[i].err || 0)));
    const y = midY - err * (plotH * 0.45);
    if (i === 0) ctx.moveTo(xAt(i), y);
    else ctx.lineTo(xAt(i), y);
  }
  ctx.stroke();

  // Danh dau su kien bat thuong
  for (let i = 0; i < n; i++) {
    const evs = tel[i].events;
    if (evs && evs.length) {
      const hasBad = evs.some((e) => e === "LANE_LOST" || e === "WATCHDOG_STOP" || e === "RECOVERY");
      ctx.fillStyle = hasBad ? "#ef4444" : "#f59e0b";
      ctx.beginPath();
      ctx.arc(xAt(i), 8, 3.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Con tro vi tri hien tai
  const cx = xAt(bbState.index);
  ctx.strokeStyle = "#f8fafc";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(cx, 2);
  ctx.lineTo(cx, h - 2);
  ctx.stroke();
}

function scrubChart(clientX) {
  const tel = bbState.telemetry;
  if (!tel.length) return;
  const rect = el.bbChart.getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / Math.max(1, rect.width)));
  const idx = Math.round(ratio * (tel.length - 1));
  stopBbPlay();
  renderBbAtIndex(idx);
}

let _chartDragging = false;
el.bbChart.addEventListener("pointerdown", (e) => {
  _chartDragging = true;
  el.bbChart.setPointerCapture(e.pointerId);
  scrubChart(e.clientX);
});
el.bbChart.addEventListener("pointermove", (e) => {
  if (_chartDragging) scrubChart(e.clientX);
});
el.bbChart.addEventListener("pointerup", () => { _chartDragging = false; });
el.bbChart.addEventListener("pointercancel", () => { _chartDragging = false; });

function stepBbIndex(delta) {
  stopBbPlay();
  renderBbAtIndex(bbState.index + delta);
}

function stopBbPlay() {
  bbState.playing = false;
  if (bbState.playTimer) {
    clearTimeout(bbState.playTimer);
    bbState.playTimer = null;
  }
  el.btnBbPlay.textContent = "Phát";
}

function scheduleNextBbTick() {
  if (!bbState.playing) return;
  const tel = bbState.telemetry;
  if (bbState.index >= tel.length - 1) {
    stopBbPlay();
    return;
  }
  const curT = Number(tel[bbState.index].t || 0);
  const nxtT = Number(tel[bbState.index + 1].t || curT + 0.05);
  const dtMs = Math.max(15, Math.min(400, ((nxtT - curT) * 1000) / bbState.speed));
  bbState.playTimer = setTimeout(() => {
    renderBbAtIndex(bbState.index + 1);
    scheduleNextBbTick();
  }, dtMs);
}

function toggleBbPlay() {
  if (!bbState.telemetry.length) return;
  if (bbState.playing) {
    stopBbPlay();
  } else {
    if (bbState.index >= bbState.telemetry.length - 1) {
      renderBbAtIndex(0);
    }
    bbState.playing = true;
    el.btnBbPlay.textContent = "Tạm dừng";
    scheduleNextBbTick();
  }
}

el.btnBbFirst.addEventListener("click", () => { stopBbPlay(); renderBbAtIndex(0); });
el.btnBbLast.addEventListener("click", () => { stopBbPlay(); renderBbAtIndex(bbState.telemetry.length - 1); });
el.btnBbPrev.addEventListener("click", () => stepBbIndex(-1));
el.btnBbNext.addEventListener("click", () => stepBbIndex(1));
el.btnBbPlay.addEventListener("click", toggleBbPlay);
el.bbSpeedSelect.addEventListener("change", (e) => {
  bbState.speed = Number(e.target.value) || 1;
});

el.btnBbToggleRaw.addEventListener("click", () => {
  bbState.showRawCam1 = !bbState.showRawCam1;
  el.btnBbToggleRaw.textContent = bbState.showRawCam1 ? "Xem ảnh vẽ làn" : "Xem ảnh gốc";
  renderBbAtIndex(bbState.index);
});

el.btnBbNextAnomaly.addEventListener("click", () => {
  const tel = bbState.telemetry;
  if (!tel.length) return;
  const curSeq = Number(tel[bbState.index].seq || 0);
  const nxt = bbState.anomalies.find((a) => Number(a.seq) > curSeq) || bbState.anomalies[0];
  if (!nxt) return;
  const idx = tel.findIndex((r) => Number(r.seq) >= Number(nxt.seq));
  if (idx >= 0) {
    stopBbPlay();
    renderBbAtIndex(idx);
  }
});

el.btnBbSaveNote.addEventListener("click", async () => {
  if (!bbState.currentId) return;
  try {
    const res = await fetch(`/api/blackbox/session/${encodeURIComponent(bbState.currentId)}/note`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ notes: el.bbNotesInput.value }),
    });
    if (!res.ok) throw new Error("HTTP " + res.status);
    el.bbNoteSavedMsg.textContent = "Đã lưu ghi chú";
    loadBlackBoxSessions();
  } catch {
    el.bbNoteSavedMsg.textContent = "Lỗi khi lưu ghi chú";
  }
});

el.btnBbExport.addEventListener("click", () => {
  if (!bbState.currentId) return;
  window.location.href = `/api/blackbox/session/${encodeURIComponent(bbState.currentId)}/export`;
});

el.btnBbDelete.addEventListener("click", async () => {
  const sid = bbState.currentId;
  if (!sid) return;
  try {
    const res = await fetch(`/api/blackbox/session/${encodeURIComponent(sid)}/delete`, { method: "POST" });
    const data = await res.json();
    if (!res.ok) {
      el.bbStatusMsg.textContent = data.error || "Không xóa được phiên đang ghi";
      return;
    }
    bbState.currentId = null;
    el.bbDetailWrap.classList.add("hidden");
    el.bbEmptyDetail.classList.remove("hidden");
    loadBlackBoxSessions();
  } catch {
    el.bbStatusMsg.textContent = "Lỗi khi xóa phiên";
  }
});
