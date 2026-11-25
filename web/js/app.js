const state = {
  mobileCamIndex: 1,
  driveMode: "AUTO",
  carRunning: false,
  carConnected: false,
  serverOnline: true,
  cam1Src: "CSI",
  cam2Src: "USB-0",
  fovPct: 100,
  maxSpeed: 85,

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

  telSpeed: document.getElementById("telSpeed"),
  telSteer: document.getElementById("telSteer"),
  kvSign: document.getElementById("kvSign"),
  uartPacket: document.getElementById("uartPacket"),

  cCam1: document.getElementById("canvasCam1"),
  cCam2: document.getElementById("canvasCam2"),
  imgCam1: document.getElementById("imgCam1"),
  imgCam2: document.getElementById("imgCam2"),
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
  const k = e.key.toLowerCase();
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
