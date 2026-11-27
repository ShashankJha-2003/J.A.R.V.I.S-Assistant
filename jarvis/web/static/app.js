/* J.A.R.V.I.S. Real-Time Expressive 360° Avatar + Direct Real-Video Talking Lips
 * - All 10 expressions active while listening, thinking, and talking (plus manual preview strip).
 * - Face & lips stay 100% static on Neutral standby AND throughout all 360° manual rotation.
 * - ONLY when J.A.R.V.I.S. is actively speaking, plays the exact chronological 30fps real-video
 *   mouth motion from `talking me.mp4` composited cleanly into the lower-face over any active expression
 *   (zero double-chin, zero seam, zero synthetic viseme jumping).
 */

(() => {
  const $ = (id) => document.getElementById(id);
  const canvas = $("avatarCanvas");
  const ctx = canvas.getContext("2d");

  const ASSET_VER = "31";
  const TURNTABLE_COUNT = 72;
  const IS_STATIC_DEMO =
    Boolean(window.JARVIS_STATIC_MODE) ||
    /github\.io$/i.test(location.hostname) ||
    location.protocol === "file:";
  const ASSET_BASE = IS_STATIC_DEMO ? "./" : "/static/";

  const EXPRESSIONS = [
    "neutral",
    "listening",
    "thinking",
    "smiling",
    "happy",
    "laughing",
    "serious",
    "sad",
    "angry",
    "surprised",
  ];

  const MOOD_PROFILE = {
    neutral: { hex: "#22d3ee", rgb: "34, 211, 238", tint: "rgba(0,0,0,0)", dur: 420, tilt: 0.000, zoom: 1.000, driftX: 0.0, driftY: 0.0 },
    listening: { hex: "#38bdf8", rgb: "56, 189, 248", tint: "rgba(56,189,248,0.018)", dur: 380, tilt: 0.004, zoom: 1.008, driftX: 0.6, driftY: -0.8 },
    thinking: { hex: "#a78bfa", rgb: "167, 139, 250", tint: "rgba(139,92,246,0.03)", dur: 480, tilt: -0.008, zoom: 1.004, driftX: -1.1, driftY: -1.3 },
    smiling: { hex: "#34d399", rgb: "52, 211, 153", tint: "rgba(52,211,153,0.018)", dur: 400, tilt: 0.003, zoom: 1.007, driftX: 0.4, driftY: -0.6 },
    happy: { hex: "#fbbf24", rgb: "251, 191, 36", tint: "rgba(251,191,36,0.025)", dur: 360, tilt: 0.006, zoom: 1.012, driftX: 0.5, driftY: -1.2 },
    laughing: { hex: "#f59e0b", rgb: "245, 158, 11", tint: "rgba(245,158,11,0.035)", dur: 320, tilt: 0.008, zoom: 1.014, driftX: 0.4, driftY: -1.8 },
    serious: { hex: "#94a3b8", rgb: "148, 163, 184", tint: "rgba(15,23,42,0.065)", dur: 460, tilt: 0.000, zoom: 1.012, driftX: 0.0, driftY: 0.3 },
    sad: { hex: "#60a5fa", rgb: "96, 165, 250", tint: "rgba(30,58,138,0.08)", dur: 560, tilt: 0.007, zoom: 0.996, driftX: -0.5, driftY: 1.8 },
    angry: { hex: "#ef4444", rgb: "239, 68, 68", tint: "rgba(185,28,28,0.06)", dur: 380, tilt: -0.005, zoom: 1.016, driftX: 0.0, driftY: 0.6 },
    surprised: { hex: "#e879f9", rgb: "232, 121, 249", tint: "rgba(217,70,239,0.025)", dur: 240, tilt: -0.004, zoom: 1.018, driftX: 0.0, driftY: -1.8 },
  };

  const DEFAULT_MOUTH_MANIFEST = {
    version: 15,
    fps: 30,
    baseWidth: 600,
    baseHeight: 848,
    crop: { x: 190, y: 358, w: 220, h: 174 },
    frames: [
      8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301,
    ],
    reel: [
      284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301,
      8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 22, 19, 16, 14, 12, 10, 8,
    ],
    closedFrame: 8,
  };

  // Offscreen compositing buffers
  const portraitCanvas = document.createElement("canvas");
  const portraitCtx = portraitCanvas.getContext("2d");
  const splitCanvas = document.createElement("canvas");
  const splitCtx = splitCanvas.getContext("2d");
  const lowerFaceCanvas = document.createElement("canvas");
  const lowerFaceCtx = lowerFaceCanvas.getContext("2d");
  const mouthCanvas = document.createElement("canvas");
  const mouthCtx = mouthCanvas.getContext("2d");
  const blinkCanvas = document.createElement("canvas");
  const blinkCtx = blinkCanvas.getContext("2d");

  // =========================================================================
  // 1. Math Utilities: Critically-Damped Spring + Smoothstep + Organic Noise
  // =========================================================================
  function smoothstep(x) {
    const t = Math.max(0, Math.min(1, x));
    return t * t * (3 - 2 * t);
  }

  function smootherstep(x) {
    const t = Math.max(0, Math.min(1, x));
    return t * t * t * (t * (t * 6 - 15) + 10);
  }

  function springStep(obj, key, target, smoothTime, dt) {
    const st = Math.max(0.001, smoothTime);
    const omega = 2.0 / st;
    const x = omega * dt;
    const exp = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x);
    const change = obj[key] - target;
    const vKey = key + "Vel";
    const vel = obj[vKey] || 0;
    const temp = (vel + omega * change) * dt;
    obj[vKey] = (vel - omega * temp) * exp;
    obj[key] = target + (change + temp) * exp;
    return obj[key];
  }

  function organicNoise(t, seed = 0) {
    return (
      Math.sin(t * 0.71 + seed) * 0.50 +
      Math.sin(t * 1.63 + seed * 2.1) * 0.32 +
      Math.sin(t * 2.89 + seed * 0.7) * 0.18
    );
  }

  // =========================================================================
  // 2. Avatar State (All 10 Expressions + Static Neutral 360° + Video Lips)
  // =========================================================================
  const avatar = {
    images: {},
    loaded: {},
    turntable: new Array(TURNTABLE_COUNT).fill(null),
    turntableLoaded: 0,

    // Exact chronological 30fps video mouth frames from talking me.mp4
    mouthManifest: DEFAULT_MOUTH_MANIFEST,
    mouthSprites: {},
    mouthLoadedCount: 0,

    // Visual UI Mode: 'face' | 'orb' | 'halo' | 'pill'
    uiMode: "face",
    cardTimer: null,
    lastOpenedUrl: "",
    lastOpenedAt: 0,

    // Manual 360° Turntable Rotation (100% static closed neutral across all 72 angles)
    targetAngleDeg: 0,
    angleDeg: 0,
    angleVel: 0,
    isDragging: false,
    dragStartX: 0,
    dragLastX: 0,
    dragMovedPx: 0,

    // Clean 2-state expression transition
    prevExpr: "neutral",
    targetExpr: "neutral",
    transStart: 0,
    transDur: 420,
    exprHoldUntil: 0,

    // Runtime voice & WebSocket
    state: "idle",
    ws: null,
    connected: false,
    speakEnabled: true,
    handsFree: false,
    pttActive: false,
    localVoiceActive: false,
    pausedForSpeech: false,
    pendingToken: null,
    recognition: null,
    audioCtx: null,
    micAnalyser: null,
    ttsAnalyser: null,
    activeAudioEl: null,
    freqBins: new Uint8Array(64),
    timeBuf: new Uint8Array(256),
    agcPeakRms: 0.16,

    // Direct 30fps real-video playback state (ONLY moves while AI is actively speaking!)
    videoLips: {
      reelPos: 0,       // continuous frame index inside manifest.reel (advances at exact 30 FPS)
      mix: 0,           // 0 = pure expression/neutral, 1 = upper expression + real-video talking lower-face
      silenceSec: 0,    // brief hold across tiny inter-syllable dips before closing on real pauses
    },

    anim: {
      tilt: 0, tiltVel: 0,
      zoom: 1, zoomVel: 0,
      x: 0, xVel: 0,
      y: 0, yVel: 0,
      nod: 0, nodVel: 0,
      audioLevel: 0, audioLevelVel: 0,
    },

    // Natural eye blinking & micro-life (paused when idle static neutral or rotating)
    blinkAmount: 0,
    blinkStartAt: 0,
    blinkDur: 180,
    nextBlinkAt: performance.now() + 2800,
    doubleBlinkPending: false,
    laughEnergy: 0,
    userSpeakingSince: 0,
    userLastLoudAt: 0,
    nodTarget: 0,
    nodResetAt: 0,
    lastFrameAt: performance.now(),
    elapsedSec: 0,
    toolTimer: null,
    turnBuffer: "",
    turnTimer: null,
    turnSource: "ptt_voice",
    sttLang: "en-IN",
  };

  // Preload all 10 calibrated expressions + blinking
  [...EXPRESSIONS, "blinking"].forEach((name) => {
    const img = new Image();
    img.src = `${ASSET_BASE}avatar/${name}.jpg?v=${ASSET_VER}`;
    img.onload = () => {
      avatar.images[name] = img;
      avatar.loaded[name] = true;
    };
  });

  // Preload all 72 closed-neutral 360° turntable frames (100% static closed lips at every angle)
  for (let i = 0; i < TURNTABLE_COUNT; i++) {
    const tImg = new Image();
    tImg.src = `${ASSET_BASE}turntable/frame_${String(i).padStart(2, "0")}.jpg?v=${ASSET_VER}`;
    tImg.onload = () => {
      avatar.turntable[i] = tImg;
      avatar.turntableLoaded++;
    };
  }

  // Preload all 35 exact chronological 30fps video mouth frames from talking me.mp4
  function preloadMouthSprites(manifest) {
    avatar.mouthManifest = manifest || DEFAULT_MOUTH_MANIFEST;
    const frames = avatar.mouthManifest.frames || DEFAULT_MOUTH_MANIFEST.frames;
    for (const fi of frames) {
      if (avatar.mouthSprites[fi]) continue;
      const img = new Image();
      img.src = `${ASSET_BASE}mouth_seq/f_${String(fi).padStart(3, "0")}.jpg?v=${ASSET_VER}`;
      img.onload = () => {
        avatar.mouthSprites[fi] = img;
        avatar.mouthLoadedCount++;
      };
    }
  }

  preloadMouthSprites(DEFAULT_MOUTH_MANIFEST);
  fetch(`${ASSET_BASE}mouth_seq/manifest.json?v=${ASSET_VER}`)
    .then((r) => (r.ok ? r.json() : null))
    .then((m) => {
      if (m && Array.isArray(m.reel) && m.crop) preloadMouthSprites(m);
    })
    .catch(() => { });

  // =========================================================================
  // 3. Expression Transitions, Blinking & Conversational Nods
  // =========================================================================
  function triggerBlink(soft = false) {
    const now = performance.now();
    if (!avatar.loaded["blinking"] || avatar.blinkStartAt > 0) return;
    if (avatar.targetExpr === "laughing" || avatar.targetExpr === "surprised") return;
    avatar.blinkStartAt = now;
    avatar.blinkDur = soft ? 155 : 185;
  }

  function triggerNod(strength = 3.8, durationMs = 230) {
    avatar.nodTarget = strength;
    avatar.nodResetAt = performance.now() + durationMs;
  }

  function inferInterimExpression(text) {
    const low = (text || "").toLowerCase();
    if (!low.trim()) return null;
    if (/\b(haha|hehe|lol|funny|joke|hilarious|laugh)\b/.test(low)) return "laughing";
    if (/\b(angry|mad|furious|hate|stupid|frustrat|annoying|shut up)\b/.test(low)) return "angry";
    if (/\b(sad|lonely|cry|bad day|feel down|upset|sorry|depress|tired)\b/.test(low)) return "sad";
    if (/\b(wow|whoa|omg|no way|shock|unbelievable|surprise|really\?)\b/.test(low)) return "surprised";
    if (/\b(shutdown|restart|danger|warning|serious|urgent|security|lock)\b/.test(low)) return "serious";
    if (/\b(awesome|amazing|fantastic|great|love|congrats|happy|thank|thanks)\b/.test(low)) return "happy";
    if (/\b(hello|hi\b|hey\b|good morning|good evening|smile)\b/.test(low)) return "smiling";
    if (/\b(think|why|how does|explain|calculate|quantum|search)\b/.test(low)) return "thinking";
    return null;
  }

  function setExpression(expr, holdMs = 0) {
    if (!EXPRESSIONS.includes(expr)) expr = "neutral";
    const now = performance.now();
    if (holdMs > 0) avatar.exprHoldUntil = now + holdMs;

    if (expr !== avatar.targetExpr) {
      const oldExpr = avatar.targetExpr;
      avatar.prevExpr = oldExpr;
      avatar.targetExpr = expr;
      avatar.transStart = now;
      avatar.transDur = (MOOD_PROFILE[expr] || MOOD_PROFILE.neutral).dur;

      const isGazeShift = oldExpr === "thinking" || expr === "thinking";
      const isBigMoodJump =
        (oldExpr === "angry" || oldExpr === "sad" || oldExpr === "serious") &&
        (expr === "smiling" || expr === "happy" || expr === "neutral");
      if (isGazeShift || isBigMoodJump || Math.random() < 0.25) {
        triggerBlink(true);
      }
      if (expr === "laughing") avatar.laughEnergy = 1.0;
      if (expr === "smiling" || expr === "listening" || expr === "happy") {
        triggerNod(2.4, 200);
      }
    }

    $("exprLabel").textContent = expr.toUpperCase();
    document.querySelectorAll(".expr-pill").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.expr === expr);
    });
  }

  function setState(newState) {
    avatar.state = newState || "idle";
    const labels = {
      idle: avatar.handsFree ? "HANDS-FREE READY" : "STANDBY",
      listening: "LISTENING...",
      processing: "THINKING...",
      executing: "EXECUTING...",
      speaking: "SPEAKING",
      awaiting_confirmation: "CONFIRM",
      error: "ALERT",
    };
    $("stateLabel").textContent = labels[avatar.state] || avatar.state.toUpperCase();
    $("btnTalk").classList.toggle("listening", avatar.pttActive || avatar.state === "listening");
  }

  // =========================================================================
  // 4. Direct Real-Video 30 FPS Mouth Playback + Expression/Rotation Physics
  // =========================================================================
  let vw = window.innerWidth;
  let vh = window.innerHeight;

  function resize() {
    vw = window.innerWidth;
    vh = window.innerHeight;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(vw * dpr);
    canvas.height = Math.round(vh * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  window.addEventListener("resize", resize);

  function updatePhysics(now) {
    const dt = Math.min(0.05, Math.max(0.001, (now - avatar.lastFrameAt) / 1000));
    avatar.lastFrameAt = now;
    avatar.elapsedSec += dt;
    const t = avatar.elapsedSec;

    // 1. Manual 360° turntable rotation spring (100% static unless manually rotated)
    const rotSmooth = avatar.isDragging ? 0.045 : 0.14;
    springStep(avatar, "angleDeg", avatar.targetAngleDeg, rotSmooth, dt);

    const normDeg = ((avatar.angleDeg % 360) + 360) % 360;
    const distFromFront = Math.min(normDeg, 360 - normDeg);
    const resetBtn = $("btnResetAngle");
    if (resetBtn) {
      if (distFromFront > 3) {
        resetBtn.classList.remove("hidden");
        resetBtn.textContent = `⟲ ${Math.round(normDeg)}° Front`;
      } else {
        resetBtn.classList.add("hidden");
      }
    }

    // 2. Audio level & direct chronological real-video mouth playback (ONLY when speaking & facing front!)
    let tgtLevel = 0;
    let voiceSounding = false;
    const facingFront = distFromFront <= 4 && !avatar.isDragging;

    if (avatar.state === "speaking") {
      const hasRealAudio = !!(avatar.ttsAnalyser && avatar.activeAudioEl && !avatar.activeAudioEl.paused);
      if (hasRealAudio) {
        avatar.ttsAnalyser.getByteTimeDomainData(avatar.timeBuf);
        avatar.ttsAnalyser.getByteFrequencyData(avatar.freqBins);
        let sq = 0;
        for (let i = 0; i < avatar.timeBuf.length; i++) {
          const s = (avatar.timeBuf[i] - 128) / 128;
          sq += s * s;
        }
        const rms = Math.sqrt(sq / avatar.timeBuf.length);
        avatar.agcPeakRms = Math.max(0.12, avatar.agcPeakRms * Math.pow(0.72, dt), rms * 1.10);
        const rmsNorm = Math.min(1.0, Math.max(0.0, (rms / avatar.agcPeakRms - 0.05) / 0.85));
        tgtLevel = rmsNorm;
        voiceSounding = rmsNorm > 0.045;
      } else {
        tgtLevel = 0.62;
        voiceSounding = true;
        for (let i = 0; i < avatar.freqBins.length; i++) {
          avatar.freqBins[i] = Math.floor(Math.max(0, 120 + Math.sin(t * 9 + i * 0.45) * 35));
        }
      }
    } else if (avatar.micAnalyser && (avatar.handsFree || avatar.pttActive) && !avatar.pausedForSpeech) {
      avatar.micAnalyser.getByteFrequencyData(avatar.freqBins);
      let sum = 0;
      for (let i = 1; i < 24; i++) sum += avatar.freqBins[i];
      tgtLevel = Math.min(1, (sum / (23 * 255)) * 2.8);

      // Empathetic listening nod when user finishes a phrase
      if (tgtLevel > 0.18) {
        if (avatar.userSpeakingSince === 0) avatar.userSpeakingSince = now;
        avatar.userLastLoudAt = now;
      } else if (
        avatar.userSpeakingSince > 0 &&
        now - avatar.userLastLoudAt > 380 &&
        now - avatar.userSpeakingSince > 750
      ) {
        triggerNod(3.2, 220);
        avatar.userSpeakingSince = 0;
      }
    } else {
      for (let i = 0; i < avatar.freqBins.length; i++) {
        avatar.freqBins[i] = Math.floor(avatar.freqBins[i] * Math.pow(0.08, dt));
      }
    }

    const reel = avatar.mouthManifest.reel || DEFAULT_MOUTH_MANIFEST.reel;
    const fps = avatar.mouthManifest.fps || 30;

    if (avatar.state === "speaking" && facingFront) {
      if (voiceSounding) {
        avatar.videoLips.silenceSec = 0;
        avatar.videoLips.mix = Math.min(1.0, avatar.videoLips.mix + dt / 0.045);
        // Play exact chronological 30fps frames of `talking me.mp4` as-is!
        avatar.videoLips.reelPos = (avatar.videoLips.reelPos + dt * fps) % reel.length;
      } else {
        // Real pause/breath between sentences: hold briefly (110ms), then smoothly close lips
        avatar.videoLips.silenceSec += dt;
        if (avatar.videoLips.silenceSec > 0.11) {
          avatar.videoLips.mix = Math.max(0.0, avatar.videoLips.mix - dt / 0.070);
          if (avatar.videoLips.mix <= 0.005) avatar.videoLips.reelPos = 0;
        } else {
          avatar.videoLips.reelPos = (avatar.videoLips.reelPos + dt * fps * 0.6) % reel.length;
        }
      }
    } else {
      // Idle Neutral, listening, thinking, OR rotating: strictly ZERO lip movement!
      avatar.videoLips.silenceSec = 0;
      avatar.videoLips.mix = Math.max(0.0, avatar.videoLips.mix - dt / 0.055);
      if (avatar.videoLips.mix <= 0.005) avatar.videoLips.reelPos = 0;
    }

    springStep(avatar.anim, "audioLevel", tgtLevel, 0.075, dt);

    // 3. Relax expression smoothly back to Neutral after hold expires when idle
    if (avatar.state === "idle" && avatar.exprHoldUntil > 0 && now > avatar.exprHoldUntil) {
      avatar.exprHoldUntil = 0;
      setExpression(avatar.handsFree ? "listening" : "neutral", 0);
    }

    // 4. Normal Human Eye Blinking (active on Neutral standby AND across expressions; paused when rotated!)
    const isStaticNeutral = avatar.state === "idle" && avatar.targetExpr === "neutral";
    const canBlink =
      facingFront &&
      avatar.loaded["blinking"] &&
      avatar.targetExpr !== "laughing" &&
      avatar.targetExpr !== "surprised";
    if (canBlink && avatar.blinkStartAt === 0 && now >= avatar.nextBlinkAt) {
      avatar.blinkStartAt = now;
      avatar.blinkDur = 195 + Math.random() * 35; // ~210ms normal human blink
      avatar.doubleBlinkPending = Math.random() < 0.12;
    }
    if (avatar.blinkStartAt > 0) {
      const el = now - avatar.blinkStartAt;
      const dur = avatar.blinkDur;
      if (el >= dur || !facingFront) {
        avatar.blinkAmount = 0;
        avatar.blinkStartAt = 0;
        if (avatar.doubleBlinkPending && facingFront) {
          avatar.doubleBlinkPending = false;
          avatar.nextBlinkAt = now + 140;
        } else {
          avatar.nextBlinkAt = now + 2600 + Math.random() * 3000;
        }
      } else {
        const u = el / dur;
        // Asymmetric real-human eyelid curve: brisk 30% drop -> 12% full shut hold -> smooth 58% lift
        if (u < 0.30) {
          avatar.blinkAmount = smootherstep(u / 0.30);
        } else if (u < 0.42) {
          avatar.blinkAmount = 1.0;
        } else {
          avatar.blinkAmount = smootherstep(1.0 - (u - 0.42) / 0.58);
        }
      }
    } else {
      avatar.blinkAmount = 0;
    }

    // 5. Normal Upright Human Posture + Calm Breathing & Subtle Micro-Life
    // Active across all front-facing states (including Neutral standby) for natural living presence;
    // smoothly zeroed during 360° manual rotation so turntable stays rock-solid & closed-lip static.
    const prof = MOOD_PROFILE[avatar.targetExpr] || MOOD_PROFILE.neutral;
    const frontFactor = smoothstep(1 - Math.min(1, distFromFront / 12));
    const postureFactor = avatar.isDragging ? 0 : frontFactor;

    // Natural calm resting human breathing (~14 breaths/min: gentle inhale rise + chest-fill expansion)
    const breathPhase = t * 1.45;
    const breathWave = Math.sin(breathPhase) * 0.72 + Math.sin(breathPhase * 2.0 - 0.9) * 0.20;
    const breathY = -breathWave * 1.85;
    const breathZoom = breathWave * 0.0032;

    // Subtle relaxed upright human posture balance (micro-drift only, never awkward sideways lean)
    const swayX = organicNoise(t * 0.42, 0.4) * 1.25;
    const swayTilt = organicNoise(t * 0.36, 2.7) * 0.0036;

    const speakTilt =
      avatar.state === "speaking"
        ? organicNoise(t * 1.85, 5.1) * (0.0045 + avatar.anim.audioLevel * 0.008)
        : 0;
    const speakLift =
      avatar.state === "speaking"
        ? -avatar.anim.audioLevel * 1.2 + organicNoise(t * 2.2, 3.3) * avatar.anim.audioLevel * 1.0
        : 0;

    avatar.laughEnergy = Math.max(0, avatar.laughEnergy - dt * 0.32);
    const laughMotion =
      avatar.targetExpr === "laughing"
        ? -Math.abs(Math.sin(t * 8.2)) * (0.8 + avatar.laughEnergy * 2.6 + avatar.anim.audioLevel * 1.5)
        : 0;

    if (avatar.nodResetAt > 0 && now > avatar.nodResetAt) {
      avatar.nodTarget = 0;
      avatar.nodResetAt = 0;
    }
    springStep(avatar.anim, "nod", avatar.nodTarget, 0.14, dt);

    springStep(avatar.anim, "tilt", (prof.tilt + swayTilt + speakTilt) * postureFactor, 0.36, dt);
    springStep(avatar.anim, "zoom", 1 + (prof.zoom - 1 + breathZoom + avatar.anim.audioLevel * 0.0035) * postureFactor, 0.40, dt);
    springStep(avatar.anim, "x", (prof.driftX + swayX) * postureFactor, 0.42, dt);
    springStep(avatar.anim, "y", (prof.driftY + breathY + speakLift + laughMotion + avatar.anim.nod) * postureFactor, 0.32, dt);
  }

  // =========================================================================
  // 5. Seamless Compositors (Staggered Expression + Clean Lower-Face Video Lips + Blink)
  // =========================================================================
  /**
   * Realistic top-down anatomical eyelid sweep (no ghost alpha fade, zero eyebrow jump!).
   * Sweeps the Poisson-locked closed-eyelids sprite smoothly downward across the eye sockets
   * (y = 0.268..0.333) as amount moves 0 -> 1, and lifts cleanly back up 1 -> 0.
   */
  function drawNaturalBlink(pCtx, blinkImg, drawX, drawY, drawW, drawH, amount, frontAlpha) {
    if (!blinkImg || amount <= 0.015 || frontAlpha <= 0.015) return;
    const bw = Math.ceil(drawW);
    const bh = Math.ceil(drawH);
    if (blinkCanvas.width !== bw || blinkCanvas.height !== bh) {
      blinkCanvas.width = bw;
      blinkCanvas.height = bh;
    }

    const bc = blinkCtx;
    bc.clearRect(0, 0, bw, bh);
    bc.globalCompositeOperation = "source-over";
    bc.globalAlpha = 1.0;
    bc.drawImage(blinkImg, 0, 0, bw, bh);

    // 1. Top-down sweeping eyelid curtain strictly below eyebrows (y=0.258..0.348)
    const yTop = bh * 0.268;
    const yBot = bh * 0.333;
    const feather = bh * 0.017;
    const curtainY = yTop + amount * (yBot - yTop + feather);
    const bandTop = bh * 0.256;
    const bandBot = bh * 0.350;

    bc.globalCompositeOperation = "destination-in";
    const sweepGrad = bc.createLinearGradient(0, Math.max(bandTop, curtainY - feather), 0, curtainY);
    sweepGrad.addColorStop(0.0, "rgba(255,255,255,1)");
    sweepGrad.addColorStop(1.0, "rgba(255,255,255,0)");
    bc.fillStyle = sweepGrad;
    bc.fillRect(0, bandTop, bw, Math.max(0, curtainY - bandTop));

    // 2. Soft top & bottom socket bounds so eyebrows and upper cheeks remain 100% locked
    const vBounds = bc.createLinearGradient(0, bandTop, 0, bandBot);
    vBounds.addColorStop(0.00, "rgba(255,255,255,0)");
    vBounds.addColorStop(0.14, "rgba(255,255,255,1)");
    vBounds.addColorStop(0.84, "rgba(255,255,255,1)");
    vBounds.addColorStop(1.00, "rgba(255,255,255,0)");
    bc.fillStyle = vBounds;
    bc.fillRect(0, bandTop, bw, bandBot - bandTop);

    // 3. Horizontal eye-region bounds (x = 0.22..0.77) keeping outer temples untouched
    const hBounds = bc.createLinearGradient(bw * 0.22, 0, bw * 0.77, 0);
    hBounds.addColorStop(0.00, "rgba(255,255,255,0)");
    hBounds.addColorStop(0.10, "rgba(255,255,255,1)");
    hBounds.addColorStop(0.90, "rgba(255,255,255,1)");
    hBounds.addColorStop(1.00, "rgba(255,255,255,0)");
    bc.fillStyle = hBounds;
    bc.fillRect(0, 0, bw, bh);
    bc.globalCompositeOperation = "source-over";

    pCtx.save();
    pCtx.globalAlpha = Math.min(1, frontAlpha);
    pCtx.drawImage(blinkCanvas, drawX, drawY, drawW, drawH);
    pCtx.restore();
  }

  function drawStaggeredExpression(pCtx, fromImg, toImg, drawX, drawY, drawW, drawH, now) {
    if (!toImg && !fromImg) return;
    if (!fromImg || fromImg === toImg) {
      pCtx.globalAlpha = 1;
      pCtx.drawImage(toImg, drawX, drawY, drawW, drawH);
      return;
    }
    const elapsed = now - avatar.transStart;
    const dur = Math.max(120, avatar.transDur);
    const staggerMs = Math.min(85, dur * 0.22);
    const upperP = smootherstep(elapsed / dur);
    const lowerP = smootherstep((elapsed - staggerMs) / Math.max(100, dur - staggerMs * 0.4));

    if (upperP >= 0.995 && lowerP >= 0.995) {
      pCtx.globalAlpha = 1;
      pCtx.drawImage(toImg, drawX, drawY, drawW, drawH);
      return;
    }

    pCtx.globalAlpha = 1;
    pCtx.drawImage(fromImg, drawX, drawY, drawW, drawH);
    if (lowerP > 0.005) {
      pCtx.globalAlpha = lowerP;
      pCtx.drawImage(toImg, drawX, drawY, drawW, drawH);
    }

    const upperExtra = Math.max(0, upperP - lowerP) / Math.max(0.001, 1 - lowerP);
    if (upperExtra > 0.015) {
      const bw = Math.ceil(drawW);
      const bh = Math.ceil(drawH);
      if (splitCanvas.width !== bw || splitCanvas.height !== bh) {
        splitCanvas.width = bw;
        splitCanvas.height = bh;
      }
      splitCtx.clearRect(0, 0, bw, bh);
      splitCtx.globalCompositeOperation = "source-over";
      splitCtx.drawImage(toImg, 0, 0, bw, bh);
      splitCtx.globalCompositeOperation = "destination-in";
      const g = splitCtx.createLinearGradient(0, bh * 0.38, 0, bh * 0.49);
      g.addColorStop(0, "rgba(255,255,255,1)");
      g.addColorStop(1, "rgba(255,255,255,0)");
      splitCtx.fillStyle = g;
      splitCtx.fillRect(0, 0, bw, bh * 0.50);
      splitCtx.globalCompositeOperation = "source-over";

      pCtx.globalAlpha = Math.min(1, upperExtra);
      pCtx.drawImage(splitCanvas, drawX, drawY, drawW, drawH);
    }
    pCtx.globalAlpha = 1;
  }

  /**
   * Composites the exact chronological 30fps real-video mouth reel from `talking me.mp4`
   * onto a coherent lower-face buffer (neutral.jpg + exact video mouth frame), feathered cleanly
   * below the nose tip (y=0.425..0.480) and covering the full chin/jaw (y=0.640..0.720) inside
   * the outer cheeks (x=0.14..0.86). Keeps 100% of upper-face expressions intact with ZERO double-chin!
   */
  function drawExactVideoTalkingLowerFace(pCtx, drawX, drawY, drawW, drawH, alpha) {
    if (alpha <= 0.01) return;

    const m = avatar.mouthManifest || DEFAULT_MOUTH_MANIFEST;
    const crop = m.crop || DEFAULT_MOUTH_MANIFEST.crop;
    const reel = m.reel || DEFAULT_MOUTH_MANIFEST.reel;
    const neutralImg = avatar.images["neutral"];
    if (!neutralImg || !reel.length) return;

    const bw = Math.ceil(drawW);
    const bh = Math.ceil(drawH);
    if (lowerFaceCanvas.width !== bw || lowerFaceCanvas.height !== bh) {
      lowerFaceCanvas.width = bw;
      lowerFaceCanvas.height = bh;
    }

    // 1. Render exact chronological 30fps video mouth frame into mouthCanvas
    const mw = crop.w;
    const mh = crop.h;
    if (mouthCanvas.width !== mw || mouthCanvas.height !== mh) {
      mouthCanvas.width = mw;
      mouthCanvas.height = mh;
    }

    const pos = ((avatar.videoLips.reelPos % reel.length) + reel.length) % reel.length;
    const idx0 = Math.floor(pos) % reel.length;
    const idx1 = (idx0 + 1) % reel.length;
    const frac = pos - Math.floor(pos);

    const img0 = avatar.mouthSprites[reel[idx0]];
    const img1 = avatar.mouthSprites[reel[idx1]];
    if (!img0) return;

    mouthCtx.clearRect(0, 0, mw, mh);
    mouthCtx.globalCompositeOperation = "source-over";
    mouthCtx.globalAlpha = 1.0;
    mouthCtx.drawImage(img0, 0, 0, mw, mh);
    if (img1 && img1 !== img0 && frac > 0.02) {
      mouthCtx.globalAlpha = frac;
      mouthCtx.drawImage(img1, 0, 0, mw, mh);
    }
    mouthCtx.globalAlpha = 1.0;

    mouthCtx.globalCompositeOperation = "destination-in";
    mouthCtx.save();
    mouthCtx.translate(mw * 0.5, mh * 0.455);
    mouthCtx.scale(1, (mh * 0.455) / (mw * 0.48));
    const rx = mw * 0.48;
    const grad = mouthCtx.createRadialGradient(0, 0, rx * 0.78, 0, 0, rx);
    grad.addColorStop(0.0, "rgba(255,255,255,1)");
    grad.addColorStop(0.75, "rgba(255,255,255,0.96)");
    grad.addColorStop(1.0, "rgba(255,255,255,0)");
    mouthCtx.fillStyle = grad;
    mouthCtx.beginPath();
    mouthCtx.arc(0, 0, rx, 0, Math.PI * 2);
    mouthCtx.fill();
    mouthCtx.restore();
    mouthCtx.globalCompositeOperation = "source-over";

    // 2. Composite exact video mouth onto clean neutral lower-face (locks jaw/chin -> zero double-chin!)
    const lc = lowerFaceCtx;
    lc.clearRect(0, 0, bw, bh);
    lc.globalCompositeOperation = "source-over";
    lc.globalAlpha = 1.0;
    lc.drawImage(neutralImg, 0, 0, bw, bh);

    const destX = (crop.x / m.baseWidth) * bw;
    const destY = (crop.y / m.baseHeight) * bh;
    const destW = (crop.w / m.baseWidth) * bw;
    const destH = (crop.h / m.baseHeight) * bh;
    lc.drawImage(mouthCanvas, destX, destY, destW, destH);

    // 3. Visually-verified coherent lower-face 2D feather mask (keeps 100% of upper-face emotions!)
    lc.globalCompositeOperation = "destination-in";
    const vMask = lc.createLinearGradient(0, bh * 0.425, 0, bh * 0.72);
    vMask.addColorStop(0.00, "rgba(255,255,255,0)");
    vMask.addColorStop(0.19, "rgba(255,255,255,1)");
    vMask.addColorStop(0.73, "rgba(255,255,255,1)");
    vMask.addColorStop(1.00, "rgba(255,255,255,0)");
    lc.fillStyle = vMask;
    lc.fillRect(0, bh * 0.42, bw, bh * 0.31);

    const hMask = lc.createLinearGradient(bw * 0.14, 0, bw * 0.86, 0);
    hMask.addColorStop(0.00, "rgba(255,255,255,0)");
    hMask.addColorStop(0.20, "rgba(255,255,255,1)");
    hMask.addColorStop(0.80, "rgba(255,255,255,1)");
    hMask.addColorStop(1.00, "rgba(255,255,255,0)");
    lc.fillStyle = hMask;
    lc.fillRect(0, 0, bw, bh);
    lc.globalCompositeOperation = "source-over";

    pCtx.save();
    pCtx.globalAlpha = Math.min(1, alpha);
    pCtx.drawImage(lowerFaceCanvas, drawX, drawY, drawW, drawH);
    pCtx.restore();
  }

  // =========================================================================
  // 5b. Alternative Visual Modes (Liquid Glass Orb, Spatial Halo, Floating Pill)
  // =========================================================================
  function renderLiquidOrb(cx, cy, prof) {
    const t = avatar.elapsedSec;
    const lvl = avatar.anim.audioLevel;
    const R = Math.min(vw * 0.24, vh * 0.24, 175);

    const aura = ctx.createRadialGradient(cx, cy, R * 0.2, cx, cy, R * 1.9);
    aura.addColorStop(0, `rgba(${prof.rgb}, ${0.32 + lvl * 0.35})`);
    aura.addColorStop(0.5, `rgba(99, 102, 241, ${0.12 + lvl * 0.16})`);
    aura.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = aura;
    ctx.beginPath();
    ctx.arc(cx, cy, R * 1.9, 0, Math.PI * 2);
    ctx.fill();

    const shells = [
      { scale: 1.06, speed: 1.4, amp: 14 + lvl * 28, alpha: 0.26, fill: prof.hex },
      { scale: 0.96, speed: -1.9, amp: 10 + lvl * 22, alpha: 0.48, fill: "#60a5fa" },
      { scale: 0.86, speed: 2.3, amp: 8 + lvl * 18, alpha: 0.82, fill: prof.hex },
    ];
    for (const sh of shells) {
      ctx.save();
      ctx.translate(cx, cy);
      ctx.beginPath();
      const pts = 84;
      for (let i = 0; i <= pts; i++) {
        const a = (i / pts) * Math.PI * 2;
        const bin = avatar.freqBins[i % 32] / 255;
        const deform =
          Math.sin(a * 3 + t * sh.speed) * sh.amp * 0.55 +
          Math.cos(a * 5 - t * sh.speed * 1.3) * sh.amp * 0.35 +
          bin * (10 + lvl * 22);
        const r = R * sh.scale + deform;
        const x = Math.cos(a) * r;
        const y = Math.sin(a) * r;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      const grad = ctx.createRadialGradient(-R * 0.28, -R * 0.32, R * 0.08, 0, 0, R * 1.25);
      grad.addColorStop(0, "rgba(255,255,255,0.95)");
      grad.addColorStop(0.28, sh.fill);
      grad.addColorStop(0.75, `rgba(${prof.rgb}, 0.28)`);
      grad.addColorStop(1, "rgba(8, 16, 32, 0.12)");
      ctx.globalAlpha = sh.alpha;
      ctx.fillStyle = grad;
      ctx.fill();
      ctx.lineWidth = 1.6;
      ctx.strokeStyle = "rgba(255,255,255,0.32)";
      ctx.stroke();
      ctx.restore();
    }

    ctx.save();
    ctx.translate(cx - R * 0.24, cy - R * 0.26);
    ctx.rotate(-0.6);
    ctx.fillStyle = "rgba(255,255,255,0.38)";
    ctx.beginPath();
    ctx.ellipse(0, 0, R * 0.26, R * 0.11, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  function renderSpatialHalo(cx, cy, prof) {
    const t = avatar.elapsedSec;
    const lvl = avatar.anim.audioLevel;
    const R = Math.min(vw * 0.30, vh * 0.28, 220);

    const rings = [
      { tiltX: 0.42, rot: t * 0.55, r: R * 1.05, alpha: 0.85, width: 3.0 },
      { tiltX: 0.58, rot: -t * 0.78 + 1.1, r: R * 0.88, alpha: 0.65, width: 2.2 },
      { tiltX: 0.34, rot: t * 1.05 + 2.4, r: R * 0.72, alpha: 0.55, width: 1.8 },
    ];
    for (const rg of rings) {
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(rg.rot * 0.35);
      ctx.scale(1, rg.tiltX + lvl * 0.08);
      ctx.strokeStyle = `rgba(${prof.rgb}, ${rg.alpha})`;
      ctx.lineWidth = rg.width;
      ctx.shadowColor = prof.hex;
      ctx.shadowBlur = 18;
      ctx.beginPath();
      const segs = 72;
      for (let i = 0; i <= segs; i++) {
        const a = (i / segs) * Math.PI * 2;
        const bin = avatar.freqBins[i % 36] / 255;
        const rr = rg.r + Math.sin(a * 6 + t * 4) * (4 + lvl * 18) + bin * 20;
        const x = Math.cos(a) * rr;
        const y = Math.sin(a) * rr;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.stroke();
      ctx.restore();
    }

    const core = ctx.createRadialGradient(cx, cy, 2, cx, cy, R * (0.42 + lvl * 0.16));
    core.addColorStop(0, "#ffffff");
    core.addColorStop(0.28, prof.hex);
    core.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = core;
    ctx.beginPath();
    ctx.arc(cx, cy, R * (0.42 + lvl * 0.16), 0, Math.PI * 2);
    ctx.fill();
  }

  function renderFloatingPill(cx, cy, prof) {
    const lvl = avatar.anim.audioLevel;
    const pillW = Math.min(440, vw * 0.82);
    const pillH = 74;
    const x0 = cx - pillW / 2;
    const y0 = cy - pillH / 2;

    ctx.save();
    ctx.shadowColor = prof.hex;
    ctx.shadowBlur = 22 + lvl * 18;
    ctx.fillStyle = "rgba(12, 18, 30, 0.82)";
    ctx.strokeStyle = `rgba(${prof.rgb}, ${0.48 + lvl * 0.45})`;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.roundRect(x0, y0, pillW, pillH, pillH / 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();

    const orbX = x0 + 38;
    const orbGrad = ctx.createRadialGradient(orbX, cy, 2, orbX, cy, 18 + lvl * 6);
    orbGrad.addColorStop(0, "#ffffff");
    orbGrad.addColorStop(0.45, prof.hex);
    orbGrad.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = orbGrad;
    ctx.beginPath();
    ctx.arc(orbX, cy, 18 + lvl * 6, 0, Math.PI * 2);
    ctx.fill();

    const bars = 28;
    const waveW = pillW - 110;
    const step = waveW / bars;
    const startX = x0 + 78;
    ctx.save();
    for (let i = 0; i < bars; i++) {
      const env = Math.sin((i / (bars - 1)) * Math.PI);
      const bin = avatar.freqBins[i % 32] / 255;
      const h = Math.max(4, env * (6 + (bin * 0.8 + lvl * 0.6) * 38));
      ctx.strokeStyle = `rgba(${prof.rgb}, ${0.35 + bin * 0.65})`;
      ctx.lineWidth = 3.2;
      ctx.lineCap = "round";
      const bx = startX + i * step;
      ctx.beginPath();
      ctx.moveTo(bx, cy - h / 2);
      ctx.lineTo(bx, cy + h / 2);
      ctx.stroke();
    }
    ctx.restore();
  }

  // =========================================================================
  // 6. Main 60 FPS Cinema Studio Render Loop
  // =========================================================================
  function render() {
    const now = performance.now();
    updatePhysics(now);

    const prof = MOOD_PROFILE[avatar.targetExpr] || MOOD_PROFILE.neutral;
    const cx = vw / 2;

    if (avatar.uiMode !== "face") {
      ctx.clearRect(0, 0, vw, vh);
      const midY = vh * 0.42;
      if (avatar.uiMode === "orb") renderLiquidOrb(cx, midY, prof);
      else if (avatar.uiMode === "halo") renderSpatialHalo(cx, midY, prof);
      else renderFloatingPill(cx, midY, prof);
      requestAnimationFrame(render);
      return;
    }

    const aspect = 600 / 848;
    const baseH = Math.min(vh * 0.82, 740);
    const baseW = Math.min(baseH * aspect, vw * 0.92);
    const finalH = baseW / aspect;
    const finalW = baseW;
    const portraitTop = Math.max(8, (vh - 180 - finalH) * 0.42);
    const cy = portraitTop + finalH * 0.44;

    ctx.clearRect(0, 0, vw, vh);

    // 1. Subtle Ambient Studio Rim Glow Behind Head & Shoulders
    const bgGlow = ctx.createRadialGradient(cx, cy, finalW * 0.2, cx, cy, finalW * 0.95);
    bgGlow.addColorStop(0, `rgba(${prof.rgb}, ${0.08 + avatar.anim.audioLevel * 0.15})`);
    bgGlow.addColorStop(0.55, `rgba(${prof.rgb}, 0.025)`);
    bgGlow.addColorStop(1, "rgba(0, 0, 0, 0)");
    ctx.fillStyle = bgGlow;
    ctx.fillRect(0, 0, vw, vh);

    // 2. Render Expressive Portrait / Static Closed-Neutral 360° Turntable
    const pw = Math.ceil(finalW * 1.12);
    const ph = Math.ceil(finalH * 1.12);
    if (portraitCanvas.width !== pw || portraitCanvas.height !== ph) {
      portraitCanvas.width = pw;
      portraitCanvas.height = ph;
    }

    const pCtx = portraitCtx;
    pCtx.clearRect(0, 0, pw, ph);
    pCtx.globalCompositeOperation = "source-over";

    const drawW = finalW * avatar.anim.zoom;
    const drawH = finalH * avatar.anim.zoom;
    const drawX = -drawW / 2;
    const drawY = -drawH / 2;

    pCtx.save();
    pCtx.translate(pw / 2 + avatar.anim.x, ph / 2 + avatar.anim.y);
    pCtx.rotate(avatar.anim.tilt);

    const normDeg = ((avatar.angleDeg % 360) + 360) % 360;
    const distFromFront = Math.min(normDeg, 360 - normDeg);
    const frontFactor = smootherstep(1 - Math.min(1, distFromFront / 9));

    // 2a. Static 100%-Closed-Neutral 360° Turntable Angle (zero lip movement while rotating!)
    if (frontFactor < 0.995) {
      const exactIdx = (normDeg / 360) * TURNTABLE_COUNT;
      const idxA = Math.floor(exactIdx) % TURNTABLE_COUNT;
      const idxB = (idxA + 1) % TURNTABLE_COUNT;
      const frac = exactIdx - Math.floor(exactIdx);
      const imgA = avatar.turntable[idxA] || avatar.images["neutral"];
      const imgB = avatar.turntable[idxB] || imgA;
      if (imgA) {
        pCtx.globalAlpha = 1;
        pCtx.drawImage(imgA, drawX, drawY, drawW, drawH);
      }
      if (imgB && imgB !== imgA && frac > 0.01) {
        pCtx.globalAlpha = frac;
        pCtx.drawImage(imgB, drawX, drawY, drawW, drawH);
      }
      pCtx.globalAlpha = 1;
    }

    // 2b. Front-Facing Expressions + Direct Real-Video Talking Lips (ONLY while speaking!)
    if (frontFactor > 0.01) {
      pCtx.save();
      if (frontFactor < 0.995) pCtx.globalAlpha = frontFactor;

      const fromImg = avatar.images[avatar.prevExpr] || avatar.images["neutral"];
      const toImg = avatar.images[avatar.targetExpr] || avatar.images["neutral"];
      if (frontFactor >= 0.995) {
        drawStaggeredExpression(pCtx, fromImg, toImg, drawX, drawY, drawW, drawH, now);
      } else if (toImg) {
        pCtx.drawImage(toImg, drawX, drawY, drawW, drawH);
      }

      // ONLY while actively speaking & facing front: real-video 30fps lips over any expression!
      if (avatar.videoLips.mix > 0.01 && frontFactor > 0.85) {
        drawExactVideoTalkingLowerFace(pCtx, drawX, drawY, drawW, drawH, avatar.videoLips.mix * frontFactor);
      }

      // Natural top-down human eye blinking (on Neutral standby AND across expressions)
      if (avatar.blinkAmount > 0.015 && avatar.images["blinking"]) {
        drawNaturalBlink(
          pCtx,
          avatar.images["blinking"],
          drawX,
          drawY,
          drawW,
          drawH,
          avatar.blinkAmount,
          frontFactor
        );
      }
      pCtx.restore();
    }

    pCtx.restore();

    // True organic 3-layer seamless studio dissolve (ZERO rectangular corners, ZERO jacket/edge cutoffs!)
    const imgL = (pw - finalW) / 2;
    const imgT = (ph - finalH) / 2;

    pCtx.globalCompositeOperation = "destination-in";

    // Layer 1: Wide bust-centered corner-dissolve oval — keeps face, collar, zippers & leather shoulders rich while rounding all 4 corners!
    pCtx.save();
    pCtx.translate(pw * 0.50, imgT + finalH * 0.51);
    const ellRx = finalW * 0.61;
    const ellRy = finalH * 0.62;
    pCtx.scale(1, ellRy / ellRx);
    const ellGrad = pCtx.createRadialGradient(0, 0, ellRx * 0.64, 0, 0, ellRx);
    ellGrad.addColorStop(0.00, "rgba(255,255,255,1)");
    ellGrad.addColorStop(0.32, "rgba(255,255,255,0.92)");
    ellGrad.addColorStop(0.60, "rgba(255,255,255,0.64)");
    ellGrad.addColorStop(0.82, "rgba(255,255,255,0.26)");
    ellGrad.addColorStop(0.94, "rgba(255,255,255,0.05)");
    ellGrad.addColorStop(1.00, "rgba(255,255,255,0)");
    pCtx.fillStyle = ellGrad;
    pCtx.beginPath();
    pCtx.arc(0, 0, ellRx, 0, Math.PI * 2);
    pCtx.fill();
    pCtx.restore();

    // Layer 2: Lower-hem mist taper — preserves full upper leather jacket & chest, then melts bottom smoothly into studio floor
    const botGrad = pCtx.createLinearGradient(0, imgT + finalH * 0.76, 0, imgT + finalH * 0.99);
    botGrad.addColorStop(0.00, "rgba(255,255,255,1)");
    botGrad.addColorStop(0.28, "rgba(255,255,255,0.88)");
    botGrad.addColorStop(0.58, "rgba(255,255,255,0.52)");
    botGrad.addColorStop(0.82, "rgba(255,255,255,0.15)");
    botGrad.addColorStop(1.00, "rgba(255,255,255,0)");
    pCtx.fillStyle = botGrad;
    pCtx.fillRect(0, 0, pw, ph);

    // Layer 3: Gentle top & outer-shoulder edge feather so jacket sleeves melt seamlessly with zero photo boundary
    const topGrad = pCtx.createLinearGradient(0, imgT, 0, imgT + finalH * 0.07);
    topGrad.addColorStop(0.00, "rgba(255,255,255,0)");
    topGrad.addColorStop(0.45, "rgba(255,255,255,0.68)");
    topGrad.addColorStop(1.00, "rgba(255,255,255,1)");
    pCtx.fillStyle = topGrad;
    pCtx.fillRect(0, 0, pw, ph);

    const sideGrad = pCtx.createLinearGradient(imgL, 0, imgL + finalW, 0);
    sideGrad.addColorStop(0.00, "rgba(255,255,255,0)");
    sideGrad.addColorStop(0.035, "rgba(255,255,255,0.14)");
    sideGrad.addColorStop(0.095, "rgba(255,255,255,0.64)");
    sideGrad.addColorStop(0.165, "rgba(255,255,255,1)");
    sideGrad.addColorStop(0.835, "rgba(255,255,255,1)");
    sideGrad.addColorStop(0.905, "rgba(255,255,255,0.64)");
    sideGrad.addColorStop(0.965, "rgba(255,255,255,0.14)");
    sideGrad.addColorStop(1.00, "rgba(255,255,255,0)");
    pCtx.fillStyle = sideGrad;
    pCtx.fillRect(0, 0, pw, ph);
    pCtx.globalCompositeOperation = "source-over";

    ctx.drawImage(portraitCanvas, cx - pw / 2, portraitTop - (ph - finalH) / 2, pw, ph);

    // 3. Clean Horizontal Voice Wave Bar Clearly Above State Chip
    const stageEl = document.querySelector(".voice-stage");
    const stageTop = stageEl ? stageEl.getBoundingClientRect().top : vh - 190;
    const waveY = stageTop - 16;
    const barCount = 36;
    const totalWaveW = Math.min(280, vw * 0.50);
    const step = totalWaveW / barCount;
    const startX = cx - totalWaveW / 2;

    ctx.save();
    for (let i = 0; i < barCount; i++) {
      const normPos = (i / (barCount - 1)) * 2 - 1;
      const env = Math.cos(normPos * Math.PI * 0.5);
      const bin = avatar.freqBins[Math.floor(Math.abs(normPos) * 26)] / 255;
      const active =
        avatar.state === "listening" || avatar.state === "speaking" || avatar.state === "processing";
      const boost = active ? 1 : 0.16;
      const barH = Math.max(2, env * (2 + (bin * 0.75 + avatar.anim.audioLevel * 0.55) * 20) * boost);
      ctx.strokeStyle = `rgba(${prof.rgb}, ${(0.22 + bin * 0.72) * Math.max(0.28, boost)})`;
      ctx.lineWidth = 2.3;
      ctx.lineCap = "round";
      const x = startX + i * step;
      ctx.beginPath();
      ctx.moveTo(x, waveY - barH / 2);
      ctx.lineTo(x, waveY + barH / 2);
      ctx.stroke();
    }
    ctx.restore();

    requestAnimationFrame(render);
  }
  requestAnimationFrame(render);

  // =========================================================================
  // 7. Neural Audio Playback (Drives Real-Video Lips Strictly While Speaking)
  // =========================================================================
  function ensureAudioContext() {
    if (!avatar.audioCtx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      avatar.audioCtx = new AC();
    }
    if (avatar.audioCtx.state === "suspended") {
      avatar.audioCtx.resume().catch(() => { });
    }
    return avatar.audioCtx;
  }

  function stopBrowserAudio() {
    if (avatar.activeAudioEl) {
      try {
        avatar.activeAudioEl.pause();
        avatar.activeAudioEl.currentTime = 0;
      } catch (_) { }
      avatar.activeAudioEl = null;
    }
    if ("speechSynthesis" in window) {
      try {
        window.speechSynthesis.cancel();
      } catch (_) { }
    }
    avatar.ttsAnalyser = null;
    avatar.pausedForSpeech = false;
  }

  async function speakInBrowser(text, base64Audio, expression, audioMime = "audio/mpeg") {
    if (!avatar.speakEnabled || !text) return;
    stopBrowserAudio();
    avatar.pausedForSpeech = true;
    avatar.agcPeakRms = 0.16;
    avatar.videoLips.reelPos = 0;
    avatar.videoLips.silenceSec = 0;
    stopRecognitionOnly();
    if (expression) setExpression(expression, 10000);
    setState("speaking");

    const onDone = () => {
      avatar.ttsAnalyser = null;
      avatar.activeAudioEl = null;
      avatar.pausedForSpeech = false;
      avatar.exprHoldUntil = performance.now() + 2400;
      if (avatar.state === "speaking") setState("idle");
      if (avatar.handsFree) startRecognition({ continuous: true });
    };

    if (base64Audio) {
      try {
        const audio = new Audio(`data:${audioMime || "audio/mpeg"};base64,${base64Audio}`);
        avatar.activeAudioEl = audio;
        const ac = ensureAudioContext();
        const src = ac.createMediaElementSource(audio);
        const analyser = ac.createAnalyser();
        analyser.fftSize = 256;
        analyser.smoothingTimeConstant = 0.50;
        src.connect(analyser);
        analyser.connect(ac.destination);
        avatar.ttsAnalyser = analyser;
        avatar.timeBuf = new Uint8Array(analyser.fftSize);
        audio.onended = onDone;
        audio.onerror = () => speakFallbackSynth(text, onDone);
        await audio.play();
        return;
      } catch (_) { }
    }
    speakFallbackSynth(text, onDone);
  }

  let cachedBrowserVoices = [];
  if ("speechSynthesis" in window) {
    const refreshVoices = () => {
      try {
        cachedBrowserVoices = window.speechSynthesis.getVoices() || [];
      } catch (_) {}
    };
    refreshVoices();
    window.speechSynthesis.onvoiceschanged = refreshVoices;
  }

  function pickIndianMaleBrowserVoice(isHindi) {
    const voices = cachedBrowserVoices.length
      ? cachedBrowserVoices
      : (window.speechSynthesis.getVoices() || []);
    if (!voices.length) return null;
    const isFemaleName = (n) => /female|swara|neerja|heera|kalpana|zira|samantha|aria|jenny|sonia|libby|natasha|ava|emma|veena/i.test(n);
    const isMaleName = (n) => /prabhat|madhur|ravi|hemant|male|ryan|daniel|george|andrew|guy|david|mark|arjun/i.test(n);
    const score = (v) => {
      const lang = String(v.lang || "");
      const name = String(v.name || "");
      let s = 0;
      if (isHindi) {
        if (/^hi-IN$/i.test(lang)) s += 60;
        else if (/^en-IN$/i.test(lang)) s += 42;
        else if (/^hi/i.test(lang)) s += 35;
      } else {
        if (/^en-IN$/i.test(lang)) s += 60;
        else if (/^hi-IN$/i.test(lang)) s += 45;
        else if (/^en-GB$/i.test(lang)) s += 20;
        else if (/^en/i.test(lang)) s += 12;
      }
      if (/prabhat|madhur|ravi|hemant/i.test(name)) s += 50;
      else if (isMaleName(name)) s += 28;
      if (isFemaleName(name)) s -= 45;
      if (/natural|online|google/i.test(name)) s += 10;
      return s;
    };
    return [...voices].sort((a, b) => score(b) - score(a))[0] || null;
  }

  function speakFallbackSynth(text, onDone) {
    if (!("speechSynthesis" in window)) {
      onDone();
      return;
    }
    try {
      const utter = new SpeechSynthesisUtterance(text);
      const isHindi = /[\u0900-\u097F]/.test(text);
      utter.lang = isHindi ? "hi-IN" : "en-IN";
      const preferred = pickIndianMaleBrowserVoice(isHindi);
      if (preferred) utter.voice = preferred;
      utter.rate = 0.98;
      utter.pitch = 0.92;
      utter.onend = onDone;
      utter.onerror = onDone;
      window.speechSynthesis.speak(utter);
    } catch (_) {
      onDone();
    }
  }

  // =========================================================================
  // 8. WebSocket Real-Time Engine Bridge
  // =========================================================================
  function openBrowserUrl(url) {
    if (!url || !/^https?:\/\//i.test(url)) return false;
    const now = performance.now();
    if (avatar.lastOpenedUrl === url && now - avatar.lastOpenedAt < 2500) {
      return true;
    }
    avatar.lastOpenedUrl = url;
    avatar.lastOpenedAt = now;

    // Do NOT pass "noopener" in features string, otherwise browsers return null even on success
    const win = window.open(url, "_blank");
    if (win) {
      try { win.opener = null; } catch (_) {}
      return true;
    }
    // Popup blocker blocked auto-open (e.g. Hands-Free without recent click): trigger OS fallback + show button
    if (!IS_STATIC_DEMO && avatar.ws && avatar.connected) {
      avatar.ws.send(JSON.stringify({ type: "open_url_os", url }));
    }
    return false;
  }

  function showBentoCard(card) {
    if (!card) return;
    const el = $("bentoCard");
    if (!el) return;
    el.classList.remove("hidden", "blocked");
    if (card.status === "blocked" || card.status === "error") el.classList.add("blocked");
    $("bentoIcon").textContent = card.icon || "⚙";
    $("bentoTitle").textContent = card.title || "Action";
    $("bentoSub").textContent = card.subtitle || "";

    const barWrap = $("bentoBarWrap");
    if (typeof card.progress === "number") {
      barWrap.classList.remove("hidden");
      $("bentoBarFill").style.width = `${Math.max(0, Math.min(100, card.progress))}%`;
    } else {
      barWrap.classList.add("hidden");
    }

    const metricsEl = $("bentoMetrics");
    const metrics = Array.isArray(card.metrics) ? card.metrics : [];
    metricsEl.innerHTML = metrics
      .map(
        (m) =>
          `<div class="bento-metric"><div class="bento-metric-label">${String(m.label || "")}</div><div class="bento-metric-val">${String(m.value || "")}</div></div>`
      )
      .join("");

    const openBtn = $("bentoOpenBtn");
    if (openBtn) {
      const targetUrl = card.url || card.open_url;
      if (targetUrl && /^https?:\/\//i.test(targetUrl)) {
        openBtn.href = targetUrl;
        openBtn.textContent = `↗ Open ${card.title || "Link"}`;
        openBtn.classList.remove("hidden");
      } else {
        openBtn.classList.add("hidden");
      }
    }

    clearTimeout(avatar.cardTimer);
    avatar.cardTimer = setTimeout(() => el.classList.add("hidden"), 9000);
  }

  function flashTool(skillName) {
    const badge = $("toolBadge");
    badge.textContent = `⚙ ${skillName}`;
    badge.classList.remove("hidden");
    clearTimeout(avatar.toolTimer);
    avatar.toolTimer = setTimeout(() => badge.classList.add("hidden"), 4500);
  }

  function showConfirm(data) {
    avatar.pendingToken = data.token;
    $("confirmText").textContent = data.description || `Confirm ${data.skill_name}?`;
    $("confirmBar").classList.remove("hidden");
    setExpression("serious", 15000);
  }

  function hideConfirm() {
    avatar.pendingToken = null;
    $("confirmBar").classList.add("hidden");
  }

  // =========================================================================
  // 8a. Standalone Live Web Engine (Runs Automatically on GitHub Pages)
  // =========================================================================
  const staticChatHistory = [];
  const STATIC_SITES = {
    google: "https://www.google.com",
    youtube: "https://www.youtube.com",
    gmail: "https://mail.google.com",
    github: "https://github.com",
    instagram: "https://www.instagram.com",
    linkedin: "https://www.linkedin.com",
    chatgpt: "https://chatgpt.com",
    gemini: "https://gemini.google.com",
    reddit: "https://www.reddit.com",
    netflix: "https://www.netflix.com",
    spotify: "https://open.spotify.com",
    whatsapp: "https://web.whatsapp.com",
    wikipedia: "https://en.wikipedia.org",
    twitter: "https://x.com",
    x: "https://x.com",
  };
  const STATIC_JOKES = [
    "Why do programmers prefer dark mode? Because light attracts bugs! Ha ha!",
    "Why did the neural network go to therapy? It had too many deep issues and kept overfitting its past! Ha ha!",
    "I told my Python script a joke about recursion, and it's still laughing... and laughing... and laughing!",
  ];

  async function deliverStaticReply(replyText, expression = "smiling", provider = "web-ai", card = null) {
    setChatTyping(false);
    $("liveJarvis").textContent = replyText;
    appendChatMessage("assistant", replyText, provider);
    staticChatHistory.push({ role: "assistant", content: replyText });
    if (staticChatHistory.length > 16) staticChatHistory.splice(0, staticChatHistory.length - 16);
    if (card) showBentoCard(card);
    setExpression(expression, 6000);
    if (avatar.speakEnabled) {
      await speakInBrowser(replyText, null, expression);
    } else {
      setState("idle");
    }
  }

  async function handleStaticCommand(rawText) {
    const cleaned = rawText.replace(/^(?:hey\s+jarvis|ok\s+jarvis|jarvis)[,\s.!?]*/i, "").trim();
    const low = cleaned.toLowerCase().replace(/[.!?]+$/g, "").trim();
    staticChatHistory.push({ role: "user", content: rawText });

    if (!cleaned || /^(?:hi|hello|hey|yo|namaste)(?:\s+(?:there|jarvis|buddy|bud|man|bro|friend))*$/i.test(low)) {
      const lines = [
        "Hey there! Good to see you — what's on your mind today?",
        "Hey! I'm live right here in your browser. Talk, type, or drag to rotate me 360°!",
      ];
      return deliverStaticReply(lines[Math.floor(Math.random() * lines.length)], "smiling", "jarvis");
    }
    if (/\b(?:what'?s up|whats up|what'?s going on|how are you|how'?s it going)\b/i.test(low)) {
      return deliverStaticReply("Doing great and running smoothly right here in your browser! How's your day going?", "happy", "jarvis");
    }
    if (/\b(?:tell me a joke|make me laugh|say something funny)\b/i.test(low)) {
      return deliverStaticReply(STATIC_JOKES[Math.floor(Math.random() * STATIC_JOKES.length)], "laughing", "jarvis");
    }
    if (/\b(?:flip a coin|toss a coin|heads or tails)\b/i.test(low)) {
      return deliverStaticReply(`I flipped a coin — it landed on ${Math.random() < 0.5 ? "Heads" : "Tails"}!`, "happy", "jarvis");
    }
    if (/\b(?:roll a dice|roll a die|roll dice)\b/i.test(low)) {
      return deliverStaticReply(`You rolled a ${1 + Math.floor(Math.random() * 6)}!`, "happy", "jarvis");
    }
    if (/^(?:क्या\s+हाल\s+है|क्या\s+हाल|कैसे\s+हो|नमस्ते|नमस्कार|और\s+भाई|सब\s+बढ़िया)$/.test(cleaned.replace(/[.!?।]+$/g, "").trim())) {
      return deliverStaticReply("सब बढ़िया है भाई! आप बताओ, आज क्या चल रहा है?", "smiling", "jarvis");
    }
    if (/^(?:kya\s+haal\s+hai|kya\s+haal|kaise\s+ho|aur\s+bhai|sab\s+badhiya|kya\s+chal\s+raha\s+hai)(?:\s+(?:bhai|yaar|bro|dost))?$/i.test(low)) {
      return deliverStaticReply("Sab badhiya hai bhai! Tum batao, aaj kya chal raha hai?", "smiling", "jarvis");
    }
    if (/\b(?:what time|current time|today'?s date|what date|kitne\s+baje|time\s+kya\s+hua)\b|(?:कितने\s+बजे|समय\s+क्या\s+हुआ)/i.test(low)) {
      const now = new Date();
      return deliverStaticReply(`It is ${now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })} on ${now.toDateString()}.`, "smiling", "clock");
    }
    if (/^(?:smile|be happy|laugh|be serious|look angry|look sad|be surprised|neutral)$/i.test(low)) {
      const map = { smile: "smiling", "be happy": "happy", laugh: "laughing", "be serious": "serious", "look angry": "angry", "look sad": "sad", "be surprised": "surprised", neutral: "neutral" };
      const expr = map[low] || "smiling";
      return deliverStaticReply(`Switching to ${expr} mode!`, expr, "expression");
    }

    // Open websites / YouTube / Google directly (English + Hinglish)
    const openMatch =
      cleaned.match(/^(?:open|launch|go\s+to|visit)\s+(?:the\s+)?(.+)$/i) ||
      cleaned.match(/^(.+?)\s+(?:khol\s+do|kholo|open\s+karo)$/i);
    if (openMatch) {
      const target = openMatch[1].trim().replace(/[.!?]+$/, "");
      const tLow = target.toLowerCase();
      const url = STATIC_SITES[tLow] || (/^https?:\/\//i.test(target) ? target : /\.[a-z]{2,}$/i.test(tLow) ? `https://${tLow}` : null);
      if (url) {
        openBrowserUrl(url);
        return deliverStaticReply(`Opening ${target} now.`, "smiling", "web", {
          icon: "🌐", title: target.toUpperCase(), subtitle: url, status: "ok", url,
          metrics: [{ label: "MODE", value: "DIRECT OPEN" }],
        });
      }
    }
    const ytMatch = cleaned.match(/\b(?:play|search)\s+(.+?)\s+on\s+youtube\b/i);
    if (ytMatch) {
      const q = ytMatch[1].trim();
      const url = `https://www.youtube.com/results?search_query=${encodeURIComponent(q)}`;
      openBrowserUrl(url);
      return deliverStaticReply(`Opening YouTube for "${q}".`, "happy", "youtube", {
        icon: "▶", title: `YouTube: ${q}`, subtitle: "Direct Browser Launch", status: "ok", url,
      });
    }

    // Live Weather via Open-Meteo (English + Hinglish, CORS-enabled, zero API key)
    const wxMatch =
      low.match(/^(?:(?:bhai|yaar|jarvis)[\s,]+)?([a-z\s,]+?)\s+(?:ka|ki|mein|me)\s+(?:weather|temperature|mausam)(?:\s+(?:batao|kaisa\s+hai|kya\s+hai))?$/i) ||
      low.match(/\b(?:weather|temperature|forecast|mausam)(?:\s+(?:in|for|at)\s+([a-z\s,]+?))?(?:\s+(?:right\s+now|currently|today|now|outside|batao|kaisa\s+hai))?$/i);
    if (wxMatch) {
      const city = ((wxMatch[1] || "Jabalpur").replace(/\s+(?:right\s+now|currently|today|now|outside|batao|kaisa\s+hai|kya\s+hai)$/i, "")).trim();
      try {
        const geo = await fetch(`https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(city)}&count=1&language=en&format=json`).then((r) => r.json());
        const loc = (geo.results || [])[0];
        if (loc) {
          const wx = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${loc.latitude}&longitude=${loc.longitude}&current=temperature_2m,relative_humidity_2m,wind_speed_10m`).then((r) => r.json());
          const cur = wx.current || {};
          const place = `${loc.name}${loc.country ? ", " + loc.country : ""}`;
          return deliverStaticReply(
            `In ${place}, it's currently ${cur.temperature_2m}°C with ${cur.relative_humidity_2m}% humidity and ${cur.wind_speed_10m} km/h wind.`,
            "smiling",
            "open-meteo",
            {
              icon: "🌤", title: place, subtitle: "Live Weather", status: "ok",
              metrics: [
                { label: "TEMP", value: `${cur.temperature_2m}°C` },
                { label: "HUMIDITY", value: `${cur.relative_humidity_2m}%` },
                { label: "WIND", value: `${cur.wind_speed_10m} km/h` },
              ],
            }
          );
        }
      } catch (_) {}
    }

    // Explicit Wikipedia lookup
    const wikiMatch = cleaned.match(/^(?:who\s+(?:is|was)|tell\s+(?:me\s+)?about|explain|define|wikipedia)\s+(.+)$/i);
    if (wikiMatch) {
      const topic = wikiMatch[1].trim().replace(/[?.!]+$/, "");
      try {
        const osr = await fetch(`https://en.wikipedia.org/w/api.php?action=opensearch&search=${encodeURIComponent(topic)}&limit=1&format=json&origin=*`).then((r) => r.json());
        const title = (osr[1] || [])[0] || topic;
        const sum = await fetch(`https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(title.replace(/ /g, "_"))}`).then((r) => r.json());
        if (sum && sum.extract) {
          const short = sum.extract.split(/(?<=[.!?])\s+/).slice(0, 2).join(" ");
          return deliverStaticReply(short, "thinking", "wikipedia", {
            icon: "📚", title: sum.title || title, subtitle: "Wikipedia Live", status: "ok",
            url: (sum.content_urls && sum.content_urls.desktop && sum.content_urls.desktop.page) || "",
          });
        }
      } catch (_) {}
    }

    // Live Conversational AI Chat (Pollinations anonymous GPT-OSS 20B — mirrors English/Hindi/Hinglish!)
    try {
      const sysPrompt =
        "You are JARVIS, a personal voice assistant. Mirror the user's language naturally (Devanagari Hindi for Hindi, casual Roman Hinglish for Hinglish, English for English). Keep replies warm and concise (1-2 short sentences like talking to a sharp friend). Never dump encyclopedia facts on casual chat.";
      const msgs = [{ role: "system", content: sysPrompt }, ...staticChatHistory.slice(-12)];
      const resp = await fetch("https://text.pollinations.ai/openai", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: "openai-fast", messages: msgs, temperature: 0.7, max_tokens: 420 }),
      });
      if (resp.ok) {
        const data = await resp.json();
        const txt = String((((data.choices || [])[0] || {}).message || {}).content || "")
          .replace(/<think>[\s\S]*?<\/think>/gi, "")
          .trim();
        if (txt) return deliverStaticReply(txt, inferInterimExpression(txt) || "smiling", "live-ai");
      }
    } catch (_) {}

    return deliverStaticReply("Good point — tell me more! For full local PC control & cloned voice, run `python main.py` locally.", "smiling", "jarvis");
  }

  function connectWebSocket() {
    if (IS_STATIC_DEMO) {
      avatar.connected = true;
      $("connDot").classList.add("online");
      setState("idle");
      updateBrainBadge({ provider: "free-ai", model: "live-web-demo" });
      if ($("btnLocalMic")) $("btnLocalMic").classList.add("hidden");
      if ($("btnVisionScreen")) $("btnVisionScreen").classList.add("hidden");
      if ($("brainPillLabel")) $("brainPillLabel").textContent = "🌐 Live Web AI";
      if (!chatHistoryLoaded) {
        chatHistoryLoaded = true;
        appendChatMessage("assistant", "Hey there! You're viewing J.A.R.V.I.S. live on GitHub Pages — talk, type, try expressions, or drag my face 360°!", "jarvis");
      }
      return;
    }

    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    avatar.ws = ws;

    ws.onopen = () => {
      avatar.connected = true;
      $("connDot").classList.add("online");
      setState("idle");
      loadChatHistory();
    };

    ws.onclose = () => {
      avatar.connected = false;
      $("connDot").classList.remove("online");
      $("stateLabel").textContent = "RECONNECTING...";
      setTimeout(connectWebSocket, 1800);
    };

    ws.onmessage = (evt) => {
      const { type, payload } = JSON.parse(evt.data);
      switch (type) {
        case "hello":
          if (payload && payload.ui_version && String(payload.ui_version) !== ASSET_VER) {
            window.location.reload();
            return;
          }
          applyStatus(payload);
          break;
        case "state_changed":
          if (avatar.activeAudioEl && payload.state === "idle") break;
          setState(payload.state);
          if (payload.state === "listening") setExpression("listening", 0);
          else if (payload.state === "processing") setExpression("thinking", 0);
          break;
        case "expression_changed":
          if (payload.expression) setExpression(payload.expression, 6000);
          break;
        case "audio_level":
          avatar.anim.audioLevel = Math.min(1, Number(payload.level || 0));
          break;
        case "transcript_final":
          $("liveYou").textContent = `"${payload.text}"`;
          appendChatMessage("user", payload.text, "voice");
          setChatTyping(true);
          triggerNod(3.4, 220);
          break;
        case "tool_called":
          setState("executing");
          setExpression("thinking", 0);
          flashTool(payload.skill);
          break;
        case "action_card":
          showBentoCard(payload);
          if (payload && payload.open_url) {
            const opened = openBrowserUrl(payload.open_url);
            if (!opened && $("bentoSub")) {
              $("bentoSub").textContent = "Allow pop-ups once in address bar for instant auto-open, or click below.";
            }
          }
          break;
        case "response_final": {
          $("liveJarvis").textContent = payload.text || "";
          setChatTyping(false);
          if (payload.text) appendChatMessage("assistant", payload.text, payload.provider || "jarvis");
          hideConfirm();
          if (payload.card) showBentoCard(payload.card);
          if (payload.open_url) {
            const opened = openBrowserUrl(payload.open_url);
            if (!opened && $("bentoSub")) {
              $("bentoSub").textContent = "Allow pop-ups once in address bar for instant auto-open, or click below.";
            }
          }
          const expr = payload.expression || "smiling";
          setExpression(expr, 5500);
          if (avatar.speakEnabled && payload.text) {
            speakInBrowser(payload.text, payload.audio_base64, expr, payload.audio_mime);
          }
          break;
        }
        case "confirmation_requested":
          showConfirm(payload);
          break;
        case "reminder_triggered":
          $("liveJarvis").textContent = payload.alert || `Reminder: ${payload.message}`;
          appendChatMessage("assistant", payload.alert || `Reminder: ${payload.message}`, "reminder");
          setExpression("surprised", 4500);
          if (avatar.speakEnabled) speakInBrowser(payload.alert || payload.message, null, "surprised");
          break;
        case "proactive_alert": {
          const alertMsg = payload.message || payload.title || "Proactive system alert.";
          $("liveJarvis").textContent = alertMsg;
          appendChatMessage("assistant", alertMsg, "proactive");
          showBentoCard({
            icon: payload.icon || "⚡",
            title: payload.title || "Proactive Alert",
            subtitle: alertMsg.slice(0, 110),
            status: payload.kind === "battery_low" || payload.kind === "memory_pressure" ? "blocked" : "ok",
            metrics: [{ label: "TRIGGER", value: String(payload.kind || "EVENT").toUpperCase() }],
          });
          setExpression(payload.expression || "serious", 5500);
          if (avatar.speakEnabled && payload.speak) {
            speakInBrowser(alertMsg, null, payload.expression || "serious");
          }
          break;
        }
        case "local_voice_status":
          $("liveJarvis").textContent = payload.message || "";
          fetch("/api/status")
            .then((r) => r.json())
            .then(applyStatus)
            .catch(() => { });
          break;
        case "error":
          setChatTyping(false);
          setExpression("sad", 4000);
          $("liveJarvis").textContent = payload.message || "An error occurred.";
          if (payload.message) appendChatMessage("assistant", payload.message, "error");
          break;
      }
    };
  }

  function updateBrainBadge(llmInfo) {
    const label = $("brainPillLabel");
    const dot = $("brainDot");
    if (!label || !llmInfo) return;
    const prov = String(llmInfo.provider || "web-knowledge").toLowerCase();
    const isCloud =
      prov === "gemini" ||
      prov === "openai" ||
      prov === "groq" ||
      prov === "ollama" ||
      prov === "free-ai";
    const names = {
      gemini: "Gemini AI",
      openai: "OpenAI",
      groq: "Groq AI",
      ollama: "Ollama AI",
      "free-ai": "Live Chat AI",
      "web-knowledge": "Knowledge AI",
      offline: "Offline Chat",
    };
    label.textContent = `🧠 ${names[prov] || prov.toUpperCase()}`;
    if (dot) dot.classList.toggle("cloud", isCloud);
  }

  function applyStatus(status) {
    if (!status) return;
    avatar.localVoiceActive = !!status.local_voice_active;
    $("btnLocalMic").classList.toggle("live", avatar.localVoiceActive);
    $("btnLocalMic").textContent = avatar.localVoiceActive ? "🖥 Py Mic ON" : "🖥 Py Mic";
    if (status.llm) updateBrainBadge(status.llm);
    if (status.config) {
      avatar.speakEnabled = !!status.config.speak_responses;
      $("btnVoice").classList.toggle("active", avatar.speakEnabled);
      $("btnVoice").textContent = avatar.speakEnabled ? "🔊 Voice ON" : "🔇 Voice OFF";
      if (status.config.language) {
        avatar.sttLang = status.config.language;
        if ($("btnLangToggle")) {
          $("btnLangToggle").textContent = avatar.sttLang.toLowerCase().startsWith("hi") ? "🌐 HI-IN" : "🌐 EN-IN";
        }
      }
    }
    if (status.pending_confirmation) showConfirm(status.pending_confirmation);
    setState(status.state || "idle");
  }

  // =========================================================================
  // 8b. Translucent Liquid-Glass Typing Chat Drawer (Unified Voice + Text Log)
  // =========================================================================
  let lastChatUserText = "";
  let lastChatUserAt = 0;
  let chatHistoryLoaded = false;

  function setChatTyping(isTyping) {
    const el = $("chatTyping");
    if (el) el.classList.toggle("hidden", !isTyping);
  }

  function appendChatMessage(role, text, metaLabel = "", timeStr = "") {
    const list = $("chatMessages");
    const content = String(text || "").trim();
    if (!list || !content) return;

    // Deduplicate immediate echo when sendCommand already pushed the typed user message
    const now = performance.now();
    if (role === "user") {
      if (content === lastChatUserText && now - lastChatUserAt < 2500) return;
      lastChatUserText = content;
      lastChatUserAt = now;
    }

    const wrap = document.createElement("div");
    wrap.className = `chat-msg ${role === "user" ? "user" : "assistant"}`;

    const body = document.createElement("div");
    body.textContent = content;
    wrap.appendChild(body);

    const stamp =
      timeStr ||
      new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    const meta = document.createElement("div");
    meta.className = "chat-meta";
    meta.textContent = metaLabel ? `${metaLabel} • ${stamp}` : stamp;
    wrap.appendChild(meta);

    list.appendChild(wrap);
    list.scrollTop = list.scrollHeight;
  }

  async function loadChatHistory() {
    if (chatHistoryLoaded) return;
    chatHistoryLoaded = true;
    const list = $("chatMessages");
    if (!list) return;
    try {
      const data = await fetch("/api/history?limit=30").then((r) => r.json());
      const msgs = Array.isArray(data.messages) ? data.messages : [];
      list.innerHTML = "";
      if (!msgs.length) {
        appendChatMessage("assistant", "Hey there! You can speak or type right here anytime.", "jarvis");
        return;
      }
      for (const m of msgs) {
        const role = m.role === "user" ? "user" : "assistant";
        appendChatMessage(role, m.content, role === "user" ? "you" : "jarvis");
      }
    } catch (_) {
      appendChatMessage("assistant", "Hey there! You can speak or type right here anytime.", "jarvis");
    }
  }

  function toggleChatDrawer(forceState) {
    const drawer = $("chatDrawer");
    const btn = $("btnChatToggle");
    if (!drawer) return;
    const shouldOpen =
      typeof forceState === "boolean" ? forceState : drawer.classList.contains("hidden");
    drawer.classList.toggle("hidden", !shouldOpen);
    if (btn) btn.classList.toggle("active", shouldOpen);
    if (shouldOpen) {
      loadChatHistory();
      const inp = $("chatInput");
      if (inp) {
        inp.focus();
        const list = $("chatMessages");
        if (list) list.scrollTop = list.scrollHeight;
      }
    }
  }

  function sendCommand(text, source = "avatar_voice") {
    const cleaned = (text || "").trim();
    if (!cleaned) return;
    stopBrowserAudio();
    $("liveYou").textContent = `"${cleaned}"`;
    appendChatMessage("user", cleaned, source === "chat_text" ? "typed" : "voice");
    setChatTyping(true);
    setState("processing");
    const preExpr = inferInterimExpression(cleaned);
    setExpression(preExpr || "thinking", 3000);
    triggerNod(3.0, 200);

    if (IS_STATIC_DEMO) {
      handleStaticCommand(cleaned);
      return;
    }

    if (avatar.ws && avatar.connected) {
      avatar.ws.send(
        JSON.stringify({
          type: "command",
          text: cleaned,
          source,
          speak: avatar.speakEnabled,
          browser_audio: true,
        })
      );
    }
  }

  // =========================================================================
  // 9. Browser Voice Recognition
  // =========================================================================
  const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;

  async function ensureMicAnalyser() {
    if (avatar.micAnalyser) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const ac = ensureAudioContext();
      const src = ac.createMediaStreamSource(stream);
      const analyser = ac.createAnalyser();
      analyser.fftSize = 128;
      analyser.smoothingTimeConstant = 0.68;
      src.connect(analyser);
      avatar.micAnalyser = analyser;
      avatar.freqBins = new Uint8Array(analyser.frequencyBinCount);
    } catch (_) { }
  }

  const TURN_INCOMPLETE_TAIL = new Set([
    "a", "an", "the", "and", "or", "but", "so", "because", "if", "when", "while",
    "that", "which", "who", "to", "for", "of", "in", "on", "at", "with", "about",
    "from", "into", "by", "um", "uh", "erm", "hmm", "like", "well", "open",
    "launch", "search", "remind", "remember", "explain", "define", "tell", "play",
    "set", "change", "adjust", "turn", "switch", "run", "delete", "kill", "thinking", "want",
    "aur", "ya", "lekin", "par", "kyunki", "toh", "phir", "mein", "ka", "ki", "ke", "ko", "se", "khol", "batao",
    "और", "या", "लेकिन", "क्योंकि", "तो", "फिर", "में", "का", "की", "के", "को", "से",
  ]);
  const TURN_SHORT_COMPLETE = new Set([
    "stop", "cancel", "yes", "no", "confirm", "abort", "help", "mute", "unmute",
    "hi", "hello", "hey", "jarvis", "hey jarvis", "ok jarvis", "bye", "goodbye",
    "pause", "resume", "next", "previous",
    "namaste", "नमस्ते", "रुको", "हाँ", "नहीं",
  ]);
  const TURN_NOISE_FILLERS = new Set(["uh", "um", "erm", "hm", "hmm", "ah", "oh", "eh", "mm", "mhm"]);

  function clearTurnBuffer() {
    if (avatar.turnTimer) {
      clearTimeout(avatar.turnTimer);
      avatar.turnTimer = null;
    }
    avatar.turnBuffer = "";
  }

  function computeTurnWaitMs(text) {
    // Unicode-aware strip preserves Devanagari Hindi & English characters cleanly
    const clean = (text || "").toLowerCase().replace(/[^\p{L}\p{M}\p{N}\s']/gu, "").trim();
    if (!clean) return 850;
    if (TURN_SHORT_COMPLETE.has(clean)) return 420;
    const words = clean.split(/\s+/);
    const last = words[words.length - 1] || "";
    if (TURN_INCOMPLETE_TAIL.has(last) || /[,\-:]$/.test((text || "").trim())) {
      return 1650;
    }
    return 850; // 700-900ms natural pause window before finalizing a spoken turn
  }

  function flushTurnBuffer() {
    if (avatar.turnTimer) {
      clearTimeout(avatar.turnTimer);
      avatar.turnTimer = null;
    }
    const text = (avatar.turnBuffer || "").trim();
    avatar.turnBuffer = "";
    if (!text) return;
    const norm = text.toLowerCase().replace(/[^\p{L}\p{M}\p{N}\s]/gu, "").trim();
    if (!norm || (norm.length < 2 && !/^\d+$/.test(norm)) || TURN_NOISE_FILLERS.has(norm)) {
      $("liveYou").textContent = "";
      return;
    }
    sendCommand(text, avatar.turnSource || "ptt_voice");
  }

  function stopRecognitionOnly() {
    clearTurnBuffer();
    if (avatar.recognition) {
      try {
        avatar.recognition.onend = null;
        avatar.recognition.stop();
      } catch (_) { }
      avatar.recognition = null;
    }
  }

  async function startRecognition({ continuous }) {
    if (!SpeechRecognitionAPI) {
      $("liveJarvis").textContent =
        "Use Chrome or Edge for browser voice, or click Py Mic / ⌨.";
      return;
    }
    stopBrowserAudio();
    stopRecognitionOnly();
    await ensureMicAnalyser();

    const rec = new SpeechRecognitionAPI();
    rec.lang = avatar.sttLang || "en-IN";
    rec.interimResults = true;
    rec.continuous = continuous;
    avatar.turnSource = continuous ? "hands_free_voice" : "ptt_voice";

    rec.onstart = () => {
      setState("listening");
      setExpression("listening", 0);
    };

    rec.onresult = (event) => {
      let interim = "";
      let finalChunk = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const res = event.results[i];
        if (res.isFinal) finalChunk += " " + res[0].transcript;
        else interim += res[0].transcript;
      }

      if (finalChunk.trim()) {
        avatar.turnBuffer = (avatar.turnBuffer + " " + finalChunk.trim()).trim();
      }

      const combinedPreview = (avatar.turnBuffer + (interim ? " " + interim.trim() : "")).trim();
      if (combinedPreview) {
        $("liveYou").textContent = combinedPreview;
        const liveMood = inferInterimExpression(combinedPreview);
        if (liveMood) setExpression(liveMood, 2600);
      }

      // While interim speech is actively arriving, keep waiting so we never cut off mid-sentence
      if (interim.trim()) {
        if (avatar.turnTimer) {
          clearTimeout(avatar.turnTimer);
          avatar.turnTimer = null;
        }
        return;
      }

      // Schedule semantic end-of-turn flush (850ms normal / 1650ms mid-thought pause / 420ms short command)
      if (avatar.turnBuffer.trim()) {
        if (avatar.turnTimer) clearTimeout(avatar.turnTimer);
        const waitMs = computeTurnWaitMs(avatar.turnBuffer);
        avatar.turnTimer = setTimeout(flushTurnBuffer, waitMs);
      }
    };

    rec.onerror = (ev) => {
      if (ev.error === "not-allowed") {
        $("liveJarvis").textContent = "Microphone access blocked — allow mic in your browser address bar.";
        avatar.handsFree = false;
        avatar.pttActive = false;
        $("btnHandsFree").classList.remove("live");
        setExpression("sad", 4000);
        setState("error");
      }
    };

    rec.onend = () => {
      if (!continuous && avatar.turnBuffer.trim()) {
        flushTurnBuffer();
      }
      if (avatar.handsFree && !avatar.pausedForSpeech) {
        try {
          rec.start();
        } catch (_) { }
      } else {
        avatar.pttActive = false;
        if (avatar.state === "listening") {
          setState("idle");
          setExpression("neutral", 0);
        }
      }
    };

    avatar.recognition = rec;
    try {
      rec.start();
    } catch (_) { }
  }

  // =========================================================================
  // 10. Manual 360° Drag-to-Rotate + Controls + Shortcuts
  // =========================================================================
  function triggerTalk() {
    ensureAudioContext();
    if (avatar.state === "speaking") interruptAll();
    if (avatar.handsFree) {
      triggerNod(3.0, 200);
      return;
    }
    avatar.pttActive = true;
    $("liveYou").textContent = "Listening...";
    startRecognition({ continuous: false });
  }

  function interruptAll() {
    stopBrowserAudio();
    if (avatar.ws && avatar.connected) avatar.ws.send(JSON.stringify({ type: "interrupt" }));
    setState("idle");
    setExpression("neutral", 0);
  }

  function resetToFront() {
    avatar.targetAngleDeg = Math.round(avatar.angleDeg / 360) * 360;
  }

  canvas.addEventListener("pointerdown", (e) => {
    avatar.isDragging = true;
    avatar.dragStartX = e.clientX;
    avatar.dragLastX = e.clientX;
    avatar.dragMovedPx = 0;
    canvas.classList.add("dragging");
    canvas.setPointerCapture(e.pointerId);
  });

  canvas.addEventListener("pointermove", (e) => {
    if (!avatar.isDragging) return;
    const dx = e.clientX - avatar.dragLastX;
    avatar.dragLastX = e.clientX;
    avatar.dragMovedPx += Math.abs(dx);
    avatar.targetAngleDeg += dx * 0.85;
  });

  function endDrag(e) {
    if (!avatar.isDragging) return;
    avatar.isDragging = false;
    canvas.classList.remove("dragging");
    try {
      canvas.releasePointerCapture(e.pointerId);
    } catch (_) { }
    if (avatar.dragMovedPx < 6) triggerTalk();
  }
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("dblclick", resetToFront);

  canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      const delta = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
      avatar.targetAngleDeg += delta * 0.14;
    },
    { passive: false }
  );

  if ($("btnResetAngle")) $("btnResetAngle").addEventListener("click", resetToFront);
  $("btnTalk").addEventListener("click", triggerTalk);

  $("btnHandsFree").addEventListener("click", () => {
    ensureAudioContext();
    avatar.handsFree = !avatar.handsFree;
    $("btnHandsFree").classList.toggle("live", avatar.handsFree);
    $("btnHandsFree").textContent = avatar.handsFree ? "⚡ Hands-Free ON" : "⚡ Hands-Free";
    if (avatar.handsFree) {
      $("liveJarvis").textContent = "Hands-free live conversation active — talk naturally anytime.";
      startRecognition({ continuous: true });
    } else {
      stopRecognitionOnly();
      setState("idle");
      setExpression("neutral", 0);
      $("liveJarvis").textContent = "Hands-free paused. Drag face to rotate or press Talk to speak.";
    }
  });

  $("btnLocalMic").addEventListener("click", () => {
    if (IS_STATIC_DEMO) return;
    const action = avatar.localVoiceActive ? "stop" : "start";
    if (avatar.ws && avatar.connected) {
      avatar.ws.send(JSON.stringify({ type: "voice_local", action, require_wake_word: true }));
    }
  });

  $("btnVoice").addEventListener("click", async () => {
    avatar.speakEnabled = !avatar.speakEnabled;
    if (!avatar.speakEnabled) stopBrowserAudio();
    $("btnVoice").classList.toggle("active", avatar.speakEnabled);
    $("btnVoice").textContent = avatar.speakEnabled ? "🔊 Voice ON" : "🔇 Voice OFF";
    if (!IS_STATIC_DEMO) {
      await fetch("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ speak_responses: avatar.speakEnabled }),
      }).catch(() => { });
    }
  });

  if ($("btnLangToggle")) {
    $("btnLangToggle").addEventListener("click", async () => {
      avatar.sttLang = avatar.sttLang === "hi-IN" ? "en-IN" : "hi-IN";
      $("btnLangToggle").textContent = avatar.sttLang === "hi-IN" ? "🌐 HI-IN" : "🌐 EN-IN";
      $("liveJarvis").textContent =
        avatar.sttLang === "hi-IN"
          ? "Hindi / Hinglish voice recognition active (hi-IN) — बोलिए भाई!"
          : "Indian English & Hinglish voice recognition active (en-IN).";
      if (!IS_STATIC_DEMO) {
        await fetch("/api/settings", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ language: avatar.sttLang }),
        }).catch(() => { });
      }
      if (avatar.handsFree) startRecognition({ continuous: true });
    });
  }

  $("btnStop").addEventListener("click", interruptAll);

  if ($("btnChatToggle")) {
    $("btnChatToggle").addEventListener("click", () => toggleChatDrawer());
  }
  async function captureAndAnalyzeBrowserScreen() {
    ensureAudioContext();
    if (navigator.mediaDevices && navigator.mediaDevices.getDisplayMedia) {
      try {
        const stream = await navigator.mediaDevices.getDisplayMedia({ video: { frameRate: 5 }, audio: false });
        const video = document.createElement("video");
        video.srcObject = stream;
        await video.play();
        const cap = document.createElement("canvas");
        const scale = Math.min(1, 1280 / Math.max(1, video.videoWidth || 1280));
        cap.width = Math.max(320, Math.round((video.videoWidth || 1280) * scale));
        cap.height = Math.max(180, Math.round((video.videoHeight || 720) * scale));
        cap.getContext("2d").drawImage(video, 0, 0, cap.width, cap.height);
        stream.getTracks().forEach((tr) => tr.stop());

        const dataUrl = cap.toDataURL("image/jpeg", 0.82);
        $("liveYou").textContent = '"Analyze my screen"';
        appendChatMessage("user", "Analyze my screen", "vision");
        setChatTyping(true);
        setState("processing");
        setExpression("thinking", 4000);
        await fetch("/api/vision/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image_base64: dataUrl,
            mime_type: "image/jpeg",
            question: "Summarize what is on my screen and point out any errors or important details.",
            speak: avatar.speakEnabled,
          }),
        });
        return;
      } catch (_) {}
    }
    sendCommand("what's on my screen", "vision_btn");
  }

  if ($("btnVisionScreen")) {
    $("btnVisionScreen").addEventListener("click", captureAndAnalyzeBrowserScreen);
  }
  if ($("btnCloseChat")) {
    $("btnCloseChat").addEventListener("click", () => toggleChatDrawer(false));
  }
  if ($("btnClearChat")) {
    $("btnClearChat").addEventListener("click", async () => {
      await fetch("/api/history", { method: "DELETE" }).catch(() => {});
      const list = $("chatMessages");
      if (list) list.innerHTML = "";
      appendChatMessage("assistant", "Chat cleared! Hey there — what's next?", "jarvis");
    });
  }
  if ($("chatForm")) {
    $("chatForm").addEventListener("submit", (e) => {
      e.preventDefault();
      ensureAudioContext();
      const inp = $("chatInput");
      const val = inp ? inp.value.trim() : "";
      if (!val) return;
      inp.value = "";
      sendCommand(val, "chat_text");
      inp.focus();
    });
  }

  if ($("exprStrip")) {
    $("exprStrip").addEventListener("click", (e) => {
      const btn = e.target.closest(".expr-pill");
      if (!btn || !btn.dataset.expr) return;
      setExpression(btn.dataset.expr, 6000);
    });
  }

  if ($("modeSwitcher")) {
    $("modeSwitcher").addEventListener("click", (e) => {
      const btn = e.target.closest(".mode-pill");
      if (!btn || !btn.dataset.mode) return;
      avatar.uiMode = btn.dataset.mode;
      document.querySelectorAll(".mode-pill").forEach((p) => p.classList.toggle("active", p === btn));
      if ($("exprStrip")) $("exprStrip").classList.toggle("hidden", avatar.uiMode !== "face");
      if ($("btnResetAngle") && avatar.uiMode !== "face") $("btnResetAngle").classList.add("hidden");
    });
  }

  if ($("bentoClose")) {
    $("bentoClose").addEventListener("click", () => $("bentoCard").classList.add("hidden"));
  }

  // Live AI Brain & LLM Setup Modal
  function syncLlmModalFields() {
    const sel = $("llmProviderSelect");
    if (!sel) return;
    const v = sel.value;
    $("groupGemini").classList.toggle("hidden", v !== "gemini" && v !== "auto");
    $("groupOpenai").classList.toggle("hidden", v !== "openai");
    $("groupGroq").classList.toggle("hidden", v !== "groq");
    $("groupOllama").classList.toggle("hidden", v !== "ollama");
  }

  async function openLlmModal() {
    const modal = $("llmModal");
    if (!modal) return;
    modal.classList.remove("hidden");
    $("llmFeedback").textContent = "";
    $("llmFeedback").classList.remove("error");
    try {
      const [diag, st] = await Promise.all([
        fetch("/api/llm/status").then((r) => r.json()),
        fetch("/api/status").then((r) => r.json()).catch(() => null),
      ]);
      if (st && st.persona && $("llmPersonaSelect")) {
        $("llmPersonaSelect").value = st.persona.name || "calm";
      }
      if (st && st.config && $("llmVoiceSelect")) {
        $("llmVoiceSelect").value = st.config.cloned_voice_enabled
          ? "cloned"
          : (st.config.edge_tts_voice || "en-IN-PrabhatNeural");
      }
      if (diag.active) {
        updateBrainBadge(diag.active);
        $("llmCurrentStatus").textContent = `Active: ${diag.active.provider} (${diag.active.model}) • Web Knowledge always ready`;
      }
      if (diag.llm_provider) $("llmProviderSelect").value = diag.llm_provider;
      if (diag.models) {
        if (diag.models.gemini) $("inputGeminiModel").value = diag.models.gemini;
        if (diag.models.openai) $("inputOpenaiModel").value = diag.models.openai;
        if (diag.models.groq) $("inputGroqModel").value = diag.models.groq;
        if (diag.models.ollama) $("inputOllamaModel").value = diag.models.ollama;
      }
      if (diag.ollama_url) $("inputOllamaUrl").value = diag.ollama_url;
      if (diag.configured) {
        $("badgeGemini").textContent = diag.configured.gemini ? "✓ Saved" : "";
        $("badgeOpenai").textContent = diag.configured.openai ? "✓ Saved" : "";
        $("badgeGroq").textContent = diag.configured.groq ? "✓ Saved" : "";
        $("badgeOllama").textContent = diag.configured.ollama
          ? `✓ Running (${(diag.ollama_models || []).length} models)`
          : "Not detected";
      }
      syncLlmModalFields();
    } catch (_) {}
  }

  async function submitLlmConfig(testConnection) {
    const fb = $("llmFeedback");
    fb.classList.remove("error");
    fb.textContent = testConnection ? "Testing live AI connection..." : "Saving configuration...";
    const body = {
      llm_provider: $("llmProviderSelect").value,
      gemini_api_key: $("inputGeminiKey").value.trim() || null,
      gemini_model: $("inputGeminiModel").value.trim() || null,
      openai_api_key: $("inputOpenaiKey").value.trim() || null,
      openai_model: $("inputOpenaiModel").value.trim() || null,
      groq_api_key: $("inputGroqKey").value.trim() || null,
      groq_model: $("inputGroqModel").value.trim() || null,
      ollama_url: $("inputOllamaUrl").value.trim() || null,
      ollama_model: $("inputOllamaModel").value.trim() || null,
      test_connection: !!testConnection,
    };
    try {
      const settingsPayload = {};
      if ($("llmPersonaSelect")) settingsPayload.persona = $("llmPersonaSelect").value;
      if ($("llmVoiceSelect")) {
        const vChoice = $("llmVoiceSelect").value;
        if (vChoice === "cloned") {
          settingsPayload.cloned_voice_enabled = true;
          settingsPayload.edge_tts_voice = "en-IN-PrabhatNeural";
        } else {
          settingsPayload.cloned_voice_enabled = false;
          settingsPayload.edge_tts_voice = vChoice;
        }
      }
      if (Object.keys(settingsPayload).length) {
        await fetch("/api/settings", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(settingsPayload),
        }).catch(() => {});
      }
      const res = await fetch("/api/llm/configure", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then((r) => r.json());
      if (res.diagnostics && res.diagnostics.active) {
        updateBrainBadge(res.diagnostics.active);
        $("llmCurrentStatus").textContent = `Active: ${res.diagnostics.active.provider} (${res.diagnostics.active.model})`;
      }
      fb.textContent = res.message || "Saved!";
      fb.classList.toggle("error", !res.ok);
      if (res.ok) {
        $("inputGeminiKey").value = "";
        $("inputOpenaiKey").value = "";
        $("inputGroqKey").value = "";
      }
    } catch (err) {
      fb.textContent = `Failed to save: ${err}`;
      fb.classList.add("error");
    }
  }

  if ($("btnAiBrain")) $("btnAiBrain").addEventListener("click", openLlmModal);
  if ($("btnCloseLlm")) $("btnCloseLlm").addEventListener("click", () => $("llmModal").classList.add("hidden"));
  if ($("llmModal")) {
    $("llmModal").addEventListener("click", (e) => {
      if (e.target === $("llmModal")) $("llmModal").classList.add("hidden");
    });
  }
  if ($("llmProviderSelect")) $("llmProviderSelect").addEventListener("change", syncLlmModalFields);
  if ($("llmForm")) {
    $("llmForm").addEventListener("submit", (e) => {
      e.preventDefault();
      submitLlmConfig(false);
    });
  }
  if ($("btnTestLlm")) $("btnTestLlm").addEventListener("click", () => submitLlmConfig(true));

  $("btnConfirmYes").addEventListener("click", () => {
    if (avatar.pendingToken && avatar.ws) {
      avatar.ws.send(JSON.stringify({ type: "confirm", token: avatar.pendingToken, approved: true }));
    }
  });
  $("btnConfirmNo").addEventListener("click", () => {
    if (avatar.pendingToken && avatar.ws) {
      avatar.ws.send(JSON.stringify({ type: "confirm", token: avatar.pendingToken, approved: false }));
    }
  });

  window.addEventListener("keydown", (e) => {
    const tag = document.activeElement ? document.activeElement.tagName : "";
    const typing = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
    if (e.code === "Escape") {
      if ($("llmModal") && !$("llmModal").classList.contains("hidden")) $("llmModal").classList.add("hidden");
      else if ($("chatDrawer") && !$("chatDrawer").classList.contains("hidden")) toggleChatDrawer(false);
      else interruptAll();
      return;
    }
    if (typing) return;
    if (e.code === "Space" && !e.repeat) {
      e.preventDefault();
      triggerTalk();
    } else if (e.code === "ArrowLeft") {
      e.preventDefault();
      avatar.targetAngleDeg -= 15;
    } else if (e.code === "ArrowRight") {
      e.preventDefault();
      avatar.targetAngleDeg += 15;
    } else if (e.key === "0" || e.key === "r" || e.key === "R") {
      resetToFront();
    } else if (e.key === "/" || e.key === "c" || e.key === "C") {
      e.preventDefault();
      toggleChatDrawer(true);
    } else if (e.key === "v" || e.key === "V") {
      e.preventDefault();
      ensureAudioContext();
      sendCommand("what's on my screen", "vision_key");
    }
  });

  connectWebSocket();
})();
