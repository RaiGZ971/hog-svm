import { useState, useRef, useCallback, useEffect } from "react";
import { useWebSocket } from "../hooks/useWebsocket";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

const VOCAB_CATEGORIES = [
  { name: "Greetings", words: ["Good morning", "Good afternoon", "Good evening", "Hello", "Goodbye", "Thank you", "You're welcome"] },
  { name: "Color", words: ["Red", "Blue", "Green", "Yellow", "White", "Black"] }
];

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
];

function drawLandmarks(ctx, landmarks, handednesses, canvasW, canvasH) {
  ctx.clearRect(0, 0, canvasW, canvasH);
  if (!landmarks || landmarks.length === 0) return;

  landmarks.forEach((hand, i) => {
    const side = handednesses?.[i]?.[0]?.categoryName;
    const pts  = hand.map(lm => ({
      x: (1 - lm.x) * canvasW,
      y: lm.y * canvasH,
    }));

    const boneColor = side === "Left" ? "rgba(99, 100, 255, 1)" : "rgba(62, 255, 159, 1)";
    const dotColor  = side === "Left" ? "#0245ed" : "#41ed02";
    const tipColor  = "#fff";

    ctx.lineWidth   = 2.5;
    ctx.strokeStyle = boneColor;
    HAND_CONNECTIONS.forEach(([a, b]) => {
      ctx.beginPath();
      ctx.moveTo(pts[a].x, pts[a].y);
      ctx.lineTo(pts[b].x, pts[b].y);
      ctx.stroke();
    });

    pts.forEach((p, idx) => {
      const isTip = [4, 8, 12, 16, 20].includes(idx);
      ctx.beginPath();
      ctx.arc(p.x, p.y, isTip ? 5 : 3.5, 0, Math.PI * 2);
      ctx.fillStyle   = isTip ? tipColor : dotColor;
      ctx.fill();
      ctx.strokeStyle = boneColor;
      ctx.lineWidth   = 1.5;
      ctx.stroke();
    });
  });
}

export default function Recognizer() {
  const videoRef      = useRef(null);
  const canvasRef     = useRef(null);
  const landmarkerRef = useRef(null);

  // Single ever-increasing timestamp — NEVER reset after landmarker is used
  const frameIdxRef = useRef(0);

  // Whether detection is active — read inside rAF loop
  const isDetectingRef = useRef(false);

  const [cameraActive,    setCameraActive]    = useState(false);
  const [cameraError,     setCameraError]     = useState(null);
  const [detectedGesture, setDetectedGesture] = useState("—");
  const [isDetecting,     setIsDetecting]     = useState(false);
  const [openCategories,  setOpenCategories]  = useState({ Greetings: true });
  const [logs,            setLogs]            = useState([]);
  const [mpReady,         setMpReady]         = useState(false);

  const addLog = useCallback((text) => {
    setLogs(prev => {
      const entry = { id: Date.now() + Math.random(), text, time: new Date().toLocaleTimeString() };
      return [entry, ...prev].slice(0, 40);
    });
  }, []);

  const handleMessage = useCallback((msg) => {
    if (msg === "__SIGNING__") {
      addLog("✋ Gesture detected — buffering...");
      return;
    }
    setDetectedGesture(msg);
    addLog(`✅ Recognized: ${msg}`);
  }, [addLog]);

  const { connectWebSocket, disconnectWebSocket, wsRef } = useWebSocket(handleMessage);

  // ── Load MediaPipe HandLandmarker once ──────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
        );
        const landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numHands: 2,
        });
        if (!cancelled) {
          landmarkerRef.current = landmarker;
          setMpReady(true);
        }
      } catch (err) {
        console.error("MediaPipe init failed:", err);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // ── Single detection + draw + send loop ──────────────────────────────────
  // detectForVideo, drawLandmarks, and WS send all happen here in one rAF
  // tick so every detected frame is sent — no separate interval, no skipping.
  useEffect(() => {
    if (!cameraActive) return;

    let rafId;
    // Target ~30fps to match browser rAF; training was 60fps but webcam
    // MediaPipe VIDEO mode is effectively capped by rAF (~30fps).
    const FPS    = 30;
    const stepMs = 1000 / FPS;

    const loop = () => {
      rafId = requestAnimationFrame(loop);

      const video      = videoRef.current;
      const canvas     = canvasRef.current;
      const landmarker = landmarkerRef.current;

      if (!video || !canvas || !landmarker) return;
      if (video.videoWidth === 0 || video.videoHeight === 0) return;

      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width  = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      // frameIdxRef never resets — guarantees strictly increasing timestamps
      const timestamp = frameIdxRef.current * stepMs;
      frameIdxRef.current += 1;

      const result = landmarker.detectForVideo(video, timestamp);

      // ── Draw landmarks ──
      const ctx = canvas.getContext("2d");
      drawLandmarks(ctx, result?.landmarks, result?.handednesses, canvas.width, canvas.height);

      if (isDetectingRef.current) {
        const ws = wsRef.current;
        if (ws && ws.readyState === WebSocket.OPEN) {
          const empty = Array.from({ length: 21 }, () => [0, 0, 0]);
          let left  = empty.map(p => [...p]);
          let right = empty.map(p => [...p]);

          if (result?.landmarks) {
            result.landmarks.forEach((handLandmarks, i) => {
              const side = result.handednesses?.[i]?.[0]?.categoryName;
              // Raw (unmirrored) coords — backend model trained on these
              const pts  = handLandmarks.map(lm => [lm.x, lm.y, lm.z]);
              if (side === "Left")  left  = pts;
              else                  right = pts;
            });
          }

          ws.send(JSON.stringify({ landmarks: [...left, ...right] }));
        }
      }
    };

    rafId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafId);
  }, [cameraActive, mpReady, wsRef]);

  // ── Camera ───────────────────────────────────────────────────────────────
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        // DO NOT reset frameIdxRef — timestamps must keep increasing
        setCameraActive(true);
        setCameraError(null);
      }
    } catch {
      setCameraError("Camera access denied or unavailable.");
    }
  }, []);

  const stopCamera = useCallback(() => {
    // Stop detecting first
    isDetectingRef.current = false;
    disconnectWebSocket();
    setIsDetecting(false);
    setDetectedGesture("—");
    setLogs([]);

    const canvas = canvasRef.current;
    if (canvas) canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);

    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    // DO NOT reset frameIdxRef — keeps incrementing across camera sessions
  }, [disconnectWebSocket]);

  // ── Detection toggle ─────────────────────────────────────────────────────
  const toggleDetection = useCallback(() => {
    setIsDetecting((prev) => {
      const next = !prev;
      isDetectingRef.current = next;

      if (next) {
        setLogs([]);
        connectWebSocket();
      } else {
        disconnectWebSocket();
        setDetectedGesture("—");
        setLogs([]);
      }
      return next;
    });
  }, [connectWebSocket, disconnectWebSocket]);

  const toggleCategory = (name) =>
    setOpenCategories((prev) => ({ ...prev, [name]: !prev[name] }));

  const canDetect = cameraActive && mpReady;

  return (
    <div style={{ width: "100%", fontFamily: "'DM Sans', sans-serif" }}>

      {/* Page header */}
      <div style={{ padding: "48px 8% 0" }}>
        <div style={{
          display: "inline-flex", alignItems: "center", gap: "6px",
          background: "#f8fafc", border: "1px solid #e2e8f0", borderRadius: "20px",
          padding: "5px 16px", fontSize: "12px", fontWeight: 600,
          color: "#64748b", letterSpacing: "0.5px", marginBottom: "16px",
        }}>
          <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#e53e3e", display: "inline-block" }} />
          Live Recognition
        </div>
        <h1 style={{
          fontSize: "clamp(22px, 3vw, 36px)", fontWeight: 700,
          letterSpacing: "-0.5px", color: "#1e293b", margin: "0 0 8px",
        }}>
          FSL Gesture Recognizer
        </h1>
        <p style={{ fontSize: "clamp(13px, 1.2vw, 15px)", color: "#64748b", margin: "0 0 28px" }}>
          {mpReady
            ? "Position your hand clearly in frame, then start recognition."
            : "Loading hand tracking model…"}
        </p>
        <hr style={{ border: "none", borderTop: "1px solid #f1f5f9", margin: 0 }} />
      </div>

      {/* Main grid */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "minmax(0, 1fr) clamp(260px, 26%, 340px)",
        gap: "24px",
        padding: "28px 8% 72px",
        alignItems: "start",
      }}>

        {/* LEFT */}
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>

          {/* Camera viewport */}
          <div style={{
            background: "#111", borderRadius: "14px", overflow: "hidden",
            aspectRatio: "16 / 9", position: "relative",
            border: isDetecting ? "2px solid #e53e3e" : "2px solid transparent",
            transition: "border-color 0.2s ease",
          }}>
            <video
              ref={videoRef}
              muted
              playsInline
              style={{
                width: "100%", height: "100%", objectFit: "cover",
                transform: "scaleX(-1)", display: cameraActive ? "block" : "none",
              }}
            />

            <canvas
              ref={canvasRef}
              style={{
                position: "absolute",
                inset: 0,
                width: "100%",
                height: "100%",
                pointerEvents: "none",
                display: cameraActive ? "block" : "none",
              }}
            />

            {!cameraActive && (
              <div style={{
                position: "absolute", inset: 0, display: "flex",
                flexDirection: "column", justifyContent: "center",
                alignItems: "center", gap: "10px",
              }}>
                <div style={{ fontSize: "48px" }}>📷</div>
                <p style={{ fontSize: "14px", color: "#666", margin: 0 }}>
                  {cameraError || "Camera not started"}
                </p>
              </div>
            )}

            {!mpReady && cameraActive && (
              <div style={{
                position: "absolute", bottom: "12px", left: "12px",
                background: "rgba(0,0,0,0.6)", color: "#fbbf24",
                padding: "5px 12px", borderRadius: "20px",
                fontSize: "12px", fontWeight: 600,
              }}>
                ⏳ Loading model…
              </div>
            )}

            {isDetecting && (
              <div style={{
                position: "absolute", top: "12px", left: "12px",
                background: "rgba(0,0,0,0.6)", color: "#fff",
                padding: "5px 12px", borderRadius: "20px",
                fontSize: "12px", fontWeight: 600,
                display: "flex", alignItems: "center", gap: "6px",
              }}>
                <span style={{
                  width: 6, height: 6, borderRadius: "50%",
                  background: "#e53e3e", display: "inline-block",
                }} />
                LIVE
              </div>
            )}
          </div>

          {/* Result card */}
          <div style={{
            background: "#f8fafc", border: "1px solid #f1f5f9",
            borderRadius: "14px", padding: "clamp(16px, 2vw, 24px)",
            display: "flex", justifyContent: "space-between", alignItems: "center",
          }}>
            <div>
              <p style={{
                fontSize: "11px", fontWeight: 600, letterSpacing: "2px",
                textTransform: "uppercase", color: "#94a3b8", margin: "0 0 6px",
              }}>
                Detected Gesture
              </p>
              <p style={{
                fontSize: "clamp(28px, 4vw, 42px)", fontWeight: 700,
                fontFamily: "'Space Mono', monospace",
                color: "#1e293b", margin: 0, letterSpacing: "-1px",
              }}>
                {detectedGesture}
              </p>
            </div>
            <div style={{
              width: "clamp(44px, 5vw, 60px)", height: "clamp(44px, 5vw, 60px)",
              borderRadius: "50%", background: "rgba(229,62,62,0.08)",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: "clamp(20px, 2.5vw, 26px)",
            }}>
              ✋
            </div>
          </div>

          {/* Controls */}
          <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
            {!cameraActive ? (
              <button onClick={startCamera} style={{
                flex: 1, background: "#e53e3e", color: "#fff", border: "none",
                borderRadius: "10px", padding: "13px 24px",
                fontSize: "clamp(13px, 1.2vw, 15px)", fontWeight: 600, cursor: "pointer",
              }}>
                Start Camera
              </button>
            ) : (
              <>
                <button
                  onClick={toggleDetection}
                  disabled={!canDetect}
                  style={{
                    flex: 1,
                    background: !canDetect ? "#94a3b8" : isDetecting ? "#111" : "#e53e3e",
                    color: "#fff", border: "none", borderRadius: "10px",
                    padding: "13px 24px", fontSize: "clamp(13px, 1.2vw, 15px)",
                    fontWeight: 600,
                    cursor: canDetect ? "pointer" : "not-allowed",
                    transition: "background 0.2s ease",
                  }}>
                  {!mpReady ? "Loading model…" : isDetecting ? "Stop Recognition" : "Start Recognition"}
                </button>
                <button onClick={stopCamera} style={{
                  background: "transparent", border: "1px solid #d1d5db",
                  borderRadius: "10px", padding: "13px 24px",
                  fontSize: "clamp(13px, 1.2vw, 15px)", fontWeight: 500,
                  cursor: "pointer", color: "#1e293b",
                }}>
                  Stop Camera
                </button>
              </>
            )}
          </div>

          {/* Console log */}
          {isDetecting && (
            <div style={{
              background: "#0f172a", borderRadius: "10px",
              padding: "12px 16px", fontFamily: "'Space Mono', monospace",
              fontSize: "12px", color: "#94a3b8",
              maxHeight: "160px", overflowY: "auto",
              display: "flex", flexDirection: "column", gap: "4px",
            }}>
              <p style={{
                margin: "0 0 8px", fontSize: "10px", letterSpacing: "2px",
                textTransform: "uppercase", color: "#475569",
              }}>
                Console
              </p>
              {logs.length === 0 && (
                <span style={{ color: "#334155" }}>Waiting for gesture...</span>
              )}
              {logs.map(log => (
                <div key={log.id} style={{ display: "flex", gap: "12px" }}>
                  <span style={{ color: "#334155", minWidth: "70px" }}>{log.time}</span>
                  <span>{log.text}</span>
                </div>
              ))}
            </div>
          )}

        </div>

        {/* RIGHT — Vocabulary panel */}
        <div style={{
          background: "#f8fafc", border: "1px solid #f1f5f9",
          borderRadius: "14px", padding: "clamp(16px, 2vw, 20px)",
          maxHeight: "74vh", overflowY: "auto",
        }}>
          <p style={{
            fontSize: "11px", fontWeight: 600, letterSpacing: "2px",
            textTransform: "uppercase", color: "#94a3b8", margin: "0 0 14px",
          }}>
            Supported Vocabulary
          </p>

          <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
            {VOCAB_CATEGORIES.map((cat) => {
              const open = openCategories[cat.name];
              return (
                <div key={cat.name}>
                  <button
                    onClick={() => toggleCategory(cat.name)}
                    style={{
                      width: "100%", display: "flex", justifyContent: "space-between",
                      alignItems: "center", padding: "9px 12px",
                      background: open ? "rgba(229,62,62,0.06)" : "white",
                      border: "1px solid #f1f5f9", borderRadius: "8px",
                      fontSize: "13px", fontWeight: 600,
                      color: open ? "#e53e3e" : "#1e293b",
                      cursor: "pointer", transition: "all 0.15s ease",
                    }}>
                    {cat.name}
                    <span style={{ fontSize: "10px", color: "#94a3b8" }}>
                      {open ? "▲" : "▼"}
                    </span>
                  </button>

                  {open && (
                    <div style={{
                      padding: "8px 4px 6px",
                      display: "flex", flexWrap: "wrap", gap: "6px",
                    }}>
                      {cat.words.map((w) => (
                        <span key={w} style={{
                          background: "white", border: "1px solid #e2e8f0",
                          borderRadius: "6px", padding: "4px 10px",
                          fontSize: "12px", color: "#475569",
                        }}>
                          {w}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

      </div>
    </div>
  );
}
