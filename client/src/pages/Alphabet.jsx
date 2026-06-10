import { useState, useRef, useCallback } from "react";
import { useWebSocket } from "../hooks/useWebsocketAlphabet";

const VOCAB_CATEGORIES = [
  { name: "Alphabets", words: [
  "A", "B", "C", "D", "E", "F", "G",
  "H", "I", "J", "K", "L", "M", "N",
  "O", "P", "Q", "R", "S", "T", "U",
  "V", "W", "X", "Y", "Z"
  ]},
];


export default function Alphabets() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const intervalRef = useRef(null);

  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [detectedGesture, setDetectedGesture] = useState("—");
  const [confidence, setConfidence] = useState(0);
  const [isDetecting, setIsDetecting] = useState(false);
  const [openCategories, setOpenCategories] = useState({ Alphabets: true });

  // ✅ WebSocket hook (FIXED to parse structured response)
  const { connectWebSocket, disconnectWebSocket, wsRef } =
    useWebSocket((data) => {
      if (data?.type === "prediction") {
        setDetectedGesture(data.label);
        setConfidence(data.confidence);
      }
    });

  // 📡 FRAME SENDER (FIXED PROTOCOL)
  const sendFrame = useCallback(() => {
    const video = videoRef.current;
    const ws = wsRef.current;

    if (!video || !ws || ws.readyState !== 1) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.width = 224;
    canvas.height = 224;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, 224, 224);

    const frame = canvas.toDataURL("image/jpeg").split(",")[1];

    ws.send(
      JSON.stringify({
        action: "predict",
        frame,
      })
    );
  }, [wsRef]);

  // 🎥 START CAMERA (UNCHANGED STYLE)
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        setCameraActive(true);
        setCameraError(null);
      }
    } catch (e) {
      setCameraError("Camera access denied or unavailable.");
    }
  }, []);

  // 🛑 STOP CAMERA (UNCHANGED STYLE)
  const stopCamera = useCallback(() => {
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }

    setCameraActive(false);
    setIsDetecting(false);
    setDetectedGesture("—");
    setConfidence(0);
  }, []);

  const toggleCategory = (name) =>
    setOpenCategories((prev) => ({ ...prev, [name]: !prev[name] }));

  return (
    <div style={{ width: "100%", fontFamily: "'DM Sans', sans-serif" }}>

      {/* HEADER (UNCHANGED) */}
      <div style={{ padding: "48px 8% 0" }}>
        <div style={{
          display: "inline-flex", alignItems: "center", gap: "6px",
          background: "#f8fafc", border: "1px solid #e2e8f0", borderRadius: "20px",
          padding: "5px 16px", fontSize: "12px", fontWeight: 600,
          color: "#64748b", letterSpacing: "0.5px", marginBottom: "16px"
        }}>
          <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#e53e3e" }} />
          Live Recognition
        </div>

        <h1 style={{
          fontSize: "clamp(22px, 3vw, 36px)", fontWeight: 700,
          color: "#1e293b", margin: "0 0 8px"
        }}>
          FSL Gesture Recognizer
        </h1>

        <p style={{ fontSize: "clamp(13px, 1.2vw, 15px)", color: "#64748b" }}>
          Position your hand clearly in frame, then start recognition.
        </p>
      </div>

      {/* GRID (UNCHANGED STYLE) */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "minmax(0, 1fr) clamp(260px, 26%, 340px)",
        gap: "24px",
        padding: "28px 8% 72px",
      }}>

        {/* LEFT */}
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>

          {/* CAMERA */}
          <div style={{
            background: "#111",
            borderRadius: "14px",
            overflow: "hidden",
            aspectRatio: "16 / 9",
            position: "relative",
            border: isDetecting ? "2px solid #e53e3e" : "2px solid transparent",
          }}>
            <video
              ref={videoRef}
              muted
              playsInline
              style={{
                width: "100%",
                height: "100%",
                objectFit: "cover",
                transform: "scaleX(-1)",
                display: cameraActive ? "block" : "none",
              }}
            />

            {!cameraActive && (
              <div style={{
                position: "absolute", inset: 0,
                display: "flex", flexDirection: "column",
                justifyContent: "center", alignItems: "center"
              }}>
                <div style={{ fontSize: "48px" }}>📷</div>
                <p style={{ fontSize: "14px", color: "#666" }}>
                  {cameraError || "Camera not started"}
                </p>
              </div>
            )}
          </div>

          {/* RESULT (UNCHANGED STYLE + confidence added) */}
          <div style={{
            background: "#f8fafc",
            border: "1px solid #f1f5f9",
            borderRadius: "14px",
            padding: "24px",
            display: "flex",
            justifyContent: "space-between"
          }}>
            <div>
              <p style={{
                fontSize: "11px",
                fontWeight: 600,
                letterSpacing: "2px",
                color: "#94a3b8"
              }}>
                Detected Gesture
              </p>

              <p style={{
                fontSize: "42px",
                fontWeight: 700,
                fontFamily: "'Space Mono', monospace",
                color: "#1e293b"
              }}>
                {detectedGesture}
              </p>

              <p style={{ fontSize: "12px", color: "#64748b" }}>
                Confidence: {(confidence * 100).toFixed(1)}%
              </p>
            </div>
          </div>

          {/* CONTROLS (UNCHANGED STYLE) */}
          <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>

            {!cameraActive ? (
              <button onClick={startCamera} style={{
                flex: 1,
                background: "#e53e3e",
                color: "#fff",
                border: "none",
                borderRadius: "10px",
                padding: "13px 24px",
                fontWeight: 600,
                cursor: "pointer",
              }}>
                Start Camera
              </button>
            ) : (
              <>
                <button
                  onClick={() => {
                    setIsDetecting((prev) => {
                      const next = !prev;

                      if (next) {
                        connectWebSocket();
                        intervalRef.current = setInterval(sendFrame, 100);
                      } else {
                        disconnectWebSocket();
                        clearInterval(intervalRef.current);
                        intervalRef.current = null;
                      }

                      return next;
                    });
                  }}
                  style={{
                    flex: 1,
                    background: isDetecting ? "#111" : "#e53e3e",
                    color: "#fff",
                    border: "none",
                    borderRadius: "10px",
                    padding: "13px 24px",
                    fontWeight: 600,
                    cursor: "pointer",
                  }}
                >
                  {isDetecting ? "Stop Recognition" : "Start Recognition"}
                </button>

                <button onClick={stopCamera} style={{
                  border: "1px solid #d1d5db",
                  borderRadius: "10px",
                  padding: "13px 24px",
                  background: "transparent",
                  cursor: "pointer"
                }}>
                  Stop Camera
                </button>
              </>
            )}
          </div>
        </div>

        {/* RIGHT PANEL (UNCHANGED) */}
        <div style={{
          background: "#f8fafc",
          border: "1px solid #f1f5f9",
          borderRadius: "14px",
          padding: "20px",
          maxHeight: "74vh",
          overflowY: "auto",
        }}>
          <p style={{
            fontSize: "11px",
            fontWeight: 600,
            letterSpacing: "2px",
            color: "#94a3b8"
          }}>
            Supported Vocabulary
          </p>

          {VOCAB_CATEGORIES.map((cat) => (
            <div key={cat.name}>
              <button
                onClick={() => toggleCategory(cat.name)}
                style={{
                  width: "100%",
                  display: "flex",
                  justifyContent: "space-between",
                  padding: "9px 12px",
                  border: "1px solid #f1f5f9",
                  borderRadius: "8px",
                  background: openCategories[cat.name] ? "rgba(229,62,62,0.06)" : "white",
                  cursor: "pointer"
                }}
              >
                {cat.name}
              </button>

              {openCategories[cat.name] && (
                <div style={{ padding: "8px", display: "flex", flexWrap: "wrap", gap: "6px" }}>
                  {cat.words.map((w) => (
                    <span key={w} style={{
                      fontSize: "12px",
                      padding: "4px 10px",
                      border: "1px solid #e2e8f0",
                      borderRadius: "6px"
                    }}>
                      {w}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>

      </div>

      {/* hidden canvas */}
      <canvas ref={canvasRef} style={{ display: "none" }} />
    </div>
  );
}
