import { useState, useRef, useCallback } from "react";

const VOCAB_CATEGORIES = [
  { name: "Alphabets", words: ["Good morning", "Good afternoon", "Good evening", "Hello", "Goodbye", "Thank you", "You're welcome"] },
  { name: "Survival", words: ["Help", "Emergency", "Police", "Hospital", "Fire", "Stop", "Wait"] },
  { name: "Number", words: ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"] },
  { name: "Calendar", words: ["January", "February", "March", "April", "May", "June", "July"] },
  { name: "Days", words: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"] },
  { name: "Family", words: ["Mother", "Father", "Sister", "Brother", "Grandmother", "Grandfather"] },
  { name: "Relationship", words: ["Friend", "Partner", "Colleague", "Neighbor"] },
  { name: "Food", words: ["Rice", "Water", "Bread", "Fish", "Vegetable", "Fruit"] },
  { name: "Color", words: ["Red", "Blue", "Green", "Yellow", "White", "Black"] },
  { name: "Drink", words: ["Water", "Coffee", "Tea", "Juice", "Milk"] },
];

export default function Recognizer() {
  const videoRef = useRef(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [detectedGesture, setDetectedGesture] = useState("—");
  const [isDetecting, setIsDetecting] = useState(false);
  const [openCategories, setOpenCategories] = useState({ Alphabets: true });

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

  const stopCamera = useCallback(() => {
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    setIsDetecting(false);
    setDetectedGesture("—");
  }, []);

  const toggleCategory = (name) =>
    setOpenCategories((prev) => ({ ...prev, [name]: !prev[name] }));

  return (
    <div style={{ width: "100%", fontFamily: "'DM Sans', sans-serif" }}>

      {/* Page header — mirrors Home's hero rhythm */}
      <div style={{ padding: "48px 8% 0" }}>
        <div style={{
          display: "inline-flex", alignItems: "center", gap: "6px",
          background: "#f8fafc", border: "1px solid #e2e8f0", borderRadius: "20px",
          padding: "5px 16px", fontSize: "12px", fontWeight: 600,
          color: "#64748b", letterSpacing: "0.5px", marginBottom: "16px"
        }}>
          <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#e53e3e", display: "inline-block" }} />
          Live Recognition
        </div>
        <h1 style={{
          fontSize: "clamp(22px, 3vw, 36px)", fontWeight: 700,
          letterSpacing: "-0.5px", color: "#1e293b", margin: "0 0 8px"
        }}>
          FSL Gesture Recognizer
        </h1>
        <p style={{ fontSize: "clamp(13px, 1.2vw, 15px)", color: "#64748b", margin: "0 0 28px" }}>
          Position your hand clearly in frame, then start recognition.
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
            background: "#111",
            borderRadius: "14px",
            overflow: "hidden",
            aspectRatio: "16 / 9",
            position: "relative",
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
                  background: "#e53e3e", display: "inline-block"
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
                textTransform: "uppercase", color: "#94a3b8", margin: "0 0 6px"
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
                <button onClick={() => setIsDetecting((d) => !d)} style={{
                  flex: 1,
                  background: isDetecting ? "#111" : "#e53e3e",
                  color: "#fff", border: "none", borderRadius: "10px",
                  padding: "13px 24px", fontSize: "clamp(13px, 1.2vw, 15px)",
                  fontWeight: 600, cursor: "pointer", transition: "background 0.2s ease",
                }}>
                  {isDetecting ? "Stop Recognition" : "Start Recognition"}
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