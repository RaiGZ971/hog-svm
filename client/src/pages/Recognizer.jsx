import { useState, useRef, useCallback } from "react";

import "../styles/components.css"

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
  const [openCategories, setOpenCategories] = useState({ Greeting: true });

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

  const toggleCategory = (name) => {
    setOpenCategories((prev) => ({
      ...prev,
      [name]: !prev[name],
    }));
  };

  return (
    <div className="recognizer-page">

      {/* CAMERA SECTION */}
      <div className="recognizer-grid">

        {/* LEFT */}
        <div className="recognizer-left">

          <div className={`camera-box ${isDetecting ? "active" : ""}`}>
            <video
              ref={videoRef}
              className="camera-video"
              muted
              playsInline
              style={{ display: cameraActive ? "block" : "none" }}
            />

            {!cameraActive && (
              <div className="camera-placeholder">
                <div style={{ fontSize: 48 }}>📷</div>
                <p>{cameraError || "Camera not started"}</p>
              </div>
            )}

            {isDetecting && (
              <div className="live-badge">
                <span className="dot"></span>
                LIVE
              </div>
            )}
          </div>

          {/* RESULT */}
          <div className="result-box">
            <p className="label">Detected Gesture</p>
            <p className="gesture">{detectedGesture}</p>
          </div>

          {/* CONTROLS */}
          <div className="controls">
            {!cameraActive ? (
              <button className="btn primary" onClick={startCamera}>
                Start Camera
              </button>
            ) : (
              <>
                <button
                  className={`btn primary ${isDetecting ? "dark" : ""}`}
                  onClick={() => setIsDetecting((d) => !d)}
                >
                  {isDetecting ? "Stop Recognition" : "Start Recognition"}
                </button>

                <button className="btn outline" onClick={stopCamera}>
                  Stop Camera
                </button>
              </>
            )}
          </div>

        </div>

        {/* RIGHT */}
        <div className="recognizer-right">

          <h3>Supported Vocabulary</h3>

          <div className="vocab-list">
            {VOCAB_CATEGORIES.map((cat) => {
              const open = openCategories[cat.name];

              return (
                <div key={cat.name} className="category">
                  <button
                    className={`category-header ${open ? "open" : ""}`}
                    onClick={() => toggleCategory(cat.name)}
                  >
                    {cat.name}
                    <span>{open ? "▲" : "▼"}</span>
                  </button>

                  {open && (
                    <div className="category-items">
                      {cat.words.map((w) => (
                        <div key={w} className="word">
                          {w}
                        </div>
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
