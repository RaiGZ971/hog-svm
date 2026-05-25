import { useNavigate } from "react-router-dom";

function Home() {
  const navigate = useNavigate();

  return (
    <main style={{ width: "100%", padding: 0, fontFamily: "'DM Sans', sans-serif" }}>

      {/* Hero */}
      <section style={{ padding: "80px 8% 64px", textAlign: "center" }}>
        <div style={{
          display: "inline-flex", alignItems: "center", gap: "6px",
          background: "#f8fafc", border: "1px solid #e2e8f0", borderRadius: "20px",
          padding: "5px 16px", fontSize: "13px", fontWeight: 600,
          color: "#64748b", letterSpacing: "0.5px", marginBottom: "28px"
        }}>
          <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#e53e3e", display: "inline-block" }} />
          Vision-based FSL Recognition
        </div>

        <h1 style={{
          fontSize: "clamp(28px, 4vw, 52px)", fontWeight: 700, lineHeight: 1.15,
          letterSpacing: "-1px", maxWidth: "720px", margin: "0 auto 20px"
        }}>
          Bridging the{" "}
          <span style={{ color: "#e53e3e" }}>FSL</span>{" "}
          Communication Gap with AI
        </h1>

        <p style={{
          fontSize: "clamp(14px, 1.5vw, 17px)", color: "#64748b",
          maxWidth: "600px", margin: "0 auto 36px", lineHeight: 1.75
        }}>
          An automated system that classifies static Filipino Sign Language alphabet
          gestures using Support Vector Machines and advanced feature extraction.
        </p>

        <div style={{ display: "flex", justifyContent: "center", gap: "12px", flexWrap: "wrap" }}>
          <button
            onClick={() => navigate("/recognizer")}
            style={{
              background: "#e53e3e", color: "#fff", border: "none",
              borderRadius: "10px", padding: "14px 28px",
              fontSize: "clamp(13px, 1.2vw, 15px)", fontWeight: 600, cursor: "pointer"
            }}>
            Start Recognition
          </button>

        </div>
      </section>

      <hr style={{ border: "none", borderTop: "1px solid #f1f5f9", margin: "0 8%" }} />

      {/* Feature Cards */}
      <section style={{ padding: "56px 8%" }}>
        <p style={{
          fontSize: "11px", fontWeight: 600, letterSpacing: "2px",
          textTransform: "uppercase", color: "#94a3b8", marginBottom: "28px"
        }}>
          How it works
        </p>

        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
          gap: "20px"
        }}>
          {[
            {
              icon: "⬡",
              title: "Image preprocessing",
              body: "Images are standardized to 224×224px. Otsu's Thresholding and morphological operations isolate the hand from noisy backgrounds.",
            },
            {
              icon: "◈",
              title: "HOG feature extraction",
              body: "The Histogram of Oriented Gradients algorithm extracts the structural contours and edge directions of each gesture — not raw pixels.",
            },
            {
              icon: "◎",
              title: "SVM classification",
              body: "A Support Vector Machine identifies each of the 26 static alphabet gestures by optimizing the margin between gesture classes.",
            },
          ].map(({ icon, title, body }) => (
            <div key={title} style={{
              background: "#f8fafc", border: "1px solid #f1f5f9",
              borderRadius: "14px", padding: "clamp(16px, 2vw, 28px)"
            }}>
              <div style={{
                width: 40, height: 40, borderRadius: "10px",
                background: "rgba(229,62,62,0.1)", display: "flex",
                alignItems: "center", justifyContent: "center",
                marginBottom: "16px", color: "#e53e3e",
                fontSize: "20px"
              }}>
                {icon}
              </div>
              <h3 style={{ fontSize: "clamp(14px, 1.2vw, 16px)", fontWeight: 600, margin: "0 0 10px", color: "#1e293b" }}>
                {title}
              </h3>
              <p style={{ fontSize: "clamp(13px, 1vw, 14px)", color: "#64748b", lineHeight: 1.7, margin: 0 }}>
                {body}
              </p>
            </div>
          ))}
        </div>
      </section>

      <hr style={{ border: "none", borderTop: "1px solid #f1f5f9", margin: "0 8%" }} />

      

    </main>
  );
}

export default Home;