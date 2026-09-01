import { useNavigate } from "react-router-dom";

function Home() {
  const navigate = useNavigate();

  return (
    <main style={{ width: "100%", padding: 0, fontFamily: "'DM Sans', sans-serif" }}>

      {/* Hero */}
      <section style={{ 
        padding: "80px 8% 64px", 
        textAlign: "center", 
        display: "flex", 
        flexDirection: "column", 
        alignItems: "center", 
        height: "auto", 
        minHeight: "fit-content", 
        background: "#f8fafc"
      }}>
        <div style={{
          display: "inline-flex", 
          alignItems: "center", 
          gap: "6px",
          background: "#ffffff", 
          border: "1px solid #e2e8f0", 
          borderRadius: "20px",
          padding: "5px 16px", 
          fontSize: "13px", 
          fontWeight: 600,
          color: "#64748b", 
          letterSpacing: "0.5px", 
          marginBottom: "28px"
        }}>
          <span style={{ 
            width: 7, 
            height: 7, 
            borderRadius: "50%", 
            background: "#e53e3e", 
            display: "inline-block" 
            }} />
          Dynamic FSL Recognition
        </div>

        <h1 style={{
          fontSize: "clamp(28px, 4vw, 52px)", 
          fontWeight: 700, 
          lineHeight: 1.15,
          letterSpacing: "-1px", 
          maxWidth: "720px", 
          margin: "0 auto 20px"
        }}>
          Translating Movement. <br />
          <span style={{ 
            color: "#e53e3e" 
            }}>
              Amplifying Voices.</span>
        </h1>
        
        <h2 style={{
          fontSize: "clamp(18px, 2vw, 24px)", 
          fontWeight: 600, 
          color: "#1e293b",
          maxWidth: "800px", 
          margin: "0 auto 24px"
        }}>
          Advancing Filipino Sign Language through the geometry of gesture.
        </h2>

        <p style={{
          fontSize: "clamp(14px, 1.5vw, 17px)", 
          color: "#64748b",
          maxWidth: "900px", 
          margin: "0 auto 36px", 
          lineHeight: 1.75, 
          textAlign: "justify"
        }}>
          Filipino Sign Language (FSL) is the vibrant, dynamic voice of the Deaf community—a language defined not by sound, but by the continuous, expressive flow of movement through space and time. To truly honor and advocate for this richness, digital spaces must learn to understand FSL in its natural state. Our research bridges this gap by moving beyond flat images, capturing the intricate geometry of gestures through 3D-landmark point clouds. Because a sign is a continuous sequence, our system utilizes weighted temporal decay alongside Quasi-Zigzag Persistence Homology (QZPH) to track how the topological structure of these gestures evolves. By cutting through topological ambiguity, our SVM-backed model translates complex spatial grammar with precision. This isn't just an algorithm; it's a step toward inclusive civic technology where every gesture is accurately recognized, respected, and understood.
        </p>

        <div style={{ 
          display: "flex", 
          flexDirection: "column", 
          alignItems: "center", 
          gap: "16px" }}>
          <button
            onClick={() => navigate("/recognizer")}
            style={{
              background: "#e53e3e", 
              color: "#fff", 
              border: "none",
              borderRadius: "10px", 
              padding: "14px 28px",
              fontSize: "clamp(13px, 1.2vw, 15px)", 
              fontWeight: 600, 
              cursor: "pointer",
              display: "flex", 
              alignItems: "center", 
              gap: "8px"
            }}>
            Launch Recognizer →
          </button>
          
          <p style={{ 
            fontSize: "12px", 
            color: "#94a3b8", 
            display: "flex", 
            alignItems: "center", 
            gap: "6px" }}>
            <span style={{ 
              color: "#e53e3e" 
              }}>
                ⓘ</span> Accessibility first. Designed for Deaf and Hard-of-Hearing users.
          </p>
        </div>
      </section>

      <hr style={{ 
        border: "none", 
        borderTop: "1px solid #f1f5f9", 
        margin: "0 8%" }} />

      {/* Feature Cards / System Architecture */}
      <section style={{ 
        padding: "56px 8%" 
        }}>
        <p style={{
          fontSize: "11px", 
          fontWeight: 600, 
          letterSpacing: "2px",
          textTransform: "uppercase", 
          color: "#94a3b8", 
          marginBottom: "28px"
        }}>
          System Architecture
        </p>

        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
          gap: "20px"
        }}>
          {[
            {
              icon: "⬡",
              title: "Pre-Processing & Keyframing",
              body: "MediaPipe extracts 3D hand landmarks from video sequences, followed by temporal keyframe selection to filter redundant frames and clean transitional noise.",
            },
            {
              icon: "◈",
              title: "Spatiotemporal Processing",
              body: "Constructs multi-scale Vietoris-Rips filtrations and applies Quasi-Zigzag Persistent Homology (QZPH) to track topological features as they appear and disappear over time.",
            },
            {
              icon: "▤",
              title: "ZZ-GRIL Feature Extraction",
              body: "Transforms dynamic spatiotemporal persistence descriptors into stable, fixed-length 1D feature vectors suitable for high-dimensional classification.",
            },
            {
              icon: "◎",
              title: "SVM Classification",
              body: "A Support Vector Machine with a Radial Basis Function (RBF) kernel analyzes the topological features to predict the corresponding word among 105 FSL gesture classes.",
            },
          ].map(({ icon, title, body }) => (
            <div key={title} style={{
              background: "#f8fafc", 
              border: "1px solid #f1f5f9",
              borderRadius: "14px", 
              padding: "clamp(16px, 2vw, 28px)"
            }}>
              <div style={{
                width: 40, 
                height: 40, 
                borderRadius: "10px",
                background: "rgba(229,62,62,0.1)", 
                display: "flex",
                alignItems: "center", 
                justifyContent: "center",
                marginBottom: "16px", 
                color: "#e53e3e",
                fontSize: "20px"
              }}>
                {icon}
              </div>
              <h3 style={{ 
                fontSize: "clamp(14px, 1.2vw, 16px)", 
                fontWeight: 600, 
                margin: "0 0 10px", 
                color: "#1e293b" }}>
                {title}
              </h3>
              <p style={{ 
                fontSize: "clamp(13px, 1vw, 14px)", 
                color: "#64748b", 
                lineHeight: 1.7, 
                margin: 0 }}>
                {body}
              </p>
            </div>
          ))}
        </div>
      </section>

      <hr style={{ 
        border: "none", 
        borderTop: "1px solid #f1f5f9", 
        margin: "0 8%" }} />

    </main>
  );
}

export default Home;