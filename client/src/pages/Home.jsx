import { useNavigate } from "react-router-dom";

function Home() {
  const navigate = useNavigate();

  return (
    <section className="section">
      <h1 className="section-title">
        QZPH-SVM FSL Recognition
      </h1>

      <p className="section-subtitle">
        Filipino Sign Language Recognition System
      </p>

      <button
        className="primary-button"
        onClick={() => navigate("/recognizer")}
      >
        Start Recognition
      </button>
    </section>
  );
}

export default Home;
