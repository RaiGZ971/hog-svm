import { useLocation, useNavigate } from "react-router-dom";
import "../styles/components.css"

function Navbar() {
  const navigate = useNavigate();
  const location = useLocation();

  const tabs = [
    { name: "home", path: "/" },
    { name: "recognizer", path: "/recognizer" }
  ];

  return (
    <nav className="primary-nav">
      <span className="logo-text">
        QZPH–SVM FSLR
      </span>

      <div className="nav-buttons">
        {tabs.map((tab) => (
          <button
            key={tab.name}
            onClick={() => navigate(tab.path)}
            className={
              location.pathname === tab.path
                ? "nav-button active"
                : "nav-button"
            }
          >
            {tab.name}
          </button>
        ))}
      </div>
    </nav>
  );
}

export default Navbar;
