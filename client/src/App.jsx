import "./App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";

import Navbar from "./components/NavBar.jsx";
import Footer from "./components/Footer.jsx";

import Home from "./pages/Home.jsx";
import Recognizer from "./pages/Recognizer.jsx";

function App() {
  return (
    <BrowserRouter>
      <div className="app-container">
        <Navbar />

        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/recognizer" element={<Recognizer />} />
          </Routes>
        </main>

        <Footer />
      </div>
    </BrowserRouter>
  );
}

export default App;
