import { useCallback, useEffect, useState } from "react";
import { Link, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { PredictView } from "./views/PredictView";
import { PerformanceView } from "./views/PerformanceView";
import { MethodView } from "./views/MethodView";
import { StyleguideView } from "./views/StyleguideView";

function useTheme() {
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    return document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
  });
  const toggle = useCallback(() => {
    setTheme((current) => {
      const next = current === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      window.localStorage.setItem("theme", next);
      return next;
    });
  }, []);
  return { theme, toggle };
}

function Header() {
  const location = useLocation();
  const [language, setLanguage] = useState<"en" | "sq">(() => {
    return window.localStorage.getItem("language") === "sq" ? "sq" : "en";
  });
  const { theme, toggle: toggleTheme } = useTheme();
  const isActive = (path: string) => location.pathname === path;

  useEffect(() => {
    document.documentElement.setAttribute("lang", language === "sq" ? "sq" : "en");
    window.localStorage.setItem("language", language);
  }, [language]);

  return (
    <header className="site-header">
      <div className="container site-header__inner">
        <Link className="wordmark" to="/">
          Football Predictor
        </Link>
        <nav className="site-nav" aria-label="Primary">
          <Link aria-current={isActive("/") ? "page" : undefined} className="site-nav__link" to="/">
            Predict
          </Link>
          <Link aria-current={isActive("/performance") ? "page" : undefined} className="site-nav__link" to="/performance">
            Performance
          </Link>
          <Link aria-current={isActive("/method") ? "page" : undefined} className="site-nav__link" to="/method">
            Method
          </Link>
        </nav>
        <div className="site-controls">
          <button type="button" className="text-button" onClick={toggleTheme} aria-label="Toggle colour theme">
            {theme === "dark" ? "Dark" : "Light"}
          </button>
          <button
            type="button"
            className="text-button"
            onClick={() => setLanguage((current) => (current === "en" ? "sq" : "en"))}
            aria-label="Toggle language"
          >
            {language === "en" ? "EN" : "SQ"}
          </button>
        </div>
      </div>
    </header>
  );
}

function Footer() {
  return (
    <footer className="site-footer">
      <div className="container">
        <p>Data source: football-data.co.uk. Software under the MIT licence.</p>
        <p className="site-footer__disclaimer">
          Estimates only. Not betting advice and not a guarantee of the result.
        </p>
        <p>
          <a href="https://github.com/ankelbega/Football-Predictor" target="_blank" rel="noreferrer">
            Source code
          </a>
        </p>
      </div>
    </footer>
  );
}

function App() {
  return (
    <>
      <Header />
      <main id="main-content">
        <Routes>
          <Route path="/" element={<PredictView />} />
          <Route path="/performance" element={<PerformanceView />} />
          <Route path="/method" element={<MethodView />} />
          <Route path="/styleguide" element={<StyleguideView />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
      <Footer />
    </>
  );
}

export default App;
