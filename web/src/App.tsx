import { useCallback, useState } from "react";
import { Link, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { useI18n } from "./i18n/useI18n";
import { Container } from "./components/Container";
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
  const { locale, setLocale, t } = useI18n();
  const { theme, toggle: toggleTheme } = useTheme();
  const isActive = (path: string) => location.pathname === path;

  return (
    <header className="site-header">
      <Container>
        <div className="site-header__inner">
          <div className="site-header__group">
            <Link className="wordmark" to="/">
              {t("app.name")}
            </Link>
            <nav className="site-nav" aria-label="Primary">
          <Link aria-current={isActive("/") ? "page" : undefined} className="site-nav__link" to="/">
            {t("nav.predict")}
          </Link>
          <Link aria-current={isActive("/performance") ? "page" : undefined} className="site-nav__link" to="/performance">
            {t("nav.performance")}
          </Link>
          <Link aria-current={isActive("/method") ? "page" : undefined} className="site-nav__link" to="/method">
            {t("nav.method")}
          </Link>
        </nav>
          </div>
          <div className="site-controls">
          <button type="button" className="text-button" onClick={toggleTheme} aria-label={t("controls.themeToggle")}>
            {theme === "dark" ? t("controls.darkTheme") : t("controls.lightTheme")}
          </button>
          <button
            type="button"
            className="text-button"
            onClick={() => setLocale(locale === "en" ? "sq" : "en")}
            aria-label={t("controls.languageToggle")}
          >
            {locale === "en" ? "EN" : "SQ"}
          </button>
        </div>
        </div>
      </Container>
    </header>
  );
}

function Footer() {
  const { t } = useI18n();
  return (
    <footer className="site-footer">
      <Container>
        <p>{t("footer.sourceCredit")}. {t("footer.licence")}</p>
        <p className="site-footer__disclaimer">{t("footer.disclaimerEstimates")}</p>
        <p>
          <a href="https://github.com/ankelbega/Football-Predictor" target="_blank" rel="noreferrer">
            {t("footer.sourceCode") ?? "Source code"}
          </a>
        </p>
      </Container>
    </footer>
  );
}

function App() {
  return (
    <>
      <Header />
      <main id="main-content">
        <Container>
          <Routes>
          <Route path="/" element={<PredictView />} />
          <Route path="/performance" element={<PerformanceView />} />
          <Route path="/method" element={<MethodView />} />
          <Route path="/styleguide" element={<StyleguideView />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
        </Container>
      </main>
      <Footer />
    </>
  );
}

export default App;
