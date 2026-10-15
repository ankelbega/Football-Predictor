import "@fontsource-variable/newsreader";
import "@fontsource-variable/instrument-sans";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import "./styles/global.css";
import { I18nProvider } from "./i18n/I18nProvider";
import App from "./App";

const rootElement = document.getElementById("root");
if (!rootElement) throw new Error("root element not found");

const storedTheme = window.localStorage.getItem("theme");
const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
const initialTheme = storedTheme === "dark" || (storedTheme === null && prefersDark) ? "dark" : "light";
document.documentElement.setAttribute("data-theme", initialTheme);

ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>
    <BrowserRouter>
      <I18nProvider>
        <App />
      </I18nProvider>
    </BrowserRouter>
  </React.StrictMode>,
);
