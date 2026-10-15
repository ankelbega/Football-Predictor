import { type ReactNode, useCallback, useEffect, useMemo, useState } from "react";
import { I18nContext, dictionaries, type I18nValue, type Locale, type TranslationKey } from "./context";

function initialLocale(): Locale {
  const stored = window.localStorage.getItem("language");
  return stored === "sq" ? "sq" : "en";
}

export function I18nProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(initialLocale);

  const setLocale = useCallback((next: Locale) => {
    setLocaleState(next);
    window.localStorage.setItem("language", next);
  }, []);

  const t = useCallback(
    (key: TranslationKey) => {
      return dictionaries[locale][key] ?? dictionaries.en[key] ?? key;
    },
    [locale],
  );

  useEffect(() => {
    document.documentElement.setAttribute("lang", locale === "sq" ? "sq" : "en");
  }, [locale]);

  const value = useMemo<I18nValue>(() => ({ locale, setLocale, t }), [locale, setLocale, t]);
  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}
