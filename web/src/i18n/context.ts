import { createContext } from "react";
import { en } from "./en";
import { sq } from "./sq";

export type Locale = "en" | "sq";

export const dictionaries = {
  en,
  sq,
};

export type TranslationKey = keyof typeof en;

export interface I18nValue {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  t: (key: TranslationKey) => string;
}

export const I18nContext = createContext<I18nValue | null>(null);
