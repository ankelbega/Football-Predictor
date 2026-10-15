import { useContext } from "react";
import { I18nContext } from "./context";

export function useI18n() {
  const value = useContext(I18nContext);
  if (!value) {
    throw new Error("I18nProvider is missing.");
  }
  return value;
}
