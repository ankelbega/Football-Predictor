import { useEffect, useMemo, useRef, useState } from "react";
import styles from "./Combobox.module.css";

interface ComboboxProps {
  id: string;
  label: string;
  options: readonly string[];
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  disabled?: boolean;
}

export function Combobox({ id, label, options, value, onChange, placeholder, disabled }: ComboboxProps) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState(value);
  const [highlightedIndex, setHighlightedIndex] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setQuery(value);
  }, [value]);

  const filtered = useMemo(() => {
    const lowered = query.trim().toLowerCase();
    if (!lowered) return [...options];
    return options.filter((option) => option.toLowerCase().includes(lowered));
  }, [options, query]);

  useEffect(() => {
    if (highlightedIndex >= filtered.length) {
      setHighlightedIndex(Math.max(0, filtered.length - 1));
    }
  }, [filtered, highlightedIndex]);

  useEffect(() => {
    function handlePointer(event: PointerEvent) {
      if (!rootRef.current?.contains(event.target as Node)) {
        setOpen(false);
        setQuery(value);
      }
    }
    document.addEventListener("pointerdown", handlePointer);
    return () => document.removeEventListener("pointerdown", handlePointer);
  }, [value]);

  function select(option: string) {
    onChange(option);
    setQuery(option);
    setOpen(false);
    inputRef.current?.focus();
  }

  return (
    <div className={styles.root} ref={rootRef}>
      <label className={styles.label} htmlFor={id}>
        {label}
      </label>
      <input
        id={id}
        ref={inputRef}
        className={styles.input}
        type="text"
        role="combobox"
        aria-expanded={open}
        aria-autocomplete="list"
        aria-controls={`${id}-listbox`}
        value={query || value}
        placeholder={placeholder}
        disabled={disabled}
        onChange={(event) => {
          setQuery(event.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onKeyDown={(event) => {
          if (event.key === "ArrowDown") {
            event.preventDefault();
            setOpen(true);
            setHighlightedIndex((index) => Math.min(index + 1, Math.max(0, filtered.length - 1)));
          } else if (event.key === "ArrowUp") {
            event.preventDefault();
            setOpen(true);
            setHighlightedIndex((index) => Math.max(0, index - 1));
          } else if (event.key === "Enter" && filtered[highlightedIndex]) {
            event.preventDefault();
            select(filtered[highlightedIndex]);
          } else if (event.key === "Escape") {
            setOpen(false);
            setQuery(value);
          }
        }}
      />
      {open && filtered.length > 0 ? (
        <ul id={`${id}-listbox`} className={styles.listbox} role="listbox">
          {filtered.map((option, index) => (
            <li key={option} role="option" aria-selected={option === value}>
              <button
                type="button"
                className={`${styles.option} ${index === highlightedIndex ? styles.optionHighlighted : ""}`}
                onMouseEnter={() => setHighlightedIndex(index)}
                onClick={() => select(option)}
              >
                {option}
              </button>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
