import type { ButtonHTMLAttributes, ReactNode } from "react";
import styles from "./Button.module.css";

type Variant = "primary" | "secondary" | "text";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  children: ReactNode;
}

export function Button({ variant = "primary", className, children, ...rest }: ButtonProps) {
  const classes = [styles.button, styles[variant], className ?? ""].join(" ").trim();
  return (
    <button type="button" className={classes} {...rest}>
      {children}
    </button>
  );
}
