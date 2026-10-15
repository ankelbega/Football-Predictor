import type { ReactNode } from "react";
import styles from "./Container.module.css";

interface ContainerProps {
  children: ReactNode;
  className?: string;
}

export function Container({ children, className }: ContainerProps) {
  const classes = className ? `${styles.container} ${className}` : styles.container;
  return <div className={classes}>{children}</div>;
}
