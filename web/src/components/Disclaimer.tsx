import styles from "./Disclaimer.module.css";

interface DisclaimerProps {
  text: string;
}

export function Disclaimer({ text }: DisclaimerProps) {
  return <p className={styles.disclaimer}>{text}</p>;
}
