import styles from "./Table.module.css";

interface Column<T> {
  key: keyof T;
  header: string;
}

interface TableProps<T extends object> {
  caption: string;
  columns: Column<T>[];
  rows: T[];
}

export function Table<T extends object>({ caption, columns, rows }: TableProps<T>) {
  return (
    <div className={styles.wrapper} role="region" aria-label={caption} tabIndex={0}>
      <table className={styles.table}>
        <caption className={styles.caption}>{caption}</caption>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={String(column.key)} scope="col">
                {column.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index}>
              {columns.map((column) => (
                <td key={String(column.key)}>{String(row[column.key])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
