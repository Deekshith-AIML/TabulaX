// Dataset helpers shared by the preview table and edit flows.

// Internal, stable per-row identifier maintained by the backend session store.
// It travels with preview records so rows can be selected/deleted, but it is
// never a real dataset column: exclude it from headers, cells and exports.
export const ROW_ID_KEY = "_tabula_row_id";

export type DataRow = Record<string, unknown>;

/** Column keys of a row, excluding internal bookkeeping fields. */
export function dataKeys(row: DataRow | undefined): string[] {
  if (!row) return [];
  return Object.keys(row).filter((key) => key !== ROW_ID_KEY);
}

/** The internal id of a row, falling back to its display index. */
export function rowIdOf(row: DataRow, index: number): number {
  const value = row[ROW_ID_KEY];
  return typeof value === "number" ? value : index;
}
