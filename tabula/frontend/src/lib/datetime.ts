// API timestamps are generated in UTC.  Responses always carry an explicit
// offset ("...+00:00"), which `new Date()` converts to the viewer's local
// zone.  Older or edge-case responses may arrive offset-less; JavaScript
// would parse those as *LOCAL* time and render the wall clock hours off
// (e.g. a 15:03 UTC instant shown as 15:03 instead of 20:33 at UTC+5:30).
// Normalise offset-less datetimes to explicit UTC so every date in the UI
// shows the same instant the server meant.
const HAS_ZONE = /(?:Z|[+-]\d{2}:?\d{2})$/i;
const DATE_ONLY = /^\d{4}-\d{2}-\d{2}$/;

export function parseApiDate(iso: string): Date {
  if (HAS_ZONE.test(iso) || DATE_ONLY.test(iso)) {
    return new Date(iso); // explicit zone, or date-only (spec: UTC midnight)
  }
  return new Date(`${iso}Z`);
}

export function formatApiDate(iso: string | null | undefined): string {
  if (!iso) return '—';
  const parsed = parseApiDate(iso);
  return Number.isNaN(parsed.getTime()) ? '—' : parsed.toLocaleString();
}
