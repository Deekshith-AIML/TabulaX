import React, { useCallback, useEffect, useState } from 'react';
import {
  Check,
  Clock,
  Database,
  Download,
  FileText,
  Loader2,
  RotateCcw,
  RefreshCw,
  Trash2,
  Upload,
} from 'lucide-react';
import { apiUrl, handleSessionExpired, SESSION_EXPIRED_MESSAGE } from '../lib/api';
import { formatApiDate } from '../lib/datetime';

interface VersionSummary {
  version_id: string;
  version_number: number;
  session_id: string;
  history_id: string | null;
  label: string | null;
  created_at: string;
  expires_at: string | null;
  restorable: boolean;
  snapshot_available?: boolean;
  row_count: number;
  columns: string[];
  size_bytes: number;
}

interface HistoryEntry {
  history_id: string;
  session_id: string | null;
  operation: string;
  status: string;
  timestamp: string;
  source_type: string | null;
  source_name: string | null;
  summary: string;
  column: string | null;
  backup_id: string | null;
  restorable: boolean;
  rollback_expires_at: string | null;
  version_id: string | null;
  version: VersionSummary | null;
  warnings: string[];
}

interface VersionEntry {
  version_id: string;
  version_number: number;
  session_id: string;
  label: string | null;
  created_at: string;
  expires_at: string;
  restorable: boolean;
  row_count: number;
  columns: string[];
  size_bytes: number;
}

/** Canonical source categories shared with the backend history API. */
type SourceCategory = 'file' | 'mysql' | 'mongodb';

const SOURCE_LABELS: Record<SourceCategory, string> = {
  file: 'File',
  mysql: 'MySQL',
  mongodb: 'MongoDB',
};

/** Clear-all confirmation titles name the active category explicitly. */
const CLEAR_TITLES: Record<SourceCategory, string> = {
  file: 'Clear all file history?',
  mysql: 'Clear all MySQL history?',
  mongodb: 'Clear all MongoDB history?',
};

const EMPTY_TEXTS: Record<SourceCategory, string> = {
  file: 'No file history yet.',
  mysql: 'No MySQL history yet.',
  mongodb: 'No MongoDB history yet.',
};

interface Props {
  sessionId: string;
  /** Active data-source section; the panel only shows this category. */
  sourceType: SourceCategory;
  /** Called after a successful working-copy restore so the UI refreshes. */
  onRestored: (resp: Record<string, unknown>) => void;
  /** Lift a backup id up so the app can open the source-restore dialog. */
  onRestoreSource?: (backupId: string) => void;
  /** Called after "Open" creates a new working session from a snapshot. */
  onOpened?: (resp: Record<string, unknown>) => void;
}

function detailOf(data: unknown, fallback: string): string {
  const d = (data as { detail?: unknown } | null)?.detail;
  if (typeof d === 'string') return d;
  if (Array.isArray(d)) {
    return d.map((x) => (x as { msg?: string })?.msg || '').filter(Boolean).join('; ') || fallback;
  }
  return fallback;
}

function fmt(iso: string | null | undefined): string {
  if (!iso) return '—';
  try {
    // Shared parser: converts offset-less API timestamps to UTC before
    // rendering so created/expires always show the same wall-clock zone.
    return formatApiDate(iso);
  } catch {
    return iso;
  }
}

function fmtBytes(n: number): string {
  if (!n) return '0 B';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Audit trail + version-snapshot rollback for the WORKING COPY.
 *
 * Scoped to the ACTIVE data-source section: only history entries of the
 * ``sourceType`` category (file / mysql / mongodb) are requested from the
 * backend and displayed; switching sections re-fetches automatically and
 * resets any selection or dialog that is no longer valid.  Entries are
 * selectable (one at a time) and expose source-type-aware actions: Open (new
 * isolated working session), Download (exact snapshot), Restore working copy,
 * Restore source database (destructive, separate dialog with typed
 * confirmation), and Delete This Entry.  Clear All History is scoped to the
 * active category only.
 */
export default function HistoryPanel({
  sessionId,
  sourceType,
  onRestored,
  onRestoreSource,
  onOpened,
}: Props) {
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [versions, setVersions] = useState<VersionEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState<{ text: string; warn: boolean } | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [opening, setOpening] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [downloadFormat, setDownloadFormat] = useState<'csv' | 'json' | 'xlsx'>('csv');
  const [actionMsg, setActionMsg] = useState('');
  const [restoreTarget, setRestoreTarget] = useState<VersionEntry | null>(null);
  const [restoring, setRestoring] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<HistoryEntry | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState('');
  const [showClearAll, setShowClearAll] = useState(false);
  const [clearingAll, setClearingAll] = useState(false);
  const [clearTyped, setClearTyped] = useState('');
  const [clearAllStats, setClearAllStats] = useState<{
    total_entries: number;
    entries_with_snapshot: number;
    entries_with_backup: number;
  } | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const token = localStorage.getItem('token');
      const headers = { Authorization: `Bearer ${token}` };
      // Always request the ACTIVE category from the backend (never filter a
      // mixed list client-side) so other sections' history cannot leak in.
      const histParams = new URLSearchParams({ source_type: sourceType });
      if (sessionId) histParams.set('session_id', sessionId);
      const [histRes, verRes] = await Promise.all([
        fetch(apiUrl(`/history/?${histParams.toString()}`), { headers }),
        sessionId
          ? fetch(apiUrl(`/history/versions/?session_id=${encodeURIComponent(sessionId)}`), {
              headers,
            })
          : Promise.resolve(null),
      ]);
      if (histRes && !histRes.ok) {
        if (histRes.status === 401) {
          handleSessionExpired();
          throw new Error(SESSION_EXPIRED_MESSAGE);
        }
        const body = await histRes.json().catch(() => ({ detail: histRes.statusText }));
        throw new Error(detailOf(body, `HTTP ${histRes.status}`));
      }
      if (histRes) {
        const body = await histRes.json();
        setHistory(Array.isArray(body.history) ? body.history : []);
      }
      if (verRes && verRes.ok) {
        const body = await verRes.json();
        setVersions(Array.isArray(body.versions) ? body.versions : []);
      } else if (verRes && verRes.status === 401) {
        handleSessionExpired();
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, [sessionId, sourceType]);

  useEffect(() => {
    load();
  }, [load]);

  // Section switched: drop transient state that referenced the previous
  // category so nothing stale (selection, dialogs, messages) can act on or
  // display another section's history.
  useEffect(() => {
    setSelectedId(null);
    setDeleteTarget(null);
    setDeleteError('');
    setRestoreTarget(null);
    setActionMsg('');
    setNotice(null);
    setDownloadFormat('csv');
    setShowClearAll(false);
    setClearTyped('');
    setClearAllStats(null);
  }, [sourceType]);

  // If the selected entry is no longer in the (possibly re-fetched) list —
  // deleted, cleared, or belonging to a newly active section — clear it so
  // the details panel never shows an invalid entry.
  useEffect(() => {
    if (selectedId && !history.some((h) => h.history_id === selectedId)) {
      setSelectedId(null);
    }
  }, [history, selectedId]);

  const selected = history.find((h) => h.history_id === selectedId) || null;

  const toggleSelect = (entry: HistoryEntry) => {
    setActionMsg('');
    setNotice(null);
    setDownloadFormat('csv');
    setSelectedId((prev) => (prev === entry.history_id ? null : entry.history_id));
  };

  const openClearAllDialog = async () => {
    setClearTyped('');
    setClearAllStats(null);
    setShowClearAll(true);
    try {
      const token = localStorage.getItem('token');
      // Scoped stats so the dialog reflects the ACTIVE category's impact.
      const res = await fetch(
        apiUrl(`/history/stats/?source_type=${encodeURIComponent(sourceType)}`),
        { headers: { Authorization: `Bearer ${token}` } }
      );
      if (res.ok) setClearAllStats(await res.json());
    } catch {
      /* stats are best-effort; dialog still works without them */
    }
  };

  interface ClearResult {
    message: string;
    partial: boolean;
    deleted?: boolean;
    entries_deleted: number;
    entries_missing: string[];
    snapshots_deleted: number;
    backups_deleted: number;
    snapshots_retained: number;
    backups_retained: number;
    failures: Array<{ target: string; id: string; error: string }>;
  }

  const resultMessage = (body: ClearResult, fallback: string): { text: string; warn: boolean } => {
    const parts = [body.message || fallback];
    const retained = body.snapshots_retained + body.backups_retained;
    if (retained > 0) {
      parts.push(
        `${retained} recovery artifact(s) retained because other history entries still need them.`
      );
    }
    if (body.failures.length > 0) {
      parts.push(
        `${body.failures.length} cleanup item(s) could not be removed and were left in place for a safe retry.`
      );
    }
    return { text: parts.join(' '), warn: body.partial || body.failures.length > 0 };
  };

  const openDeleteDialog = (entry: HistoryEntry) => {
    setDeleteError('');
    setDeleteTarget(entry);
  };

  const deleteEntry = async (entry: HistoryEntry) => {
    if (deleting) return;
    setDeleting(true);
    setDeleteError('');
    try {
      const token = localStorage.getItem('token');
      const res = await fetch(
        apiUrl(`/history/${encodeURIComponent(entry.history_id)}`),
        {
          method: 'DELETE',
          headers: { Authorization: `Bearer ${token}` },
        }
      );
      if (!res.ok) {
        if (res.status === 401) {
          handleSessionExpired();
          throw new Error(SESSION_EXPIRED_MESSAGE);
        }
        const body = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(detailOf(body, `HTTP ${res.status}`));
      }
      const body = (await res.json()) as ClearResult;
      // The backend confirmed the persistent deletion before any state change.
      setHistory((prev) => prev.filter((h) => h.history_id !== entry.history_id));
      setSelectedId((prev) => (prev === entry.history_id ? null : prev));
      setDeleteTarget(null);
      setNotice(resultMessage(body, 'History entry deleted.'));
      // Re-sync from the authoritative API (best effort; a failed refresh
      // cannot resurrect the entry because the backend already deleted it).
      load();
    } catch (err) {
      // Failed request: keep the entry visible and surface the real error.
      setDeleteError((err as Error).message);
    } finally {
      setDeleting(false);
    }
  };

  const confirmClearAll = async () => {
    if (clearingAll) return;
    setClearingAll(true);
    setError('');
    try {
      const token = localStorage.getItem('token');
      // Scoped to the ACTIVE category: the backend re-validates source_type
      // and ownership, so other categories can never be cleared by mistake.
      const res = await fetch(apiUrl('/history/clear_all/'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          confirm: true,
          typed_confirmation: clearTyped,
          source_type: sourceType,
        }),
      });
      if (!res.ok) {
        if (res.status === 401) {
          handleSessionExpired();
          throw new Error(SESSION_EXPIRED_MESSAGE);
        }
        const body = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(detailOf(body, `HTTP ${res.status}`));
      }
      const body = (await res.json()) as ClearResult;
      setShowClearAll(false);
      setSelectedId(null);
      const outcome = resultMessage(
        body,
        `${SOURCE_LABELS[sourceType]} history cleared.`
      );
      setNotice(outcome);
      const fullyCleared = !body.partial && body.failures.length === 0;
      if (fullyCleared) {
        // The authoritative response confirmed every entry was removed.
        setHistory([]);
      }
      // Always re-fetch from the backend: a partial failure shows the
      // entries that actually remain, and a failed refresh keeps the
      // previous (honest) list instead of falsely showing an empty one.
      load();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setClearingAll(false);
    }
  };

  const openEntry = async (entry: HistoryEntry) => {
    if (!entry.version?.restorable || opening) return;
    setOpening(true);
    setActionMsg('');
    try {
      const token = localStorage.getItem('token');
      const res = await fetch(apiUrl('/history/open_version/'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          version_id: entry.version.version_id,
          history_id: entry.history_id,
        }),
      });
      if (!res.ok) {
        if (res.status === 401) {
          handleSessionExpired();
          throw new Error(SESSION_EXPIRED_MESSAGE);
        }
        const body = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(detailOf(body, `HTTP ${res.status}`));
      }
      const body = await res.json();
      onOpened?.(body);
      setSelectedId(null);
      load();
    } catch (err) {
      setActionMsg((err as Error).message);
    } finally {
      setOpening(false);
    }
  };

  const downloadEntry = async (entry: HistoryEntry) => {
    if (!entry.version?.restorable || downloading) return;
    setDownloading(true);
    setActionMsg('');
    try {
      const token = localStorage.getItem('token');
      const res = await fetch(
        apiUrl(
          `/history/versions/${encodeURIComponent(entry.version.version_id)}/download/?format=${downloadFormat}`
        ),
        { headers: { Authorization: `Bearer ${token}` } }
      );
      if (!res.ok) {
        if (res.status === 401) {
          handleSessionExpired();
          throw new Error(SESSION_EXPIRED_MESSAGE);
        }
        const body = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(detailOf(body, `HTTP ${res.status}`));
      }
      const blob = await res.blob();
      const cd = res.headers.get('Content-Disposition') || '';
      const match = /filename="([^"]+)"/.exec(cd);
      const name =
        match?.[1] || `version_v${entry.version.version_number}.${downloadFormat}`;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = name;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      const converted = res.headers.get('X-Format-Converted') === 'true';
      setActionMsg(
        converted
          ? `Downloaded ${name} — exported from the stored snapshot (format converted).`
          : `Downloaded ${name} — the exact saved snapshot.`
      );
    } catch (err) {
      setActionMsg((err as Error).message);
    } finally {
      setDownloading(false);
    }
  };

  const confirmRestore = async (version: VersionEntry) => {
    setRestoring(true);
    setError('');
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(apiUrl('/history/restore_working/'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          session_id: version.session_id,
          version_id: version.version_id,
          confirm: true,
        }),
      });
      if (!response.ok) {
        if (response.status === 401) {
          handleSessionExpired();
          throw new Error(SESSION_EXPIRED_MESSAGE);
        }
        const body = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(detailOf(body, `HTTP ${response.status}`));
      }
      const body = await response.json();
      setRestoreTarget(null);
      onRestored(body);
      load();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setRestoring(false);
    }
  };

  const sourceLabel = (h: HistoryEntry): string => {
    if (h.source_name) return h.source_name;
    if (h.source_type === 'mysql') return 'MySQL database';
    if (h.source_type === 'mongodb') return 'MongoDB collection';
    return 'Uploaded file';
  };

  const capabilities = (h: HistoryEntry) => {
    const v = h.version;
    const snapshotOk = Boolean(v && v.restorable);
    const snapshotExpired = Boolean(v) && !snapshotOk;
    return {
      canOpen: snapshotOk,
      canDownload: snapshotOk,
      canRestoreWorking: snapshotOk && Boolean(h.session_id),
      canRestoreSource: Boolean(h.restorable && h.backup_id),
      snapshotOk,
      snapshotExpired,
      noSnapshot: !v,
    };
  };

  return (
    <section className="mb-8 rounded-2xl border border-neutral-200 bg-white p-6 shadow-lg animate-slide-in">
      <div className="flex items-center justify-between mb-4">
        <h2 className="section-title flex items-center text-xl font-semibold text-neutral-700 m-0">
          <Clock size={20} className="mr-2 text-primary" />
          History &amp; Rollback
          <span
            data-testid="history-source-label"
            className="ml-3 text-sm font-medium text-neutral-500"
          >
            {SOURCE_LABELS[sourceType]} History
          </span>
        </h2>
        <div className="flex items-center gap-2">
          <button
            onClick={openClearAllDialog}
            disabled={history.length === 0 || loading}
            title={
              history.length === 0
                ? `There is no ${SOURCE_LABELS[sourceType].toLowerCase()} history to clear`
                : `Permanently clear ALL ${SOURCE_LABELS[sourceType]} history for your account (typed confirmation required)`
            }
            className="btn bg-white hover:bg-red-50 text-red-700 border border-red-200 text-sm py-1.5 px-3 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
          >
            <Trash2 size={14} className="mr-1.5" />
            Clear All History
          </button>
          <button
            onClick={load}
            disabled={loading}
            className="btn bg-white hover:bg-neutral-100 text-neutral-700 border border-neutral-300 text-sm py-1.5 px-3 disabled:opacity-50 flex items-center"
          >
            {loading ? (
              <Loader2 size={14} className="mr-1.5 animate-spin" />
            ) : (
              <RefreshCw size={14} className="mr-1.5" />
            )}
            Refresh
          </button>
        </div>
      </div>

      <p className="text-xs text-neutral-500 mb-3">
        Showing only {SOURCE_LABELS[sourceType].toLowerCase()}-source history for the active
        section. Working-copy versions are kept for 7 days. Opening or downloading a
        version never modifies data. Restoring a working copy never writes to a
        connected source database; restoring a live database is a separate,
        explicitly confirmed action.
      </p>

      {error && (
        <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-md px-3 py-2 mb-3">
          {error}
        </p>
      )}

      {notice && (
        <p
          className={`text-sm rounded-md px-3 py-2 mb-3 border ${
            notice.warn
              ? 'text-amber-700 bg-amber-50 border-amber-200'
              : 'text-emerald-700 bg-emerald-50 border-emerald-200'
          }`}
        >
          {notice.text}
        </p>
      )}

      {sessionId && versions.length > 0 && (
        <div className="mb-5">
          <h3 className="text-sm font-semibold text-neutral-700 mb-2">Version snapshots</h3>
          <ul className="space-y-2">
            {versions.map((v) => (
              <li
                key={v.version_id}
                className="flex items-center justify-between border border-neutral-200 rounded-lg px-3 py-2 text-sm"
              >
                <div className="min-w-0">
                  <p className="font-medium text-neutral-800">
                    v{v.version_number}
                    {v.label ? ` — ${v.label}` : ''}
                    <span className="ml-2 text-xs text-neutral-500">
                      {v.row_count} rows · {v.columns.length} cols · {fmt(v.created_at)}
                    </span>
                  </p>
                  <p className="text-xs text-neutral-500">
                    {v.restorable ? `Restorable until ${fmt(v.expires_at)}` : 'Expired — not restorable'}
                  </p>
                </div>
                <button
                  onClick={() => setRestoreTarget(v)}
                  disabled={!v.restorable || restoring}
                  title="Restore the working copy to this snapshot (source database untouched)"
                  className="btn bg-white hover:bg-blue-50 text-blue-700 border border-blue-200 text-xs py-1 px-2.5 disabled:opacity-50 flex items-center shrink-0 ml-3"
                >
                  <RotateCcw size={12} className="mr-1" />
                  Restore
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      {loading && history.length === 0 && (
        <div className="flex items-center text-sm text-neutral-500 py-4">
          <Loader2 size={16} className="mr-2 animate-spin" /> Loading history…
        </div>
      )}

      {!loading && history.length === 0 && (
        <p className="text-sm text-neutral-500">
          {EMPTY_TEXTS[sourceType]} Clear All History is unavailable until an
          operation is recorded in this section.
        </p>
      )}

      {history.length > 0 && (
        <>
          <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
            <h3 className="text-sm font-semibold text-neutral-700 m-0">
              {SOURCE_LABELS[sourceType]} History entries ({history.length}) — select one
              to view details
            </h3>
          </div>
          <ul className="divide-y divide-neutral-100 border border-neutral-200 rounded-lg">
            {history.map((h) => {
              const isSel = h.history_id === selectedId;
              const caps = capabilities(h);
              return (
                <li
                  key={h.history_id}
                  onClick={() => toggleSelect(h)}
                  role="button"
                  tabIndex={0}
                  aria-pressed={isSel}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      toggleSelect(h);
                    }
                  }}
                  className={`px-3 py-2.5 text-sm cursor-pointer transition-colors ${
                    isSel
                      ? 'bg-blue-50 ring-2 ring-inset ring-primary/40'
                      : 'hover:bg-neutral-50'
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <span
                      aria-hidden="true"
                      className={`mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full border ${
                        isSel ? 'border-primary bg-primary text-white' : 'border-neutral-300'
                      }`}
                    >
                      {isSel && <Check size={11} strokeWidth={3} />}
                    </span>
                    <div className="min-w-0 flex-1">
                      <p className="font-medium text-neutral-800">
                        {h.operation}
                        {h.status && h.status !== 'success' && (
                          <span
                            className={`ml-2 px-1.5 py-0.5 rounded text-[10px] ${
                              h.status === 'partial'
                                ? 'bg-amber-100 text-amber-800'
                                : 'bg-red-100 text-red-800'
                            }`}
                          >
                            {h.status}
                          </span>
                        )}
                        {caps.snapshotOk && (
                          <span className="ml-2 px-1.5 py-0.5 bg-blue-100 text-blue-800 rounded text-[10px]">
                            snapshot v{h.version?.version_number}
                          </span>
                        )}
                        {caps.snapshotExpired && (
                          <span className="ml-2 px-1.5 py-0.5 bg-neutral-100 text-neutral-500 rounded text-[10px]">
                            snapshot expired
                          </span>
                        )}
                        {caps.canRestoreSource && (
                          <span className="ml-2 px-1.5 py-0.5 bg-amber-100 text-amber-800 rounded text-[10px]">
                            source restorable until {fmt(h.rollback_expires_at)}
                          </span>
                        )}
                      </p>
                      <p className="text-xs text-neutral-600 mt-0.5">{h.summary}</p>
                      {h.warnings.length > 0 && (
                        <p className="text-xs text-amber-700 mt-0.5">{h.warnings.join(' ')}</p>
                      )}
                    </div>
                    <span className="text-xs text-neutral-400 whitespace-nowrap shrink-0">
                      {fmt(h.timestamp)}
                    </span>
                  </div>
                </li>
              );
            })}
          </ul>
        </>
      )}

      {/* Selected-entry detail + actions (one entry at a time) */}
      {selected && (
        <div className="mt-4 rounded-xl border border-neutral-200 bg-neutral-50/70 p-4">
          <h4 className="text-sm font-semibold text-neutral-800 mb-3 flex items-center">
            <FileText size={15} className="mr-1.5 text-primary" />
            Selected entry
          </h4>
          <dl className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-1.5 text-xs mb-3">
            <div className="flex justify-between gap-2">
              <dt className="text-neutral-500">Operation</dt>
              <dd className="font-medium text-neutral-800">{selected.operation}</dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt className="text-neutral-500">Timestamp</dt>
              <dd className="font-medium text-neutral-800">{fmt(selected.timestamp)}</dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt className="text-neutral-500">Dataset / source</dt>
              <dd className="font-medium text-neutral-800 truncate" title={sourceLabel(selected)}>
                {sourceLabel(selected)}
              </dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt className="text-neutral-500">Source type</dt>
              <dd className="font-medium text-neutral-800">
                {selected.source_type || 'file'}
                {selected.column ? ` · column “${selected.column}”` : ''}
              </dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt className="text-neutral-500">Version</dt>
              <dd className="font-medium text-neutral-800">
                {selected.version
                  ? `v${selected.version.version_number} · ${selected.version.row_count} rows · ${selected.version.columns.length} cols · ${fmtBytes(selected.version.size_bytes)}`
                  : '—'}
              </dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt className="text-neutral-500">Version id</dt>
              <dd className="font-mono text-[10px] text-neutral-600 truncate" title={selected.version?.version_id || ''}>
                {selected.version?.version_id || '—'}
              </dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt className="text-neutral-500">Rollback window</dt>
              <dd className="font-medium text-neutral-800">
                {selected.restorable && selected.rollback_expires_at
                  ? `source restore until ${fmt(selected.rollback_expires_at)}`
                  : selected.rollback_expires_at
                    ? `expired ${fmt(selected.rollback_expires_at)}`
                    : '—'}
              </dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt className="text-neutral-500">Saved snapshot</dt>
              <dd className="font-medium text-neutral-800">
                {capabilities(selected).snapshotOk
                  ? `available (v${selected.version?.version_number}, verified)`
                  : capabilities(selected).snapshotExpired
                    ? 'expired — open/download unavailable'
                    : 'not recorded for this entry'}
              </dd>
            </div>
          </dl>

          <div className="text-xs text-neutral-600 mb-3 space-y-0.5">
            <p>
              <span className={capabilities(selected).canOpen ? 'text-emerald-700' : 'text-neutral-400'}>
                {capabilities(selected).canOpen ? '✓' : '✕'}
              </span>{' '}
              Open in workspace — loads the exact snapshot into a new isolated working copy
              (historical snapshot stays read-only).
            </p>
            <p>
              <span className={capabilities(selected).canDownload ? 'text-emerald-700' : 'text-neutral-400'}>
                {capabilities(selected).canDownload ? '✓' : '✕'}
              </span>{' '}
              Download — the exact saved snapshot (never the current dataset); never modifies data.
            </p>
            <p>
              <span className={capabilities(selected).canRestoreWorking ? 'text-emerald-700' : 'text-neutral-400'}>
                {capabilities(selected).canRestoreWorking ? '✓' : '✕'}
              </span>{' '}
              Restore as working copy — replaces the working dataset only; the live source
              database is never touched.
            </p>
            <p>
              <span className={capabilities(selected).canRestoreSource ? 'text-amber-600' : 'text-neutral-400'}>
                {capabilities(selected).canRestoreSource ? '✓' : '✕'}
              </span>{' '}
              Restore source database to this version — a real database mutation behind a typed
              RESTORE confirmation.
              {!capabilities(selected).canRestoreSource && selected.version && (
                <span className="text-neutral-400"> (no live-source backup linked to this entry)</span>
              )}
            </p>
          </div>

          {actionMsg && (
            <p className="text-xs text-neutral-700 bg-white border border-neutral-200 rounded-md px-3 py-2 mb-3">
              {actionMsg}
            </p>
          )}

          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={() => openEntry(selected)}
              disabled={!capabilities(selected).canOpen || opening}
              title="Open this exact version in the transformation workspace (new working copy)"
              className="btn bg-blue-600 hover:bg-blue-700 text-white border border-blue-700 text-sm py-1.5 px-3 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
            >
              {opening ? <Loader2 size={14} className="mr-1.5 animate-spin" /> : <Upload size={14} className="mr-1.5" />}
              Open in workspace
            </button>

            <span className="flex items-center">
              <select
                value={downloadFormat}
                onChange={(e) => setDownloadFormat(e.target.value as 'csv' | 'json' | 'xlsx')}
                disabled={!capabilities(selected).canDownload}
                className="mr-1.5 rounded-md border border-neutral-300 bg-white text-sm py-1.5 px-2 text-neutral-700 disabled:opacity-50"
                aria-label="Download format"
              >
                <option value="csv">CSV</option>
                <option value="json">JSON</option>
                <option value="xlsx">XLSX</option>
              </select>
              <button
                onClick={() => downloadEntry(selected)}
                disabled={!capabilities(selected).canDownload || downloading}
                title="Download this exact saved version (does not change the current dataset)"
                className="btn bg-white hover:bg-neutral-100 text-neutral-700 border border-neutral-300 text-sm py-1.5 px-3 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
              >
                {downloading ? <Loader2 size={14} className="mr-1.5 animate-spin" /> : <Download size={14} className="mr-1.5" />}
                Download
              </button>
            </span>

            <button
              onClick={() => {
                if (selected.version) {
                  setRestoreTarget({
                    ...selected.version,
                    label: selected.version.label ?? null,
                    expires_at: selected.version.expires_at || '',
                  });
                }
              }}
              disabled={!capabilities(selected).canRestoreWorking || restoring}
              title="Make this version the current working dataset (source database untouched)"
              className="btn bg-white hover:bg-blue-50 text-blue-700 border border-blue-200 text-sm py-1.5 px-3 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
            >
              <RotateCcw size={14} className="mr-1.5" />
              Restore as working copy
            </button>

            {capabilities(selected).canRestoreSource && onRestoreSource && (
              <button
                onClick={() => onRestoreSource(selected.backup_id!)}
                title="Restore the LIVE source database from this entry's verified backup (typed RESTORE confirmation required)"
                className="btn bg-white hover:bg-amber-50 text-amber-700 border border-amber-300 text-sm py-1.5 px-3 flex items-center"
              >
                <Database size={14} className="mr-1.5" />
                Restore source database to this version…
              </button>
            )}

            <button
              onClick={() => openDeleteDialog(selected)}
              disabled={deleting}
              title="Delete this history entry and its unshared recovery snapshot (active dataset and source databases untouched)"
              className="btn bg-white hover:bg-red-50 text-red-700 border border-red-200 text-sm py-1.5 px-3 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
            >
              <Trash2 size={14} className="mr-1.5" />
              Delete This Entry
            </button>
          </div>
        </div>
      )}

      {restoreTarget && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
          <div className="bg-white rounded-xl shadow-2xl max-w-md w-full p-5">
            <h3 className="text-lg font-semibold text-neutral-800 mb-2">
              Restore working copy to v{restoreTarget.version_number}?
            </h3>
            <p className="text-sm text-neutral-600 mb-4">
              This replaces the current working copy ({restoreTarget.row_count} rows) with the
              snapshot taken at {fmt(restoreTarget.created_at)}. The connected source database is
              NOT modified. The historical snapshot itself is preserved, and the restore is
              recorded in history so you can roll forward again.
            </p>
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setRestoreTarget(null)}
                disabled={restoring}
                className="btn bg-white hover:bg-neutral-100 text-neutral-700 border border-neutral-300 text-sm py-1.5 px-3"
              >
                Cancel
              </button>
              <button
                onClick={() => confirmRestore(restoreTarget)}
                disabled={restoring}
                className="btn bg-blue-600 hover:bg-blue-700 text-white border border-blue-700 text-sm py-1.5 px-3 disabled:opacity-50 flex items-center"
              >
                {restoring && <Loader2 size={14} className="mr-1.5 animate-spin" />}
                Restore working copy
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete one history entry — explicit confirmation with entry details */}
      {deleteTarget && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
          <div className="bg-white rounded-xl shadow-2xl max-w-md w-full p-5">
            <h3 className="text-lg font-semibold text-neutral-800 mb-2">
              Delete this history entry?
            </h3>
            <dl className="text-sm text-neutral-700 mb-3 space-y-1 border border-neutral-200 rounded-md bg-neutral-50 px-3 py-2">
              <div className="flex justify-between gap-3">
                <dt className="text-neutral-500">Entry</dt>
                <dd className="font-medium text-right truncate" title={sourceLabel(deleteTarget)}>
                  {sourceLabel(deleteTarget)}
                </dd>
              </div>
              <div className="flex justify-between gap-3">
                <dt className="text-neutral-500">Source</dt>
                <dd className="font-medium text-right">
                  {deleteTarget.source_type || 'file'}
                </dd>
              </div>
              <div className="flex justify-between gap-3">
                <dt className="text-neutral-500">Operation</dt>
                <dd className="font-medium text-right">{deleteTarget.operation}</dd>
              </div>
              <div className="flex justify-between gap-3">
                <dt className="text-neutral-500">Timestamp</dt>
                <dd className="font-medium text-right">{fmt(deleteTarget.timestamp)}</dd>
              </div>
            </dl>
            <p className="text-sm text-neutral-600 mb-3">
              This permanently removes the entry from your history. If it owns a
              recovery snapshot, that snapshot is deleted too, which may
              permanently remove your ability to restore that version. Snapshots
              still needed by other history entries are kept. The active
              dataset, uploaded files, and connected source databases are NOT
              modified.
            </p>
            {deleteTarget.restorable && deleteTarget.backup_id && (
              <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-md px-3 py-2 mb-3">
                This entry holds a source-database backup; deleting it removes
                that restore path from TabulaX.
              </p>
            )}
            {deleteError && (
              <p className="text-xs text-red-600 bg-red-50 border border-red-200 rounded-md px-3 py-2 mb-3">
                {deleteError} — the entry was not deleted.
              </p>
            )}
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setDeleteTarget(null)}
                disabled={deleting}
                className="btn bg-white hover:bg-neutral-100 text-neutral-700 border border-neutral-300 text-sm py-1.5 px-3"
              >
                Cancel
              </button>
              <button
                onClick={() => deleteEntry(deleteTarget)}
                disabled={deleting}
                className="btn bg-red-600 hover:bg-red-700 text-white border border-red-700 text-sm py-1.5 px-3 disabled:opacity-50 flex items-center"
              >
                {deleting && <Loader2 size={14} className="mr-1.5 animate-spin" />}
                Delete Entry
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Clear ALL history for the ACTIVE source category — typed confirmation */}
      {showClearAll && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
          <div className="bg-white rounded-xl shadow-2xl max-w-md w-full p-5">
            <h3 className="text-lg font-semibold text-neutral-800 mb-2">
              {CLEAR_TITLES[sourceType]}
            </h3>
            <p className="text-sm text-neutral-600 mb-3">
              This deletes every {SOURCE_LABELS[sourceType].toLowerCase()} history entry in
              your account
              {clearAllStats
                ? ` (${clearAllStats.total_entries} entries, ${clearAllStats.entries_with_snapshot} with version snapshots, ${clearAllStats.entries_with_backup} with source-restore backups)`
                : ''}{' '}
              and their unshared recovery snapshots. History from other sections (file,
              MySQL, MongoDB) is NOT touched. Snapshots referenced by another surviving
              entry are retained. The active working dataset and connected source
              databases are NOT modified.
            </p>
            {sourceType !== 'file' && (
              <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-md px-3 py-2 mb-3">
                Deleting {SOURCE_LABELS[sourceType].toLowerCase()} history entries that are
                the only recovery path for a live database removes your ability to restore
                that database from TabulaX.
              </p>
            )}
            {sourceType === 'file' && (
              <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-md px-3 py-2 mb-3">
                Deleting the only history entry that references a snapshot permanently
                removes your ability to restore that file version.
              </p>
            )}
            <label className="block text-xs text-neutral-600 mb-1">
              Type <span className="font-mono font-semibold">CLEAR HISTORY</span> to confirm:
            </label>
            <input
              type="text"
              value={clearTyped}
              onChange={(e) => setClearTyped(e.target.value)}
              placeholder="CLEAR HISTORY"
              aria-label="Type CLEAR HISTORY to confirm"
              className="w-full rounded-md border border-neutral-300 text-sm py-1.5 px-2 mb-4 font-mono focus:outline-none focus:ring-2 focus:ring-primary/40"
            />
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setShowClearAll(false)}
                disabled={clearingAll}
                className="btn bg-white hover:bg-neutral-100 text-neutral-700 border border-neutral-300 text-sm py-1.5 px-3"
              >
                Cancel
              </button>
              <button
                onClick={confirmClearAll}
                disabled={clearingAll || clearTyped !== 'CLEAR HISTORY'}
                className="btn bg-red-600 hover:bg-red-700 text-white border border-red-700 text-sm py-1.5 px-3 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
              >
                {clearingAll && <Loader2 size={14} className="mr-1.5 animate-spin" />}
                Permanently Clear History
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
