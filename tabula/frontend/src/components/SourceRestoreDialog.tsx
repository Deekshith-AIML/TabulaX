import React, { useEffect, useRef, useState } from 'react';
import { Database, Loader2, RotateCcw } from 'lucide-react';
import { apiUrl, handleSessionExpired, SESSION_EXPIRED_MESSAGE } from '../lib/api';
import { formatApiDate } from '../lib/datetime';

interface MysqlConn {
  host?: string;
  port?: string | number;
  user?: string;
  password?: string;
  database?: string;
}

interface RestorePreview {
  backup_id: string;
  kind: string;
  source_name: string | null;
  created_at: string | null;
  expires_at: string | null;
  restorable: boolean;
  requires_typed_confirmation: string;
  snapshot: Record<string, unknown>;
  conflicts: Record<string, unknown> | null;
  live: Record<string, unknown> | null;
  warnings: string[];
}

interface Props {
  backupId: string;
  mysqlConnection?: MysqlConn;
  mongoUri?: string;
  /** The caller's active session, so the restore can hand back a refreshed one. */
  sessionId?: string;
  onClose: () => void;
  /** Called with the restore response (counts, warnings) on success. */
  onDone: (resp: Record<string, unknown>) => void;
}

function detailOf(data: unknown, fallback: string): string {
  const d = (data as { detail?: unknown } | null)?.detail;
  if (typeof d === 'string') return d;
  if (Array.isArray(d)) {
    return d.map((x) => (x as { msg?: string })?.msg || '').filter(Boolean).join('; ') || fallback;
  }
  return fallback;
}

/**
 * Two-step dialog for restoring a deleted source database from a verified
 * backup artifact: preview (counts, conflicts, expiry) then an execute step
 * gated by typed "RESTORE" confirmation.  The current source state is backed
 * up first server-side; failures abort before any data is written.
 */
export default function SourceRestoreDialog({
  backupId,
  mysqlConnection,
  mongoUri,
  sessionId,
  onClose,
  onDone,
}: Props) {
  const [phase, setPhase] = useState<'loading' | 'preview' | 'error'>('loading');
  const [preview, setPreview] = useState<RestorePreview | null>(null);
  const [loadError, setLoadError] = useState('');
  const [submitError, setSubmitError] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [typed, setTyped] = useState('');
  const [overwrite, setOverwrite] = useState(false);
  const mounted = useRef(true);

  const isMysql =
    preview?.kind?.startsWith('mysql') ||
    (!preview && Boolean(mysqlConnection));

  const connectionPayload = (): Record<string, unknown> => {
    if (isMysql) {
      return {
        mysql_connection: {
          host: mysqlConnection?.host,
          port: Number(mysqlConnection?.port) || 3306,
          user: mysqlConnection?.user,
          password: mysqlConnection?.password || '',
          database: mysqlConnection?.database,
        },
      };
    }
    return { mongo_connection: { uri: mongoUri } };
  };

  useEffect(() => {
    mounted.current = true;
    (async () => {
      try {
        const token = localStorage.getItem('token');
        const res = await fetch(apiUrl('/restore_source/preview/'), {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({ backup_id: backupId, ...connectionPayload() }),
        });
        if (!res.ok) {
          if (res.status === 401) {
            handleSessionExpired();
            throw new Error(SESSION_EXPIRED_MESSAGE);
          }
          const data = await res.json().catch(() => null);
          throw new Error(detailOf(data, `Preview failed (${res.statusText})`));
        }
        const body = (await res.json()) as RestorePreview;
        if (!mounted.current) return;
        setPreview(body);
        setPhase('preview');
      } catch (err) {
        if (!mounted.current) return;
        setLoadError((err as Error).message);
        setPhase('error');
      }
    })();
    return () => {
      mounted.current = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [backupId]);

  const submit = async () => {
    if (!preview || typed !== 'RESTORE' || submitting) return;
    setSubmitting(true);
    setSubmitError('');
    try {
      const token = localStorage.getItem('token');
      const res = await fetch(apiUrl('/restore_source/'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          backup_id: backupId,
          confirm: true,
          typed_confirmation: typed,
          overwrite_conflicts: overwrite,
          // Lets the server retire the stale session and return a refreshed
          // one (live data) so the UI updates without a manual reconnect.
          ...(sessionId ? { session_id: sessionId } : {}),
          ...connectionPayload(),
        }),
      });
      if (!res.ok) {
        if (res.status === 401) {
          handleSessionExpired();
          throw new Error(SESSION_EXPIRED_MESSAGE);
        }
        const data = await res.json().catch(() => null);
        throw new Error(detailOf(data, `Restore failed (${res.statusText})`));
      }
      const body = (await res.json()) as Record<string, unknown>;
      onDone(body);
    } catch (err) {
      setSubmitError((err as Error).message);
    } finally {
      if (mounted.current) setSubmitting(false);
    }
  };

  const hasConflicts = Boolean(preview?.conflicts);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm animate-fade-in"
      role="dialog"
      aria-modal="true"
      aria-label="Restore from backup"
    >
      <div className="bg-white rounded-2xl shadow-float w-full max-w-xl max-h-[85vh] overflow-y-auto">
        <div className="flex items-start justify-between px-6 pt-6 pb-4 border-b border-neutral-200">
          <div className="flex items-start gap-3">
            <span className="mt-0.5 inline-flex h-9 w-9 items-center justify-center rounded-full bg-blue-100 text-blue-600">
              <RotateCcw size={18} />
            </span>
            <div>
              <h2 className="text-lg font-semibold text-neutral-800">Restore source database</h2>
              <p className="text-xs text-neutral-500 mt-0.5">
                Writes to the live database — current state is backed up first
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            disabled={submitting}
            className="p-1.5 rounded-full text-neutral-400 hover:bg-neutral-100 hover:text-neutral-600"
            aria-label="Close"
          >
            ×
          </button>
        </div>

        <div className="px-6 py-5 space-y-4 text-sm">
          {phase === 'loading' && (
            <div className="flex items-center text-neutral-600 py-6">
              <Loader2 size={18} className="mr-2 animate-spin" /> Loading restore preview…
            </div>
          )}

          {phase === 'error' && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-red-700">
              <p className="font-medium">Restore unavailable</p>
              <p className="mt-1">{loadError}</p>
              <p className="mt-2 text-xs text-red-600">
                The live source was NOT modified.
              </p>
            </div>
          )}

          {phase === 'preview' && preview && (
            <>
              <dl className="grid grid-cols-2 gap-x-4 gap-y-2">
                <div>
                  <dt className="text-xs text-neutral-500">Backup</dt>
                  <dd className="font-medium text-neutral-800">{preview.kind}</dd>
                </div>
                <div>
                  <dt className="text-xs text-neutral-500">Source</dt>
                  <dd className="font-medium text-neutral-800">{preview.source_name || '—'}</dd>
                </div>
                <div>
                  <dt className="text-xs text-neutral-500">Snapshot taken</dt>
                  <dd className="text-neutral-700">
                    {formatApiDate(preview.created_at)}
                  </dd>
                </div>
                <div>
                  <dt className="text-xs text-neutral-500">Expires</dt>
                  <dd className="text-neutral-700">
                    {formatApiDate(preview.expires_at)}
                  </dd>
                </div>
              </dl>

              {!preview.restorable && (
                <p className="bg-red-50 border border-red-200 rounded-lg p-3 text-red-700">
                  This backup has expired and can no longer be restored.
                </p>
              )}

              {preview.snapshot && Object.keys(preview.snapshot).length > 0 && (
                <div className="bg-neutral-50 border border-neutral-200 rounded-lg p-3">
                  <p className="text-xs font-semibold uppercase tracking-wider text-neutral-500 mb-1">
                    Snapshot
                  </p>
                  <pre className="text-xs text-neutral-700 whitespace-pre-wrap">
                    {JSON.stringify(preview.snapshot, null, 1)}
                  </pre>
                </div>
              )}

              {preview.conflicts && (
                <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-amber-800">
                  <p className="font-medium">Conflicts detected</p>
                  <pre className="text-xs mt-1 whitespace-pre-wrap">
                    {JSON.stringify(preview.conflicts, null, 1)}
                  </pre>
                </div>
              )}

              {preview.warnings.length > 0 && (
                <ul className="space-y-1">
                  {preview.warnings.map((w, i) => (
                    <li key={i} className="text-xs text-amber-700">
                      • {w}
                    </li>
                  ))}
                </ul>
              )}

              {hasConflicts && (
                <label className="flex items-start gap-2 text-sm text-neutral-700 select-none">
                  <input
                    type="checkbox"
                    checked={overwrite}
                    onChange={(e) => setOverwrite(e.target.checked)}
                    className="mt-0.5"
                  />
                  <span>
                    Overwrite conflicting rows/documents (existing data that differs from the
                    snapshot will be replaced)
                  </span>
                </label>
              )}

              <label className="block">
                <span className="text-sm text-neutral-700">
                  Type <strong>RESTORE</strong> (uppercase) to confirm writing to the live source:
                </span>
                <input
                  type="text"
                  value={typed}
                  onChange={(e) => setTyped(e.target.value)}
                  disabled={submitting || !preview.restorable}
                  className="mt-1.5 w-full border border-neutral-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
                  placeholder="RESTORE"
                  autoFocus
                />
              </label>

              {submitError && (
                <p className="bg-red-50 border border-red-200 rounded-lg p-3 text-red-700">
                  {submitError}
                  <span className="block text-xs mt-1 text-red-600">
                    If this failed during the write, the pre-restore backup is listed in History.
                  </span>
                </p>
              )}
            </>
          )}
        </div>

        {phase === 'preview' && preview && (
          <div className="px-6 pb-5 flex justify-end gap-2 border-t border-neutral-100 pt-4">
            <button
              onClick={onClose}
              disabled={submitting}
              className="btn bg-white hover:bg-neutral-100 text-neutral-700 border border-neutral-300 text-sm py-2 px-4"
            >
              Cancel
            </button>
            <button
              onClick={submit}
              disabled={typed !== 'RESTORE' || submitting || !preview.restorable}
              className="btn bg-blue-600 hover:bg-blue-700 text-white border border-blue-700 text-sm py-2 px-4 disabled:opacity-50 flex items-center"
            >
              {submitting && <Loader2 size={14} className="mr-1.5 animate-spin" />}
              <Database size={14} className="mr-1.5" />
              Restore to source
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
