import React, { useEffect, useRef, useState } from 'react';
import { AlertTriangle, Database, Loader2, X } from 'lucide-react';
import { apiUrl, handleSessionExpired, SESSION_EXPIRED_MESSAGE } from '../lib/api';
import { dataKeys } from '../lib/data';

export type SourceDeleteTarget =
  | 'mysql_rows'
  | 'mysql_columns'
  | 'mongo_documents'
  | 'mongo_fields';

interface MysqlConn {
  host?: string;
  port?: string | number;
  user?: string;
  password?: string;
  database?: string;
}

interface PreviewInfo {
  preview_token: string;
  target: SourceDeleteTarget;
  source: string;
  database: string;
  table?: string;
  collection?: string;
  primary_key?: string[];
  columns?: string[];
  fields?: string[];
  field_counts?: Record<string, number>;
  requested_count: number;
  matched_count: number;
  sample: Record<string, unknown>[];
  warnings: string[];
  irreversible: boolean;
  backup: {
    kind: string;
    format: string;
    expires_in_days?: number;
    restoration?: string;
  };
}

interface Props {
  target: SourceDeleteTarget;
  sessionId: string;
  rowIds: number[];
  fields: string[];
  mysqlConnection?: MysqlConn;
  mongoUri?: string;
  onClose: () => void;
  onDone: (resp: Record<string, unknown>) => void;
}

const TITLES: Record<SourceDeleteTarget, string> = {
  mysql_rows: 'Permanently delete rows from MySQL',
  mysql_columns: 'Permanently drop columns from MySQL',
  mongo_documents: 'Permanently delete MongoDB documents',
  mongo_fields: 'Permanently remove fields from MongoDB',
};

const VERBS: Record<SourceDeleteTarget, [string, string]> = {
  mysql_rows: ['Delete row(s) from', 'rows from'],
  mysql_columns: ['Drop column(s) from', 'columns from'],
  mongo_documents: ['Delete document(s) from', 'documents from'],
  mongo_fields: ['Remove field(s) from', 'fields from'],
};

function detailOf(data: unknown, fallback: string): string {
  const d = (data as { detail?: unknown } | null)?.detail;
  if (typeof d === 'string') return d;
  if (Array.isArray(d)) {
    return (
      d
        .map((x) => (x as { msg?: string })?.msg || '')
        .filter(Boolean)
        .join('; ') || fallback
    );
  }
  return fallback;
}

/**
 * Two-step dialog for PERMANENT deletion of source database data:
 *   1. server-side preview (issues a 15-minute HMAC preview_token) with an
 *      explicit summary of exactly what would be touched;
 *   2. the delete itself, gated by a backup checkbox (default on) and an
 *      exact typed "DELETE" confirmation.
 * Any failure is shown with an explicit "source NOT modified" guarantee —
 * the backend aborts before touching data whenever preview/backup/consent
 * checks fail.
 */
export default function SourceDeleteDialog({
  target,
  sessionId,
  rowIds,
  fields,
  mysqlConnection,
  mongoUri,
  onClose,
  onDone,
}: Props) {
  const [phase, setPhase] = useState<'loading' | 'preview' | 'error'>('loading');
  const [preview, setPreview] = useState<PreviewInfo | null>(null);
  const [loadError, setLoadError] = useState('');
  const [submitError, setSubmitError] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [typed, setTyped] = useState('');
  const [requireBackup, setRequireBackup] = useState(true);
  const [allowPartial, setAllowPartial] = useState(false);
  const mounted = useRef(true);

  const isRowTarget =
    target === 'mysql_rows' || target === 'mongo_documents';
  const isMysql = target.startsWith('mysql');
  const namespace = preview
    ? `${preview.database}.${preview.table || preview.collection || ''}`
    : '';

  // Preview endpoints take `mysql_connection`; MySQL delete endpoints take
  // `connection_details`. Mongo uses `mongo_connection` on both.
  const connectionPayload = (forDelete: boolean): Record<string, unknown> => {
    if (isMysql) {
      const conn = {
        host: mysqlConnection?.host,
        port: Number(mysqlConnection?.port) || 3306,
        user: mysqlConnection?.user,
        password: mysqlConnection?.password || '',
        database: mysqlConnection?.database,
      };
      return forDelete ? { connection_details: conn } : { mysql_connection: conn };
    }
    return { mongo_connection: { uri: mongoUri } };
  };

  const selectionPayload = (): Record<string, unknown> => {
    if (isRowTarget) return { row_ids: rowIds };
    if (target === 'mysql_columns') return { columns: fields };
    return { fields };
  };

  useEffect(() => {
    mounted.current = true;
    (async () => {
      try {
        const token = localStorage.getItem('token');
        const res = await fetch(apiUrl('/source_delete/preview/'), {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({
            session_id: sessionId,
            target,
            ...selectionPayload(),
            ...connectionPayload(false),
          }),
        });
        if (!res.ok) {
          if (res.status === 401) {
            handleSessionExpired();
            throw new Error(SESSION_EXPIRED_MESSAGE);
          }
          const data = await res.json().catch(() => null);
          throw new Error(detailOf(data, `Preview failed (${res.statusText})`));
        }
        const body = (await res.json()) as PreviewInfo;
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
  }, [target]);

  const submit = async () => {
    if (!preview || typed !== 'DELETE' || submitting) return;
    setSubmitting(true);
    setSubmitError('');
    try {
      const token = localStorage.getItem('token');
      const res = await fetch(apiUrl(`/source_delete/${target}/`), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          session_id: sessionId,
          target,
          preview_token: preview.preview_token,
          confirm: true,
          typed_confirmation: typed,
          require_backup: requireBackup,
          ...(isRowTarget ? { allow_partial: allowPartial } : {}),
          ...selectionPayload(),
          ...connectionPayload(true),
        }),
      });
      if (!res.ok) {
        if (res.status === 401) {
          handleSessionExpired();
          throw new Error(SESSION_EXPIRED_MESSAGE);
        }
        const data = await res.json().catch(() => null);
        throw new Error(detailOf(data, `Deletion failed (${res.statusText})`));
      }
      const body = (await res.json()) as Record<string, unknown>;
      onDone(body);
    } catch (err) {
      setSubmitError((err as Error).message);
    } finally {
      if (mounted.current) setSubmitting(false);
    }
  };

  const [verb] = VERBS[target];
  const partialPending =
    preview != null && preview.matched_count < preview.requested_count;
  const sampleKeys = preview?.sample?.length ? dataKeys(preview.sample[0]) : [];

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm animate-fade-in"
      role="dialog"
      aria-modal="true"
      aria-label={TITLES[target]}
    >
      <div className="bg-white rounded-2xl shadow-float w-full max-w-xl max-h-[85vh] overflow-y-auto">
        <div className="flex items-start justify-between px-6 pt-6 pb-4 border-b border-neutral-200">
          <div className="flex items-start gap-3">
            <span className="mt-0.5 inline-flex h-9 w-9 items-center justify-center rounded-full bg-red-100 text-red-600">
              <Database size={18} />
            </span>
            <div>
              <h2 className="text-lg font-semibold text-neutral-800">
                {TITLES[target]}
              </h2>
              <p className="text-xs text-neutral-500 mt-0.5">
                Permanent, source-level change — not a working-copy edit
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            disabled={submitting}
            className="p-1.5 rounded-full text-neutral-400 hover:bg-neutral-100 hover:text-neutral-600 transition-colors"
            aria-label="Close"
          >
            <X size={18} />
          </button>
        </div>

        <div className="px-6 py-5 space-y-4">
          {phase === 'loading' && (
            <div className="flex items-center justify-center py-10 text-neutral-500">
              <Loader2 size={20} className="animate-spin mr-2" />
              Inspecting the source…
            </div>
          )}

          {phase === 'error' && (
            <div>
              <div className="bg-red-50 border-l-4 border-error p-3 text-sm text-red-700">
                {loadError}
              </div>
              <p className="mt-3 text-sm text-neutral-600">
                The source was <strong>not modified</strong>. Fix the issue
                (e.g. reconnect the table/collection) and try again.
              </p>
              <div className="mt-4 flex justify-end">
                <button onClick={onClose} className="btn text-sm py-2 px-4">
                  Close
                </button>
              </div>
            </div>
          )}

          {phase === 'preview' && preview && (
            <>
              <div className="rounded-lg border border-neutral-200 bg-neutral-50 p-4 text-sm space-y-2">
                <div className="flex justify-between gap-4">
                  <span className="text-neutral-500">Target</span>
                  <span className="font-medium text-neutral-800 text-right break-all">
                    {namespace}
                  </span>
                </div>
                <div className="flex justify-between gap-4">
                  <span className="text-neutral-500">Operation</span>
                  <span className="font-medium text-neutral-800 text-right">
                    {isRowTarget
                      ? `${verb} ${namespace}`
                      : target === 'mysql_columns'
                        ? `Drop ${preview.columns?.length} column(s) via ALTER TABLE`
                        : `$unset ${preview.fields?.length} field(s) on all documents`}
                  </span>
                </div>
                <div className="flex justify-between gap-4">
                  <span className="text-neutral-500">Matched in source</span>
                  <span className="font-medium text-neutral-800">
                    {preview.matched_count} of {preview.requested_count} selected
                  </span>
                </div>
                {preview.primary_key && preview.primary_key.length > 0 && (
                  <div className="flex justify-between gap-4">
                    <span className="text-neutral-500">
                      {preview.source === 'mongodb' ? 'Identity' : 'Primary key'}
                    </span>
                    <span className="font-mono text-xs text-neutral-800 text-right break-all">
                      {preview.primary_key.join(', ')}
                    </span>
                  </div>
                )}
                {preview.fields && preview.field_counts && (
                  <div className="flex justify-between gap-4">
                    <span className="text-neutral-500">Documents with field</span>
                    <span className="font-medium text-neutral-800 text-right">
                      {preview.fields
                        .map((f) => `${f}: ${preview.field_counts?.[f] ?? 0}`)
                        .join(', ')}
                    </span>
                  </div>
                )}
                <div className="flex justify-between gap-4">
                  <span className="text-neutral-500">Backup</span>
                  <span className="font-medium text-neutral-800 text-right">
                    {preview.backup.format.toUpperCase()} (.gz), expires in{' '}
                    {preview.backup.expires_in_days ?? 7} days
                  </span>
                </div>
              </div>

              {preview.warnings.length > 0 && (
                <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800 space-y-1">
                  {preview.warnings.map((w, i) => (
                    <p key={i}>• {w}</p>
                  ))}
                </div>
              )}

              <div className="rounded-lg border-2 border-red-300 bg-red-50 p-3 text-sm text-red-700 flex gap-2">
                <AlertTriangle size={16} className="shrink-0 mt-0.5" />
                <span>
                  This changes the <strong>source database itself</strong> and
                  cannot be undone from TabulaX. The UI will refresh from the
                  live source afterwards. Restore from the backup if needed.
                </span>
              </div>

              {preview.sample.length > 0 && (
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wider text-neutral-500 mb-2">
                    Affected data (first {preview.sample.length})
                  </p>
                  <div className="overflow-x-auto rounded-lg border border-neutral-200">
                    <table className="min-w-full text-xs">
                      <thead className="bg-neutral-50">
                        <tr>
                          {sampleKeys.slice(0, 8).map((k) => (
                            <th
                              key={k}
                              className="px-3 py-2 text-left font-semibold text-neutral-600 whitespace-nowrap"
                            >
                              {k}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-neutral-100">
                        {preview.sample.slice(0, 5).map((row, i) => (
                          <tr key={i}>
                            {sampleKeys.slice(0, 8).map((k) => (
                              <td
                                key={k}
                                className="px-3 py-1.5 text-neutral-700 whitespace-nowrap"
                              >
                                {String(row[k])}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              <label className="flex items-start gap-2 text-sm text-neutral-700 select-none">
                <input
                  type="checkbox"
                  checked={requireBackup}
                  onChange={(e) => setRequireBackup(e.target.checked)}
                  disabled={submitting}
                  className="mt-0.5"
                />
                <span>
                  Create a restore backup before deleting
                  <span className="block text-xs text-neutral-500">
                    Recommended: deletion aborts if the backup cannot be stored.
                  </span>
                </span>
              </label>

              {partialPending && (
                <label className="flex items-start gap-2 text-sm text-neutral-700 select-none">
                  <input
                    type="checkbox"
                    checked={allowPartial}
                    onChange={(e) => setAllowPartial(e.target.checked)}
                    disabled={submitting}
                    className="mt-0.5"
                  />
                  <span>
                    Allow deleting only the{' '}
                    <strong>{preview.matched_count}</strong> row(s)/document(s)
                    that still exist (some selected ones are gone)
                  </span>
                </label>
              )}

              <div>
                <label
                  htmlFor="confirm-delete-word"
                  className="block text-sm font-medium text-neutral-700 mb-1.5"
                >
                  Type <code className="px-1.5 py-0.5 rounded bg-neutral-100 font-mono text-red-700">DELETE</code> to
                  confirm
                </label>
                <input
                  id="confirm-delete-word"
                  type="text"
                  value={typed}
                  onChange={(e) => setTyped(e.target.value)}
                  disabled={submitting}
                  autoComplete="off"
                  spellCheck={false}
                  placeholder="DELETE"
                  className="w-full rounded-lg border border-neutral-300 px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-error/40 focus:border-error"
                />
                <p className="mt-1 text-xs text-neutral-500">
                  Preview token valid for 15 minutes.
                </p>
              </div>

              {submitError && (
                <div className="rounded-lg border-l-4 border-error bg-red-50 p-3 text-sm text-red-700">
                  {submitError}
                  <p className="mt-1 text-xs">
                    The source was <strong>not modified</strong> (deletion only
                    runs after every check passes).
                  </p>
                </div>
              )}

              <div className="flex justify-end gap-2 pt-1">
                <button
                  onClick={onClose}
                  disabled={submitting}
                  className="btn text-sm py-2 px-4 disabled:opacity-50"
                >
                  Cancel
                </button>
                <button
                  onClick={submit}
                  disabled={typed !== 'DELETE' || submitting}
                  className="btn btn-error text-sm py-2 px-4 disabled:opacity-50 flex items-center"
                >
                  {submitting && (
                    <Loader2 size={14} className="animate-spin mr-1.5" />
                  )}
                  {submitting
                    ? 'Deleting…'
                    : `Yes, permanently delete (${preview.matched_count})`}
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
