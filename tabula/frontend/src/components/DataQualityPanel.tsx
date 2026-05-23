import React, { useCallback, useEffect, useState } from 'react';
import { AlertTriangle, CheckCircle2, Info, Loader2, RefreshCw } from 'lucide-react';
import { apiUrl, handleSessionExpired, SESSION_EXPIRED_MESSAGE } from '../lib/api';

interface Issue {
  severity: string;
  message: string;
}

interface ColumnProfile {
  name: string;
  family: string;
  dtype: string;
  null_count: number;
  blank_count: number;
  missing_pct: number;
  unique_count: number;
  unique_pct: number;
  is_constant: boolean;
  is_mixed_type: boolean;
  samples: unknown[];
  issues: Issue[];
  numeric?: {
    min?: number;
    max?: number;
    mean?: number;
    median?: number;
    std?: number;
    outliers?: number;
  } | null;
  categorical?: { top_values?: { value: string; count: number; pct: number }[] } | null;
  text?: { avg_length?: number } | null;
}

interface Profile {
  generated_at: string;
  approximate: boolean;
  total_rows: number;
  profiled_rows: number;
  overview: {
    rows: number;
    columns: number;
    columns_with_missing: number;
    duplicate_rows: number;
    warnings: Issue[];
  };
  columns: ColumnProfile[];
  potential_keys: string[];
}

interface Props {
  sessionId: string;
  columns: string[];
  /** Latest examples the user typed (optional) for preflight validation. */
  examples?: string;
  selectedColumn?: string;
  transformationId?: string;
  /**
   * Bumped by the app whenever the authoritative dataset changes (apply,
   * restore, session swap) so same-session updates re-run the profile.
   */
  refreshKey?: number;
}

function detailOf(data: unknown, fallback: string): string {
  const d = (data as { detail?: unknown } | null)?.detail;
  if (typeof d === 'string') return d;
  if (Array.isArray(d)) {
    return d.map((x) => (x as { msg?: string })?.msg || '').filter(Boolean).join('; ') || fallback;
  }
  return fallback;
}

function SeverityIcon({ severity }: { severity: string }) {
  if (severity === 'error') return <AlertTriangle size={14} className="text-red-500 shrink-0" />;
  if (severity === 'warning') return <AlertTriangle size={14} className="text-amber-500 shrink-0" />;
  return <Info size={14} className="text-sky-500 shrink-0" />;
}

/**
 * Read-only data-quality profile plus a preflight validation summary for the
 * currently selected column/examples.  Runs automatically when a session
 * exists; never mutates data and never calls an LLM.
 */
export default function DataQualityPanel({
  sessionId,
  columns,
  examples,
  selectedColumn,
  transformationId,
  refreshKey,
}: Props) {
  const [profile, setProfile] = useState<Profile | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [expanded, setExpanded] = useState<string | null>(null);
  const [validation, setValidation] = useState<{
    ok: boolean;
    errors: string[];
    warnings: string[];
    info: string[];
  } | null>(null);
  const [validating, setValidating] = useState(false);

  const loadProfile = useCallback(async () => {
    if (!sessionId) return;
    setLoading(true);
    setError('');
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(apiUrl('/quality/profile/'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ session_id: sessionId }),
      });
      if (!response.ok) {
        if (response.status === 401) {
          handleSessionExpired();
          throw new Error(SESSION_EXPIRED_MESSAGE);
        }
        const body = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(detailOf(body, `HTTP ${response.status}`));
      }
      setProfile((await response.json()) as Profile);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  useEffect(() => {
    setProfile(null);
    setValidation(null);
    if (sessionId) loadProfile();
    // refreshKey: dataset changed under the SAME session (apply/restore) —
    // the profile must follow the server state, not the first load.
  }, [sessionId, loadProfile, refreshKey]);

  // Preflight validation runs alongside generation; manual re-check button.
  const runValidation = useCallback(async () => {
    if (!sessionId || !selectedColumn || !examples?.trim()) {
      setValidation(null);
      return;
    }
    const pairs: { input: string; output: string }[] = [];
    for (const line of examples.split('\n')) {
      const s = line.trim();
      if (!s) continue;
      const idx = s.indexOf('→') >= 0 ? s.indexOf('→') : s.indexOf('->');
      if (idx < 0) continue;
      const arrowLen = s.startsWith('->', idx) ? 2 : 1;
      pairs.push({
        input: s.slice(0, idx).trim(),
        output: s.slice(idx + arrowLen).trim(),
      });
    }
    if (pairs.length === 0) {
      setValidation(null);
      return;
    }
    setValidating(true);
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(apiUrl('/quality/validate/'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          session_id: sessionId,
          column: selectedColumn,
          examples: pairs,
          transformation_id: transformationId || null,
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
      setValidation({
        ok: Boolean(body.ok),
        errors: Array.isArray(body.errors) ? body.errors : [],
        warnings: Array.isArray(body.warnings) ? body.warnings : [],
        info: Array.isArray(body.info) ? body.info : [],
      });
    } catch (err) {
      setValidation({
        ok: false,
        errors: [(err as Error).message],
        warnings: [],
        info: [],
      });
    } finally {
      setValidating(false);
    }
  }, [sessionId, selectedColumn, examples, transformationId]);

  // Re-run preflight when the generated transformation changes.
  useEffect(() => {
    runValidation();
  }, [runValidation]);

  if (!sessionId) return null;

  return (
    <section className="mb-8 rounded-2xl border border-neutral-200 bg-white p-6 shadow-lg animate-slide-in">
      <div className="flex items-center justify-between mb-4">
        <h2 className="section-title flex items-center text-xl font-semibold text-neutral-700 m-0">
          <CheckCircle2 size={20} className="mr-2 text-primary" />
          Data Quality
        </h2>
        <button
          onClick={loadProfile}
          disabled={loading}
          className="btn bg-white hover:bg-neutral-100 text-neutral-700 border border-neutral-300 text-sm py-1.5 px-3 disabled:opacity-50 flex items-center"
        >
          {loading ? (
            <Loader2 size={14} className="mr-1.5 animate-spin" />
          ) : (
            <RefreshCw size={14} className="mr-1.5" />
          )}
          Refresh profile
        </button>
      </div>

      {error && (
        <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-md px-3 py-2 mb-3">
          {error}
        </p>
      )}

      {loading && !profile && (
        <div className="flex items-center text-sm text-neutral-500 py-6">
          <Loader2 size={16} className="mr-2 animate-spin" /> Profiling dataset…
        </div>
      )}

      {profile && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
            <div className="bg-neutral-50 border border-neutral-200 rounded-lg p-3">
              <p className="text-xs text-neutral-500">Rows</p>
              <p className="font-semibold text-neutral-800">
                {profile.total_rows.toLocaleString()}
                {profile.approximate && (
                  <span className="ml-1 text-[10px] text-amber-600 font-normal">(sampled)</span>
                )}
              </p>
            </div>
            <div className="bg-neutral-50 border border-neutral-200 rounded-lg p-3">
              <p className="text-xs text-neutral-500">Columns</p>
              <p className="font-semibold text-neutral-800">{profile.overview.columns}</p>
            </div>
            <div className="bg-neutral-50 border border-neutral-200 rounded-lg p-3">
              <p className="text-xs text-neutral-500">Duplicate rows</p>
              <p className="font-semibold text-neutral-800">{profile.overview.duplicate_rows}</p>
            </div>
            <div className="bg-neutral-50 border border-neutral-200 rounded-lg p-3">
              <p className="text-xs text-neutral-500">Potential keys</p>
              <p className="font-semibold text-neutral-800 truncate" title={profile.potential_keys.join(', ')}>
                {profile.potential_keys.length > 0 ? profile.potential_keys.join(', ') : '—'}
              </p>
            </div>
          </div>

          {profile.overview.warnings.length > 0 && (
            <ul className="space-y-1">
              {profile.overview.warnings.map((w, i) => (
                <li key={i} className="flex items-start gap-2 text-sm text-neutral-600">
                  <SeverityIcon severity={w.severity} />
                  <span>{w.message}</span>
                </li>
              ))}
            </ul>
          )}

          <div className="overflow-x-auto border border-neutral-200 rounded-lg">
            <table className="min-w-full text-sm divide-y divide-neutral-200">
              <thead className="bg-neutral-50">
                <tr>
                  <th className="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-neutral-500">
                    Column
                  </th>
                  <th className="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-neutral-500">
                    Type
                  </th>
                  <th className="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-neutral-500">
                    Missing %
                  </th>
                  <th className="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-neutral-500">
                    Unique
                  </th>
                  <th className="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-neutral-500">
                    Issues
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-neutral-100">
                {profile.columns.map((col) => (
                  <React.Fragment key={col.name}>
                    <tr
                      className="hover:bg-neutral-50 cursor-pointer"
                      onClick={() => setExpanded(expanded === col.name ? null : col.name)}
                    >
                      <td className="px-3 py-2 font-medium text-neutral-800">
                        {col.name}
                        {profile.potential_keys.includes(col.name) && (
                          <span className="ml-1.5 px-1.5 py-0.5 bg-green-100 text-green-700 rounded text-[10px]">
                            key?
                          </span>
                        )}
                      </td>
                      <td className="px-3 py-2 text-neutral-600">{col.family}</td>
                      <td className="px-3 py-2 text-neutral-600">{col.missing_pct}%</td>
                      <td className="px-3 py-2 text-neutral-600">{col.unique_count}</td>
                      <td className="px-3 py-2">
                        {col.issues.length === 0 ? (
                          <span className="text-neutral-400">—</span>
                        ) : (
                          <span className="text-xs text-amber-700">
                            {col.issues.length} issue(s)
                          </span>
                        )}
                      </td>
                    </tr>
                    {expanded === col.name && (
                      <tr className="bg-neutral-50/60">
                        <td colSpan={5} className="px-3 py-3 text-xs text-neutral-600 space-y-1">
                          <p>
                            dtype: <code>{col.dtype}</code>
                            {col.numeric?.min !== undefined && (
                              <> · min {col.numeric.min} · max {col.numeric.max} · mean {col.numeric.mean}</>
                            )}
                          </p>
                          {col.samples.length > 0 && (
                            <p>
                              samples:{' '}
                              {col.samples.slice(0, 5).map((s) => JSON.stringify(s)).join(', ')}
                            </p>
                          )}
                          {col.categorical?.top_values && (
                            <p>
                              top:{' '}
                              {col.categorical.top_values
                                .slice(0, 5)
                                .map((t) => `${t.value} (${t.pct}%)`)
                                .join(', ')}
                            </p>
                          )}
                          {col.issues.map((iss, i) => (
                            <p key={i} className="flex items-start gap-1.5">
                              <SeverityIcon severity={iss.severity} />
                              <span>{iss.message}</span>
                            </p>
                          ))}
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {selectedColumn && examples?.trim() && (
        <div className="mt-4 border-t border-neutral-200 pt-4">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-semibold text-neutral-700">
              Transformation preflight — {selectedColumn}
            </h3>
            <button
              onClick={runValidation}
              disabled={validating}
              className="btn bg-white hover:bg-neutral-100 text-neutral-700 border border-neutral-300 text-xs py-1 px-2 disabled:opacity-50"
            >
              {validating ? 'Checking…' : 'Re-check'}
            </button>
          </div>
          {validation === null && !validating && (
            <p className="text-xs text-neutral-500">
              Add examples to preflight the rule against this column.
            </p>
          )}
          {validating && (
            <p className="text-xs text-neutral-500 flex items-center">
              <Loader2 size={12} className="mr-1.5 animate-spin" /> Validating…
            </p>
          )}
          {validation && !validating && (
            <div className="space-y-1">
              {validation.errors.map((e, i) => (
                <p key={`e${i}`} className="text-xs text-red-700 flex items-start gap-1.5">
                  <AlertTriangle size={12} className="mt-0.5 shrink-0" /> {e}
                </p>
              ))}
              {validation.warnings.map((w, i) => (
                <p key={`w${i}`} className="text-xs text-amber-700 flex items-start gap-1.5">
                  <AlertTriangle size={12} className="mt-0.5 shrink-0" /> {w}
                </p>
              ))}
              {validation.info.map((inf, i) => (
                <p key={`i${i}`} className="text-xs text-sky-700 flex items-start gap-1.5">
                  <Info size={12} className="mt-0.5 shrink-0" /> {inf}
                </p>
              ))}
              {validation.ok && validation.errors.length === 0 && (
                <p className="text-xs text-emerald-700 flex items-center gap-1.5">
                  <CheckCircle2 size={12} /> Preflight passed — the rule reproduces every example.
                </p>
              )}
            </div>
          )}
        </div>
      )}

      {columns.length === 0 && (
        <p className="text-xs text-neutral-400 mt-2">No columns loaded yet.</p>
      )}
    </section>
  );
}
