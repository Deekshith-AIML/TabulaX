import React, { useCallback, useEffect, useState } from 'react';
import { Cpu, Download, KeyRound, Loader2, ShieldCheck } from 'lucide-react';
import { apiUrl, handleSessionExpired, SESSION_EXPIRED_MESSAGE } from '../lib/api';

interface OllamaStatus {
  base_url: string;
  model: string;
  reachable: boolean;
  model_installed: boolean;
  auto_download: boolean;
  error: string | null;
  installed_models?: string[];
}

interface Status {
  provider: string;
  configured: boolean;
  model: string;
  personal_key_supported: boolean;
  deterministic_always_available: boolean;
  max_calls_per_request: number;
  ollama?: OllamaStatus;
}

interface Props {
  /** Current transient personal key (React state only — never localStorage). */
  apiKey: string;
  onApiKeyChange: (key: string) => void;
}

/**
 * AI settings: provider status, a transient personal OpenAI key (kept in React
 * state only — never stored in localStorage/MongoDB), and an explicit local
 * Ollama model download with size/hardware disclosure.
 */
export default function LlmSettings({ apiKey, onApiKeyChange }: Props) {
  const [status, setStatus] = useState<Status | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [pulling, setPulling] = useState(false);
  const [pullMsg, setPullMsg] = useState('');
  const [showKey, setShowKey] = useState(false);

  const loadStatus = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const token = localStorage.getItem('token');
      const res = await fetch(apiUrl('/llm/status/'), {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) {
        if (res.status === 401) {
          handleSessionExpired();
          throw new Error(SESSION_EXPIRED_MESSAGE);
        }
        throw new Error(`HTTP ${res.status}`);
      }
      setStatus((await res.json()) as Status);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadStatus();
  }, [loadStatus]);

  const pullModel = async () => {
    if (
      !window.confirm(
        'Download the local AI model? This uses several GB of disk and can take minutes. ' +
          'It runs entirely on your machine (no data leaves it).'
      )
    ) {
      return;
    }
    setPulling(true);
    setPullMsg('');
    setError('');
    try {
      const token = localStorage.getItem('token');
      const res = await fetch(apiUrl('/llm/ollama/pull/'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ confirm: true }),
      });
      const body = await res.json().catch(() => null);
      if (!res.ok) {
        if (res.status === 401) {
          handleSessionExpired();
          throw new Error(SESSION_EXPIRED_MESSAGE);
        }
        const d = (body as { detail?: string } | null)?.detail;
        throw new Error(d || `HTTP ${res.status}`);
      }
      setPullMsg(`Model ready: ${JSON.stringify(body)}`);
      loadStatus();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setPulling(false);
    }
  };

  return (
    <details className="mb-6 rounded-xl border border-neutral-200 bg-white/80 p-4 text-sm">
      <summary className="cursor-pointer font-semibold text-neutral-700 flex items-center">
        <KeyRound size={16} className="mr-2 text-primary" />
        AI settings (optional)
      </summary>
      <div className="mt-3 space-y-4">
        {error && (
          <p className="text-red-600 bg-red-50 border border-red-200 rounded-md px-3 py-2">{error}</p>
        )}

        <div className="flex items-center justify-between">
          <p className="text-neutral-600">
            Server provider:{' '}
            <strong>{status ? status.provider : '…'}</strong>
            {status && (
              <span className={status.configured ? 'text-emerald-600' : 'text-neutral-400'}>
                {' '}({status.configured ? 'configured' : 'not configured'})
              </span>
            )}
          </p>
          <button
            onClick={loadStatus}
            disabled={loading}
            className="text-xs text-neutral-500 hover:text-primary disabled:opacity-50"
          >
            {loading ? '…' : 'Refresh'}
          </button>
        </div>

        <p className="text-xs text-neutral-500">
          Deterministic rules (case, numbers, date templates, exact lookups) always run first and
          never need an AI provider. The AI is only a fallback for ambiguous examples, and your
          data never leaves this machine unless you configure a hosted provider.
        </p>

        <div>
          <label className="block text-xs font-semibold uppercase tracking-wider text-neutral-500 mb-1">
            Personal OpenAI API key (optional, this browser session only)
          </label>
          <div className="flex gap-2">
            <input
              type={showKey ? 'text' : 'password'}
              value={apiKey}
              onChange={(e) => onApiKeyChange(e.target.value)}
              placeholder="sk-…"
              autoComplete="off"
              spellCheck={false}
              className="flex-1 border border-neutral-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/40"
            />
            <button
              type="button"
              onClick={() => setShowKey((v) => !v)}
              className="text-xs text-neutral-500 hover:text-neutral-700 border border-neutral-300 rounded-lg px-2"
            >
              {showKey ? 'Hide' : 'Show'}
            </button>
          </div>
          <p className="mt-1.5 text-xs text-neutral-500 flex items-start gap-1.5">
            <ShieldCheck size={12} className="mt-0.5 shrink-0 text-emerald-600" />
            The key is kept only in this page&apos;s memory for the current session. It is never
            saved to this computer, our servers, or history — and is only sent to api.openai.com
            over HTTPS when you generate or classify.
          </p>
        </div>

        {status?.ollama && (
          <div className="border border-neutral-200 rounded-lg p-3 bg-neutral-50">
            <p className="font-medium text-neutral-700 flex items-center">
              <Cpu size={14} className="mr-1.5" />
              Local model (Ollama) — {status.ollama.model}
            </p>
            <p className="text-xs text-neutral-600 mt-1">
              {status.ollama.reachable
                ? status.ollama.model_installed
                  ? 'Server reachable; model installed.'
                  : 'Server reachable; model not downloaded yet.'
                : status.ollama.error || 'Server not reachable.'}
            </p>
            <p className="text-xs text-neutral-500 mt-1">
              Runs fully offline and free. Requires ~8 GB RAM for llama3.2 and a few GB of disk.
            </p>
            {!status.ollama.model_installed && (
              <button
                onClick={pullModel}
                disabled={pulling || !status.ollama.reachable}
                className="mt-2 btn bg-white hover:bg-neutral-100 text-neutral-700 border border-neutral-300 text-xs py-1.5 px-3 disabled:opacity-50 flex items-center"
              >
                {pulling ? (
                  <Loader2 size={12} className="mr-1.5 animate-spin" />
                ) : (
                  <Download size={12} className="mr-1.5" />
                )}
                Download model (several GB)
              </button>
            )}
            {pullMsg && <p className="text-xs text-emerald-700 mt-2">{pullMsg}</p>}
          </div>
        )}
      </div>
    </details>
  );
}
