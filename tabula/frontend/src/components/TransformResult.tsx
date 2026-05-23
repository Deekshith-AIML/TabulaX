import React, { useState, useRef, useEffect } from "react";
import Papa from "papaparse";
import { Download, RotateCcw, Check, AlertCircle, CheckCircle, AlertTriangle, UploadCloud, X, Database, Eye, EyeOff, Loader2, Code2, Save } from "lucide-react";
import { apiUrl, handleSessionExpired, SESSION_EXPIRED_MESSAGE } from "../lib/api";
import { dataKeys } from "../lib/data";

// Define a specific interface for connection information (can be shared if identical to DatabaseConnect)
interface ConnectionInfo {
  host?: string;
  port?: string;
  user?: string;
  password?: string;
  database?: string;
  table?: string;
  uri?: string;
}

// Define an interface for preview data rows (example, adjust as needed)
interface PreviewDataRow {
  [key: string]: any; // Allows any string keys
}

// Enriched reporting returned by /apply/ (spec §15): which transformation
// family ran, which provider served it, and the per-row outcome tallies, so
// the UI can state whether the transformation ACTUALLY succeeded instead of
// showing a generic green when rows failed or need review.
interface ApplyMetrics {
  family: string;
  provider: string;
  status: "success" | "review" | "partial";
  processed_rows: number;
  transformed_rows: number;
  failed_rows: number;
  review_rows: number;
  warnings: string[];
  row_errors: string[];
}

interface TransformResultProps {
  selectedColumn: string;
  code: string;
  transformationId: string;
  sessionId: string;
  llmAvailable?: boolean;
  setTransformedPreview: (preview: PreviewDataRow[]) => void;
  transformedPreview: PreviewDataRow[];
  isDatabaseMode: boolean;
  isMongoMode?: boolean;
  connectionInfo: ConnectionInfo;
  selectedDb: string;
  selectedTable: string;
  setReadyToUpdate: (ready: boolean) => void;
  originalPreview: PreviewDataRow[]; // Prop from App.tsx, assumed to be the true original
  setTransformedColumns: (updater: (prevCols: string[]) => string[]) => void; // Changed to accept an updater function
  // Applies a server dataset response (preview/columns/row counts) to the app
  // state; used by Revert so the whole UI stays in sync with the session.
  onDatasetReset?: (resp: Record<string, unknown>) => void;
  // Called after any operation that invalidates the row/column undo stack.
  clearEditUndo?: () => void;
  // Sync the authoritative row count after apply (the server always reports
  // the FULL dataset, never just the preview sample).
  setRowCount?: (n: number) => void;
}

const TransformResult: React.FC<TransformResultProps> = ({
  selectedColumn,
  code,
  transformationId,
  sessionId,
  llmAvailable,
  setTransformedPreview,
  transformedPreview, // This is the data that gets updated with transformations
  isDatabaseMode,
  isMongoMode,
  connectionInfo,
  selectedDb,
  selectedTable,
  setReadyToUpdate,
  originalPreview, // Use this prop for reverting CSV mode
  setTransformedColumns, // Use this prop from App.tsx
  onDatasetReset,
  clearEditUndo,
  setRowCount,
}) => {
  const [isApplying, setIsApplying] = useState(false);
  const [showRevertMenu, setShowRevertMenu] = useState(false);
  const [transformedDataForDownload, setTransformedDataForDownload] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const revertButtonRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isCodeVisible, setIsCodeVisible] = useState(true);
  const [isApplied, setIsApplied] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false); // New state for download loading
  // Outcome of the last apply (family / provider / row tallies).  Null until
  // an apply runs; drives the status panel so we never show a generic green.
  const [applyMetrics, setApplyMetrics] = useState<ApplyMetrics | null>(null);

  // Click outside handler for the revert menu
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        revertButtonRef.current &&
        menuRef.current &&
        !revertButtonRef.current.contains(event.target as Node) &&
        !menuRef.current.contains(event.target as Node)
      ) {
        setShowRevertMenu(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleRevertClick = () => {
    setShowRevertMenu(!showRevertMenu);
  };

  const handleApply = async () => {
    if (!selectedColumn || !transformationId) {
      setError("Please select a column and generate a transformation first.");
      return;
    }
    setIsApplying(true);
    setError(null);

    const sourceType = isMongoMode ? "mongodb_stored" : isDatabaseMode ? "database" : "file";

    try {
      const token = localStorage.getItem('token');
      // The client never sends raw Python. It references a transformation the
      // server stored under this user's account during /generate_function/.
      const payload: Record<string, unknown> = {
        transformation_id: transformationId,
        selected_column: selectedColumn,
        source_type: sourceType,
      };

      if (sourceType === "file" || sourceType === "mongodb_stored") {
        // These sources are read back from a server-side session.
        if (sessionId) payload.session_id = sessionId;
      } else if (sourceType === "database") {
        // MySQL tables snapshot into a server-side session too; send its
        // handle so apply sees edits (deleted rows/columns). db_info remains
        // for the explicit live write-back step.
        if (sessionId) payload.session_id = sessionId;
        payload.db_info = {
          ...connectionInfo,
          database: selectedDb,
          table: selectedTable,
          ...(isDatabaseMode && {
            host: connectionInfo.host,
            port: connectionInfo.port,
            user: connectionInfo.user,
            password: connectionInfo.password,
          }),
        };
      }

      const response = await fetch(apiUrl('/apply/'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        if (response.status === 401) {
          handleSessionExpired();
          throw new Error(SESSION_EXPIRED_MESSAGE);
        }
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errorData.detail || 'Failed to apply transformation');
      }

      const result = await response.json();
      if (result.preview) {
        setTransformedPreview(result.preview);
        // The server reports the FULL transformed dataset (every row of the
        // authoritative session, not the preview sample).
        if (typeof result.row_count === "number") setRowCount?.(result.row_count);
        if (Array.isArray(result.transformed_columns)) {
          // Visible dataset columns from the server (internal row ids excluded).
          setTransformedColumns(() => result.transformed_columns);
        } else if (result.preview.length > 0) {
          setTransformedColumns(() => dataKeys(result.preview[0]));
        }
        // Record the outcome so the status panel can report family, provider
        // and honest row tallies (partial / review, never a blanket green).
        setApplyMetrics({
          family: result.family || "",
          provider: result.provider || "",
          status: result.status || (result.failed_rows > 0 ? "partial" : "success"),
          processed_rows: result.processed_rows ?? 0,
          transformed_rows: result.transformed_rows ?? 0,
          failed_rows: result.failed_rows ?? 0,
          review_rows: result.review_rows ?? 0,
          warnings: Array.isArray(result.warnings) ? result.warnings : [],
          row_errors: Array.isArray(result.row_errors) ? result.row_errors : [],
        });
        setIsApplied(true);
        if (setReadyToUpdate) setReadyToUpdate(true);
      } else {
        setError("Transformation applied, but no preview data returned.");
      }
      // The undo stack describes rows/columns as they were before the apply;
      // restoring them now would produce an inconsistent dataset.
      clearEditUndo?.();
    } catch (err) {
      console.error("Transformation error:", err);
      setError((err as Error).message);
      if (setReadyToUpdate) setReadyToUpdate(false);
    } finally {
      setIsApplying(false);
    }
  };

  const handleDownloadTransformedFile = async () => {
    setIsDownloading(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');
      const downloadUrl = sessionId
        ? `${apiUrl('/download/')}?session_id=${encodeURIComponent(sessionId)}`
        : apiUrl('/download/');
      const response = await fetch(downloadUrl, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (!response.ok) {
        if (response.status === 401) {
          handleSessionExpired();
          throw new Error(SESSION_EXPIRED_MESSAGE);
        }
        // Try to parse error from backend if it sends JSON, otherwise use statusText
        let errorDetail = response.statusText;
        try {
          const errorData = await response.json();
          errorDetail = errorData.detail || errorDetail;
        } catch (e) {
          // Ignore if response is not JSON
        }
        throw new Error(errorDetail || 'Failed to download transformed data');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "transformed_data.csv"; // You can make the filename dynamic if needed
      document.body.appendChild(a); // Required for Firefox
      a.click();
      window.URL.revokeObjectURL(url);
      a.remove();

    } catch (err) {
      console.error("Download error:", err);
      setError((err as Error).message);
    } finally {
      setIsDownloading(false);
    }
  };

  const handleRevert = async () => {
    setError(null);
    if (sessionId) {
      // Revert the session's working copy on the server so the preview,
      // exports and any later refresh all agree with what is shown here.
      try {
        const token = localStorage.getItem('token');
        const response = await fetch(apiUrl('/revert_session/'), {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`,
          },
          body: JSON.stringify({ session_id: sessionId }),
        });
        if (!response.ok) {
          if (response.status === 401) {
            handleSessionExpired();
            throw new Error(SESSION_EXPIRED_MESSAGE);
          }
          const errorData = await response.json().catch(() => ({ detail: response.statusText }));
          throw new Error(errorData.detail || 'Failed to revert the dataset');
        }
        const result = await response.json();
        onDatasetReset?.(result);
      } catch (err) {
        console.error("Revert error:", err);
        setError((err as Error).message);
        return;
      }
    } else {
      // Live database preview (no session): client-side display revert only.
      setTransformedPreview(originalPreview);
      setTransformedColumns(() =>
        originalPreview.length > 0 ? dataKeys(originalPreview[0]) : []
      );
    }
    clearEditUndo?.();
    setIsApplied(false);
    setApplyMetrics(null);
    if (setReadyToUpdate) setReadyToUpdate(false);
    setShowRevertMenu(false);
  };

  useEffect(() => {
    // Reset applied state if the stored transformation or column changes
    setIsApplied(false);
    setApplyMetrics(null);
    if (setReadyToUpdate) setReadyToUpdate(false);
  }, [transformationId, selectedColumn, setReadyToUpdate]);

  if (!transformationId || !code) {
    return (
      <div className="p-4 bg-blue-50 border border-blue-200 rounded-md text-blue-700 text-sm flex items-center">
        <Code2 size={18} className="inline mr-2.5 shrink-0" />
        Generated Python function will appear here once examples are provided and classified.
      </div>
    );
  }

  return (
    <div className="space-y-5 p-5 bg-white rounded-lg border border-neutral-200 shadow-md">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold text-neutral-700 flex items-center">
          <Code2 size={20} className="mr-2 text-primary" />
          Generated Transformation Code
        </h3>
        <button 
          onClick={() => setIsCodeVisible(!isCodeVisible)}
          className="p-1.5 rounded-md hover:bg-neutral-100 text-neutral-500 transition-colors"
          title={isCodeVisible ? "Hide Code" : "Show Code"}
        >
          {isCodeVisible ? <EyeOff size={18} /> : <Eye size={18} />}
        </button>
      </div>

      {isCodeVisible && (
        <pre className="code-block p-4 bg-neutral-800 text-neutral-100 rounded-md text-sm overflow-x-auto max-h-60">
          <code>{code}</code>
        </pre>
      )}

      {llmAvailable === false && (
        <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded px-2.5 py-1.5">
          AI assistant is off — this was generated by a deterministic rule. Open-ended patterns need an
          <code className="font-mono"> OPENAI_API_KEY</code>.
        </p>
      )}

       {error && (
        <div className="bg-red-50 border-l-4 border-red-500 text-red-700 p-3 rounded-md flex items-start text-sm" role="alert">
          <AlertTriangle size={18} className="mr-2.5 shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      {isApplied && applyMetrics && (
        <div className={`rounded-md border p-3 text-sm ${
          applyMetrics.status === 'partial' ? 'bg-red-50 border-red-200'
          : applyMetrics.status === 'review' ? 'bg-amber-50 border-amber-200'
          : 'bg-green-50 border-green-200'
        }`}>
          <div className={`flex flex-wrap items-center gap-x-4 gap-y-1 ${
            applyMetrics.status === 'partial' ? 'text-red-700'
            : applyMetrics.status === 'review' ? 'text-amber-700'
            : 'text-green-700'
          }`}>
            <span className="flex items-center">
              {applyMetrics.status === 'partial'
                ? <AlertTriangle size={15} className="mr-1.5" />
                : applyMetrics.status === 'review'
                ? <AlertTriangle size={15} className="mr-1.5" />
                : <CheckCircle size={15} className="mr-1.5" />}
              <span className="font-medium">{applyMetrics.family || "Transformation"}</span>
            </span>
            {applyMetrics.provider && (
              <span>Provider: <span className="font-medium">{applyMetrics.provider}</span></span>
            )}
            <span>
              Rows: <span className="font-medium">{applyMetrics.transformed_rows}</span>
              {" / "}{applyMetrics.processed_rows} processed
            </span>
            {applyMetrics.failed_rows > 0 && (
              <span className="text-red-600">Failed: <span className="font-medium">{applyMetrics.failed_rows}</span></span>
            )}
            {applyMetrics.review_rows > 0 && (
              <span className="text-amber-600">Review: <span className="font-medium">{applyMetrics.review_rows}</span></span>
            )}
          </div>
          {applyMetrics.warnings.slice(0, 2).map((w, i) => (
            <p key={i} className={`mt-1.5 text-xs ${
              applyMetrics.status === 'partial' ? 'text-red-600'
              : applyMetrics.status === 'review' ? 'text-amber-600'
              : 'text-green-600'
            }`}>{w}</p>
          ))}
        </div>
      )}

      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 pt-2 border-t border-neutral-200">
        <p className={`text-sm ${
          isLoading ? 'text-neutral-500'
          : isApplied && applyMetrics?.status === 'partial' ? 'text-red-600 font-medium'
          : isApplied && applyMetrics?.status === 'review' ? 'text-amber-600 font-medium'
          : isApplied ? 'text-green-600 font-medium'
          : 'text-neutral-500'
        } flex items-center`}>
          {isLoading ? (
            <><Loader2 size={16} className="animate-spin mr-2" /> Applying...</>
          ) : isApplied && applyMetrics?.status === 'partial' ? (
            <><AlertTriangle size={16} className="mr-2" /> Applied with {applyMetrics.failed_rows} failed row(s) — review before exporting.</>
          ) : isApplied && applyMetrics?.status === 'review' ? (
            <><AlertTriangle size={16} className="mr-2" /> Applied; {applyMetrics.review_rows} row(s) flagged for review.</>
          ) : isApplied ? (
            <><CheckCircle size={16} className="mr-2" /> Transformation applied to preview.</>
          ) : (
            <>Apply the generated code to see a preview of the transformation.</>
          )}
        </p>
        <div className="flex items-center gap-3 flex-shrink-0">
          {isApplied && transformedPreview && transformedPreview.length > 0 && (
            <button
              onClick={handleDownloadTransformedFile}
              disabled={isDownloading || isLoading}
              className="btn bg-green-500 hover:bg-green-600 text-white text-sm py-2 px-3.5 disabled:opacity-60"
            >
              {isDownloading ? <Loader2 size={16} className="animate-spin mr-1.5" /> : <Download size={16} className="mr-1.5" />}
              {isDownloading ? 'Downloading...' : 'Download File'}
            </button>
          )}
          <button
            onClick={handleRevert}
            disabled={isLoading || !originalPreview || originalPreview.length === 0}
            className="btn bg-neutral-100 hover:bg-neutral-200 text-neutral-700 text-sm py-2 px-3.5 disabled:opacity-60"
          >
            <RotateCcw size={16} className="mr-1.5" />
            Revert
          </button>
          <button 
            onClick={handleApply}
            disabled={isLoading}
            className="btn btn-primary text-sm py-2 px-3.5 disabled:bg-primary/60"
          >
            <Save size={16} className="mr-1.5" />
            Apply to Preview
          </button>
        </div>
      </div>
    </div>
  );
};

export default TransformResult;
