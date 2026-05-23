import React, { useState, useEffect } from "react";
import { useDropzone } from 'react-dropzone';
import { FileUp, Database, Code, Download, X, ArrowRight, Wand2, LogOut, Trash2, Undo2 } from 'lucide-react';
import './index.css';

// Components
import DatabaseConnect from './components/DatabaseConnect';
import ColumnSelector from './components/ColumnSelector';
import ExampleInput from './components/ExampleInput';
import TransformResult from './components/TransformResult';
import LandingPage from './components/LandingPage';
import MongoConnect from './components/MongoConnect';
import SourceDeleteDialog, { SourceDeleteTarget } from './components/SourceDeleteDialog';
import SourceRestoreDialog from './components/SourceRestoreDialog';
import DataQualityPanel from './components/DataQualityPanel';
import HistoryPanel from './components/HistoryPanel';
import LlmSettings from './components/LlmSettings';
import { useAuth } from './components/AuthContext';
import { apiUrl, handleSessionExpired, SESSION_EXPIRED_MESSAGE } from './lib/api';
import { ROW_ID_KEY, dataKeys, rowIdOf } from './lib/data';

// Shared ConnectionInfo type
interface ConnectionInfo {
  host?: string;
  port?: string;
  user?: string;
  password?: string;
  database?: string; 
  table?: string;    
  uri?: string;      
}

// One entry per deletion, undone newest-first.
type UndoOp =
  | { kind: 'rows'; rows: Record<string, unknown>[] }
  | { kind: 'columns'; cols: Record<string, unknown>[] };

// Remembers the active file session across refreshes so the UI reloads the
// server's (edited) dataset instead of stale or missing data.
const SESSION_KEY = 'tabula.activeSession';
interface SavedSession {
  type: 'file';
  session_id: string;
  source_name?: string | null;
}
// The undo stack is per-session and must survive a page refresh — otherwise a
// reload wipes the deleted rows and "Undo last deletion" becomes impossible
// even though the server still holds them in history.  Persisted next to the
// active session and keyed by its id so a different session never reuses it.
const UNDO_KEY = 'tabula.undoStack';
interface SavedUndo {
  session_id: string;
  stack: UndoOp[];
}

function App() {
  const { isAuthenticated, user, logout } = useAuth();
  
  const [file, setFile] = useState<File | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [previewData, setPreviewData] = useState<any[]>([]);
  const [unmodifiedPreviewData, setUnmodifiedPreviewData] = useState<any[]>([]);
  const [selectedColumn, setSelectedColumn] = useState('');
  const [examples, setExamples] = useState('');
  const [classification, setClassification] = useState('');
  const [generatedCode, setGeneratedCode] = useState('');
  // Server-side handles: the raw code is never sent back to /apply/*.  These
  // track the session of the loaded dataset and the stored transformation.
  const [sessionId, setSessionId] = useState('');
  const [transformationId, setTransformationId] = useState('');
  const [llmAvailable, setLlmAvailable] = useState(false);
  const [error, setError] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [transformedColumns, setTransformedColumns] = useState<string[]>([]);
  const [transformedCsv, setTransformedCsv] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('file'); // 'file', 'database', 'mongodb', or 'sql-view'
  const [isDatabaseMode, setIsDatabaseMode] = useState(false); // True for MySQL, could be reused or have a specific mongo one
  const [isMongoMode, setIsMongoMode] = useState(false);
  const [mongoUri, setMongoUri] = useState('mongodb://localhost:27017/');
  const [selectedMongoDb, setSelectedMongoDb] = useState('');
  const [selectedMongoCollection, setSelectedMongoCollection] = useState('');
  
  // Use the shared ConnectionInfo type for the state
  const [connectionInfo, setConnectionInfo] = useState<ConnectionInfo>({
    host: 'localhost', // Default for MySQL
    port: '3306',    // Default for MySQL
    user: 'root',     // Default for MySQL
    password: '',   // Default for MySQL
    // uri will be set for MongoDB mode
  });

  const [selectedDb, setSelectedDb] = useState('');
  const [selectedTable, setSelectedTable] = useState('');
  const [readyToUpdate, setReadyToUpdate] = useState(false);
  const [sqlContent, setSqlContent] = useState('');

  // Dataset editing: selection, undo stack and row counts synced with the
  // server-side session (see /delete_rows/, /restore_columns/, ...).
  const [rowCount, setRowCount] = useState(0);
  // Bumped whenever the authoritative dataset changes (apply, restore,
  // session swap) so panels keyed on the session alone still re-fetch.
  const [datasetVersion, setDatasetVersion] = useState(0);
  const [selectedRowIds, setSelectedRowIds] = useState<Set<number>>(new Set());
  const [undoStack, setUndoStack] = useState<UndoOp[]>([]);
  const [isEditing, setIsEditing] = useState(false);
  const [showColumnPicker, setShowColumnPicker] = useState(false);
  const [pendingColumns, setPendingColumns] = useState<string[]>([]);
  // Permanent SOURCE deletion (destructive dialog) and its success banner.
  const [sourceDeleteTarget, setSourceDeleteTarget] = useState<SourceDeleteTarget | null>(null);
  const [notice, setNotice] = useState('');
  // Source restore from a verified backup (opens from History).
  const [sourceRestoreId, setSourceRestoreId] = useState<string | null>(null);
  // Transient personal AI key — React state only, never localStorage.
  const [llmApiKey, setLlmApiKey] = useState('');

  // Persist the undo stack to localStorage, keyed to the active session so a
  // different session never inherits stale rows.  Writes are best-effort: a
  // full/ unavailable localStorage must never break an edit.
  const persistUndo = (sid: string, stack: UndoOp[]) => {
    try {
      if (!sid || stack.length === 0) {
        localStorage.removeItem(UNDO_KEY);
        return;
      }
      localStorage.setItem(UNDO_KEY, JSON.stringify({ session_id: sid, stack }));
    } catch {
      /* quota/private mode: undo just won't survive a reload */
    }
  };

  // Read back the persisted undo stack for a session (empty if none/expired).
  const loadPersistedUndo = (sid: string): UndoOp[] => {
    try {
      const raw = localStorage.getItem(UNDO_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw) as SavedUndo;
      if (!parsed || parsed.session_id !== sid || !Array.isArray(parsed.stack)) {
        return [];
      }
      return parsed.stack;
    } catch {
      return [];
    }
  };

  const onDrop = (acceptedFiles: File[]) => {
    if (acceptedFiles && acceptedFiles.length > 0) {
      setFile(acceptedFiles[0]);
      handleFileUploadFromDrop(acceptedFiles[0]);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/json': ['.json'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/sql': ['.sql'],
      'text/plain': ['.sql']
    },
    maxFiles: 1,
    multiple: false
  });

  // Restore the active file session after a refresh/navigation so the preview
  // always reflects the server's current dataset (edits included).
  useEffect(() => {
    if (!isAuthenticated) return;
    const raw = localStorage.getItem(SESSION_KEY);
    if (!raw) return;
    let saved: SavedSession | null = null;
    try {
      saved = JSON.parse(raw) as SavedSession;
    } catch {
      localStorage.removeItem(SESSION_KEY);
      return;
    }
    if (!saved || saved.type !== 'file' || !saved.session_id) return;

    let cancelled = false;
    (async () => {
      try {
        const token = localStorage.getItem('token');
        const response = await fetch(
          apiUrl(`/session_preview/?session_id=${encodeURIComponent(saved!.session_id)}`),
          { headers: { 'Authorization': `Bearer ${token}` } }
        );
        if (!response.ok) {
          if (response.status === 401) {
            handleSessionExpired();
            return;
          }
          // Session expired (12h TTL) or was deleted: forget it instead of
          // showing stale data.  History snapshots survive for 7 days and
          // can recreate the working copy from the History panel.
          localStorage.removeItem(SESSION_KEY);
          return;
        }
        const body = await response.json();
        if (cancelled) return;
        setSessionId(body.session_id || saved!.session_id);
        setColumns(Array.isArray(body.columns) ? body.columns : []);
        setPreviewData(Array.isArray(body.preview) ? body.preview : []);
        setUnmodifiedPreviewData(Array.isArray(body.original_preview) ? body.original_preview : []);
        setRowCount(typeof body.row_count === 'number'
          ? body.row_count
          : (Array.isArray(body.preview) ? body.preview.length : 0));
        setSelectedRowIds(new Set());
        // Restore any in-progress undo stack from localStorage so a reload
        // doesn't strand a "delete → undo" flow with the rows unrecoverable
        // from the UI.  (Server still has them in history either way.)
        setUndoStack(loadPersistedUndo(body.session_id || saved!.session_id));
        setTransformedColumns([]);
        setShowColumnPicker(false);
        setPendingColumns([]);
        if (saved!.source_name) {
          setFile(new File([], saved!.source_name));
        }
        setActiveTab('file');
        setIsDatabaseMode(false);
        setIsMongoMode(false);
        // Surface the history origin of this working copy after a refresh
        // (the marker is stored server-side in the session's source_meta).
        const histOrigin = body.source_meta?.history;
        if (histOrigin?.version_number != null) {
          setNotice(
            `Working copy opened from history (v${histOrigin.version_number}` +
            `${histOrigin.operation ? `, ${histOrigin.operation}` : ''}). ` +
            'The saved historical snapshot is preserved.'
          );
        }
      } catch (err) {
        console.error('Session restore failed:', err);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [isAuthenticated]);

  const handleFileUploadFromDrop = async (file: File) => {
    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const token = localStorage.getItem('token');
      const response = await fetch(apiUrl('/upload_file/'), {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: formData,
      });

      if (!response.ok) {
        if (response.status === 401) {
          handleSessionExpired();
          throw new Error(SESSION_EXPIRED_MESSAGE);
        }
        let errorMessage = `HTTP error! status: ${response.status}`;
        try {
          const errorData = await response.json();
          if (errorData && errorData.detail) {
            errorMessage = errorData.detail;
          }
        } catch (e) {
          // If response is not JSON or doesn't have detail, stick with the HTTP status
        }
        throw new Error(errorMessage);
      }

      const data = await response.json();
      // Capture the server-side session handle for subsequent /apply/ calls.
      setSessionId(data.session_id || '');
      setSelectedRowIds(new Set());
      setUndoStack([]);
      persistUndo(data.session_id || '', []);
      setShowColumnPicker(false);
      setPendingColumns([]);
      
      if (data.file_type === 'sql') {
        // Handle SQL file differently
        setSqlContent(data.sql_content);
        setColumns([]);
        setPreviewData([]);
        setUnmodifiedPreviewData([]);
        setRowCount(0);
        localStorage.removeItem(SESSION_KEY);
        // Show SQL content in a formatted way
        setActiveTab('sql-view');
      } else {
        // Handle CSV, JSON, and XLSX files
        setColumns(data.columns);
        setPreviewData(data.preview);
        setUnmodifiedPreviewData(data.preview);
        setRowCount(typeof data.row_count === 'number'
          ? data.row_count
          : (Array.isArray(data.preview) ? data.preview.length : 0));
        if (data.session_id) {
          const saved: SavedSession = {
            type: 'file',
            session_id: data.session_id,
            source_name: file.name,
          };
          localStorage.setItem(SESSION_KEY, JSON.stringify(saved));
        }
      }
      setError('');
    } catch (err) {
      console.error('Upload error:', err);
      setError('Failed to upload file: ' + (err as Error).message);
      setFile(null); // Reset file selection on error
    } finally {
      setIsUploading(false);
    }
  };

  // Handle manual download of transformed CSV
  const handleManualDownload = () => {
    if (!transformedCsv) {
      setError('No transformed CSV available to download');
      return;
    }
    
    const blob = new Blob([transformedCsv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'transformed_data.csv';
    a.click();
    window.URL.revokeObjectURL(url);
  };

  // Handle download from server
  const handleDownload = async () => {
    try {
      const token = localStorage.getItem('token');
      const downloadUrl = sessionId
        ? `${apiUrl('/download/')}?session_id=${encodeURIComponent(sessionId)}`
        : apiUrl('/download/');
      const response = await fetch(downloadUrl, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (!response.ok) {
        if (response.status === 401) {
          handleSessionExpired();
          throw new Error(SESSION_EXPIRED_MESSAGE);
        }
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const csvText = await response.text();
      const blob = new Blob([csvText], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'transformed_data.csv';
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Download error:', err);
      setError('Failed to download CSV: ' + (err as Error).message);
    }
  };

  // Dismiss error message
  const dismissError = () => setError('');
  const normalize = (s: string) => s.toLowerCase().replace(/\s+/g, '');

  const handleRemoveFile = () => {
    setFile(null);
    setColumns([]);
    setPreviewData([]);
    setUnmodifiedPreviewData([]);
    setSelectedColumn('');
    setExamples('');
    setClassification('');
    setGeneratedCode('');
    setSessionId('');
    setTransformationId('');
    setLlmAvailable(false);
    setTransformedColumns([]);
    setTransformedCsv(null);
    setError('');
    setRowCount(0);
    setSelectedRowIds(new Set());
    setUndoStack([]);
    localStorage.removeItem(UNDO_KEY);
    setShowColumnPicker(false);
    setPendingColumns([]);
    localStorage.removeItem(SESSION_KEY);
    // Reset tab to file if it was sql-view or a DB view
    if (activeTab === 'sql-view' || activeTab === 'database' || activeTab === 'mongodb') {
      setActiveTab('file');
    }
    setIsDatabaseMode(false); // Reset MySQL mode
    setIsMongoMode(false); // Reset Mongo mode
    setSqlContent('');
  };

  const canEdit = Boolean(sessionId);

  // Apply a dataset response from an edit/undo/revert endpoint to the UI.
  const applyEditResponse = (resp: Record<string, unknown>) => {
    if (Array.isArray(resp.preview)) setPreviewData(resp.preview);
    if (Array.isArray(resp.columns)) {
      const cols: string[] = resp.columns;
      setColumns(cols);
      if (selectedColumn && !cols.includes(selectedColumn)) {
        setSelectedColumn('');
      }
      if (!Array.isArray(resp.transformed_columns)) {
        setTransformedColumns((prev: string[]) => prev.filter((c) => cols.includes(c)));
      }
    }
    if (Array.isArray(resp.transformed_columns)) {
      setTransformedColumns(resp.transformed_columns);
    }
    if (typeof resp.row_count === 'number') setRowCount(resp.row_count);
    setSelectedRowIds(new Set());
    setDatasetVersion((v) => v + 1);
  };

  // Re-fetch a session's preview from the server so the UI state is always
  // the authoritative dataset (used after any restore/open/swap).
  const refreshFromBackend = async (sid: string, fallbackNotice?: string) => {
    if (!sid) return;
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(
        apiUrl(`/session_preview/?session_id=${encodeURIComponent(sid)}`),
        { headers: { 'Authorization': `Bearer ${token}` } }
      );
      if (!response.ok) {
        if (response.status === 401) {
          handleSessionExpired();
          return;
        }
        // Never fall back to stale data silently — say it out loud.
        setNotice(
          `${fallbackNotice || 'The dataset was updated on the server.'} ` +
          'The preview could not be reloaded automatically — reconnect the dataset to see the latest state.'
        );
        return;
      }
      const body = await response.json();
      setSessionId(body.session_id || sid);
      if (Array.isArray(body.columns)) setColumns(body.columns);
      if (Array.isArray(body.preview)) setPreviewData(body.preview);
      if (Array.isArray(body.original_preview)) {
        setUnmodifiedPreviewData(body.original_preview);
      }
      if (typeof body.row_count === 'number') setRowCount(body.row_count);
      setSelectedRowIds(new Set());
      setTransformedColumns([]);
      setShowColumnPicker(false);
      setPendingColumns([]);
      setDatasetVersion((v) => v + 1);
      // Surface the history marker stored server-side (e.g. "restored to
      // vN") so a refresh/reopen never silently loses the context.
      const histOrigin = body.source_meta?.history;
      if (histOrigin?.version_number != null) {
        setNotice(
          `${fallbackNotice || 'Working copy updated.'} ` +
          `Server state: v${histOrigin.version_number}` +
          `${histOrigin.operation ? ` (${histOrigin.operation})` : ''}.`
        );
      } else if (fallbackNotice) {
        setNotice(fallbackNotice);
      }
    } catch (err) {
      console.error('Session refresh failed:', err);
    }
  };

  // Centralize "make THIS response the active dataset": adopt the session id,
  // clear every stale piece of working state (undo, selections, pending
  // transforms), refresh from the server, and route to the matching tab.
  const activateDataset = (resp: Record<string, unknown>, notice?: string) => {
    const sid = typeof resp.session_id === 'string' ? resp.session_id : '';
    if (!sid) {
      // Honest failure: no usable session came back — warn, never fake it.
      if (notice) {
        setError('');
        setNotice(notice);
      }
      return;
    }
    const src = typeof resp.source === 'string' ? resp.source : 'file';
    const name =
      typeof resp.source_name === 'string' && resp.source_name
        ? resp.source_name
        : file?.name || 'dataset';
    setSessionId(sid);
    applyEditResponse(resp);
    setUnmodifiedPreviewData(Array.isArray(resp.preview) ? resp.preview : []);
    setSelectedColumn('');
    setExamples('');
    setClassification('');
    setGeneratedCode('');
    setTransformationId('');
    setTransformedCsv(null);
    setUndoStack([]);
    persistUndo(sid, []);
    setSelectedRowIds(new Set());
    setShowColumnPicker(false);
    setPendingColumns([]);
    setFile(new File([], name));
    if (src === 'mysql' || src === 'database') {
      setActiveTab('database');
      setIsDatabaseMode(true);
      setIsMongoMode(false);
    } else if (src === 'mongodb') {
      setActiveTab('mongodb');
      setIsDatabaseMode(false);
      setIsMongoMode(true);
    } else {
      setActiveTab('file');
      setIsDatabaseMode(false);
      setIsMongoMode(false);
    }
    if (src === 'file') {
      localStorage.setItem(
        SESSION_KEY,
        JSON.stringify({ type: 'file', session_id: sid, source_name: name })
      );
    } else {
      localStorage.removeItem(SESSION_KEY);
    }
    setError('');
    // Authoritative rehydrate: the server's session_preview wins over the
    // operation's own response payload.
    void refreshFromBackend(sid, notice);
  };

  // "Open" a history version: the backend created a NEW isolated working
  // session from the exact snapshot; adopt it as the active dataset and route
  // to the matching workspace tab.  The historical snapshot stays untouched.
  const handleOpenVersion = (resp: Record<string, unknown>) => {
    const fromV = resp.from_version_number != null ? `v${resp.from_version_number}` : 'history';
    const fromOp = typeof resp.from_operation === 'string' ? `, ${resp.from_operation}` : '';
    activateDataset(
      resp,
      `Opened ${fromV}${fromOp} as a new working copy. The historical snapshot is preserved.`
    );
  };

  const postEdit = async (path: string, payload: unknown) => {
    const token = localStorage.getItem('token');
    const response = await fetch(apiUrl(path), {
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
      throw new Error(errorData.detail || `Request failed: ${response.statusText}`);
    }
    return response.json();
  };

  const dismissNotice = () => setNotice('');

  // True only for database-backed sessions whose source the API can actually
  // modify (file sessions never qualify: the original file is untouched).
  const canSourceDelete = (isDatabaseMode || isMongoMode) && Boolean(sessionId);

  // Apply a successful permanent-deletion response: the backend re-synced the
  // session from the live source (new session id, or none if emptied), so the
  // UI must swap sessions, drop undo history and show the honest outcome.
  const applySourceDeleteResponse = (resp: Record<string, unknown>) => {
    setSourceDeleteTarget(null);
    setSessionId(typeof resp.session_id === 'string' ? resp.session_id : '');
    applyEditResponse(resp);
    setUndoStack([]);
    persistUndo(typeof resp.session_id === 'string' ? resp.session_id : '', []);
    setShowColumnPicker(false);
    setPendingColumns([]);
    const warnings = Array.isArray(resp.warnings)
      ? (resp.warnings as string[]).filter(Boolean)
      : [];
    const message = String(resp.message || 'Permanent deletion completed.');
    const text = [message, ...warnings].join(' ');
    if (resp.partial_failure) {
      // Partial: some changes applied, some failed — show it as an error so
      // it cannot be mistaken for a clean success, but the view IS refreshed.
      setNotice('');
      setError(text);
    } else {
      setError('');
      setNotice(text);
    }
  };

  const openSourceDeleteRows = () => {
    if (!sessionId || selectedRowIds.size === 0 || isEditing) return;
    setNotice('');
    setSourceDeleteTarget(isMongoMode ? 'mongo_documents' : 'mysql_rows');
  };

  const openSourceDeleteColumns = () => {
    if (!sessionId || pendingColumns.length === 0 || isEditing) return;
    setNotice('');
    setSourceDeleteTarget(isMongoMode ? 'mongo_fields' : 'mysql_columns');
  };

  const handleDeleteRows = async () => {
    if (!sessionId || selectedRowIds.size === 0 || isEditing) return;
    const ids = Array.from(selectedRowIds);
    if (!window.confirm(`Delete ${ids.length} selected row(s)? You can undo this.`)) {
      return;
    }
    setIsEditing(true);
    try {
      const body = await postEdit('/delete_rows/', {
        session_id: sessionId,
        row_ids: ids,
      });
      applyEditResponse(body);
      const entry: UndoOp = { kind: 'rows', rows: body.deleted_rows || [] };
      setUndoStack((prev) => {
        const next = [...prev, entry];
        persistUndo(sessionId, next);
        return next;
      });
      setError('');
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsEditing(false);
    }
  };

  const handleDeleteColumns = async () => {
    if (!sessionId || pendingColumns.length === 0 || isEditing) return;
    if (!window.confirm(
      `Delete column(s): ${pendingColumns.join(', ')}? You can undo this.`
    )) {
      return;
    }
    setIsEditing(true);
    try {
      const body = await postEdit('/delete_columns/', {
        session_id: sessionId,
        columns: pendingColumns,
      });
      applyEditResponse(body);
      const entry: UndoOp = { kind: 'columns', cols: body.deleted_columns || [] };
      setUndoStack((prev) => {
        const next = [...prev, entry];
        persistUndo(sessionId, next);
        return next;
      });
      setShowColumnPicker(false);
      setPendingColumns([]);
      setError('');
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsEditing(false);
    }
  };

  const handleUndo = async () => {
    const last = undoStack[undoStack.length - 1];
    if (!last || !sessionId || isEditing) return;
    setIsEditing(true);
    try {
      const body = last.kind === 'rows'
        ? await postEdit('/restore_rows/', { session_id: sessionId, rows: last.rows })
        : await postEdit('/restore_columns/', { session_id: sessionId, columns: last.cols });
      applyEditResponse(body);
      setUndoStack((prev) => {
        const next = prev.slice(0, -1);
        persistUndo(sessionId, next);
        return next;
      });
      setError('');
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsEditing(false);
    }
  };

  const toggleRowSelection = (id: number) => {
    setSelectedRowIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const allVisibleIds = previewData.map((row, index) => rowIdOf(row, index));
  const allVisibleSelected =
    allVisibleIds.length > 0 && allVisibleIds.every((id) => selectedRowIds.has(id));
  const toggleSelectAll = () => {
    setSelectedRowIds(allVisibleSelected ? new Set() : new Set(allVisibleIds));
  };

  const toggleColumnPick = (name: string) => {
    setPendingColumns((prev) =>
      prev.includes(name) ? prev.filter((c) => c !== name) : [...prev, name]
    );
  };

  // Switching section (Files / Database / MongoDB) always starts from a clean
  // slate: the shared workspace (preview, columns, transformation, edits,
  // connection selections, errors) must not leak into the next section.
  const handleSectionChange = (tab: string) => {
    if (tab === activeTab) return;
    setFile(null);
    setColumns([]);
    setPreviewData([]);
    setUnmodifiedPreviewData([]);
    setSelectedColumn('');
    setExamples('');
    setClassification('');
    setGeneratedCode('');
    setSessionId('');
    setTransformationId('');
    setLlmAvailable(false);
    setTransformedColumns([]);
    setTransformedCsv(null);
    setError('');
    setSqlContent('');
    setReadyToUpdate(false);
    setRowCount(0);
    setSelectedRowIds(new Set());
    setUndoStack([]);
    persistUndo('', []);
    setIsEditing(false);
    setShowColumnPicker(false);
    setPendingColumns([]);
    setSourceDeleteTarget(null);
    setSourceRestoreId(null);
    setNotice('');
    setConnectionInfo({ host: 'localhost', port: '3306', user: 'root', password: '' });
    setSelectedDb('');
    setSelectedTable('');
    setMongoUri('mongodb://localhost:27017/');
    setSelectedMongoDb('');
    setSelectedMongoCollection('');
    // Drop the stored file-session pointer so a refresh cannot resurrect it.
    localStorage.removeItem(SESSION_KEY);
    setActiveTab(tab);
    setIsDatabaseMode(tab === 'database');
    setIsMongoMode(tab === 'mongodb');
  };

  // If not authenticated, show landing page
  if (!isAuthenticated) {
    return <LandingPage />;
  }

  // Update the file upload UI section
  const renderFileUploadSection = () => (
    <div className="mb-8 animate-slide-in">
      <h2 className="section-title flex items-center text-xl font-semibold text-neutral-700 mb-3">
        <FileUp size={20} className="mr-2 text-primary" />
        Upload Data File
      </h2>
      <div 
        {...getRootProps()} 
        className={`bg-neutral-50/70 rounded-2xl border-2 border-dashed transition-all duration-300 ease-in-out cursor-pointer ${
          isDragActive ? 'border-primary bg-blue-50 scale-[1.02] shadow-lg ring-4 ring-primary/10' : isUploading ? 'border-neutral-400' : 'border-neutral-300 hover:border-primary/70 hover:bg-white hover:shadow-md'
        } ${file ? 'p-4' : 'p-8'}`}
      >
        <input {...getInputProps()} className="hidden" />
        <div className={`text-center ${file ? 'px-2' : ''}`}>
          {isUploading ? (
            <div className="flex flex-col items-center justify-center h-24">
              <div className="animate-spin rounded-full h-10 w-10 border-t-2 border-b-2 border-primary"></div>
              <p className="mt-3 text-sm text-neutral-600">Uploading...</p>
            </div>
          ) : file ? (
            <div className="flex items-center justify-between w-full h-24">
              <div className="flex items-center space-x-3 overflow-hidden mr-2">
                <FileUp size={28} className="text-primary flex-shrink-0" />
                <div>
                  <p className="text-md font-medium text-neutral-800 truncate" title={file.name}>
                    {file.name}
                  </p>
                  <p className="text-xs text-neutral-500">({(file.size / 1024).toFixed(1)} KB)</p>
                </div>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation(); // Prevent dropzone activation
                  handleRemoveFile();
                }}
                className="p-1.5 rounded-full text-neutral-500 hover:bg-red-100 hover:text-red-600 transition-colors duration-150 flex-shrink-0"
                title="Remove file"
              >
                <X size={18} />
              </button>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-24">
              <svg className="mx-auto h-12 w-12 text-primary/60 mb-2 animate-float" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <p className="mt-2 text-sm text-neutral-600">
                Drag & drop your file here, or <span className="font-medium text-primary hover:text-primary-dark cursor-pointer">browse</span>
              </p>
              <p className="mt-1 text-xs text-neutral-500">Supported: CSV, JSON, XLSX, SQL</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );

  return (
    <div className="relative min-h-screen bg-gradient-to-br from-neutral-50 to-blue-100 py-8 px-4 sm:px-6 lg:px-8 text-neutral-800">
      {/* Ambient background: slow drifting gradient orbs (decorative only) */}
      <div className="bg-orbs" aria-hidden="true">
        <div className="bg-orb bg-orb-1" />
        <div className="bg-orb bg-orb-2" />
        <div className="bg-orb bg-orb-3" />
      </div>

      <div className="relative z-10 max-w-6xl mx-auto"> {/* Increased max-width for a bit more space */}
        {/* Header */}
        <header className="flex justify-between items-center mb-10 animate-fade-in">
          <div className="text-left"> {/* Changed from text-center for alignment */}
            <h1 className="text-4xl font-extrabold text-neutral-900 tracking-tight">
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-primary via-accent to-secondary animate-shimmer">
                TabulaX
              </span>
            </h1>
            <p className="text-lg text-neutral-600 mt-1">
              Transform your data with ease and intelligence.
            </p>
          </div>
          {isAuthenticated && user && (
            <div className="flex items-center space-x-3">
              <div className="text-right">
                <p className="text-sm text-neutral-600">{user.username}</p>
                {/* <p className=\"text-xs text-neutral-500\">Active</p> */}
            </div>
            <button 
              onClick={logout}
                className="p-2 rounded-full hover:bg-neutral-200 text-neutral-600 hover:text-error transition-colors duration-150"
              title="Logout"
            >
                <LogOut className="h-5 w-5" />
            </button>
          </div>
          )}
        </header>

        {/* Main Content Container */}
        <main className="bg-white/90 backdrop-blur-xl rounded-2xl shadow-2xl overflow-hidden ring-1 ring-black/5">
          {/* Top accent line */}
          <div className="h-1 w-full bg-gradient-to-r from-primary via-accent to-secondary opacity-80" />
          {/* Error Alert - No changes here for now, but could be standardized component later */}
          {error && (
            <div className="bg-red-50 border-l-4 border-error p-4 animate-fade-in">
              <div className="flex items-start">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-red-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3 flex-1">
                  <p className="text-sm text-red-700">{error}</p>
                </div>
                <div className="ml-auto pl-3">
                  <div className="-mx-1.5 -my-1.5">
                    <button
                      onClick={dismissError}
                      className="inline-flex rounded-md p-1.5 text-red-500 hover:bg-red-100 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
                    >
                      <span className="sr-only">Dismiss</span>
                      <X size={16} />
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {notice && (
            <div className="bg-emerald-50 border-l-4 border-success p-4 animate-fade-in">
              <div className="flex items-start">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-success" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 0016 0zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3 flex-1">
                  <p className="text-sm text-emerald-800">{notice}</p>
                </div>
                <div className="ml-auto pl-3">
                  <div className="-mx-1.5 -my-1.5">
                    <button
                      onClick={dismissNotice}
                      className="inline-flex rounded-md p-1.5 text-success hover:bg-emerald-100 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-success"
                    >
                      <span className="sr-only">Dismiss</span>
                      <X size={16} />
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Tabs */}
          {isAuthenticated && (
            <div className="border-b border-neutral-200 bg-neutral-50/80 backdrop-blur rounded-t-2xl">
              <nav className="flex items-center gap-1.5 px-4 py-2.5">
              <button
                  onClick={() => handleSectionChange('file')}
                  className={`relative py-2 px-4 text-center rounded-full font-semibold text-sm transition-all duration-200 flex items-center space-x-2 ${
                  activeTab === 'file'
                    ? 'bg-white text-primary shadow-sm ring-1 ring-primary/20'
                      : 'text-neutral-500 hover:text-primary hover:bg-white/60'
                }`}
              >
                  <FileUp size={16} />
                  <span>Files</span>
              </button>
              <button
                  onClick={() => handleSectionChange('database')}
                  className={`relative py-2 px-4 text-center rounded-full font-semibold text-sm transition-all duration-200 flex items-center space-x-2 ${
                  activeTab === 'database'
                    ? 'bg-white text-primary shadow-sm ring-1 ring-primary/20'
                      : 'text-neutral-500 hover:text-primary hover:bg-white/60'
                }`}
              >
                  <Database size={16} />
                  <span>Database</span>
                </button>
                <button
                  onClick={() => handleSectionChange('mongodb')}
                  className={`relative py-2 px-4 text-center rounded-full font-semibold text-sm transition-all duration-200 flex items-center space-x-2 ${
                  activeTab === 'mongodb'
                    ? 'bg-white text-primary shadow-sm ring-1 ring-primary/20'
                      : 'text-neutral-500 hover:text-primary hover:bg-white/60'
                  }`}
                >
                  <Database size={16} /> {/* Replace with a MongoDB icon if available */}
                  <span>MongoDB</span>
              </button>
            </nav>
          </div>
          )}

          {/* key forces a fresh entrance animation whenever the section changes */}
          <div className="p-6 md:p-8 animate-fade-in" key={activeTab}>
            {!isAuthenticated ? (
                <LandingPage />
            ) : activeTab === 'database' ? (
              <DatabaseConnect 
                setColumns={setColumns}
                setPreviewData={setPreviewData}
                setUnmodifiedPreviewData={setUnmodifiedPreviewData}
                selectedColumn={selectedColumn}
                generatedCode={generatedCode}
                transformationId={transformationId}
                setSessionId={setSessionId}
                setTransformedCsv={setTransformedCsv}
                setIsDatabaseMode={setIsDatabaseMode}
                setConnectionInfo={setConnectionInfo}
                setSelectedDb={setSelectedDb}
                setSelectedTable={setSelectedTable}
                connectionInfo={connectionInfo}
                selectedDb={selectedDb}
                selectedTable={selectedTable}
              />
            ) : activeTab === 'mongodb' ? (
              <MongoConnect 
                setColumns={setColumns}
                setPreviewData={setPreviewData}
                setUnmodifiedPreviewData={setUnmodifiedPreviewData}
                setIsMongoMode={setIsMongoMode}
                mongoUri={mongoUri}
                setMongoUri={setMongoUri}
                setSelectedMongoDb={setSelectedMongoDb}
                setSelectedMongoCollection={setSelectedMongoCollection}
                selectedMongoDb={selectedMongoDb}
                selectedMongoCollection={selectedMongoCollection}
                selectedColumn={selectedColumn}
                transformationId={transformationId}
                setSessionId={setSessionId}
              />
            ) : activeTab === 'sql-view' ? (
              <div className="mb-8 animate-slide-in">
                <h2 className="section-title flex items-center">
                  <Code size={18} className="mr-2 text-primary" />
                  SQL Content
                </h2>
                <pre className="bg-gray-50 p-4 rounded-lg overflow-x-auto">
                  <code className="text-sm">{sqlContent}</code>
                </pre>
              </div>
            ) : (
              renderFileUploadSection()
            )}

            {/* Data quality profile + transformation preflight (read-only) */}
            {isAuthenticated && sessionId && (
              <DataQualityPanel
                sessionId={sessionId}
                columns={columns}
                examples={examples}
                selectedColumn={selectedColumn}
                transformationId={transformationId}
                refreshKey={datasetVersion}
              />
            )}

            {/* Transformation Section */}
            {columns.length > 0 && (
              <div className="mb-8 rounded-2xl border border-neutral-200 bg-white p-6 shadow-lg animate-slide-in stagger-1">
                <h2 className="section-title flex items-center text-xl font-semibold text-neutral-700 mb-5">
                  <Wand2 size={20} className="mr-2 text-accent" />
                  Define Transformation
                </h2>
                <LlmSettings apiKey={llmApiKey} onApiKeyChange={setLlmApiKey} />
                <div className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-start">
                    <div className="space-y-4 p-4 bg-neutral-50 rounded-lg border border-neutral-200">
                      <ColumnSelector 
                        columns={columns}
                        selectedColumn={selectedColumn}
                        setSelectedColumn={setSelectedColumn}
                      />
                      
                      <ExampleInput 
                        selectedColumn={selectedColumn}
                        examples={examples}
                        setExamples={setExamples}
                        setClassification={setClassification}
                        setGeneratedCode={setGeneratedCode}
                        setTransformationId={setTransformationId}
                        setLlmAvailable={setLlmAvailable}
                        llmApiKey={llmApiKey}
                      />
                    </div>
                    
                    <div className="sticky top-6">
                      {classification && (
                        <div className="p-4 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg border border-blue-200 shadow-md animate-fade-in">
                          <h3 className="text-md font-semibold text-neutral-700 mb-2 flex items-center">
                            <Code size={18} className="mr-2 text-primary" />
                            Detected Pattern
                          </h3>
                          <div className="p-3 bg-white rounded-md shadow-sm">
                            <div className="flex items-center">
                              <div className="w-2.5 h-2.5 bg-primary rounded-full mr-2.5 shrink-0"></div>
                              <p className="text-neutral-800 font-medium text-sm">{classification}</p>
                            </div>
                            {/* <p className=\"text-xs text-neutral-500 mt-1.5\">
                              The transformation function will be tailored for this pattern.
                            </p> */}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                  
                  <TransformResult 
                    selectedColumn={selectedColumn}
                    code={generatedCode}
                    transformationId={transformationId}
                    sessionId={sessionId}
                    llmAvailable={llmAvailable}
                    transformedPreview={previewData}
                    isDatabaseMode={isDatabaseMode || isMongoMode}
                    isMongoMode={isMongoMode}
                    connectionInfo={
                      isMongoMode 
                        ? { uri: mongoUri, database: selectedMongoDb, table: selectedMongoCollection }
                        : { ...connectionInfo, database: selectedDb, table: selectedTable }
                    }
                    selectedDb={isMongoMode ? selectedMongoDb : selectedDb}
                    selectedTable={isMongoMode ? selectedMongoCollection : selectedTable}
                    setReadyToUpdate={setReadyToUpdate}
                    originalPreview={unmodifiedPreviewData}
                    setTransformedColumns={setTransformedColumns}
                    onDatasetReset={applyEditResponse}
                    clearEditUndo={() => setUndoStack([])}
                    setRowCount={setRowCount}
                    setTransformedPreview={(data) => {
                      setPreviewData(data);
                      // The apply covered the FULL column — bump so panels
                      // (data quality) re-fetch the new dataset state.
                      setDatasetVersion((v) => v + 1);
                    }}
                  />
                </div>
              </div>
            )}

            {/* CSV Preview */}
            {(previewData.length > 0 || canEdit) && (
              <div className="animate-slide-in stagger-2 mt-10">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-xl font-semibold text-neutral-700 flex items-center m-0">
                    <Code size={20} className="mr-2 text-neutral-600" />
                    Data Preview
                  </h2>
                  
                  {transformedCsv && (
                    <div className="flex space-x-2">
                      <button
                        onClick={handleManualDownload}
                        className="btn bg-neutral-700 text-white hover:bg-neutral-800 focus:ring-neutral-500 text-sm py-2 px-3.5"
                      >
                        <Download size={16} className="mr-1.5" />
                          Download Results
                      </button>
                    </div>
                  )}
                </div>

                {/* Row/column editing toolbar (edits persist to the session) */}
                {canEdit && (
                  <div className="mb-4 p-3 bg-neutral-50 border border-neutral-200 rounded-lg flex flex-wrap items-center gap-3 text-sm">
                    <label className="flex items-center gap-2 text-neutral-600 select-none">
                      <input
                        type="checkbox"
                        checked={allVisibleSelected}
                        onChange={toggleSelectAll}
                        disabled={previewData.length === 0 || isEditing}
                        aria-label="Select all preview rows"
                      />
                      {selectedRowIds.size > 0
                        ? `${selectedRowIds.size} row(s) selected`
                        : 'Select rows'}
                    </label>
                    <button
                      onClick={handleDeleteRows}
                      disabled={selectedRowIds.size === 0 || isEditing}
                      className="btn bg-red-50 hover:bg-red-100 text-red-700 border border-red-200 text-sm py-1.5 px-3 disabled:opacity-50 flex items-center"
                    >
                      <Trash2 size={14} className="mr-1.5" />
                      Delete selected rows
                    </button>
                    <button
                      onClick={() => setShowColumnPicker((v) => !v)}
                      disabled={columns.length === 0 || isEditing}
                      className="btn bg-white hover:bg-neutral-100 text-neutral-700 border border-neutral-300 text-sm py-1.5 px-3 disabled:opacity-50"
                    >
                      Delete columns…
                    </button>
                    {canSourceDelete && (
                      <button
                        onClick={openSourceDeleteRows}
                        disabled={selectedRowIds.size === 0 || isEditing}
                        title="Permanently delete the selected rows from the connected database (not just the working copy)"
                        className="btn bg-red-600 hover:bg-red-700 text-white border border-red-700 text-sm py-1.5 px-3 disabled:opacity-50 flex items-center"
                      >
                        <Database size={14} className="mr-1.5" />
                        Delete from source DB…
                      </button>
                    )}
                    {undoStack.length > 0 && (
                      <button
                        onClick={handleUndo}
                        disabled={isEditing}
                        className="btn bg-white hover:bg-neutral-100 text-neutral-700 border border-neutral-300 text-sm py-1.5 px-3 disabled:opacity-50 flex items-center"
                        title={`Undo: ${undoStack[undoStack.length - 1].kind}`}
                      >
                        <Undo2 size={14} className="mr-1.5" />
                        Undo last deletion ({undoStack.length})
                      </button>
                    )}
                    <span className="text-xs text-neutral-400 italic">
                      {canSourceDelete
                        ? 'Edits apply to the working copy — source unchanged'
                        : 'Edits apply to the working copy — original file never modified'}
                    </span>
                    <span className="ml-auto text-xs text-neutral-500">
                      {rowCount > 0 ? `${rowCount} row(s) total` : 'Empty dataset'}
                    </span>
                  </div>
                )}

                {showColumnPicker && canEdit && (
                  <div className="mb-4 p-3 bg-white border border-neutral-200 rounded-lg">
                    <p className="text-xs font-semibold uppercase tracking-wider text-neutral-500 mb-2">
                      Choose columns to delete
                    </p>
                    <div className="flex flex-wrap gap-x-4 gap-y-1.5 mb-3">
                      {columns.map((col) => (
                        <label key={col} className="flex items-center gap-1.5 text-sm text-neutral-700 select-none">
                          <input
                            type="checkbox"
                            checked={pendingColumns.includes(col)}
                            onChange={() => toggleColumnPick(col)}
                            disabled={isEditing}
                          />
                          {col}
                        </label>
                      ))}
                    </div>
                    <div className="flex gap-2">
                      <button
                        onClick={handleDeleteColumns}
                        disabled={pendingColumns.length === 0 || isEditing}
                        className="btn bg-red-50 hover:bg-red-100 text-red-700 border border-red-200 text-sm py-1.5 px-3 disabled:opacity-50 flex items-center"
                      >
                        <Trash2 size={14} className="mr-1.5" />
                        Delete {pendingColumns.length || ''} column(s)
                      </button>
                      {canSourceDelete && (
                        <button
                          onClick={openSourceDeleteColumns}
                          disabled={pendingColumns.length === 0 || isEditing}
                          title={
                            isMongoMode
                              ? 'Permanently remove the chosen fields from every document in the collection'
                              : 'Permanently DROP the chosen columns from the live table (ALTER TABLE)'
                          }
                          className="btn bg-red-600 hover:bg-red-700 text-white border border-red-700 text-sm py-1.5 px-3 disabled:opacity-50 flex items-center"
                        >
                          <Database size={14} className="mr-1.5" />
                          Delete from source DB…
                        </button>
                      )}
                      <button
                        onClick={() => { setShowColumnPicker(false); setPendingColumns([]); }}
                        className="btn bg-white hover:bg-neutral-100 text-neutral-700 border border-neutral-300 text-sm py-1.5 px-3"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                )}
                
                {previewData.length > 0 ? (
                <div className="bg-white rounded-lg overflow-hidden shadow-xl border border-neutral-200">
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-neutral-200">
                      <thead className="bg-neutral-50">
                        <tr>
                          {canEdit && (
                            <th className="px-3 py-3 text-left bg-neutral-50">
                              <input
                                type="checkbox"
                                checked={allVisibleSelected}
                                onChange={toggleSelectAll}
                                disabled={isEditing}
                                aria-label="Select all preview rows"
                              />
                            </th>
                          )}
                          {previewData.length > 0 && dataKeys(previewData[0]).map((key) => {
                            const isSelectedBase = normalize(key) === normalize(selectedColumn);
                            const isTransformed = transformedColumns.some(col => normalize(col) === normalize(key));
                            // A column is the target of transformation if it's selected AND code has been generated for it
                            const isTransformationTarget = isSelectedBase && generatedCode;

                            let thClass = 'px-5 py-3 text-left text-xs font-semibold uppercase tracking-wider whitespace-nowrap ';

                            if (isTransformationTarget && isTransformed) {
                              // Column was selected, code generated, and transformation applied (now shows transformed data)
                              thClass += 'bg-gradient-to-r from-green-50 via-green-100 to-green-100 text-green-800';
                            } else if (isTransformationTarget) {
                              // Column is selected and code generated, but not yet applied (or reverted)
                              thClass += 'bg-gradient-to-r from-yellow-50 via-yellow-100 to-yellow-100 text-yellow-800';
                            } else if (isTransformed) {
                              // Other columns that might be affected by transformation (e.g. new columns)
                              thClass += 'bg-neutral-100 text-neutral-700'; 
                            } else {
                              thClass += 'text-neutral-600';
                            }

                            return (
                              <th key={key} className={thClass}>
                                {key}
                                {isTransformationTarget && isTransformed && (
                                  <span className="ml-1.5 px-1.5 py-0.5 bg-green-200 text-green-800 rounded-full text-[10px] font-medium">Transformed</span>
                                )}
                                {isTransformationTarget && !isTransformed && (
                                  <span className="ml-1.5 px-1.5 py-0.5 bg-yellow-200 text-yellow-800 rounded-full text-[10px] font-medium">Selected</span>
                                )}
                              </th>
                            );
                          })}
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-neutral-200">
                        {previewData.map((row, index) => {
                          const rowId = rowIdOf(row, index);
                          return (
                          <tr key={rowId} className="hover:bg-neutral-50 transition-colors duration-150">
                            {canEdit && (
                              <td className="px-3 py-3.5">
                                <input
                                  type="checkbox"
                                  checked={selectedRowIds.has(rowId)}
                                  onChange={() => toggleRowSelection(rowId)}
                                  disabled={isEditing}
                                  aria-label={`Select row ${index + 1}`}
                                />
                              </td>
                            )}
                            {Object.entries(row)
                              .filter(([key]) => key !== ROW_ID_KEY)
                              .map(([key, value]) => {
                              const isSelectedBase = normalize(key) === normalize(selectedColumn);
                              const isTransformed = transformedColumns.some(col => normalize(col) === normalize(key));
                              const isTransformationTarget = isSelectedBase && generatedCode; 

                              let tdClass = 'px-5 py-3.5 whitespace-nowrap text-sm ';

                              if (isTransformationTarget && isTransformed) {
                                tdClass += 'bg-green-50 text-green-800 font-medium';
                              } else if (isTransformationTarget) {
                                tdClass += 'bg-yellow-50 text-yellow-800';
                              } else if (isTransformed) {
                                tdClass += 'bg-neutral-50 text-neutral-800'; 
                              } else {
                                tdClass += 'text-neutral-700';
                              }
                              
                              return (
                                <td key={key} className={tdClass}>
                                {String(value)}
                              </td>
                              );
                            })}
                          </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  {previewData.length > 0 && (
                    <div className="py-2.5 px-4 bg-neutral-50 text-xs text-neutral-500 border-t border-neutral-200">
                      Showing {previewData.length}{rowCount > previewData.length ? ` of ${rowCount}` : ''} preview rows. {transformedColumns.length > 0 ? `Column(s) ${transformedColumns.join(', ')} affected by transformation.` : ''}
                  </div>
                  )}
                </div>
                ) : (
                  <div className="bg-white rounded-lg border border-dashed border-neutral-300 p-6 text-center text-sm text-neutral-500">
                    The dataset has no rows left.{' '}
                    {undoStack.length > 0 && (
                      <button
                        onClick={handleUndo}
                        disabled={isEditing}
                        className="btn bg-white hover:bg-neutral-100 text-neutral-700 border border-neutral-300 text-sm py-1.5 px-3 disabled:opacity-50 inline-flex items-center ml-2"
                      >
                        <Undo2 size={14} className="mr-1.5" />
                        Undo last deletion
                      </button>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Audit trail + working-copy version rollback (7-day window).
                History is scoped to the ACTIVE data-source section (Files /
                MySQL / MongoDB) via sourceType; it is shown before any upload
                too, so a section can always audit its own past operations. */}
            {isAuthenticated && (
              <HistoryPanel
                sessionId={sessionId}
                sourceType={
                  activeTab === 'database'
                    ? 'mysql'
                    : activeTab === 'mongodb'
                      ? 'mongodb'
                      : 'file' // 'file' and 'sql-view' (uploaded SQL) are Files
                }
                onOpened={handleOpenVersion}
                onRestored={(resp) => {
                  // A restore may isolate the snapshot into a NEW session
                  // (when the original session expired) — activateDataset
                  // adopts it, clears stale state, and re-syncs the preview
                  // from the server so the UI never shows abandoned data.
                  activateDataset(
                    resp,
                    typeof resp.message === 'string'
                      ? resp.message
                      : 'Working copy restored.'
                  );
                }}
                onRestoreSource={(backupId) => setSourceRestoreId(backupId)}
              />
            )}
          </div>
        </main>

        {/* Permanent source deletion: preview + typed DELETE confirmation */}
        {sourceDeleteTarget && sessionId && (
          <SourceDeleteDialog
            target={sourceDeleteTarget}
            sessionId={sessionId}
            rowIds={Array.from(selectedRowIds)}
            fields={pendingColumns}
            mysqlConnection={{
              host: connectionInfo.host,
              port: connectionInfo.port,
              user: connectionInfo.user,
              password: connectionInfo.password,
              database: selectedDb,
            }}
            mongoUri={mongoUri}
            onClose={() => setSourceDeleteTarget(null)}
            onDone={applySourceDeleteResponse}
          />
        )}

        {/* Restore a previous source deletion from a verified backup */}
        {sourceRestoreId && (
          <SourceRestoreDialog
            backupId={sourceRestoreId}
            mysqlConnection={{
              host: connectionInfo.host,
              port: connectionInfo.port,
              user: connectionInfo.user,
              password: connectionInfo.password,
              database: selectedDb,
            }}
            mongoUri={mongoUri}
            sessionId={sessionId}
            onClose={() => setSourceRestoreId(null)}
            onDone={(resp) => {
              setSourceRestoreId(null);
              const warnings = Array.isArray(resp.warnings)
                ? (resp.warnings as string[]).filter(Boolean)
                : [];
              const message = String(resp.message || 'Source restore completed.');
              const text = [message, ...warnings].join(' ');
              const hasSession =
                typeof resp.session_id === 'string' && Boolean(resp.session_id);
              if (hasSession) {
                // The server restored, verified, and returned a REFRESHED
                // session (live data) — adopt it so the preview updates
                // without a manual reconnect.
                activateDataset(resp, text);
              } else {
                // No refreshed session came back (empty source, refresh
                // failure, or an older backend) — say so instead of leaving
                // stale data on screen as if nothing happened.
                setError('');
                setNotice(
                  [
                    text,
                    'The view was not refreshed automatically — reconnect the source to load the restored data.',
                  ].join(' ')
                );
              }
            }}
          />
        )}

        {/* Footer */}
        <div className="mt-8 text-center text-sm text-gray-500">
          <p>CSV Transformation Studio — Transform your data with ease</p>
        </div>
      </div>
    </div>
  );
}

export default App;