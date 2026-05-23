import React, { useState, useEffect } from 'react';
import { Database, AlertTriangle, CheckCircle, Loader2, UploadCloud, RotateCcw, Link, X, Settings2, Eye } from 'lucide-react';
import { apiUrl, handleSessionExpired, SESSION_EXPIRED_MESSAGE } from '../lib/api';

// Re-defined here for clarity, or import from a shared types file if available
interface ConnectionInfo {
  host?: string;
  port?: string;
  user?: string;
  password?: string;
  database?: string;
  table?: string;
  uri?: string;
}

interface DatabaseConnectProps {
  setColumns: (columns: string[]) => void;
  setPreviewData: (data: any[]) => void;
  setUnmodifiedPreviewData: (data: any[]) => void;
  selectedColumn: string;
  generatedCode: string;
  transformationId: string;
  setSessionId: (id: string) => void;
  setTransformedCsv: (csv: string | null) => void;
  setIsDatabaseMode: (isDbMode: boolean) => void;
  setConnectionInfo: (info: ConnectionInfo) => void;
  setSelectedDb: (db: string) => void;
  setSelectedTable: (table: string) => void;
  connectionInfo: ConnectionInfo;
  selectedDb: string;
  selectedTable: string;
}

const DatabaseConnect: React.FC<DatabaseConnectProps> = ({
  setColumns,
  setPreviewData,
  setUnmodifiedPreviewData,
  selectedColumn,
  generatedCode,
  transformationId,
  setSessionId,
  setTransformedCsv,
  setIsDatabaseMode,
  setConnectionInfo,
  setSelectedDb,
  setSelectedTable,
  connectionInfo: currentConnectionInfo, // Renamed to avoid conflict with local state
  selectedDb: currentSelectedDb,
  selectedTable: currentSelectedTable,
}) => {
  const [localConnectionInfo, setLocalConnectionInfo] = useState<ConnectionInfo>(currentConnectionInfo);
  const [databases, setDatabases] = useState<string[]>([]);
  const [tables, setTables] = useState<string[]>([]);
  const [localSelectedDb, setLocalSelectedDb] = useState(currentSelectedDb);
  const [localSelectedTable, setLocalSelectedTable] = useState(currentSelectedTable);
  const [isLoading, setIsLoading] = useState(false);
  const [isConnected, setIsConnected] = useState(false); 
  const [isDataFetched, setIsDataFetched] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isUpdating, setIsUpdating] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Sync with parent state if it changes
  useEffect(() => setLocalConnectionInfo(currentConnectionInfo), [currentConnectionInfo]);
  useEffect(() => setLocalSelectedDb(currentSelectedDb), [currentSelectedDb]);
  useEffect(() => setLocalSelectedTable(currentSelectedTable), [currentSelectedTable]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setLocalConnectionInfo(prev => ({ ...prev, [name]: value }));
  };

  const handleConnect = async () => {
    setIsLoading(true);
    setError(null);
    setIsConnected(false);
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(apiUrl('/connect_database/'), {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(localConnectionInfo),
      });
      if (!response.ok) {
        if (response.status === 401) {
          handleSessionExpired();
          throw new Error(SESSION_EXPIRED_MESSAGE);
        }
        const errData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errData.detail || 'Failed to connect to database');
      }
      const data = await response.json();
      setDatabases(data.databases || []);
      setIsConnected(true);
      setConnectionInfo(localConnectionInfo); // Update parent state
      setError(null); // Clear previous errors
      setTables([]); // Reset tables when (re)connecting
      setLocalSelectedTable('');
      setColumns([]);
      setPreviewData([]);
      setUnmodifiedPreviewData([]);
    } catch (err) {
      console.error("Connection error:", err);
      setError((err as Error).message);
      setIsConnected(false);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchTables = async (databaseName: string) => {
    if (!databaseName) return;
    setIsLoading(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');
      
      // Prepare the connection info WITH the selected database name
      const connectionInfoForTables = {
        ...localConnectionInfo, // Spread existing host, user, pass, port
        database: databaseName, // Add/override the database field
      };

      // The backend endpoint /get_tables/ expects connection info in the body
      // and no longer uses query parameters for db_name.
      const response = await fetch(
        apiUrl('/get_tables/'),
        {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
          body: JSON.stringify(connectionInfoForTables), // Send connection info with database name
        }
      );
      if (!response.ok) {
        if (response.status === 401) {
          handleSessionExpired();
          throw new Error(SESSION_EXPIRED_MESSAGE);
        }
        const errData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errData.detail || 'Failed to fetch tables');
      }
      const data = await response.json();
      setTables(data.tables || []);
      setSelectedDb(databaseName); // Update parent state
      // Also update localConnectionInfo to include the database, 
      // so subsequent calls like fetchColumnsAndPreview have it.
      setLocalConnectionInfo(connectionInfoForTables);
      setConnectionInfo(connectionInfoForTables); // And parent state

      setLocalSelectedTable(''); // Reset selected table when DB changes
      setColumns([]);
      setPreviewData([]);
      setUnmodifiedPreviewData([]);
    } catch (err) {
      console.error("Fetch tables error:", err);
      setError((err as Error).message);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchColumnsAndPreview = async (tableName: string) => {
    if (!localSelectedDb || !tableName) return;
    setIsLoading(true);
    setError(null);
    setIsDataFetched(false);
    try {
      const token = localStorage.getItem('token');

      // localConnectionInfo should already have the database name set by fetchTables
      // Ensure localConnectionInfo is up-to-date if relying on it directly from state that might not have refreshed.
      // The `localConnectionInfo` state *should* be updated by the corrected `fetchTables`.
      // If `localSelectedDb` is the source of truth for the DB name for this call:
      const connectionInfoForColumns = {
        ...localConnectionInfo,
        database: localSelectedDb, // Explicitly ensure the selected DB is part of the body
      };

      // Backend expects table_name as a query parameter and connection info (with DB) in body.
      const response = await fetch(
        `${apiUrl('/get_columns/')}?table_name=${encodeURIComponent(tableName)}`, // table_name as query param
        {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
          body: JSON.stringify(connectionInfoForColumns), // Send connection info (host, user, pass, port, database) in body
        }
      );
      if (!response.ok) {
        if (response.status === 401) {
          handleSessionExpired();
          throw new Error(SESSION_EXPIRED_MESSAGE);
        }
        const errData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errData.detail || 'Failed to fetch data');
      }
      const data = await response.json();
      setColumns(data.columns || []);
      setPreviewData(data.preview || []);
      setUnmodifiedPreviewData(data.preview || []);
      // The backend snapshot this table into a server-side session (same as
      // MongoDB); keep the handle so row/column edits, undo and apply work.
      setSessionId(data.session_id || '');
      setSelectedTable(tableName); // Update parent state
      setIsDataFetched(true);
      setIsDatabaseMode(true); // Ensure App.tsx knows we are in DB mode
    } catch (err) {
      console.error("Fetch data error:", err);
      setError((err as Error).message);
      setIsDataFetched(false);
    } finally {
      setIsLoading(false);
    }
  };

  const handleUpdateDatabase = async () => {
    if (!selectedColumn || !transformationId || !localSelectedDb || !localSelectedTable) {
      setError("Missing information for database update. Ensure column, transformation, database, and table are set.");
      return;
    }
    setIsUpdating(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');

      // Prepare the payload according to ApplyToSQLPayload model.
      // The client references the stored transformation_id — never raw code —
      // and must set confirm=true to actually modify the table.
      const currentDbName = localSelectedDb?.trim(); // Trim whitespace and handle potential null/undefined

      if (!currentDbName) { // Check if trimmed database name is empty, null, or undefined
        throw new Error("No database selected or the selected database name is invalid.");
      }

      const payload = {
        connection_details: {
          host: localConnectionInfo.host,
          user: localConnectionInfo.user,
          password: localConnectionInfo.password,
          port: localConnectionInfo.port ? parseInt(localConnectionInfo.port, 10) : 3306,
          database: currentDbName, // Use the trimmed database name
        },
        table_name: localSelectedTable,
        column_to_transform: selectedColumn,
        transformation_id: transformationId,
        confirm: true,
      };

      const response = await fetch(apiUrl('/apply_to_database/'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        if (response.status === 401) {
          handleSessionExpired();
          throw new Error(SESSION_EXPIRED_MESSAGE);
        }
        const errData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errData.detail || 'Failed to update database');
      }
      // Re-fetch preview data to show updated table
      await fetchColumnsAndPreview(localSelectedTable);
      alert("Database updated successfully!"); // Consider a more subtle notification
    } catch (err) {
      console.error("Update database error:", err);
      setError((err as Error).message);
    } finally {
      setIsUpdating(false);
    }
  };

  const inputField = (name: keyof ConnectionInfo, label: string, type: string = "text", placeholder?: string) => (
    <div>
      <label htmlFor={name} className="block text-sm font-medium text-neutral-600 mb-1">
        {label}
      </label>
      <input
        type={type}
        id={name}
        name={name}
        value={(localConnectionInfo as any)[name] || ''}
        onChange={handleInputChange}
        placeholder={placeholder}
        className="input w-full px-3.5 py-2 border border-neutral-300 rounded-md shadow-sm focus:ring-1 focus:ring-primary focus:border-primary text-sm"
        disabled={isLoading || isConnected}
      />
    </div>
  );

  
  return (
    <div className="space-y-6 animate-slide-in">
      <h2 className="text-xl font-semibold text-neutral-700 flex items-center">
        <Database size={20} className="mr-2 text-primary" />
        Connect to MySQL Database
      </h2>
      
      {error && (
        <div className="bg-red-50 border-l-4 border-red-500 text-red-700 p-3.5 rounded-md flex items-start text-sm" role="alert">
          <AlertTriangle size={18} className="mr-2.5 shrink-0 mt-0.5" />
          <span>{error}</span>
          <button onClick={() => setError(null)} className="ml-auto text-red-600 hover:text-red-800 p-1 -m-1 rounded-full"><X size={16}/></button>
        </div>
      )}
      
      {!isConnected ? (
        <div className="p-5 bg-neutral-50 rounded-lg border border-neutral-200 space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {inputField("host", "Host", "text", "e.g., localhost or 127.0.0.1")}
                {inputField("user", "User", "text", "e.g., root")}
                {inputField("password", "Password", "password", "Database password")}
                {inputField("port", "Port", "text", "e.g., 3306")}
            </div>
            <button
                onClick={handleConnect}
                disabled={isLoading}
                className="btn btn-primary w-full flex items-center justify-center text-sm py-2.5"
            >
                {isLoading ? <Loader2 size={18} className="animate-spin mr-2" /> : <Link size={16} className="mr-2" />}
                {isLoading ? 'Connecting...' : 'Connect to Server'}
            </button>
            </div>
      ) : (
        <div className="p-3 bg-green-50 border border-green-300 rounded-md text-green-700 text-sm flex items-center justify-between">
          <div className="flex items-center">
            <CheckCircle size={18} className="mr-2.5 shrink-0" />
            Connected to {localConnectionInfo.host} (User: {localConnectionInfo.user})
          </div>
          <button
            onClick={() => { setIsConnected(false); setDatabases([]); setTables([]); setColumns([]); setPreviewData([]); setUnmodifiedPreviewData([]); setLocalSelectedDb(''); setLocalSelectedTable(''); setError(null); setIsDataFetched(false); }}
            className="btn bg-red-100 text-red-600 hover:bg-red-200 text-xs py-1 px-2.5"
          >
            <X size={14} className="mr-1"/> Disconnect
          </button>
        </div>
      )}

      {isConnected && databases.length > 0 && (
        <div className="space-y-3">
          <div>
            <label htmlFor="db-select" className="block text-sm font-medium text-neutral-700 mb-1">
              Select Database
            </label>
            <select 
              id="db-select"
              value={localSelectedDb}
              onChange={(e) => { setLocalSelectedDb(e.target.value); fetchTables(e.target.value); }}
              className="select w-full px-3.5 py-2.5 border border-neutral-300 rounded-md shadow-sm focus:ring-1 focus:ring-primary focus:border-primary text-sm"
              disabled={isLoading}
            >
              <option value="" disabled>-- Choose a database --</option>
              {databases.map(db => <option key={db} value={db}>{db}</option>)}
            </select>
            </div>
            
          {tables.length > 0 && localSelectedDb && (
            <div>
              <label htmlFor="table-select" className="block text-sm font-medium text-neutral-700 mb-1">
                Select Table
              </label>
              <select 
                id="table-select"
                value={localSelectedTable}
                onChange={(e) => { setLocalSelectedTable(e.target.value); fetchColumnsAndPreview(e.target.value); }}
                className="select w-full px-3.5 py-2.5 border border-neutral-300 rounded-md shadow-sm focus:ring-1 focus:ring-primary focus:border-primary text-sm"
                disabled={isLoading}
              >
                <option value="" disabled>-- Choose a table --</option>
                {tables.map(table => <option key={table} value={table}>{table}</option>)}
              </select>
              </div>
            )}
          </div>
      )}
      
      {isDataFetched && localSelectedTable && (
         <div className="p-3 bg-blue-50 border border-blue-300 rounded-md text-blue-700 text-sm flex items-center">
            <Eye size={18} className="mr-2.5 shrink-0" />
            Previewing data from '{localSelectedTable}'. You can now define transformations.
        </div>
      )}

      {isDataFetched && selectedColumn && generatedCode && localSelectedTable && (
        <div className="mt-6 pt-5 border-t border-neutral-200 space-y-3">
          <h3 className="text-md font-semibold text-neutral-700 flex items-center">
            <Settings2 size={18} className="mr-2 text-accent"/>
            Database Actions
          </h3>
          <p className="text-sm text-neutral-600">
            Apply the generated transformation directly to the '{localSelectedTable}' table in the '{localSelectedDb}' database.
            This action will modify the data in the selected column.
          </p>
            <button
              onClick={handleUpdateDatabase}
            disabled={isUpdating || isLoading}
            className="btn btn-success w-full flex items-center justify-center text-sm py-2.5"
          >
            {isUpdating ? <Loader2 size={18} className="animate-spin mr-2" /> : <UploadCloud size={16} className="mr-2" />}
            {isUpdating ? 'Updating Database...' : `Update '${selectedColumn}' in '${localSelectedTable}'`}
            </button>
        </div>
      )}
    </div>
  );
};

export default DatabaseConnect;