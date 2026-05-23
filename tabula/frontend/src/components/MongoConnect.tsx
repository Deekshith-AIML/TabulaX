import React, { useState } from 'react';
import { AlertCircle, Database } from 'lucide-react';
import { apiUrl, handleSessionExpired, SESSION_EXPIRED_MESSAGE } from '../lib/api';

interface MongoConnectionInfoFromApp {
  uri: string;
}

interface MongoConnectProps {
  setColumns: (columns: string[]) => void;
  setPreviewData: (data: any[]) => void;
  setUnmodifiedPreviewData: (data: any[]) => void;
  setIsMongoMode: (mode: boolean) => void;
  mongoUri: string; // From App.tsx state
  setMongoUri: (uri: string) => void; // To update App.tsx state
  setSelectedMongoDb: (db: string) => void;
  setSelectedMongoCollection: (collection: string) => void;
  selectedMongoDb: string;
  selectedMongoCollection: string;
  selectedColumn: string;
  transformationId: string;
  setSessionId: (id: string) => void;
}

const MongoConnect: React.FC<MongoConnectProps> = ({
  setColumns,
  setPreviewData,
  setUnmodifiedPreviewData,
  setIsMongoMode,
  mongoUri,
  setMongoUri,
  setSelectedMongoDb,
  setSelectedMongoCollection,
  selectedMongoDb,
  selectedMongoCollection,
  selectedColumn,
  transformationId,
  setSessionId,
}) => {
  const [databases, setDatabases] = useState<string[]>([]);
  const [collections, setCollections] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isUpdating, setIsUpdating] = useState(false);

  const handleConnect = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');
      const res = await fetch(apiUrl('/connect_mongo/'), {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ uri: mongoUri }),
      });
      if (!res.ok) {
        if (res.status === 401) {
          handleSessionExpired();
          throw new Error(SESSION_EXPIRED_MESSAGE);
        }
        const errorData = await res.json().catch(() => ({ detail: 'Connection failed' }));
        throw new Error(errorData.detail || 'Connection failed');
      }
      const data = await res.json();
      setDatabases(data.databases);
      setConnected(true);
      setIsMongoMode(true);
      setError(null);
    } catch (err) {
      setError('Connection failed: ' + (err as Error).message);
      setConnected(false);
    } finally {
      setLoading(false);
    }
  };

  const fetchCollections = async (dbName: string) => {
    setSelectedMongoDb(dbName);
    setLoading(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');
      const collectionsRes = await fetch(`${apiUrl('/get_mongo_collections/')}?database_name=${encodeURIComponent(dbName)}`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ uri: mongoUri }), 
      });

      if (!collectionsRes.ok) {
        if (collectionsRes.status === 401) {
          handleSessionExpired();
          throw new Error(SESSION_EXPIRED_MESSAGE);
        }
        const errorData = await collectionsRes.json().catch(() => ({ detail: 'Failed to fetch collections' }));
        throw new Error(errorData.detail || 'Failed to fetch collections');
      }
      const data = await collectionsRes.json();
      setCollections(data.collections);
      setError(null);
    } catch (err) {
      setError('Error fetching collections: ' + (err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const fetchData = async (collectionName: string) => {
    setSelectedMongoCollection(collectionName);
    setLoading(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');
      const dataRes = await fetch(`${apiUrl('/get_mongo_data/')}?database_name=${encodeURIComponent(selectedMongoDb)}&collection_name=${encodeURIComponent(collectionName)}`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}` 
        },
        body: JSON.stringify({ uri: mongoUri }),
      });

      if (!dataRes.ok) {
        if (dataRes.status === 401) {
          handleSessionExpired();
          throw new Error(SESSION_EXPIRED_MESSAGE);
        }
        const errorData = await dataRes.json().catch(() => ({ detail: 'Failed to fetch data' }));
        throw new Error(errorData.detail || 'Failed to fetch data');
      }
      const data = await dataRes.json();
      
      // The backend stored this collection snapshot in a server-side session;
      // capture its handle so /apply/ can read it back without sending code.
      setSessionId(data.session_id || '');
      
      // Filter out _id from columns
      const filteredColumns = data.columns.filter((col: string) => col !== '_id');
      
      // Filter out _id from preview data rows
      const filteredPreview = data.preview.map((row: any) => {
        const { _id, ...rest } = row;
        return rest;
      });

      setColumns(filteredColumns);
      setPreviewData(filteredPreview);
      setUnmodifiedPreviewData(filteredPreview); // Also use filtered preview for unmodified state
      setError(null);
    } catch (err) {
      setError('Error fetching data: ' + (err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const handleUpdateMongoDatabase = async () => {
    if (!selectedMongoDb || !selectedMongoCollection || !selectedColumn || !transformationId) {
      setError('Please select a database, collection, column, and generate a transformation first.');
      return;
    }
    setIsUpdating(true);
    setError(null);

    // JSON body matching ApplyToMongoPayload. References the stored
    // transformation_id (never raw code) and requires confirm=true to write.
    const payload = {
      conn_info_json: JSON.stringify({ uri: mongoUri }),
      database_name: selectedMongoDb,
      collection_name: selectedMongoCollection,
      column_to_transform: selectedColumn,
      transformation_id: transformationId,
      confirm: true,
    };

    try {
      const token = localStorage.getItem('token');
      const res = await fetch(apiUrl('/apply_to_mongodb/'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(payload),
      });

      const data = await res.json().catch(() => ({ detail: 'MongoDB update failed' }));
      if (!res.ok) {
        if (res.status === 401) {
          handleSessionExpired();
          throw new Error(SESSION_EXPIRED_MESSAGE);
        }
        throw new Error(data.detail || 'MongoDB update failed');
      }
      alert(data.message || 'MongoDB updated successfully!');
      fetchData(selectedMongoCollection);
    } catch (err) {
      setError('Update failed: ' + (err as Error).message);
    } finally {
      setIsUpdating(false);
    }
  };

  return (
    <div className="glass-card p-6 mb-8 animate-fade-in">
      <h2 className="section-title flex items-center">
        <Database size={18} className="mr-2 text-primary" />
        MongoDB Connection
      </h2>
      {error && (
        <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-3 mb-4 rounded-md flex items-center" role="alert">
          <AlertCircle size={20} className="mr-2" />
          <p className="text-sm">{error}</p>
        </div>
      )}
      {!connected ? (
        <form onSubmit={handleConnect} className="space-y-4">
          <div>
            <label htmlFor="mongoUri" className="block text-gray-700 text-sm font-medium mb-1">MongoDB Connection URI</label>
            <input
              id="mongoUri"
              type="text"
              placeholder="mongodb://user:password@host:port/database"
              value={mongoUri}
              onChange={(e) => setMongoUri(e.target.value)}
              className="input"
              required
            />
             <p className="text-xs text-gray-500 mt-1">Example: mongodb://localhost:27017/mydatabase (database name in URI is optional here)</p>
          </div>
          <button type="submit" disabled={loading} className="btn btn-primary w-full md:w-auto">
            {loading ? 'Connecting...' : 'Connect to MongoDB'}
          </button>
        </form>
      ) : (
        <div className="space-y-6 mt-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-white rounded-lg shadow p-4">
              <h3 className="text-md font-medium text-gray-700 mb-3">Databases</h3>
              <div className="max-h-60 overflow-y-auto border rounded-lg">
                {databases.map(db => (
                  <div
                    key={db}
                    onClick={() => fetchCollections(db)}
                    className={`p-2 hover:bg-blue-50 cursor-pointer transition-colors duration-150 ${
                      selectedMongoDb === db ? 'bg-blue-100 border-l-4 border-primary' : ''
                    }`}
                  >
                    {db}
                  </div>
                ))}
              </div>
            </div>
            {selectedMongoDb && (
              <div className="bg-white rounded-lg shadow p-4">
                <h3 className="text-md font-medium text-gray-700 mb-3">Collections in {selectedMongoDb}</h3>
                <div className="max-h-60 overflow-y-auto border rounded-lg">
                  {collections.map(col => (
                    <div
                      key={col}
                      onClick={() => fetchData(col)}
                      className={`p-2 hover:bg-blue-50 cursor-pointer transition-colors duration-150 ${
                        selectedMongoCollection === col ? 'bg-blue-100 border-l-4 border-secondary' : ''
                      }`}
                    >
                      {col}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {selectedMongoCollection && transformationId && selectedColumn && (
            <button
              onClick={handleUpdateMongoDatabase}
              className="btn btn-success mt-4 w-full md:w-auto"
              disabled={isUpdating}
            >
              {isUpdating ? 'Updating MongoDB...' : 'Update MongoDB Collection'}
            </button>
          )}
        </div>
      )}
    </div>
  );
};

export default MongoConnect; 