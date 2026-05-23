import React from "react";
import { ChevronDown } from 'lucide-react';

interface ColumnSelectorProps {
  columns: string[];
  selectedColumn: string;
  setSelectedColumn: (column: string) => void;
}

const ColumnSelector: React.FC<ColumnSelectorProps> = ({ 
  columns, 
  selectedColumn, 
  setSelectedColumn 
}) => {
  if (!columns || columns.length === 0) {
    return null; // Or a placeholder if columns are loading/not available
  }

  return (
    <div className="mb-4">
      <label htmlFor="column-selector" className="block text-sm font-medium text-neutral-700 mb-1.5">
        Select Column to Transform
      </label>
      <div className="relative">
        <select 
          id="column-selector"
          value={selectedColumn}
          onChange={(e) => setSelectedColumn(e.target.value)}
          className="select appearance-none w-full px-4 py-2.5 border border-neutral-300 rounded-md shadow-sm bg-white focus:ring-2 focus:ring-primary/50 focus:border-primary transition-colors duration-150 text-neutral-800"
        >
          <option value="" disabled>-- Select a column --</option>
          {columns.map((col) => (
            <option key={col} value={col}>{col}</option>
          ))}
        </select>
        <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-neutral-500">
          <ChevronDown size={18} />
        </div>
      </div>
    </div>
  );
};

export default ColumnSelector;