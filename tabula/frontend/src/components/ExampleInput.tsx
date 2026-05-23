import React, { useState, ChangeEvent } from "react";
import { AlertCircle, Lightbulb, Wand2, Zap, Sparkles } from 'lucide-react';
import { apiUrl, handleSessionExpired, SESSION_EXPIRED_MESSAGE } from '../lib/api';
import { parseExamplesText, ExamplePair } from '../lib/examples';

interface ExampleInputProps {
  selectedColumn: string;
  examples: string;
  setExamples: (examples: string) => void;
  setClassification: (classification: string) => void;
  setGeneratedCode: (code: string) => void;
  setTransformationId: (id: string) => void;
  setLlmAvailable: (available: boolean) => void;
  /** Transient personal key for this session only (never persisted). */
  llmApiKey?: string;
}

const ExampleInput: React.FC<ExampleInputProps> = ({ 
  selectedColumn, 
  examples, 
  setExamples, 
  setClassification, 
  setGeneratedCode,
  setTransformationId,
  setLlmAvailable,
  llmApiKey,
}) => {
  const [isClassifying, setIsClassifying] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentClassification, setCurrentClassification] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [llmActive, setLlmActive] = useState<boolean | null>(null);
  // Task/labels decided by /classify/, reused by /generate_function/ so the
  // pair costs at most one model call.
  const [classifyTask, setClassifyTask] = useState<string | null>(null);
  const [classifyLabels, setClassifyLabels] = useState<string[]>([]);
  // What the planner decided (strategy, description, ambiguities).
  const [genNotice, setGenNotice] = useState<{ kind: 'info' | 'warning'; text: string } | null>(null);

  // Sent only with this request; the backend never stores it.
  const llmConfigPayload = llmApiKey?.trim()
    ? { llm_config: { provider: 'openai', api_key: llmApiKey.trim() } }
    : {};

  // Validate the textarea locally so malformed lines surface immediately with
  // line numbers; the same rules run again on the server.
  const validateExamples = (): ExamplePair[] | null => {
    if (!selectedColumn || !examples.trim()) {
      setError("Please select a column and provide examples.");
      return null;
    }
    const { pairs, errors } = parseExamplesText(examples);
    if (errors.length > 0) {
      setError(errors.join(" "));
      return null;
    }
    return pairs;
  };

  const handleClassify = async () => {
    const pairs = validateExamples();
    if (!pairs) return;
    setIsClassifying(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');
      const classifyResponse = await fetch(apiUrl('/classify/'), {
        method: "POST",
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ column: selectedColumn, examples: pairs, ...llmConfigPayload }),
      });

      if (!classifyResponse.ok) {
        if (classifyResponse.status === 401) {
          handleSessionExpired();
          throw new Error(SESSION_EXPIRED_MESSAGE);
        }
        const errorData = await classifyResponse.json().catch(() => ({ detail: classifyResponse.statusText }));
        throw new Error(errorData.detail || `Classification failed: ${classifyResponse.statusText}`);
      }

      const classifyData = await classifyResponse.json();
      setClassification(classifyData.classification);
      setCurrentClassification(classifyData.classification);
      setClassifyTask(classifyData.task ?? null);
      setClassifyLabels(Array.isArray(classifyData.labels) ? classifyData.labels : []);
      setGenNotice(null);
      // A configured model was consulted but could not answer (rate limit /
      // quota / network).  Say WHAT failed now, so the user isn't surprised
      // by a generic error at Generate time. Deterministic rules are unaffected.
      if (typeof classifyData.llm_error === 'string' && classifyData.llm_error) {
        setGenNotice({
          kind: 'warning',
          text:
            'The AI model could not be reached: ' + classifyData.llm_error +
            ' Deterministic transformations (case, numeric, prefix, templates) ' +
            'still work — retry in a moment for model-backed tasks, or add a ' +
            'personal key in AI settings.',
        });
      }
    } catch (err) {
      console.error("Classification error:", err);
      setError("Error classifying examples: " + (err as Error).message);
    } finally {
      setIsClassifying(false);
    }
  };

  const handleGenerate = async () => {
    const pairs = validateExamples();
    if (!pairs) return;
    if (!currentClassification) {
      setError("Please classify examples first and ensure all fields are filled.");
      return;
    }
    setIsGenerating(true);
    setError(null);
    setGenNotice(null);
    try {
      const token = localStorage.getItem('token');
      const payload: Record<string, unknown> = {
        column: selectedColumn,
        examples: pairs,
        classification: currentClassification,
        ...llmConfigPayload,
      };
      // Reuse the /classify/ decision so generate doesn't pay for a second
      // model call (older servers ignore these extra fields).
      if (classifyTask) payload.task = classifyTask;
      if (classifyLabels.length > 0) payload.labels = classifyLabels;
      const codeResponse = await fetch(apiUrl('/generate_function/'), {
        method: "POST",
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!codeResponse.ok) {
        if (codeResponse.status === 401) {
          handleSessionExpired();
          throw new Error(SESSION_EXPIRED_MESSAGE);
        }
        const errorData = await codeResponse.json().catch(() => ({ detail: codeResponse.statusText }));
        throw new Error(errorData.detail || `Code generation failed: ${codeResponse.statusText}`);
      }

      const codeData = await codeResponse.json();
      const modelBacked = codeData.strategy === 'model';
      if (modelBacked) {
        // No reusable Python function: rows are processed by the model at
        // apply time.  Keep a visible placeholder so the rest of the UI
        // (target-column highlighting, result panel) works unchanged.
        setGeneratedCode(
          codeData.code ||
          '# Model-backed transformation: rows are processed by the AI model\n' +
          '# when you apply. Unseen inputs are handled by meaning, not lookup.'
        );
      } else {
        setGeneratedCode(codeData.code || "");
      }
      setTransformationId(codeData.transformation_id || "");
      const ambiguities: string[] = Array.isArray(codeData.ambiguities)
        ? codeData.ambiguities.filter((a: unknown) => typeof a === 'string' && a)
        : [];
      if (ambiguities.length > 0) {
        setGenNotice({ kind: 'warning', text: ambiguities.join(' ') });
      } else if (modelBacked) {
        // Lead with the inferred transformation family (spec §15) so the user
        // knows which kind of rule was detected before they apply it.
        const family = typeof codeData.family === 'string' ? codeData.family : '';
        setGenNotice({
          kind: 'info',
          text: (family ? family + ': ' : '') + (codeData.description ||
            'This transformation needs the AI model: it will run over every row when you apply.'),
        });
      } else if (codeData.description) {
        setGenNotice({ kind: 'info', text: codeData.description });
      }
      const available = Boolean(codeData.llm_available);
      setLlmActive(available);
      setLlmAvailable(available);
    } catch (err) {
      console.error("Generation error:", err);
      setError("Error generating function: " + (err as Error).message);
    } finally {
      setIsGenerating(false);
    }
  };

  if (!selectedColumn) {
    return (
      <div className="p-3.5 bg-yellow-50 border border-yellow-300 rounded-md text-yellow-800 text-sm flex items-center">
        <Lightbulb size={18} className="inline mr-2.5 shrink-0" /> 
        Please select a column first to provide examples.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {error && (
        <div className="bg-red-50 border-l-4 border-red-500 text-red-700 p-3.5 rounded-md flex items-start text-sm" role="alert">
          <AlertCircle size={18} className="mr-2.5 shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}
      {llmActive === false && (
        <div className="bg-amber-50 border-l-4 border-amber-400 text-amber-800 p-3.5 rounded-md flex items-start text-sm" role="status">
          <Sparkles size={18} className="mr-2.5 shrink-0 mt-0.5" />
          <span>
            AI assistant is off. Supported patterns (case changes, numbers, dates,
            base64, roman numerals, and saved lookups) still work — open-ended
            patterns need an <code className="font-mono">OPENAI_API_KEY</code>.
          </span>
        </div>
      )}
      {genNotice && (
        <div
          role="status"
          className={`p-3.5 border-l-4 rounded-md text-sm flex items-start ${
            genNotice.kind === 'warning'
              ? 'bg-amber-50 border-amber-400 text-amber-800'
              : 'bg-blue-50 border-blue-400 text-blue-800'
          }`}
        >
          {genNotice.kind === 'warning'
            ? <AlertCircle size={18} className="mr-2.5 shrink-0 mt-0.5" />
            : <Sparkles size={18} className="mr-2.5 shrink-0 mt-0.5" />}
          <span>{genNotice.text}</span>
        </div>
      )}
      <div>
        <label htmlFor="example-input" className="block text-sm font-medium text-neutral-700 mb-1.5">
          Provide Examples (one per line: input → output)
      </label>
      <textarea
          id="example-input"
          value={examples}
          onChange={(e: ChangeEvent<HTMLTextAreaElement>) => setExamples(e.target.value)}
          placeholder={`e.g., transforming "apple" to "APPLE":\napple -> APPLE\nbanana → BANANA`}
          className="textarea w-full px-3.5 py-2.5 border border-neutral-300 rounded-md shadow-sm focus:ring-2 focus:ring-primary/50 focus:border-primary transition-colors duration-150 text-neutral-800 min-h-[100px] placeholder:text-neutral-400"
        rows={4}
        />
      </div>
      
      <div className="grid sm:grid-cols-2 gap-3">
        <button 
          onClick={handleClassify} 
          disabled={!examples.trim() || isClassifying || isGenerating}
          className="btn btn-accent w-full flex items-center justify-center text-sm py-2.5 disabled:bg-accent/60"
        >
            {isClassifying ? (
              <>
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Classifying...
              </>
            ) : (
              <>
              <Wand2 size={16} className="mr-2" />
                Classify Examples
              </>
            )}
        </button>
        
        <button 
          onClick={handleGenerate} 
          className="btn btn-primary w-full flex items-center justify-center text-sm py-2.5 disabled:bg-primary/60"
          disabled={!currentClassification || isGenerating || isClassifying}
        >
            {isGenerating ? (
              <>
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Generating...
              </>
            ) : (
              <>
              <Zap size={16} className="mr-2" />
                Generate Function
              </>
            )}
        </button>
      </div>
    </div>
  );
};

export default ExampleInput;