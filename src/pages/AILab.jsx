import React, { useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  BrainCircuit,
  Terminal,
  Activity,
  Lock,
  Database,
  Cpu,
  CheckCircle,
  AlertTriangle,
  ArrowRight,
  RefreshCw,
  Save,
  Trash2,
  Download,
  FileJson
} from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import Papa from 'papaparse';
import * as XLSX from 'xlsx';
import { loadModel } from '../ml/predict';

const DEV_KEY = "Password@123";
const rawApiUrl = import.meta.env.VITE_API_URL || 
  (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
    ? 'http://localhost:5001' 
    : '/api');

// Clean URL: remove trailing slash
const API_URL = rawApiUrl.endsWith('/') ? rawApiUrl.slice(0, -1) : rawApiUrl;

export default function AILab() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [password, setPassword] = useState("");
  const [authError, setAuthError] = useState("");

  const [uploadedFile, setUploadedFile] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState({ msg: "Ready to train", current: 0, total: 100, mae: null });
  const [logs, setLogs] = useState([]);
  const [results, setResults] = useState(null);
  const [activeTab, setActiveTab] = useState('data');

  useEffect(() => {
    if (localStorage.getItem('lifespan_lab_auth') === 'true') {
      setIsAuthenticated(true);
    }
    fetchModelInfo();
  }, []);

  const fetchModelInfo = async () => {
    try {
      const resp = await fetch(`${API_URL}/model-info`);
      if (resp.ok) {
        const info = await resp.json();
        setResults(info);
      }
    } catch (e) {
      console.warn("Could not fetch model info", e);
    }
  };

  const addLog = (msg, type = 'info') => {
    setLogs(prev => [...prev, { time: new Date().toLocaleTimeString(), msg, type }].slice(-20));
  };

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;
    setUploadedFile(file);
    addLog(`File attached: ${file.name}. Ready for remote training.`, 'success');
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  const startTraining = async () => {
    setIsTraining(true);
    setLogs([]);
    addLog("Connecting to Remote Neural Engine...", 'info');
    setProgress({ msg: "Uploading dataset...", current: 20, total: 100 });

    try {
      const formData = new FormData();
      if (uploadedFile) {
        formData.append('file', uploadedFile);
      } else {
        // Fallback to synthetic if no file
        return startSyntheticTraining();
      }

      const response = await fetch(`${API_URL}/train`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) throw new Error("Training failed on server");

      const data = await response.json();
      addLog("Training complete! Model reloaded on server.", 'success');
      setProgress({ msg: "Training Finished", current: 100, total: 100, mae: data.metrics.mae });
      fetchModelInfo();
      setActiveTab('results');
    } catch (err) {
      addLog(`Training failed: ${err.message}`, 'error');
    } finally {
      setIsTraining(false);
    }
  };

  const startSyntheticTraining = async () => {
    setIsTraining(true);
    addLog("Requesting high-quality synthetic cohort (10,000 records)...", 'info');
    try {
      const response = await fetch(`${API_URL}/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ synthetic: true, n: 10000 })
      });

      if (!response.ok) throw new Error("Synthetic training failed");

      const data = await response.json();
      addLog("Synthetic training successful.", 'success');
      setProgress({ msg: "Training Finished", current: 100, total: 100, mae: data.metrics.mae });
      fetchModelInfo();
    } catch (err) {
      addLog(`Error: ${err.message}`, 'error');
    } finally {
      setIsTraining(false);
    }
  };

  const downloadReport = () => {
    if (!results) return;
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `LifeSpanX_Model_Report_${new Date().toISOString().split('T')[0]}.json`;
    a.click();
  };

  const handleLogin = (e) => {
    e.preventDefault();
    if (password === DEV_KEY) {
      setIsAuthenticated(true);
      localStorage.setItem('lifespan_lab_auth', 'true');
    } else {
      setAuthError("Invalid Developer Key");
    }
  };

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen flex items-center justify-center p-6 bg-slate-950">
        <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} className="glass-panel p-10 max-w-md w-full border-teal/30 text-center">
          <div className="w-16 h-16 bg-teal/20 rounded-2xl flex items-center justify-center text-teal mx-auto mb-6"><Lock size={32} /></div>
          <h1 className="text-3xl font-display font-bold text-white mb-2">Neural Lab Access</h1>
          <p className="text-gray-400 mb-8">Unauthorized access to model training is prohibited.</p>
          <form onSubmit={handleLogin} className="space-y-4">
            <input type="password" placeholder="Developer Key" value={password} onChange={(e) => setPassword(e.target.value)} className="w-full bg-slate-900 border border-teal/20 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-teal text-center font-mono" />
            {authError && <p className="text-rose-500 text-sm font-bold">{authError}</p>}
            <button type="submit" className="w-full bg-teal text-slate-950 py-3 rounded-xl font-bold hover:bg-teal-400 transition-colors flex items-center justify-center gap-2">Unlock Lab <ArrowRight size={18} /></button>
          </form>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen pt-24 pb-12 px-6 max-w-6xl mx-auto">
      <div className="flex justify-between items-center mb-12">
        <div>
          <h1 className="text-4xl font-display font-bold bg-clip-text text-transparent bg-gradient-to-r from-teal to-blue-500 mb-2">AI Neural Lab</h1>
          <div className="flex items-center gap-2 text-xs font-bold text-teal/70 uppercase tracking-widest"><Cpu size={14} /> Remote RandomForest Training Console</div>
        </div>
        <button onClick={() => { localStorage.removeItem('lifespan_lab_auth'); setIsAuthenticated(false); }} className="text-gray-500 hover:text-white text-sm">Logout</button>
      </div>

      <div className="grid lg:grid-cols-3 gap-8">
        <div className="lg:col-span-1 space-y-6">
          <div className="glass-panel p-6">
            <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2"><Database size={18} className="text-blue-400" /> Training Data</h3>
            <div className="space-y-4">
              <div {...getRootProps()} className={`border-2 border-dashed rounded-xl p-8 text-center transition-all ${isDragActive ? 'border-teal bg-teal/5' : 'border-border-dark/30 hover:border-teal/50 cursor-pointer'}`}>
                <input {...getInputProps()} />
                <Database className="w-10 h-10 text-gray-500 mx-auto mb-3" />
                <p className="text-sm text-gray-400">{uploadedFile ? uploadedFile.name : "Drag & drop dataset"}</p>
              </div>
              <div className="relative">
                <div className="absolute inset-0 flex items-center"><div className="w-full border-t border-border-dark/20"></div></div>
                <div className="relative flex justify-center text-xs uppercase"><span className="bg-surface-dark px-2 text-gray-500">OR</span></div>
              </div>
              <button onClick={startSyntheticTraining} className="w-full py-3 border border-blue-500/30 rounded-xl text-blue-400 hover:bg-blue-500/10 transition-colors text-sm font-bold flex items-center justify-center gap-2">
                <Cpu size={16} /> Use Synthetic Cohort
              </button>
            </div>
          </div>

          {(uploadedFile || results) && (
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="glass-panel p-6 bg-teal/5 border-teal/20">
              <h3 className="text-sm font-bold text-teal uppercase tracking-widest mb-4">Training Readiness</h3>
              <div className="space-y-3">
                <div className="flex justify-between text-sm"><span className="text-gray-400">Features</span><span className="text-white font-bold">14 Dimensions</span></div>
                <div className="flex justify-between text-sm"><span className="text-gray-400">Method</span><span className="text-white font-bold">RandomForestRegressor</span></div>
                <button disabled={isTraining} onClick={startTraining} className="w-full mt-4 bg-teal text-slate-950 py-3 rounded-xl font-bold flex items-center justify-center gap-2 hover:scale-[1.02] active:scale-95 transition-all disabled:opacity-50">
                  {isTraining ? <RefreshCw className="animate-spin" size={18} /> : <BrainCircuit size={18} />}
                  {isTraining ? "Remote Processing..." : "Execute Cloud Training"}
                </button>
              </div>
            </motion.div>
          )}
        </div>

        <div className="lg:col-span-2 space-y-6">
          <div className="glass-panel p-8">
            <div className="flex justify-between items-end mb-6">
              <div>
                <h3 className="text-xl font-bold text-white mb-1">Live Model Status</h3>
                <p className="text-sm text-gray-400">{results ? `Trained at ${new Date(results.trained_at).toLocaleString()}` : "No model metrics available"}</p>
              </div>
              <div className="text-right">
                <div className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-1">Architecture</div>
                <div className="text-teal font-mono">Random Forest Ensembling</div>
              </div>
            </div>

            <div className="space-y-8">
              <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
                <div className="p-4 bg-slate-900/50 rounded-2xl border border-border-dark/10">
                  <div className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-1">R² Score</div>
                  <div className="text-2xl font-bold text-blue-400 font-mono">{results?.metrics?.r2 ? results.metrics.r2.toFixed(4) : "---"}</div>
                </div>
                <div className="p-4 bg-slate-900/50 rounded-2xl border border-border-dark/10">
                  <div className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-1">Mean Abs Error</div>
                  <div className="text-2xl font-bold text-teal font-mono">{results?.metrics?.mae ? results.metrics.mae.toFixed(2) : "---"}</div>
                </div>
                <div className="p-4 bg-slate-900/50 rounded-2xl border border-border-dark/10 hidden md:block">
                  <div className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-1">Samples</div>
                  <div className="text-2xl font-bold text-amber-500 font-mono">{results?.n_samples || 0}</div>
                </div>
              </div>

              {results?.importances && (
                <div>
                  <h4 className="text-xs font-black text-gray-500 uppercase tracking-widest mb-4">Top Feature Importances</h4>
                  <div className="space-y-3">
                    {Object.entries(results.importances).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([name, val]) => (
                      <div key={name}>
                        <div className="flex justify-between text-[10px] font-bold mb-1"><span className="uppercase text-gray-400">{name}</span><span className="text-teal">{(val * 100).toFixed(1)}%</span></div>
                        <div className="w-full h-1.5 bg-slate-900 rounded-full overflow-hidden"><motion.div initial={{ width: 0 }} animate={{ width: `${val * 100}%` }} className="h-full bg-teal shadow-[0_0_8px_rgba(20,255,236,0.5)]" /></div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="glass-panel p-6 bg-black/40 border-slate-800">
            <h3 className="text-sm font-bold text-gray-500 uppercase tracking-widest mb-4 flex items-center gap-2"><Terminal size={14} /> Training Console Output</h3>
            <div className="font-mono text-xs space-y-1 h-32 overflow-y-auto scrollbar-hide">
              {logs.length === 0 && <p className="text-gray-700 italic">No output yet...</p>}
              {logs.map((log, i) => (
                <div key={i} className="flex gap-4"><span className="text-gray-600">[{log.time}]</span><span className={log.type === 'success' ? 'text-teal' : log.type === 'error' ? 'text-rose-500' : 'text-blue-400'}>{log.msg}</span></div>
              ))}
            </div>
          </div>

          {results && (
            <div className="flex gap-4">
              <button onClick={downloadReport} className="flex-1 bg-white/5 border border-white/10 text-white py-4 rounded-2xl font-bold hover:bg-white/10 transition-all flex items-center justify-center gap-2">
                <Download size={18} /> Download Model Report
              </button>
              <button onClick={() => window.location.reload()} className="flex-1 bg-teal text-slate-950 py-4 rounded-2xl font-bold hover:shadow-lg transition-all flex items-center justify-center gap-2">
                <RefreshCw size={18} /> Refresh Metrics
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
