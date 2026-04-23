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
  Trash2
} from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import Papa from 'papaparse';
import * as XLSX from 'xlsx';
import { trainModel, generateSyntheticData } from '../ml/trainer';
import { loadModel } from '../ml/predict';

const DEV_KEY = "lifespan-dev-2026"; // Simple developer key for lab access

export default function AILab() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [password, setPassword] = useState("");
  const [authError, setAuthError] = useState("");

  const [trainingData, setTrainingData] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState({ msg: "Ready to train", current: 0, total: 10, mae: null });
  const [logs, setLogs] = useState([]);
  const [trainedModel, setTrainedModel] = useState(null);
  const [activeTab, setActiveTab] = useState('data'); // 'data', 'training', 'results'

  // Authentication logic
  const handleLogin = (e) => {
    e.preventDefault();
    if (password === DEV_KEY) {
      setIsAuthenticated(true);
      localStorage.setItem('lifespan_lab_auth', 'true');
    } else {
      setAuthError("Invalid Developer Key");
    }
  };

  useEffect(() => {
    if (localStorage.getItem('lifespan_lab_auth') === 'true') {
      setIsAuthenticated(true);
    }
    
    // Check for existing custom model
    const saved = localStorage.getItem('lifespanx_custom_model');
    if (saved) {
      setTrainedModel(JSON.parse(saved));
    }
  }, []);

  const addLog = (msg, type = 'info') => {
    setLogs(prev => [...prev, { time: new Date().toLocaleTimeString(), msg, type }].slice(-20));
  };

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    addLog(`Reading file: ${file.name}...`, 'info');

    const handleData = (json) => {
      setTrainingData(json);
      addLog(`Dataset loaded: ${json.length} records.`, 'success');
      setActiveTab('training');
    };

    if (file.name.endsWith('.csv')) {
      Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => handleData(results.data),
        error: (err) => addLog(`Error parsing CSV: ${err.message}`, 'error')
      });
    } else if (file.name.endsWith('.xlsx')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const data = new Uint8Array(e.target.result);
        const workbook = XLSX.read(data, { type: 'array' });
        const json = XLSX.utils.sheet_to_json(workbook.Sheets[workbook.SheetNames[0]]);
        handleData(json);
      };
      reader.readAsArrayBuffer(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  const startTraining = async () => {
    if (!trainingData) return;
    setIsTraining(true);
    setLogs([]);
    addLog("Initializing Neural Engine...", 'info');
    
    try {
      const model = await trainModel(trainingData, { nTrees: 15, lr: 0.25 }, (msg, stats) => {
        addLog(msg, 'info');
        if (stats) setProgress(prev => ({ ...prev, msg, ...stats }));
      });
      
      setTrainedModel(model);
      addLog("Training complete! High-accuracy model generated.", 'success');
      setActiveTab('results');
    } catch (err) {
      addLog(`Training failed: ${err.message}`, 'error');
    } finally {
      setIsTraining(false);
    }
  };

  const useSyntheticData = () => {
    addLog("Generating synthetic cohort (500 records)...", 'info');
    const data = generateSyntheticData(500);
    setTrainingData(data);
    addLog("Synthetic data ready.", 'success');
    setActiveTab('training');
  };

  const saveModelToApp = async () => {
    if (!trainedModel) return;
    localStorage.setItem('lifespanx_custom_model', JSON.stringify(trainedModel));
    await loadModel(); // Re-trigger load in predict.js
    addLog("Model deployed to Application State.", 'success');
    alert("Model deployed successfully! The entire app is now using your custom model.");
  };

  const deleteCustomModel = () => {
    localStorage.removeItem('lifespanx_custom_model');
    setTrainedModel(null);
    loadModel();
    addLog("Custom model removed. App reverted to Baseline.", 'info');
  };

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen flex items-center justify-center p-6 bg-slate-950">
        <motion.div 
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="glass-panel p-10 max-w-md w-full border-teal/30 text-center"
        >
          <div className="w-16 h-16 bg-teal/20 rounded-2xl flex items-center justify-center text-teal mx-auto mb-6">
            <Lock size={32} />
          </div>
          <h1 className="text-3xl font-display font-bold text-white mb-2">Neural Lab Access</h1>
          <p className="text-gray-400 mb-8">Unauthorized access to model training is prohibited. Please enter your developer key.</p>
          
          <form onSubmit={handleLogin} className="space-y-4">
            <input 
              type="password" 
              placeholder="Developer Key"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full bg-slate-900 border border-teal/20 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-teal text-center font-mono"
            />
            {authError && <p className="text-rose-500 text-sm font-bold">{authError}</p>}
            <button type="submit" className="w-full bg-teal text-slate-950 py-3 rounded-xl font-bold hover:bg-teal-400 transition-colors flex items-center justify-center gap-2">
              Unlock Lab <ArrowRight size={18} />
            </button>
          </form>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen pt-24 pb-12 px-6 max-w-6xl mx-auto">
      <div className="flex justify-between items-center mb-12">
        <div>
          <h1 className="text-4xl font-display font-bold bg-clip-text text-transparent bg-gradient-to-r from-teal to-blue-500 mb-2">
            AI Neural Lab
          </h1>
          <div className="flex items-center gap-2 text-xs font-bold text-teal/70 uppercase tracking-widest">
            <Cpu size={14} /> Model Training & Architecture Console
          </div>
        </div>
        
        <div className="flex gap-4">
          {trainedModel && (
            <button 
              onClick={deleteCustomModel}
              className="flex items-center gap-2 px-4 py-2 bg-rose-500/10 text-rose-500 rounded-lg text-sm font-bold border border-rose-500/20 hover:bg-rose-500 hover:text-white transition-all"
            >
              <Trash2 size={16} /> Reset Engine
            </button>
          )}
          <button 
            onClick={() => { localStorage.removeItem('lifespan_lab_auth'); setIsAuthenticated(false); }}
            className="text-gray-500 hover:text-white text-sm"
          >
            Logout
          </button>
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-8">
        
        {/* Left Column: Data & Controls */}
        <div className="lg:col-span-1 space-y-6">
          
          <div className="glass-panel p-6">
            <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
              <Database size={18} className="text-blue-400" /> Training Data
            </h3>
            
            <div className="space-y-4">
              <div 
                {...getRootProps()} 
                className={`border-2 border-dashed rounded-xl p-8 text-center transition-all ${
                  isDragActive ? 'border-teal bg-teal/5' : 'border-border-dark/30 hover:border-teal/50 cursor-pointer'
                }`}
              >
                <input {...getInputProps()} />
                <Database className="w-10 h-10 text-gray-500 mx-auto mb-3" />
                <p className="text-sm text-gray-400">Drag & drop dataset</p>
              </div>

              <div className="relative">
                <div className="absolute inset-0 flex items-center"><div className="w-full border-t border-border-dark/20"></div></div>
                <div className="relative flex justify-center text-xs uppercase"><span className="bg-surface-dark px-2 text-gray-500">OR</span></div>
              </div>

              <button 
                onClick={useSyntheticData}
                className="w-full py-3 border border-blue-500/30 rounded-xl text-blue-400 hover:bg-blue-500/10 transition-colors text-sm font-bold flex items-center justify-center gap-2"
              >
                <Cpu size={16} /> Use Synthetic Cohort
              </button>
            </div>
          </div>

          {trainingData && (
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="glass-panel p-6 bg-teal/5 border-teal/20">
              <h3 className="text-sm font-bold text-teal uppercase tracking-widest mb-4">Training Readiness</h3>
              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Sample Size</span>
                  <span className="text-white font-bold">{trainingData.length} records</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Feature Count</span>
                  <span className="text-white font-bold">17 Inputs</span>
                </div>
                <button 
                  disabled={isTraining}
                  onClick={startTraining}
                  className="w-full mt-4 bg-teal text-slate-950 py-3 rounded-xl font-bold flex items-center justify-center gap-2 hover:scale-[1.02] active:scale-95 transition-all disabled:opacity-50"
                >
                  {isTraining ? <RefreshCw className="animate-spin" size={18} /> : <BrainCircuit size={18} />}
                  {isTraining ? "Neural Processing..." : "Execute Training"}
                </button>
              </div>
            </motion.div>
          )}
        </div>

        {/* Center/Right: Console & Progress */}
        <div className="lg:col-span-2 space-y-6">
          
          {/* Progress Section */}
          <div className="glass-panel p-8">
            <div className="flex justify-between items-end mb-6">
              <div>
                <h3 className="text-xl font-bold text-white mb-1">Neural Engine Status</h3>
                <p className="text-sm text-gray-400">{progress.msg}</p>
              </div>
              <div className="text-right">
                <div className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-1">Architecture</div>
                <div className="text-teal font-mono">GBDT v3.0 Ensembling</div>
              </div>
            </div>

            <div className="space-y-8">
              <div>
                <div className="flex justify-between text-xs font-bold mb-2">
                  <span className="text-gray-500 uppercase tracking-tighter">Tree Building Progress</span>
                  <span className="text-teal">{Math.round((progress.current / progress.total) * 100)}%</span>
                </div>
                <div className="w-full h-3 bg-slate-900 rounded-full overflow-hidden border border-border-dark/10">
                  <motion.div 
                    initial={{ width: 0 }}
                    animate={{ width: `${(progress.current / progress.total) * 100}%` }}
                    className="h-full bg-gradient-to-r from-teal to-blue-500 shadow-[0_0_10px_rgba(20,255,236,0.3)]"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
                <div className="p-4 bg-slate-900/50 rounded-2xl border border-border-dark/10">
                  <div className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-1">Mean Abs Error</div>
                  <div className="text-2xl font-bold text-blue-400 font-mono">{progress.mae ? progress.mae.toFixed(4) : "---"}</div>
                </div>
                <div className="p-4 bg-slate-900/50 rounded-2xl border border-border-dark/10">
                  <div className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-1">Active Trees</div>
                  <div className="text-2xl font-bold text-teal font-mono">{progress.current}</div>
                </div>
                <div className="p-4 bg-slate-900/50 rounded-2xl border border-border-dark/10 hidden md:block">
                  <div className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-1">Learning Rate</div>
                  <div className="text-2xl font-bold text-amber-500 font-mono">0.25</div>
                </div>
              </div>
            </div>
          </div>

          {/* Training Logs */}
          <div className="glass-panel p-6 bg-black/40 border-slate-800">
            <h3 className="text-sm font-bold text-gray-500 uppercase tracking-widest mb-4 flex items-center gap-2">
              <Terminal size={14} /> Training Console Output
            </h3>
            <div className="font-mono text-xs space-y-1 h-48 overflow-y-auto scrollbar-hide">
              {logs.length === 0 && <p className="text-gray-700 italic">No output yet. Initialize training to see neural logs...</p>}
              {logs.map((log, i) => (
                <div key={i} className="flex gap-4">
                  <span className="text-gray-600">[{log.time}]</span>
                  <span className={
                    log.type === 'success' ? 'text-teal' : 
                    log.type === 'error' ? 'text-rose-500' : 'text-blue-400'
                  }>
                    {log.msg}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Results & Deployment */}
          <AnimatePresence>
            {trainedModel && (
              <motion.div 
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="glass-panel p-8 bg-gradient-to-br from-blue-500/10 to-teal/10 border-blue-500/20"
              >
                <div className="flex flex-col md:flex-row justify-between items-center gap-6">
                  <div>
                    <div className="flex items-center gap-2 text-teal font-bold mb-2">
                      <CheckCircle size={20} /> New Model Ready
                    </div>
                    <p className="text-sm text-gray-400 max-w-md">
                      Your neural engine has been successfully calibrated. Accuracy (MAE) is <span className="text-white font-bold">{trainedModel.metrics.mae}</span>.
                      Deploy this model to make it active across the entire platform.
                    </p>
                  </div>
                  <button 
                    onClick={saveModelToApp}
                    className="whitespace-nowrap bg-white text-slate-950 px-8 py-4 rounded-2xl font-bold hover:shadow-[0_0_20px_rgba(255,255,255,0.3)] transition-all flex items-center gap-2"
                  >
                    <Save size={18} /> Deploy to Application
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
