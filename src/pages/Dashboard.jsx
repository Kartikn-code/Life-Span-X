import React, { useEffect, useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useUser } from '../context/UserContext';
import { predictLifespan, predictLifespanFast, loadModel } from '../ml/predict';
import { supabase } from '../lib/supabase';
import LifeScoreGauge from '../components/LifeScoreGauge';
import RiskRadar from '../components/RiskRadar';
import { AreaChart, Area, CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { AlertTriangle, TrendingUp, Settings, Activity, Download, Heart, ClipboardList, Shield, Brain, Zap, CheckCircle, RefreshCcw } from 'lucide-react';
import { motion } from 'framer-motion';
import { useTheme } from '../context/ThemeContext';
import { generateLongevityAudit } from '../utils/pdfGenerator';
import { incrementGlobalCounter } from '../utils/stats';
import ModelInsights from '../components/ModelInsights';

export default function Dashboard() {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const navigate = useNavigate();
  const { userData, updateUserData, predictions, setPredictions, engineEnabled } = useUser();
  const [loading, setLoading] = useState(true);
  const [trainingProgress, setTrainingProgress] = useState({ model: '', current: 0, total: 100 });
  const [isTraining, setIsTraining] = useState(false);
  const [error, setError] = useState(null);
  const hasSaved = useRef(false);
  const runningRef = useRef(false);

  useEffect(() => {
    // Safety check: if no data, go back
    if (!userData || Object.keys(userData).length === 0) {
      navigate('/onboarding');
      return;
    }

    const runPrediction = async () => {
      if (runningRef.current) return;
      if (!engineEnabled) {
        setError("Neural Engine has to be turned on to run calculations.");
        setLoading(false);
        return;
      }
      
      runningRef.current = true;
      try {
        // Step 1: Set Initial Baseline (Instant)
        if (!predictions || predictions.isBaseline) {
          const baseline = predictLifespanFast(userData);
          if (baseline) setPredictions({ ...baseline, isBaseline: true });
          setLoading(false);
        }

        // Step 2: Background AI Calibration
        setIsTraining(true);
        await loadModel();
        
        // Step 3: Run Full AI Prediction
        const result = await predictLifespan(userData);
        
        if (result) {
          setPredictions(result);
          setIsTraining(false);

          // Step 4: Background Save (Non-blocking)
          if (userData.isNewEntry && !hasSaved.current) {
            hasSaved.current = true;
            const dataToSave = { ...userData };
            delete dataToSave.isNewEntry;

            const newEntry = {
                id: crypto.randomUUID ? crypto.randomUUID() : Date.now().toString(),
                date: new Date().toISOString(),
                userdata: dataToSave,
                prediction: result.prediction || 75,
                base: result.base || 80,
                score: Math.min(100, Math.max(0, ((result.prediction || 75) / 100) * 100))
            };
            
            supabase.from('patient_records').insert([newEntry]).then(() => {
              incrementGlobalCounter(1);
            }).catch(err => console.error('Supabase error:', err));
            
            updateUserData({ isNewEntry: false });
          }
        }
        runningRef.current = false;
      } catch (err) {
        console.error('Prediction error:', err);
        setError("Engine Calibration Error: " + err.message);
        setLoading(false);
        setIsTraining(false);
        runningRef.current = false;
      }
    };

    if (engineEnabled && (!predictions || !predictions.prediction || predictions.isBaseline)) {
      setError(null);
      runPrediction();
    } else if (!engineEnabled) {
      setError("Neural Engine has to be turned on to run calculations.");
      setLoading(false);
    } else {
      setLoading(false);
    }
  }, [userData, predictions, navigate, setPredictions, engineEnabled]);

  if (error) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-background-light dark:bg-background-dark p-6">
        <div className="glass-panel p-8 text-center max-w-md border-red-500/30">
          <div className="w-16 h-16 bg-red-500/10 rounded-full flex items-center justify-center mx-auto mb-6">
            <Activity className="text-red-500 w-8 h-8" />
          </div>
          <h2 className="text-2xl font-bold mb-4 text-text-light dark:text-white">Assessment Interrupted</h2>
          <p className="text-slate-600 dark:text-gray-400 mb-8">{error}</p>
          <button onClick={() => navigate('/onboarding')} className="btn-primary w-full bg-red-500 hover:bg-red-600 border-none">
            Restart Assessment
          </button>
        </div>
      </div>
    );
  }

  if (loading && !predictions) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background-light dark:bg-background-dark">
        <Activity className="w-12 h-12 text-teal animate-pulse" />
      </div>
    );
  }

  // Robust Defaults
  const p = predictions || {};
  const prediction = p.prediction || 75;
  const biologicalAge = p.biologicalAge || 30;
  const base = p.base || 80;
  const featureImportance = Array.isArray(p.featureImportance) ? p.featureImportance : [];
  const recommendations = p.recommendations || { positive: [], negative: [] };
  const survivalCurve = p.survivalCurve || [];
  const modelUsed = p.modelUsed || 'Neural Engine';

  const currentAge = parseFloat(userData?.age) || 30;
  const score = Math.min(100, Math.max(0, (prediction / 100) * 100));
  
  const riskLevel = score < 50 ? 'HIGH' : score < 75 ? 'MODERATE' : 'LOW';
  const riskColor = riskLevel === 'HIGH' ? 'text-danger border-danger' : riskLevel === 'MODERATE' ? 'text-amber border-amber' : 'text-teal border-teal';

  const getVal = (val, dflt) => parseFloat(val) || dflt;
  const adjustedRadarData = [
    { subject: 'Diet', score: 50 + (getVal(userData?.fruit_intake, 2) + getVal(userData?.vegetable_intake, 3)) * 5 - getVal(userData?.processed_food, 1) * 20, ideal: 90 },
    { subject: 'Exercise', score: 30 + getVal(userData?.exercise_level, 1) * 20, ideal: 90 },
    { subject: 'Sleep', score: 100 - Math.abs(7.5 - getVal(userData?.sleep_hours, 7.5)) * 15, ideal: 95 },
    { subject: 'Stress', score: 100 - (getVal(userData?.stress_level, 3) - 1) * 20, ideal: 90 }, 
    { subject: 'Clean Living', score: 100 - getVal(userData?.smoking, 0) * 40 - getVal(userData?.alcohol, 0) * 20, ideal: 95 },
    { subject: 'Health Base', score: 100 - getVal(userData?.heart_disease, 0) * 30 - getVal(userData?.diabetes, 0) * 20, ideal: 90 }
  ];

  const topRisks = featureImportance.filter(f => f?.impact < 0).sort((a, b) => b.impact - a.impact).slice(0, 3);
  const topProtective = featureImportance.filter(f => f?.impact > 0).sort((a, b) => b.impact - a.impact).slice(0, 3);

  const handleDownload = () => {
    generateLongevityAudit(userData, predictions, 'dashboard-content');
  };

  return (
    <motion.div 
      id="dashboard-content"
      initial={{ opacity: 0 }} animate={{ opacity: 1 }}
      className="min-h-screen pt-8 pb-12 px-8 max-w-[1400px] mx-auto print:pt-8 print:px-0"
    >
      <div className="flex flex-col xl:flex-row justify-between items-start xl:items-center mb-12 gap-6">
        <div className="space-y-1">
          <h1 className="text-5xl font-black tracking-tighter text-slate-950 dark:text-white flex items-center gap-3">
            LifeScope Insights
            {isTraining && <RefreshCcw className="w-6 h-6 text-teal animate-spin" />}
          </h1>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-teal text-[10px] font-black tracking-[0.3em] uppercase bg-teal/5 px-3 py-1 rounded-full border border-teal/20">
              <Zap className="w-3 h-3" /> Bio-Digital Twin Analysis
            </div>
            <div className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-500">
              Model: <span className="text-slate-900 dark:text-gray-300">{modelUsed}</span>
            </div>
          </div>
        </div>

        <div className="flex flex-wrap gap-3 print:hidden">
          <button onClick={() => navigate('/history')} className="px-5 py-2.5 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-sm font-bold flex items-center gap-2 hover:bg-slate-50 dark:hover:bg-slate-800 transition-all">
            <ClipboardList className="w-4 h-4 text-slate-500" /> History
          </button>
          <button onClick={handleDownload} className="px-5 py-2.5 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-sm font-bold flex items-center gap-2 hover:bg-slate-50 dark:hover:bg-slate-800 transition-all">
            <Download className="w-4 h-4 text-slate-500" /> Download Report
          </button>
          <button onClick={() => navigate('/simulate')} className="px-5 py-2.5 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-sm font-bold flex items-center gap-2 hover:bg-slate-50 dark:hover:bg-slate-800 transition-all">
            <Settings className="w-4 h-4 text-slate-500" /> Neural Simulator
          </button>
          <button onClick={() => navigate('/action-plan')} className="px-6 py-2.5 rounded-2xl bg-teal text-slate-950 text-sm font-black flex items-center gap-2 hover:shadow-lg hover:shadow-teal/20 transition-all active:scale-95">
            <Activity className="w-4 h-4" /> Professional Action Plan
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-12">
        {[
          { label: 'Metabolic Efficiency', value: (100 - (biologicalAge/currentAge)*20).toFixed(1) + '%', icon: Activity, color: 'text-teal' },
          { label: 'Risk Percentile', value: (score > 80 ? '98th' : score > 60 ? '75th' : '42nd'), icon: Shield, color: 'text-blue-500' },
          { label: 'Longevity Delta', value: (prediction - base > 0 ? '+' : '') + (prediction - base).toFixed(1) + 'y', icon: TrendingUp, color: 'text-emerald-500' },
          { label: 'Data Points', value: '142+', icon: Brain, color: 'text-purple-500' }
        ].map((m, i) => (
          <div key={i} className="bg-white dark:bg-slate-900 p-6 rounded-3xl border border-slate-200 dark:border-slate-800 shadow-sm flex items-center gap-4">
            <div className={`p-3 rounded-2xl bg-slate-50 dark:bg-slate-800/50 ${m.color}`}>
              <m.icon className="w-5 h-5" />
            </div>
            <div>
              <div className="text-[10px] font-black uppercase tracking-widest text-slate-500 mb-0.5">{m.label}</div>
              <div className="text-xl font-black text-slate-950 dark:text-white">{m.value}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="grid xl:grid-cols-12 gap-8 items-start">
        <div className="xl:col-span-4 space-y-8">
          <div className="bg-white dark:bg-slate-900 p-10 rounded-[2.5rem] border border-slate-200 dark:border-slate-800 shadow-xl flex flex-col items-center relative overflow-hidden">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-teal via-blue-500 to-purple-500 opacity-50" />
            <LifeScoreGauge biologicalAge={biologicalAge} chronologicalAge={currentAge} yearsPredicted={prediction} />
            <div className={`mt-10 px-8 py-2.5 rounded-full border-2 ${riskColor} font-black text-[10px] tracking-[0.3em] uppercase bg-white dark:bg-slate-950 shadow-sm`}>
              {riskLevel} Risk Profile Status
            </div>
            <div className="mt-12 w-full grid grid-cols-2 gap-4">
              <div className="p-4 rounded-2xl bg-slate-50 dark:bg-slate-800/30 border border-slate-100 dark:border-slate-800/50">
                <div className="text-[10px] font-black text-slate-500 uppercase mb-1">Chrono Age</div>
                <div className="text-2xl font-black text-slate-950 dark:text-white">{currentAge} <span className="text-xs">yrs</span></div>
              </div>
              <div className="p-4 rounded-2xl bg-slate-50 dark:bg-slate-800/30 border border-slate-100 dark:border-slate-800/50">
                <div className="text-[10px] font-black text-slate-500 uppercase mb-1">Biological</div>
                <div className="text-2xl font-black text-teal">{biologicalAge} <span className="text-xs">yrs</span></div>
              </div>
            </div>
          </div>
        </div>

        <div className="xl:col-span-8 space-y-8">
          <div className="grid lg:grid-cols-2 gap-8">
            <div className="bg-white dark:bg-slate-900 p-8 rounded-[2rem] border border-slate-200 dark:border-slate-800 shadow-sm">
              <h3 className="font-black text-xs uppercase tracking-[0.2em] text-slate-950 dark:text-white mb-8">Survival Probability Path</h3>
              <div className="h-56 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={survivalCurve}>
                    <XAxis dataKey="age" hide />
                    <YAxis hide />
                    <Area type="monotone" dataKey="probability" stroke="#00F5D4" fill="#00F5D4" fillOpacity={0.1} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className="bg-white dark:bg-slate-900 p-8 rounded-[2rem] border border-slate-200 dark:border-slate-800 shadow-sm">
              <h3 className="font-black text-xs uppercase tracking-[0.2em] text-slate-950 dark:text-white mb-8">Biomarker Equilibrium</h3>
              <RiskRadar data={adjustedRadarData} />
            </div>
          </div>
          <ModelInsights />
        </div>
      </div>
    </motion.div>
  );
}
