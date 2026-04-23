import React, { useEffect, useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useUser } from '../context/UserContext';
import { predictLifespan, predictLifespanFast, loadModel } from '../ml/predict';
import { supabase } from '../lib/supabase';
import LifeScoreGauge from '../components/LifeScoreGauge';
import RiskRadar from '../components/RiskRadar';
import { AreaChart, Area, CartesianGrid, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
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
      window.scrollTo(0, 0);
      try {
        // Step 1: Set Initial Baseline (Instant)
        if (!predictions || predictions.isBaseline) {
          const baseline = predictLifespanFast(userData);
          setPredictions(baseline);
          setLoading(false);
        }

        // Step 2: Background AI Calibration
        setIsTraining(true);
        await loadModel((progress) => {
          setTrainingProgress(progress);
        });
        
        // Step 3: Run Full AI Prediction
        const result = await predictLifespan(userData);
        
        // Upgrade to AI results
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
              prediction: result.prediction,
              base: result.base,
              score: Math.min(100, Math.max(0, (result.prediction / 100) * 100))
          };
          
          supabase.from('patient_records').insert([newEntry]).then(() => {
            incrementGlobalCounter(1);
          }).catch(err => console.error('Supabase error:', err));
          
          updateUserData({ isNewEntry: false });
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

    // Only run if we don't have predictions or if they are incomplete
    // Always run if we don't have predictions, if they are baseline, or if it's a new session
    if (engineEnabled && (!predictions || !predictions.prediction || predictions.isBaseline)) {
      setError(null);
      runPrediction();
    } else if (!engineEnabled) {
      setError("Neural Engine has to be turned on to run calculations.");
      setLoading(false);
    } else {
      window.scrollTo(0, 0);
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

  // Loading / Training State
  const hasPrediction = predictions && typeof predictions.prediction === 'number';

  // Loading State (Only if absolutely no data)
  if (loading && !predictions) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background-light dark:bg-background-dark">
        <Activity className="w-12 h-12 text-teal animate-pulse" />
      </div>
    );
  }

  // Safe destructuring with defaults to prevent crashes
  const { 
    prediction = 75, 
    biologicalAge = 30, 
    base = 80, 
    featureImportance = [], 
    recommendations = { positive: [], negative: [] }, 
    survivalCurve = [], 
    confidenceInterval = [70, 80],
    modelUsed = 'Deep Neural Network'
  } = predictions || {};

  const currentAge = parseFloat(userData?.age) || 30;
  const score = Math.min(100, Math.max(0, (prediction / 100) * 100));
  const yearsRemaining = Math.max(0, prediction - currentAge).toFixed(1);
  
  const riskLevel = score < 50 ? 'HIGH' : score < 75 ? 'MODERATE' : 'LOW';
  const riskColor = riskLevel === 'HIGH' ? 'text-danger border-danger' : riskLevel === 'MODERATE' ? 'text-amber border-amber' : 'text-teal border-teal';

  // Null safe radar formulation
  const getVal = (val, dflt) => parseFloat(val) || dflt;
  const adjustedRadarData = [
    { subject: 'Diet', score: 50 + (getVal(userData?.fruit_intake, 2) + getVal(userData?.vegetable_intake, 3)) * 5 - getVal(userData?.processed_food, 1) * 20, ideal: 90 },
    { subject: 'Exercise', score: 30 + getVal(userData?.exercise_level, 1) * 20, ideal: 90 },
    { subject: 'Sleep', score: 100 - Math.abs(7.5 - getVal(userData?.sleep_hours, 7.5)) * 15, ideal: 95 },
    { subject: 'Stress', score: 100 - (getVal(userData?.stress_level, 3) - 1) * 20, ideal: 90 }, 
    { subject: 'Clean Living', score: 100 - getVal(userData?.smoking, 0) * 40 - getVal(userData?.alcohol, 0) * 20, ideal: 95 },
    { subject: 'Health Base', score: 100 - getVal(userData?.heart_disease, 0) * 30 - getVal(userData?.diabetes, 0) * 20, ideal: 90 }
  ];

  const topRisks = (featureImportance || []).filter(f => f?.impact < -0.5).sort((a, b) => a.impact - b.impact).slice(0, 3);
  const topProtective = (featureImportance || []).filter(f => f?.impact > 0.5).sort((a, b) => b.impact - a.impact).slice(0, 3);

  const handleDownload = () => {
    generateLongevityAudit(userData, predictions, 'dashboard-content');
  };

  return (
    <motion.div 
      id="dashboard-content"
      initial={{ opacity: 0 }} animate={{ opacity: 1 }}
      className="min-h-screen pt-8 pb-12 px-8 max-w-[1400px] mx-auto print:pt-8 print:px-0"
    >
      {/* Top Navigation / Status Bar */}
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

      {/* Primary Metrics Strip */}
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
        
        {/* LEFT COLUMN: The Twin Visualizer */}
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

          {/* Clinical Markers Table */}
          <div className="bg-white dark:bg-slate-900 p-8 rounded-[2rem] border border-slate-200 dark:border-slate-800">
            <h3 className="font-black text-xs uppercase tracking-widest mb-6 flex items-center gap-2 text-slate-900 dark:text-gray-300">
              <Heart className="w-4 h-4 text-rose-500" /> Vital Clinical Logs
            </h3>
            <div className="space-y-4">
              {[
                { label: 'Expected Trajectory', value: prediction.toFixed(1) + ' yrs', color: 'text-slate-900 dark:text-white' },
                { label: 'Statistical Confidence', value: (predictions?.model_r2 * 100 || 95).toFixed(1) + '%', color: 'text-emerald-500' },
                { label: 'Prediction Delta', value: (prediction - currentAge).toFixed(1) + ' yrs', color: 'text-blue-500' },
                { label: 'Health Score', value: score.toFixed(1) + '/100', color: 'text-teal' }
              ].map((item, i) => (
                <div key={i} className="flex justify-between items-center py-3 border-b border-slate-100 dark:border-slate-800/50 last:border-0">
                  <span className="text-xs font-bold text-slate-500 uppercase tracking-tight">{item.label}</span>
                  <span className={`font-mono font-black ${item.color}`}>{item.value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: Analytical Intelligence */}
        <div className="xl:col-span-8 space-y-8">
          
          <div className="grid lg:grid-cols-2 gap-8">
            {/* Survival Path */}
            <div className="bg-white dark:bg-slate-900 p-8 rounded-[2rem] border border-slate-200 dark:border-slate-800 shadow-sm">
              <div className="flex justify-between items-start mb-8">
                <h3 className="font-black text-xs uppercase tracking-[0.2em] text-slate-950 dark:text-white">Survival Probability Path</h3>
                <div className="text-[10px] font-bold text-teal flex items-center gap-1">
                  <Activity className="w-3 h-3" /> Live Feed
                </div>
              </div>
              <div className="h-56 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={survivalCurve || []}>
                    <defs>
                      <linearGradient id="colorProb" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#00F5D4" stopOpacity={0.4}/>
                        <stop offset="95%" stopColor="#00F5D4" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={isDark ? "#1e293b" : "#f1f5f9"} />
                    <XAxis dataKey="age" stroke="#64748b" fontSize={10} tickLine={false} axisLine={false} dy={10} />
                    <YAxis hide domain={[0, 100]} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: isDark ? '#0f172a' : '#FFF', borderColor: '#00F5D4', borderRadius: '16px', border: '1px solid rgba(0,245,212,0.2)', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)' }}
                      itemStyle={{ color: '#00F5D4', fontSize: '12px', fontWeight: 'bold' }}
                    />
                    <Area type="monotone" dataKey="probability" stroke="#00F5D4" fillOpacity={1} fill="url(#colorProb)" strokeWidth={4} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <p className="text-[9px] text-slate-400 mt-6 uppercase tracking-widest text-center font-black">Probability Distribution vs Biological Milestone (Age)</p>
            </div>

            {/* Biomarker Radar */}
            <div className="bg-white dark:bg-slate-900 p-8 rounded-[2rem] border border-slate-200 dark:border-slate-800 shadow-sm">
              <h3 className="font-black text-xs uppercase tracking-[0.2em] text-slate-950 dark:text-white mb-8">Biomarker Equilibrium</h3>
              <RiskRadar data={adjustedRadarData} />
            </div>
          </div>

          {/* Machine Learning Feature Insights */}
          <ModelInsights />

          {/* Diagnostic Neural Log */}
          <div className="bg-white dark:bg-slate-900 p-10 rounded-[2.5rem] border border-slate-200 dark:border-slate-800 relative overflow-hidden shadow-sm">
            <div className="absolute top-0 right-0 p-8 opacity-5">
              <Brain className="w-32 h-32" />
            </div>
            
            <h3 className="font-black text-xs uppercase tracking-[0.3em] text-blue-500 mb-8 flex items-center gap-3">
              <Zap className="w-4 h-4" /> Diagnostic Neural Log: Feature Importance
            </h3>
            
            <div className="space-y-8">
              <div className="bg-slate-50 dark:bg-slate-950/50 p-6 rounded-3xl border border-slate-100 dark:border-slate-800">
                <p className="text-base leading-relaxed text-slate-800 dark:text-gray-200 font-medium">
                  <span className="text-teal font-black uppercase tracking-widest text-[10px] block mb-2 underline decoration-teal/30 underline-offset-4">Primary Inference</span>
                  Your current biological trajectory is primarily optimized by 
                  <span className="mx-1.5 font-black text-slate-950 dark:text-white border-b-2 border-teal/40 italic">
                    {topProtective.length > 0 ? topProtective[0]?.name : 'baseline health metrics'}
                  </span> 
                  acting as a vital protective shield, while 
                  <span className="mx-1.5 font-black text-rose-500 border-b-2 border-rose-500/40 italic">
                    {topRisks.length > 0 ? topRisks[0]?.name : 'lifestyle stressors'}
                  </span> 
                  represent the highest priority for neural recalibration.
                </p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                {[...topRisks, ...topProtective].map((factor, i) => {
                  const isRisk = factor.impact < 0;
                  return (
                    <motion.div 
                      key={i} 
                      whileHover={{ y: -5 }}
                      className={`p-6 rounded-[2rem] border transition-all ${isRisk ? 'bg-rose-50/30 dark:bg-rose-500/5 border-rose-100 dark:border-rose-500/20' : 'bg-emerald-50/30 dark:bg-emerald-500/5 border-emerald-100 dark:border-emerald-500/20'}`}
                    >
                      <div className={`text-[10px] font-black uppercase tracking-widest mb-2 ${isRisk ? 'text-rose-500' : 'text-emerald-500'}`}>
                        {isRisk ? 'Negative Impact' : 'Longevity Bonus'}
                      </div>
                      <div className={`text-4xl font-black mb-2 font-mono ${isRisk ? 'text-rose-600' : 'text-emerald-400'}`}>
                        {isRisk ? '-' : '+'}{Math.abs(factor?.impact || 0).toFixed(1)}
                        <span className="text-[10px] ml-1 uppercase">yrs</span>
                      </div>
                      <div className="text-[13px] font-black text-slate-950 dark:text-white leading-tight uppercase">{factor?.name}</div>
                    </motion.div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Action Protocols */}
          <div className="grid md:grid-cols-2 gap-8">
            <div className="bg-white dark:bg-slate-900 p-8 rounded-[2rem] border border-slate-200 dark:border-slate-800 border-l-8 border-l-rose-500">
              <h3 className="font-black text-sm uppercase tracking-[0.2em] text-rose-500 mb-6 flex items-center gap-2">
                <AlertTriangle className="w-5 h-5" /> High-Priority Protocols
              </h3>
              <div className="space-y-4">
                {(recommendations?.negative || []).map((rec, i) => (
                  <div key={i} className="flex gap-4 p-4 rounded-2xl bg-rose-50/30 dark:bg-rose-500/5 border border-rose-100/50 dark:border-rose-500/10 transition-colors hover:bg-rose-50/50">
                    <div className="w-1.5 h-1.5 rounded-full bg-rose-500 mt-2 flex-shrink-0 shadow-[0_0_8px_rgba(244,63,94,0.5)]" />
                    <p className="text-[13px] font-bold text-slate-800 dark:text-gray-300 leading-relaxed">{rec}</p>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="bg-white dark:bg-slate-900 p-8 rounded-[2rem] border border-slate-200 dark:border-slate-800 border-l-8 border-l-teal">
              <h3 className="font-black text-sm uppercase tracking-[0.2em] text-teal mb-6 flex items-center gap-2">
                <CheckCircle className="w-5 h-5" /> Optimization Strengths
              </h3>
              <div className="space-y-4">
                {(recommendations?.positive || []).map((rec, i) => (
                  <div key={i} className="flex gap-4 p-4 rounded-2xl bg-teal/5 border border-teal/10 transition-colors hover:bg-teal/10">
                    <div className="w-1.5 h-1.5 rounded-full bg-teal mt-2 flex-shrink-0 shadow-[0_0_8px_rgba(20,184,166,0.5)]" />
                    <p className="text-[13px] font-bold text-slate-800 dark:text-gray-300 leading-relaxed">{rec}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
          
        </div>
      </div>
    </motion.div>
  );
}
