import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useUser } from '../context/UserContext';
import { predictLifespan } from '../ml/predict';
import LifeScoreGauge from '../components/LifeScoreGauge';
import { motion } from 'framer-motion';
import { ArrowLeft, Save, Settings, Zap, Activity } from 'lucide-react';

export default function Simulate() {
  const navigate = useNavigate();
  const { userData, predictions } = useUser();
  const [simData, setSimData] = useState(null);
  const [basePrediction, setBasePrediction] = useState(0);

  // Initial values mapped from user data
  const [sliders, setSliders] = useState({
    cigarettes: 0,
    alcohol_units: 0,
    exercise_days: parseInt(userData?.exercise_freq) || 0,
    sleep: parseFloat(userData?.sleep_hours) || 7,
    stress: parseInt(userData?.stress) || 5,
    fruit_veg: parseInt(userData?.fruit_veg) || 3,
    water: parseFloat(userData?.water) || 2,
    bmi: userData ? (parseFloat(userData.weight) / Math.pow(parseFloat(userData.height)/100, 2)).toFixed(1) : 22
  });

  useEffect(() => {
    if (!userData || !predictions) {
      navigate('/dashboard');
      return;
    }
    
    // Onboarding uses '1'/'0' strings for boolean fields
    const initCigs = userData.smoking === '1' ? 10 : 0;
    const initAlc = userData.alcohol === '1' ? 5 : 0;
    const initEx = parseInt(userData.exercise_level) * 2 || 0;
    const initStress = parseInt(userData.stress_level) * 2 || 5;

    setSliders(prev => ({
      ...prev,
      cigarettes: initCigs,
      alcohol_units: initAlc,
      exercise_days: Math.min(7, initEx),
      stress: Math.min(10, initStress)
    }));

    setBasePrediction(predictions.prediction);
  }, [userData, predictions, navigate]);

  useEffect(() => {
    if (!userData) return;

    // Map simulation sliders back to the categorical/numeric inputs the engine expects
    const simulatedUser = {
      ...userData,
      smoking: sliders.cigarettes > 0 ? 'yes' : 'never',
      alcohol: sliders.alcohol_units > 0 ? 'yes' : 'no',
      exercise_level: Math.min(3, Math.floor(sliders.exercise_days / 2)).toString(),
      sleep_hours: sliders.sleep.toString(),
      stress_level: Math.min(5, Math.ceil(sliders.stress / 2)).toString(),
      bmi: sliders.bmi
    };

    // Debounce the prediction call to avoid API flooding
    const timer = setTimeout(async () => {
      const result = await predictLifespan(simulatedUser);
      setSimData(result);
    }, 300);

    return () => clearTimeout(timer);
  }, [sliders, userData]);

  if (!userData || !simData) return null;

  const handleSliderChange = (e) => {
    const { name, value } = e.target;
    setSliders(prev => ({ ...prev, [name]: parseFloat(value) }));
  };

  const yearsGained = (simData.prediction - basePrediction).toFixed(1);
  const currentAge = parseFloat(userData?.age) || 30;
  
  // Biological Age Heuristic for Simulation:
  // If prediction increases, bio age decreases.
  const biologicalAge = Math.max(18, currentAge - (simData.prediction - basePrediction) * 0.5);

  const saveScenario = () => {
    // Save to local storage for "Compare scenarios panel"
    const saved = JSON.parse(localStorage.getItem('lifespan_scenarios') || '[]');
    saved.push({
      date: new Date().toISOString(),
      sliders,
      prediction: simData.prediction,
      gained: yearsGained
    });
    localStorage.setItem('lifespan_scenarios', JSON.stringify(saved));
    alert('Scenario saved!');
  };

  return (
    <div className="min-h-screen pt-28 pb-12 px-8 max-w-7xl mx-auto">
      <div className="flex items-center justify-between mb-12">
        <div>
          <button onClick={() => navigate('/dashboard')} className="group text-slate-500 hover:text-teal flex items-center gap-2 mb-2 transition-colors text-xs font-black uppercase tracking-widest">
            <ArrowLeft className="w-4 h-4 transition-transform group-hover:-translate-x-1" /> Back to Intelligence Dashboard
          </button>
          <h1 className="text-4xl font-black tracking-tighter text-slate-950 dark:text-white">Neural Trajectory Simulator</h1>
        </div>
        <button onClick={saveScenario} className="px-6 py-2.5 rounded-2xl bg-teal text-slate-950 text-sm font-black flex items-center gap-2 hover:shadow-lg hover:shadow-teal/20 transition-all active:scale-95">
          <Save className="w-4 h-4" /> Save Scenario
        </button>
      </div>

      <div className="grid lg:grid-cols-12 gap-12 items-start">
        {/* Left Column: Sliders */}
        <div className="lg:col-span-7 bg-white dark:bg-slate-900 p-10 rounded-[2.5rem] border border-slate-200 dark:border-slate-800 shadow-sm">
          <div className="flex items-center gap-3 mb-8">
            <div className="p-2.5 bg-teal/10 rounded-xl">
              <Settings className="w-5 h-5 text-teal" />
            </div>
            <div>
              <h2 className="text-xl font-black text-slate-950 dark:text-white leading-tight">Counterfactual Inputs</h2>
              <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Adjust modifiable longevity factors</p>
            </div>
          </div>

          <div className="space-y-8 pr-4">
            
            <div className="space-y-3">
              <div className="flex justify-between items-end">
                <label className="text-xs font-black text-slate-900 dark:text-gray-300 uppercase tracking-tight">Cigarettes / Day</label>
                <span className="font-mono text-lg font-black text-teal">{sliders.cigarettes}</span>
              </div>
              <input type="range" name="cigarettes" min="0" max="40" step="1" value={sliders.cigarettes} onChange={handleSliderChange} className="w-full h-2 bg-slate-100 dark:bg-slate-800 rounded-lg appearance-none cursor-pointer accent-teal" />
            </div>

            <div className="space-y-3">
              <div className="flex justify-between items-end">
                <label className="text-xs font-black text-slate-900 dark:text-gray-300 uppercase tracking-tight">Alcohol Units / Week</label>
                <span className="font-mono text-lg font-black text-teal">{sliders.alcohol_units}</span>
              </div>
              <input type="range" name="alcohol_units" min="0" max="35" step="1" value={sliders.alcohol_units} onChange={handleSliderChange} className="w-full h-2 bg-slate-100 dark:bg-slate-800 rounded-lg appearance-none cursor-pointer accent-teal" />
            </div>

            <div className="space-y-3">
              <div className="flex justify-between items-end">
                <label className="text-xs font-black text-slate-900 dark:text-gray-300 uppercase tracking-tight">Exercise Days / Week</label>
                <span className="font-mono text-lg font-black text-teal">{sliders.exercise_days}</span>
              </div>
              <input type="range" name="exercise_days" min="0" max="7" step="1" value={sliders.exercise_days} onChange={handleSliderChange} className="w-full h-2 bg-slate-100 dark:bg-slate-800 rounded-lg appearance-none cursor-pointer accent-teal" />
            </div>

            <div className="space-y-3">
              <div className="flex justify-between items-end">
                <label className="text-xs font-black text-slate-900 dark:text-gray-300 uppercase tracking-tight">Sleep Hours</label>
                <span className="font-mono text-lg font-black text-teal">{sliders.sleep}h</span>
              </div>
              <input type="range" name="sleep" min="4" max="10" step="0.5" value={sliders.sleep} onChange={handleSliderChange} className="w-full h-2 bg-slate-100 dark:bg-slate-800 rounded-lg appearance-none cursor-pointer accent-teal" />
            </div>

            <div className="space-y-3">
              <div className="flex justify-between items-end">
                <label className="text-xs font-black text-slate-900 dark:text-gray-300 uppercase tracking-tight">Stress Level (1-10)</label>
                <span className="font-mono text-lg font-black text-teal">{sliders.stress}</span>
              </div>
              <input type="range" name="stress" min="1" max="10" step="1" value={sliders.stress} onChange={handleSliderChange} className="w-full h-2 bg-slate-100 dark:bg-slate-800 rounded-lg appearance-none cursor-pointer accent-teal" />
            </div>

            <div className="space-y-3">
              <div className="flex justify-between items-end">
                <label className="text-xs font-black text-slate-900 dark:text-gray-300 uppercase tracking-tight">Target BMI</label>
                <span className="font-mono text-lg font-black text-teal">{sliders.bmi}</span>
              </div>
              <input type="range" name="bmi" min="15" max="45" step="0.1" value={sliders.bmi} onChange={handleSliderChange} className="w-full h-2 bg-slate-100 dark:bg-slate-800 rounded-lg appearance-none cursor-pointer accent-teal" />
            </div>

          </div>
        </div>

        {/* Right Column: Live Gauge */}
        <div className="lg:col-span-5 space-y-8">
          <div className="bg-white dark:bg-slate-900 p-10 rounded-[2.5rem] border border-slate-200 dark:border-slate-800 shadow-xl flex flex-col items-center justify-center relative overflow-hidden">
            <div className="absolute top-0 left-0 w-full h-1 bg-teal/30" />
            
            <h3 className="text-[10px] font-black uppercase tracking-[0.3em] text-slate-400 mb-8">Live Neural Projection</h3>
            
            <LifeScoreGauge biologicalAge={biologicalAge} chronologicalAge={currentAge} yearsPredicted={simData.prediction} />
            
            <div className="mt-12 w-full space-y-4">
              <div className="flex justify-between items-center py-3 border-b border-slate-100 dark:border-slate-800/50">
                <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Base Projection</span>
                <span className="font-mono font-black text-slate-900 dark:text-white">{basePrediction.toFixed(1)} <span className="text-[10px]">yrs</span></span>
              </div>
              
              <div className="flex justify-between items-center py-3">
                <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Net Trajectory Shift</span>
                <motion.div 
                  key={yearsGained}
                  initial={{ scale: 1.2 }}
                  animate={{ scale: 1 }}
                  className={`text-2xl font-black font-mono ${parseFloat(yearsGained) > 0 ? 'text-teal' : parseFloat(yearsGained) < 0 ? 'text-rose-500' : 'text-slate-500'}`}
                >
                  {parseFloat(yearsGained) > 0 ? '+' : ''}{yearsGained} <span className="text-xs font-sans">Yrs</span>
                </motion.div>
              </div>
            </div>
          </div>

          <div className="bg-teal/5 border border-teal/10 p-6 rounded-3xl">
             <p className="text-[11px] text-teal-800 dark:text-teal-400 leading-relaxed font-bold">
               <Zap className="w-3 h-3 inline mr-2" />
               Counterfactual logic is processed in-browser. Adjusting these parameters simulates biological outcomes based on actuarial Gompertz mortality laws.
             </p>
          </div>
        </div>
      </div>
    </div>
  );
}
