import React, { useEffect, useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useUser } from '../context/UserContext';
import { calculateLifespan } from '../ml/healthPredictEngine';
import { supabase } from '../lib/supabase';
import LifeScoreGauge from '../components/LifeScoreGauge';
import RiskRadar from '../components/RiskRadar';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { AlertTriangle, TrendingUp, Settings, Activity, Download, Apple, Dumbbell, Heart, ClipboardList, Shield } from 'lucide-react';
import { motion } from 'framer-motion';

export default function Dashboard() {
  const navigate = useNavigate();
  const { userData, updateUserData, predictions, setPredictions } = useUser();
  const [loading, setLoading] = useState(true);
  const hasSaved = useRef(false);

  useEffect(() => {
    if (!userData || Object.keys(userData).length === 0) {
      navigate('/onboarding');
      return;
    }

    const runPrediction = async () => {
      // Wait briefly so loading animation displays smoothly
      await new Promise(r => setTimeout(r, 600));
      const result = calculateLifespan(userData);
      
      if (userData.isNewEntry && !hasSaved.current) {
        hasSaved.current = true;
        
        // Remove the flag so it doesn't get saved into the DB
        const dataToSave = { ...userData };
        delete dataToSave.isNewEntry;

        const newEntry = {
            id: Date.now().toString(),
            date: new Date().toISOString(),
            userdata: dataToSave,
            prediction: result.prediction,
            base: result.base,
            score: Math.min(100, Math.max(0, (result.prediction / 100) * 100))
        };
        
        try {
          const { error } = await supabase
            .from('patient_records')
            .insert([newEntry]);
            
          if (error) {
            console.error('Error saving to Supabase:', error);
          }
        } catch (err) {
          console.error('Failed to save to Supabase:', err);
        }

        // Clear the flag from UserContext so reloads/loads don't duplicate
        updateUserData({ isNewEntry: false });
      }

      setPredictions(result);
      setLoading(false);
    };

    if (!predictions) {
      runPrediction();
    } else {
      setLoading(false);
    }
  }, [userData, predictions, navigate, setPredictions]);

  if (loading || !predictions) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-teal"></div>
      </div>
    );
  }

  const { prediction, base, modifiers } = predictions;
  const currentAge = parseFloat(userData.age) || 30;
  const score = Math.min(100, Math.max(0, (prediction / 100) * 100));
  const yearsRemaining = Math.max(0, prediction - currentAge).toFixed(1);
  
  const riskLevel = score < 50 ? 'HIGH' : score < 75 ? 'MODERATE' : 'LOW';
  const riskColor = riskLevel === 'HIGH' ? 'text-danger border-danger' : riskLevel === 'MODERATE' ? 'text-amber border-amber' : 'text-teal border-teal';

  // Radar Data Formulation
  const adjustedRadarData = [
    { subject: 'Diet', score: Math.min(100, Math.max(0, 50 + (modifiers.diet || 0) * 10)), ideal: 80 },
    { subject: 'Exercise', score: Math.min(100, Math.max(0, 50 + (modifiers.exercise || 0) * 10)), ideal: 90 },
    { subject: 'Sleep', score: Math.min(100, Math.max(0, 50 + (modifiers.sleep || 0) * 10)), ideal: 85 },
    { subject: 'Relaxation', score: Math.min(100, Math.max(0, 80 + (modifiers.stress || 0) * 5)), ideal: 90 }, 
    { subject: 'Clean Living', score: Math.min(100, Math.max(0, 80 + ((modifiers.smoking || 0) + (modifiers.alcohol || 0)) * 5)), ideal: 95 },
    { subject: 'Health Base', score: Math.min(100, Math.max(0, 80 + (modifiers.chronic || 0) * 5)), ideal: 90 }
  ];

  const comparisonData = [
    { name: 'You', val: prediction },
    { name: 'Avg', val: base },
    { name: 'Top 10%', val: base + 10 }
  ];

  const sortedModifiers = Object.entries(modifiers).sort((a, b) => a[1] - b[1]);
  const topRisks = sortedModifiers.filter(m => m[1] < 0).slice(0, 3);

  // Generate Personalized Recommendations
  const generateRoutines = () => {
    let diet = { title: "Balanced Maintenance", desc: "Maintain your current diet with a focus on whole foods, lean proteins, and hydration." };
    let workout = { title: "Active Lifestyle", desc: "Continue your current physical activities, ensuring a mix of cardio and strength training." };
    let lifestyle = { title: "Stress Management", desc: "Incorporate 10 minutes of daily mindfulness or meditation to maintain optimal mental health." };

    const bmi = userData.weight && userData.height ? (userData.weight / Math.pow(userData.height/100, 2)) : 25;
    
    // Diet Logic
    if (bmi > 25) {
      diet = { title: "Caloric Deficit & High Protein", desc: "Focus on a mild caloric deficit. Increase lean protein and fiber intake to promote satiety while reducing processed carbs." };
    } else if (modifiers.chronic && modifiers.chronic < 0) {
      diet = { title: "Mediterranean Protocol", desc: "Strict adherence to a Mediterranean diet: high in omega-3s, olive oil, nuts, and zero processed sugars to support heart health." };
    } else if (userData.diet === 'vegan' || userData.diet === 'vegetarian') {
      diet = { title: "Plant-Based Optimization", desc: "Ensure adequate B12, Iron, and complete amino acid profiles through diverse plant sources or supplementation." };
    }

    // Workout Logic
    const age = parseInt(userData.age) || 30;
    const exerciseLvl = parseInt(userData.exercise_freq) || 0;
    
    if (exerciseLvl <= 1) {
      workout = { title: "Foundation Building", desc: "Start with 20 minutes of daily Zone 2 cardio (brisk walking). Add 2 days of light resistance training to build metabolic baseline." };
    } else if (age > 60) {
      workout = { title: "Longevity & Mobility", desc: "Prioritize joint mobility, balance exercises (yoga/tai chi), and moderate strength training to preserve bone density." };
    } else if (exerciseLvl >= 4) {
      workout = { title: "Athletic Optimization", desc: "Maintain high-intensity interval training (HIIT) combined with heavy compound lifts 3-4x weekly. Focus on recovery." };
    }

    // Lifestyle/Stress Logic
    const stressLvl = parseInt(userData.stress) || 5;
    if (stressLvl > 7) {
      lifestyle = { title: "Active Decompression", desc: "Critical: Implement non-negotiable downtime. Recommended 15 mins daily breathwork (e.g. box breathing) and strict sleep hygiene." };
    } else if (modifiers.smoking && modifiers.smoking < 0) {
      lifestyle = { title: "Cessation Protocol", desc: "Prioritize smoking cessation programs. Replace the habit trigger with physical activity or mindfulness exercises." };
    }

    return { diet, workout, lifestyle };
  };

  const routines = generateRoutines();

  const handleDownload = () => {
    window.print();
  };

  return (
    <motion.div 
      initial={{ opacity: 0 }} animate={{ opacity: 1 }}
      className="min-h-screen pt-24 pb-12 px-6 max-w-7xl mx-auto print:pt-8 print:px-0 print:bg-white print:text-black"
    >
      <div className="flex flex-col md:flex-row justify-between items-center mb-8 print:mb-4">
        <h1 className="text-3xl font-display font-bold print:text-black">LifeLytics Health Dashboard</h1>
        <div className="flex flex-wrap gap-4 mt-4 md:mt-0 print:hidden">
          <button onClick={() => navigate('/history')} className="btn-secondary flex items-center gap-2 py-2 text-sm bg-surface hover:bg-surface/80 border border-teal/50 text-teal">
            <ClipboardList className="w-4 h-4" /> Patient Records
          </button>
          <button onClick={() => navigate('/science')} className="btn-secondary flex items-center gap-2 py-2 text-sm bg-surface hover:bg-surface/80 border border-blue-500/50 text-blue-400">
            <Shield className="w-4 h-4" /> About Model
          </button>
          <button onClick={handleDownload} className="btn-secondary flex items-center gap-2 py-2 text-sm bg-surface hover:bg-surface/80 border border-border">
            <Download className="w-4 h-4" /> Download Report
          </button>
          <button onClick={() => navigate('/simulate')} className="btn-secondary flex items-center gap-2 py-2 text-sm">
            <Settings className="w-4 h-4" /> Simulator
          </button>
          <button onClick={() => navigate('/action-plan')} className="btn-primary flex items-center gap-2 py-2 text-sm">
            <TrendingUp className="w-4 h-4" /> Action Plan
          </button>
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-8">
        {/* Left Column */}
        <div className="lg:col-span-1 space-y-8">
          <div className="glass-panel p-6 flex flex-col items-center print:border-gray-200 print:shadow-none print:bg-gray-50">
            <LifeScoreGauge score={score} yearsPredicted={prediction} />
            <div className={`mt-6 px-4 py-1 rounded-full border ${riskColor} bg-surface font-bold text-sm tracking-widest print:bg-white`}>
              {riskLevel} RISK
            </div>
            <p className="text-gray-400 text-sm mt-4 text-center print:text-gray-600">
              Based on {predictions.modelUsed}
            </p>
          </div>

          <div className="glass-panel p-6 print:border-gray-200 print:shadow-none">
            <h3 className="font-semibold mb-4 text-gray-300 print:text-gray-800">Quick Metrics</h3>
            <div className="space-y-4">
              <div className="flex justify-between border-b border-border/50 pb-2 print:border-gray-300">
                <span className="text-gray-400 print:text-gray-600">Current Age</span>
                <span className="font-mono text-lg print:text-black">{currentAge}</span>
              </div>
              <div className="flex justify-between border-b border-border/50 pb-2 print:border-gray-300">
                <span className="text-gray-400 print:text-gray-600">Predicted Lifespan</span>
                <span className="font-mono text-lg text-teal print:text-teal-700">{prediction}</span>
              </div>
              <div className="flex justify-between border-b border-border/50 pb-2 print:border-gray-300">
                <span className="text-gray-400 print:text-gray-600">Expected Range</span>
                <span className="font-mono text-lg print:text-black">{Math.max(40, prediction - 4)} - {Math.min(100, prediction + 5)} yrs</span>
              </div>
              <div className="flex justify-between pb-2">
                <span className="text-gray-400 print:text-gray-600">Estimated Years Left</span>
                <span className="font-mono text-lg print:text-black">{yearsRemaining}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column */}
        <div className="lg:col-span-2 space-y-8">
          
          <div className="grid md:grid-cols-2 gap-8 print:grid-cols-2 print:gap-4">
            {/* Radar Chart */}
            <div className="glass-panel p-6 print:border-gray-200 print:shadow-none">
              <h3 className="font-semibold mb-4 text-gray-300 print:text-gray-800">Lifestyle Balance</h3>
              <RiskRadar data={adjustedRadarData} />
            </div>

            {/* Comparison Chart */}
            <div className="glass-panel p-6 flex flex-col justify-between print:border-gray-200 print:shadow-none">
              <h3 className="font-semibold mb-4 text-gray-300 print:text-gray-800">Cohort Comparison</h3>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={comparisonData} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                    <XAxis type="number" domain={[40, 100]} hide />
                    <YAxis dataKey="name" type="category" stroke="#9CA3AF" axisLine={false} tickLine={false} />
                    <Tooltip cursor={{ fill: '#1F2937' }} contentStyle={{ backgroundColor: '#111827', borderColor: '#374151' }} />
                    <Bar dataKey="val" radius={[0, 4, 4, 0]} barSize={20}>
                      {comparisonData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={index === 0 ? '#00F5D4' : index === 1 ? '#F5A623' : '#3B82F6'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Explainable AI: Why this prediction? */}
          <div className="glass-panel p-6 print:border-gray-200 print:shadow-none">
            <h3 className="font-semibold mb-2 text-gray-300 print:text-gray-800">Why this prediction? (Feature Contributions)</h3>
            
            {/* Dynamic AI Narrative */}
            <div className="bg-surface/50 p-4 rounded-xl border border-border/30 mb-6 print:bg-gray-50 print:border-gray-200">
              <p className="text-sm leading-relaxed text-gray-300 print:text-gray-800">
                <strong className="text-teal print:text-teal-700">AI Explanation:</strong> Your predicted {prediction < base ? 'lower' : 'higher'} lifespan of <strong className="text-white print:text-black">{prediction} years</strong> is primarily influenced by 
                {topRisks.length > 0 
                  ? ` negative impacts from ${topRisks.map(r => r[0].replace('_', ' ')).join(', ')}` 
                  : ' protective lifestyle factors and a lack of significant chronic risks'}. 
                {sortedModifiers.filter(m => m[1] > 0).length > 0 && ` Positive factors such as ${sortedModifiers.filter(m => m[1] > 0).slice(0, 2).map(r => r[0].replace('_', ' ')).join(' and ')} are adding beneficial years to your trajectory.`}
              </p>
            </div>

            <div className="grid md:grid-cols-3 gap-4">
              {topRisks.length > 0 ? topRisks.map(([factor, impact]) => (
                <div key={factor} className="bg-surface border border-border/50 rounded-xl p-4 print:bg-white print:border-gray-300">
                  <div className="flex items-center gap-2 mb-2">
                    <AlertTriangle className="text-danger w-4 h-4 print:text-red-600" />
                    <span className="capitalize font-medium text-sm print:text-gray-800">{factor.replace('_', ' ')}</span>
                  </div>
                  <div className="text-xl font-bold font-mono text-danger print:text-red-600">
                    reduces by {Math.abs(impact).toFixed(1)} <span className="text-xs text-gray-400 font-sans font-normal print:text-gray-500">yrs</span>
                  </div>
                </div>
              )) : (
                <div className="col-span-3 text-center text-gray-400 py-6 print:text-gray-600">
                  No major modifiable risks identified. Great job!
                </div>
              )}
            </div>
          </div>

          {/* Personalized Health Protocol */}
          <div className="glass-panel p-6 print:border-gray-200 print:shadow-none print:break-inside-avoid">
            <h3 className="text-2xl font-display font-semibold mb-6 text-white print:text-black">Personalized Longevity Protocol</h3>
            <div className="space-y-6">
              
              <div className="flex gap-4 items-start">
                <div className="bg-emerald-500/20 p-3 rounded-xl print:bg-green-100">
                  <Apple className="text-emerald-400 w-6 h-6 print:text-green-700" />
                </div>
                <div>
                  <h4 className="text-lg font-bold text-emerald-300 print:text-green-800 mb-1">Nutrition: {routines.diet.title}</h4>
                  <p className="text-gray-300 text-sm leading-relaxed print:text-gray-700">{routines.diet.desc}</p>
                </div>
              </div>

              <div className="flex gap-4 items-start">
                <div className="bg-blue-500/20 p-3 rounded-xl print:bg-blue-100">
                  <Dumbbell className="text-blue-400 w-6 h-6 print:text-blue-700" />
                </div>
                <div>
                  <h4 className="text-lg font-bold text-blue-300 print:text-blue-800 mb-1">Physical Activity: {routines.workout.title}</h4>
                  <p className="text-gray-300 text-sm leading-relaxed print:text-gray-700">{routines.workout.desc}</p>
                </div>
              </div>

              <div className="flex gap-4 items-start">
                <div className="bg-purple-500/20 p-3 rounded-xl print:bg-purple-100">
                  <Heart className="text-purple-400 w-6 h-6 print:text-purple-700" />
                </div>
                <div>
                  <h4 className="text-lg font-bold text-purple-300 print:text-purple-800 mb-1">Lifestyle & Mind: {routines.lifestyle.title}</h4>
                  <p className="text-gray-300 text-sm leading-relaxed print:text-gray-700">{routines.lifestyle.desc}</p>
                </div>
              </div>

            </div>
          </div>
          
        </div>
      </div>
    </motion.div>
  );
}
