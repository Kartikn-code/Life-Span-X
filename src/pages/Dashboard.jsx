import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useUser } from '../context/UserContext';
import { predictLifespan, loadModel } from '../ml/predict';
import LifeScoreGauge from '../components/LifeScoreGauge';
import RiskRadar from '../components/RiskRadar';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { AlertTriangle, TrendingUp, Settings, Activity } from 'lucide-react';
import { motion } from 'framer-motion';

export default function Dashboard() {
  const navigate = useNavigate();
  const { userData, predictions, setPredictions } = useUser();
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!userData || Object.keys(userData).length === 0) {
      navigate('/onboarding');
      return;
    }

    const runPrediction = async () => {
      await loadModel();
      const result = await predictLifespan(userData);
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
  const radarData = [
    { subject: 'Diet', score: 50 + (modifiers.diet || 0) * 10, ideal: 80 },
    { subject: 'Exercise', score: 50 + (modifiers.exercise || 0) * 10, ideal: 90 },
    { subject: 'Sleep', score: 50 + (modifiers.sleep || 0) * 10, ideal: 85 },
    { subject: 'Stress', score: 80 + (modifiers.stress || 0) * 10, ideal: 50 }, // lower is better, but radar usually higher is better. Let's map it:
    { subject: 'Vices', score: 50 + ((modifiers.smoking || 0) + (modifiers.alcohol || 0)) * 5, ideal: 95 },
    { subject: 'Medical', score: 50 + (modifiers.chronic || 0) * 5, ideal: 90 }
  ];

  // Adjust radar scores to be 0-100 logic where higher = healthier
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

  // Top Risk Factors
  const sortedModifiers = Object.entries(modifiers).sort((a, b) => a[1] - b[1]);
  const topRisks = sortedModifiers.filter(m => m[1] < 0).slice(0, 3);

  return (
    <motion.div 
      initial={{ opacity: 0 }} animate={{ opacity: 1 }}
      className="min-h-screen pt-24 pb-12 px-6 max-w-7xl mx-auto"
    >
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-display font-bold">Health Dashboard</h1>
        <div className="flex gap-4">
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
          <div className="glass-panel p-6 flex flex-col items-center">
            <LifeScoreGauge score={score} yearsPredicted={prediction} />
            <div className={`mt-6 px-4 py-1 rounded-full border ${riskColor} bg-surface font-bold text-sm tracking-widest`}>
              {riskLevel} RISK
            </div>
            <p className="text-gray-400 text-sm mt-4 text-center">
              Based on {predictions.modelUsed}
            </p>
          </div>

          <div className="glass-panel p-6">
            <h3 className="font-semibold mb-4 text-gray-300">Quick Metrics</h3>
            <div className="space-y-4">
              <div className="flex justify-between border-b border-border/50 pb-2">
                <span className="text-gray-400">Current Age</span>
                <span className="font-mono text-lg">{currentAge}</span>
              </div>
              <div className="flex justify-between border-b border-border/50 pb-2">
                <span className="text-gray-400">Predicted Lifespan</span>
                <span className="font-mono text-lg text-teal">{prediction}</span>
              </div>
              <div className="flex justify-between pb-2">
                <span className="text-gray-400">Estimated Years Left</span>
                <span className="font-mono text-lg">{yearsRemaining}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column */}
        <div className="lg:col-span-2 space-y-8">
          {/* Persona Badge */}
          <div className="glass-panel p-6 border-l-4 border-l-amber flex gap-4 items-start">
            <div className="bg-amber/20 p-3 rounded-xl">
              <Activity className="text-amber w-6 h-6" />
            </div>
            <div>
              <h3 className="text-xl font-bold text-amber mb-2">Persona: The Stressed Professional</h3>
              <p className="text-gray-300 text-sm leading-relaxed">
                High stress, sedentary lifestyle, average diet. Peer group: 38% of your age cohort. 
                Your predicted lifespan is {((base+10) - prediction).toFixed(1)} years below your cohort's top performers.
              </p>
              <button onClick={() => navigate('/simulate')} className="text-teal hover:underline text-sm mt-3 inline-block">
                See what changed their outcome &rarr;
              </button>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            {/* Radar Chart */}
            <div className="glass-panel p-6">
              <h3 className="font-semibold mb-4 text-gray-300">Lifestyle Balance</h3>
              <RiskRadar data={adjustedRadarData} />
            </div>

            {/* Comparison Chart */}
            <div className="glass-panel p-6 flex flex-col justify-between">
              <h3 className="font-semibold mb-4 text-gray-300">Cohort Comparison</h3>
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

          {/* Top Risk Factors */}
          <div className="glass-panel p-6">
            <h3 className="font-semibold mb-4 text-gray-300">Top Modifiable Risks</h3>
            <div className="grid md:grid-cols-3 gap-4">
              {topRisks.length > 0 ? topRisks.map(([factor, impact]) => (
                <div key={factor} className="bg-surface border border-border/50 rounded-xl p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <AlertTriangle className="text-danger w-4 h-4" />
                    <span className="capitalize font-medium text-sm">{factor.replace('_', ' ')}</span>
                  </div>
                  <div className="text-2xl font-bold font-mono text-danger">{impact.toFixed(1)} <span className="text-xs text-gray-400 font-sans font-normal">yrs</span></div>
                  <button onClick={() => navigate('/simulate')} className="w-full mt-3 bg-danger/10 hover:bg-danger/20 text-danger text-xs py-2 rounded-lg transition-colors">
                    Fix this
                  </button>
                </div>
              )) : (
                <div className="col-span-3 text-center text-gray-400 py-6">
                  No major modifiable risks identified. Great job!
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
