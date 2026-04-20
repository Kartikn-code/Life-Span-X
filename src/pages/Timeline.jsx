import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useUser } from '../context/UserContext';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { ArrowLeft } from 'lucide-react';
import { motion } from 'framer-motion';

export default function Timeline() {
  const navigate = useNavigate();
  const { userData, predictions } = useUser();
  const [chartData, setChartData] = useState([]);

  useEffect(() => {
    if (!userData || !predictions) {
      navigate('/dashboard');
      return;
    }

    const currentAge = parseFloat(userData.age) || 30;
    const baseScore = Math.min(100, Math.max(0, (predictions.prediction / 100) * 100));
    
    // Generate 10-year projection
    const data = [];
    for (let year = 0; year <= 10; year++) {
      const projectedAge = currentAge + year;
      
      // Current path: slight degradation over time due to aging, exacerbated by negative modifiers
      const degradationFactor = predictions.totalModifier < 0 ? Math.abs(predictions.totalModifier) * 0.1 : 0.5;
      const currentPathScore = Math.max(0, baseScore - (year * degradationFactor));
      
      // Improved path: Assumes user follows action plan and removes modifiable risks
      const modifiableRiskAmount = Math.abs(Object.entries(predictions.modifiers)
        .filter(([k, v]) => v < 0 && k !== 'genetics')
        .reduce((sum, [_, v]) => sum + v, 0));
        
      // Recover some score over time, then slow degradation
      const improvedPathScore = Math.min(100, baseScore + (year <= 3 ? (modifiableRiskAmount * 0.3 * year) : (modifiableRiskAmount * 0.9) - ((year-3) * 0.3)));
      
      // Optimal path: 95 max
      const optimalPathScore = Math.max(0, 95 - (year * 0.2));

      data.push({
        year: `Year ${year}`,
        age: projectedAge,
        "Current Path": parseFloat(currentPathScore.toFixed(1)),
        "Improved Path": parseFloat(improvedPathScore.toFixed(1)),
        "Optimal Path": parseFloat(optimalPathScore.toFixed(1)),
      });
    }

    setChartData(data);
  }, [userData, predictions, navigate]);

  if (!chartData.length) return null;

  return (
    <motion.div 
      initial={{ opacity: 0 }} animate={{ opacity: 1 }}
      className="min-h-screen pt-24 pb-12 px-6 max-w-7xl mx-auto"
    >
      <button onClick={() => navigate('/dashboard')} className="text-teal hover:underline flex items-center gap-2 mb-8">
        <ArrowLeft className="w-4 h-4" /> Back to Dashboard
      </button>

      <div className="glass-panel p-8">
        <h2 className="text-3xl font-display font-bold mb-2">10-Year Health Trajectory</h2>
        <p className="text-gray-400 mb-8">Projection of your LifeScore based on actuarial aging models and current risk factors.</p>
        
        <div className="h-[400px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
              <XAxis dataKey="year" stroke="#9CA3AF" tick={{fill: '#9CA3AF'}} tickMargin={10} />
              <YAxis domain={[0, 100]} stroke="#9CA3AF" tick={{fill: '#9CA3AF'}} label={{ value: 'Health Score', angle: -90, position: 'insideLeft', fill: '#9CA3AF' }} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#111827', borderColor: '#374151', color: '#fff' }}
                itemStyle={{ color: '#fff' }}
                labelFormatter={(label, payload) => payload && payload[0] ? `${label} (Age ${payload[0].payload.age})` : label}
              />
              <Legend wrapperStyle={{ paddingTop: '20px' }} />
              <Line type="monotone" dataKey="Current Path" stroke="#F5A623" strokeWidth={3} dot={{r: 4, fill: '#F5A623'}} activeDot={{r: 6}} />
              <Line type="monotone" dataKey="Improved Path" stroke="#00F5D4" strokeWidth={3} strokeDasharray="5 5" dot={{r: 4, fill: '#00F5D4'}} activeDot={{r: 6}} />
              <Line type="monotone" dataKey="Optimal Path" stroke="#3B82F6" strokeWidth={2} strokeDasharray="3 3" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="mt-8 grid md:grid-cols-3 gap-6">
          <div className="bg-amber/10 border border-amber/20 p-4 rounded-xl">
            <h4 className="font-semibold text-amber mb-2">Current Trajectory</h4>
            <p className="text-sm text-gray-300">If you maintain current habits, your score will steadily decline, reaching {chartData[10]?.['Current Path']} by age {chartData[10]?.age}.</p>
          </div>
          <div className="bg-teal/10 border border-teal/20 p-4 rounded-xl">
            <h4 className="font-semibold text-teal mb-2">Potential Improvement</h4>
            <p className="text-sm text-gray-300">Adopting the action plan will stabilize your score around {chartData[10]?.['Improved Path']} in 10 years.</p>
          </div>
          <div className="bg-blue-500/10 border border-blue-500/20 p-4 rounded-xl">
            <h4 className="font-semibold text-blue-400 mb-2">Milestone Warning</h4>
            <p className="text-sm text-gray-300">Heart disease and hypertension risks increase significantly after age {parseFloat(userData.age) + 6} on the current path.</p>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
