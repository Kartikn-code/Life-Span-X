import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useUser } from '../context/UserContext';
import { runSimulation } from '../ml/counterfactual';
import LifeScoreGauge from '../components/LifeScoreGauge';
import { motion } from 'framer-motion';
import { ArrowLeft, Save } from 'lucide-react';

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
    
    // Set initial cigarettes based on string band
    let initCigs = 0;
    if (userData.smoking === '1-10') initCigs = 5;
    if (userData.smoking === '11-20') initCigs = 15;
    if (userData.smoking === '20+') initCigs = 25;
    
    // Set initial alcohol based on string band
    let initAlc = 0;
    if (userData.alcohol === '1-7') initAlc = 4;
    if (userData.alcohol === '8-14') initAlc = 11;
    if (userData.alcohol === '15-21') initAlc = 18;
    if (userData.alcohol === '21+') initAlc = 25;

    setSliders(prev => ({
      ...prev,
      cigarettes: initCigs,
      alcohol_units: initAlc
    }));

    setBasePrediction(predictions.prediction);
  }, [userData, predictions, navigate]);

  useEffect(() => {
    if (userData) {
      // Run simulation on every slider change
      const result = runSimulation(userData, sliders);
      setSimData(result);
    }
  }, [sliders, userData]);

  if (!userData || !simData) return null;

  const handleSliderChange = (e) => {
    const { name, value } = e.target;
    setSliders(prev => ({ ...prev, [name]: parseFloat(value) }));
  };

  const yearsGained = (simData.prediction - basePrediction).toFixed(1);
  const score = Math.min(100, Math.max(0, (simData.prediction / 100) * 100));

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
    <div className="min-h-screen pt-24 pb-12 px-6 max-w-7xl mx-auto">
      <button onClick={() => navigate('/dashboard')} className="text-teal hover:underline flex items-center gap-2 mb-8">
        <ArrowLeft className="w-4 h-4" /> Back to Dashboard
      </button>

      <div className="grid lg:grid-cols-2 gap-12">
        {/* Left Column: Sliders */}
        <div className="glass-panel p-8">
          <h2 className="text-2xl font-display font-bold mb-6">Counterfactual Simulator</h2>
          <p className="text-gray-400 mb-8 text-sm">Adjust the modifiable factors below to see how they impact your lifespan trajectory in real-time.</p>

          <div className="space-y-6 max-h-[60vh] overflow-y-auto pr-4 custom-scrollbar">
            
            <div>
              <div className="flex justify-between mb-1 text-sm">
                <label className="text-gray-300">Cigarettes / Day</label>
                <span className="font-mono text-teal">{sliders.cigarettes}</span>
              </div>
              <input type="range" name="cigarettes" min="0" max="40" step="1" value={sliders.cigarettes} onChange={handleSliderChange} className="w-full accent-teal" />
            </div>

            <div>
              <div className="flex justify-between mb-1 text-sm">
                <label className="text-gray-300">Alcohol Units / Week</label>
                <span className="font-mono text-teal">{sliders.alcohol_units}</span>
              </div>
              <input type="range" name="alcohol_units" min="0" max="35" step="1" value={sliders.alcohol_units} onChange={handleSliderChange} className="w-full accent-teal" />
            </div>

            <div>
              <div className="flex justify-between mb-1 text-sm">
                <label className="text-gray-300">Exercise Days / Week</label>
                <span className="font-mono text-teal">{sliders.exercise_days}</span>
              </div>
              <input type="range" name="exercise_days" min="0" max="7" step="1" value={sliders.exercise_days} onChange={handleSliderChange} className="w-full accent-teal" />
            </div>

            <div>
              <div className="flex justify-between mb-1 text-sm">
                <label className="text-gray-300">Sleep Hours</label>
                <span className="font-mono text-teal">{sliders.sleep}</span>
              </div>
              <input type="range" name="sleep" min="4" max="10" step="0.5" value={sliders.sleep} onChange={handleSliderChange} className="w-full accent-teal" />
            </div>

            <div>
              <div className="flex justify-between mb-1 text-sm">
                <label className="text-gray-300">Stress Level (1-10)</label>
                <span className="font-mono text-teal">{sliders.stress}</span>
              </div>
              <input type="range" name="stress" min="1" max="10" step="1" value={sliders.stress} onChange={handleSliderChange} className="w-full accent-teal" />
            </div>

            <div>
              <div className="flex justify-between mb-1 text-sm">
                <label className="text-gray-300">Fruit & Veg (Servings/day)</label>
                <span className="font-mono text-teal">{sliders.fruit_veg}</span>
              </div>
              <input type="range" name="fruit_veg" min="0" max="10" step="1" value={sliders.fruit_veg} onChange={handleSliderChange} className="w-full accent-teal" />
            </div>

            <div>
              <div className="flex justify-between mb-1 text-sm">
                <label className="text-gray-300">Water Intake (L/day)</label>
                <span className="font-mono text-teal">{sliders.water}</span>
              </div>
              <input type="range" name="water" min="0" max="4" step="0.5" value={sliders.water} onChange={handleSliderChange} className="w-full accent-teal" />
            </div>

            <div>
              <div className="flex justify-between mb-1 text-sm">
                <label className="text-gray-300">Target BMI</label>
                <span className="font-mono text-teal">{sliders.bmi}</span>
              </div>
              <input type="range" name="bmi" min="15" max="45" step="0.1" value={sliders.bmi} onChange={handleSliderChange} className="w-full accent-teal" />
            </div>

          </div>
        </div>

        {/* Right Column: Live Gauge */}
        <div className="flex flex-col space-y-8">
          <div className="glass-panel p-8 flex flex-col items-center justify-center flex-1">
            <h3 className="text-lg font-semibold mb-8 text-gray-300">Simulated Trajectory</h3>
            <LifeScoreGauge score={score} yearsPredicted={simData.prediction} />
            
            <div className="mt-8 text-center space-y-2">
              <div className="text-gray-400">Base Prediction: <span className="text-white font-mono">{basePrediction} years</span></div>
              <div className="text-gray-400">With these changes: <span className="text-white font-mono">{simData.prediction.toFixed(1)} years</span></div>
              
              <motion.div 
                key={yearsGained}
                initial={{ scale: 1.2, color: '#fff' }}
                animate={{ scale: 1, color: yearsGained > 0 ? '#00F5D4' : yearsGained < 0 ? '#FF4D4F' : '#9CA3AF' }}
                className="text-3xl font-bold font-mono mt-4"
              >
                {yearsGained > 0 ? '+' : ''}{yearsGained} <span className="text-sm font-sans font-normal">years gained</span>
              </motion.div>
            </div>
          </div>

          <button onClick={saveScenario} className="btn-secondary flex items-center justify-center gap-2 w-full py-4">
            <Save className="w-5 h-5" /> Save Scenario
          </button>
        </div>
      </div>
    </div>
  );
}
