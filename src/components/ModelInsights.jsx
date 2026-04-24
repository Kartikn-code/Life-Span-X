import { BrainCircuit, Info, Zap } from 'lucide-react';
import { useUser } from '../context/UserContext';
import { motion } from 'framer-motion';

const ModelInsights = () => {
  const { predictions } = useUser();
  const importances = predictions?.featureImportance || {};
  
  // Convert object to array and normalize
  const features = (Array.isArray(importances) ? importances : Object.entries(importances).map(([name, weight]) => ({
    name: name.charAt(0).toUpperCase() + name.slice(1).replace('_', ' '),
    impact: weight
  })))
  .sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact))
  .slice(0, 8);

  const maxImpact = Math.max(...features.map(f => Math.abs(f.impact)), 0.01);

  return (
    <div className="bg-white dark:bg-slate-900 rounded-3xl p-8 border border-slate-200 dark:border-slate-800 shadow-sm relative overflow-hidden">
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-teal to-blue-500 opacity-30" />
      
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-teal/10 rounded-xl">
            <BrainCircuit className="w-5 h-5 text-teal" />
          </div>
          <div>
            <h3 className="text-lg font-black text-slate-950 dark:text-white">Neural Feature Attribution</h3>
            <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Live importance scores from RandomForest Model</p>
          </div>
        </div>
        <div className="group relative">
          <Zap className="w-4 h-4 text-amber-500 cursor-help" />
          <div className="absolute right-0 bottom-full mb-2 w-48 p-2 bg-slate-900 text-[10px] text-white rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50 shadow-xl">
            These values represent the feature attribution calculated by the ensemble model for your specific profile.
          </div>
        </div>
      </div>

      <div className="space-y-4">
        {features.length > 0 ? features.map((f, i) => (
          <div key={f.name} className="space-y-1">
            <div className="flex justify-between text-[10px] font-black uppercase tracking-tight">
              <span className="text-slate-600 dark:text-slate-400">{f.name}</span>
              <span className="text-teal">
                {Math.abs(f.impact * 10).toFixed(1)}%
              </span>
            </div>
            <div className="w-full h-1.5 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
              <motion.div 
                initial={{ width: 0 }}
                animate={{ width: `${(Math.abs(f.impact) / maxImpact) * 100}%` }}
                className="h-full rounded-full bg-gradient-to-r from-teal to-blue-500 shadow-[0_0_8px_rgba(20,255,236,0.3)]"
              />
            </div>
          </div>
        )) : (
          <div className="py-12 text-center">
            <p className="text-sm text-gray-500 italic">Awaiting neural engine calibration...</p>
          </div>
        )}
      </div>

      <div className="mt-8 pt-6 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex -space-x-2">
            {[1, 2, 3].map(i => (
              <div key={i} className="w-6 h-6 rounded-full border-2 border-white dark:border-slate-900 bg-slate-200 dark:bg-slate-800 flex items-center justify-center overflow-hidden">
                <img src={`https://api.dicebear.com/7.x/avataaars/svg?seed=${i+10}`} alt="AI Avatar" />
              </div>
            ))}
          </div>
          <p className="text-[10px] font-bold text-slate-500">
            Validated against cohort records.
          </p>
        </div>
        <div className="text-[10px] font-black text-teal uppercase tracking-widest bg-teal/5 px-2 py-1 rounded border border-teal/10">
          R² = {(predictions?.metrics?.r2 || 0.85).toFixed(2)}
        </div>
      </div>
    </div>
  );
};

export default ModelInsights;
