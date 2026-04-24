import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

export default function LifeScoreGauge({ biologicalAge = 30, chronologicalAge = 30, yearsPredicted = 75 }) {
  const [animatedBio, setAnimatedBio] = useState(0);

  useEffect(() => {
    setAnimatedBio(biologicalAge);
  }, [biologicalAge]);

  const size = 280;
  const center = size / 2;
  const strokeWidth = 14;
  const radius = (size / 2) - strokeWidth;
  const circumference = radius * 2 * Math.PI;
  
  const progress = (animatedBio / 100) * circumference;
  const isHealthy = biologicalAge <= chronologicalAge;
  const delta = Math.abs(biologicalAge - chronologicalAge).toFixed(1);

  return (
    <div className="flex flex-col items-center select-none">
      <p className="text-[10px] font-black uppercase tracking-[0.4em] text-slate-400 dark:text-gray-500 mb-6">
        Lifespan Prediction
      </p>

      <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
        <div className={`absolute inset-4 rounded-full blur-3xl opacity-10 transition-colors duration-1000 ${isHealthy ? 'bg-teal' : 'bg-rose-500'}`} />
        
        <svg width={size} height={size} className="transform -rotate-90 relative z-10">
          <defs>
            <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor={isHealthy ? "#00F5D4" : "#f43f5e"} />
              <stop offset="100%" stopColor={isHealthy ? "#0ea5e9" : "#94a3b8"} />
            </linearGradient>
          </defs>

          <circle
            cx={center}
            cy={center}
            r={radius}
            fill="transparent"
            stroke="currentColor"
            strokeWidth={strokeWidth}
            className="text-slate-100 dark:text-slate-800/40"
          />

          <motion.circle
            cx={center}
            cy={center}
            r={radius}
            fill="transparent"
            stroke="url(#gaugeGradient)"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: circumference - progress }}
            className="drop-shadow-[0_0_12px_rgba(0,245,212,0.3)]"
          />
        </svg>

        <div className="absolute inset-0 flex flex-col items-center justify-center z-20">
          <div className="text-center">
            <div className="flex items-baseline justify-center mb-2">
              <motion.span className="text-5xl font-black font-display tracking-tighter leading-none text-slate-950 dark:text-white">
                {typeof yearsPredicted === 'number' ? yearsPredicted.toFixed(1) : '75.0'}
              </motion.span>
            </div>
            
            <div className="mb-8">
               <div className={`inline-flex items-center px-3 py-1 rounded-full text-[11px] font-black uppercase tracking-widest ${isHealthy ? 'bg-teal/10 text-teal-800 dark:text-teal' : 'bg-rose-500/10 text-rose-800 dark:text-rose-500'}`}>
                  {isHealthy ? 'Optimal' : 'Elevated'} • {delta}Y {isHealthy ? 'Gain' : 'Loss'}
               </div>
            </div>

            <div className="flex flex-col items-center">
              <div className="w-10 h-[1px] bg-slate-200 dark:bg-slate-800 mb-4" />
              <p className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-400 mb-1">Biological Age</p>
              <div className="text-2xl font-black text-slate-950 dark:text-white flex items-baseline gap-1">
                {typeof biologicalAge === 'number' ? biologicalAge.toFixed(1) : '30.0'}
                <span className="text-[10px] text-slate-400 font-black uppercase">yrs</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
