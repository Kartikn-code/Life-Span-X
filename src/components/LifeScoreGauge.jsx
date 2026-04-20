import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

export default function LifeScoreGauge({ score, yearsPredicted }) {
  const [animatedScore, setAnimatedScore] = useState(0);

  useEffect(() => {
    const duration = 1200; // 1.2s
    const steps = 60;
    const stepTime = duration / steps;
    const increment = score / steps;
    let current = 0;

    const timer = setInterval(() => {
      current += increment;
      if (current >= score) {
        setAnimatedScore(score);
        clearInterval(timer);
      } else {
        setAnimatedScore(current);
      }
    }, stepTime);

    return () => clearInterval(timer);
  }, [score]);

  const radius = 100;
  const stroke = 20;
  const normalizedRadius = radius - stroke * 2;
  const circumference = normalizedRadius * 2 * Math.PI;
  const strokeDashoffset = circumference - (animatedScore / 100) * circumference;

  let colorClass = 'text-danger';
  if (animatedScore > 40) colorClass = 'text-amber';
  if (animatedScore > 70) colorClass = 'text-teal';

  return (
    <div className="flex flex-col items-center justify-center relative w-64 h-64 mx-auto">
      <svg height={radius * 2} width={radius * 2} className="transform -rotate-90">
        {/* Background Circle */}
        <circle
          stroke="#1F2937"
          fill="transparent"
          strokeWidth={stroke}
          r={normalizedRadius}
          cx={radius}
          cy={radius}
        />
        {/* Progress Circle */}
        <motion.circle
          stroke="currentColor"
          fill="transparent"
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circumference + ' ' + circumference}
          style={{ strokeDashoffset }}
          r={normalizedRadius}
          cx={radius}
          cy={radius}
          className={`transition-all duration-300 ${colorClass}`}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset }}
          transition={{ duration: 1.2, ease: "easeOut" }}
        />
      </svg>
      <div className="absolute flex flex-col items-center justify-center">
        <span className={`text-4xl font-bold font-display ${colorClass}`}>
          {Math.round(animatedScore)}
        </span>
        <span className="text-sm text-gray-400">LifeScore</span>
        <span className="text-lg font-semibold mt-2">{yearsPredicted}</span>
        <span className="text-xs text-gray-500">Years Predicted</span>
      </div>
    </div>
  );
}
