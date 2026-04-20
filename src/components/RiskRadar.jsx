import React from 'react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from 'recharts';

export default function RiskRadar({ data }) {
  // data format: [{ subject: 'Smoking', A: 120, B: 110, fullMark: 150 }]
  return (
    <div className="w-full h-64">
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart cx="50%" cy="50%" outerRadius="80%" data={data}>
          <PolarGrid stroke="#374151" />
          <PolarAngleAxis dataKey="subject" tick={{ fill: '#9CA3AF', fontSize: 12 }} />
          <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
          
          {/* Ideal line */}
          <Radar name="Ideal" dataKey="ideal" stroke="#00F5D4" strokeDasharray="3 3" fill="transparent" />
          
          {/* User's score */}
          <Radar name="You" dataKey="score" stroke="#F5A623" fill="#F5A623" fillOpacity={0.5} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
