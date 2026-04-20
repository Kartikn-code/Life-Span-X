import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import Papa from 'papaparse';
import { calculateActuarialLifespan } from '../ml/actuarial';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, PieChart, Pie, ScatterChart, Scatter, ZAxis } from 'recharts';
import { ArrowLeft, UploadCloud, Users, Activity, AlertTriangle } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';

export default function DoctorPortal() {
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setLoading(true);

    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        processData(results.data);
      }
    });
  }, []);

  const loadSampleData = () => {
    // Generate synthetic data on the fly for the demo
    setLoading(true);
    const synthetic = Array.from({ length: 50 }, (_, i) => ({
      id: i + 1,
      age: Math.floor(Math.random() * 50) + 20, // 20-70
      gender: Math.random() > 0.5 ? 'male' : 'female',
      weight: Math.floor(Math.random() * 40) + 50,
      height: Math.floor(Math.random() * 30) + 150,
      smoking: ['never', 'ex', '1-10', '11-20', '20+'][Math.floor(Math.random() * 5)],
      alcohol: ['0', '1-7', '8-14', '15-21', '21+'][Math.floor(Math.random() * 5)],
      exercise_freq: Math.floor(Math.random() * 8).toString(),
      stress: Math.floor(Math.random() * 10) + 1,
      sleep_hours: Math.floor(Math.random() * 5) + 4,
      diet: ['omnivore', 'vegetarian', 'vegan'][Math.floor(Math.random() * 3)]
    }));
    
    setTimeout(() => {
      processData(synthetic);
    }, 500);
  };

  const processData = (rawData) => {
    // Run prediction for each row
    const processed = rawData.map(row => {
      const result = calculateActuarialLifespan(row);
      return {
        ...row,
        prediction: result.prediction,
        score: Math.min(100, Math.max(0, (result.prediction / 100) * 100)),
        riskLevel: result.prediction < 65 ? 'High' : result.prediction < 75 ? 'Moderate' : 'Low'
      };
    });

    // Calculate aggregate stats
    const avgScore = processed.reduce((acc, row) => acc + row.score, 0) / processed.length;
    const avgPrediction = processed.reduce((acc, row) => acc + row.prediction, 0) / processed.length;
    
    const riskDistribution = [
      { name: 'Low Risk', value: processed.filter(r => r.riskLevel === 'Low').length, color: '#00F5D4' },
      { name: 'Moderate Risk', value: processed.filter(r => r.riskLevel === 'Moderate').length, color: '#F5A623' },
      { name: 'High Risk', value: processed.filter(r => r.riskLevel === 'High').length, color: '#FF4D4F' }
    ];

    // Age groups for scatter
    const scatterData = processed.map(r => ({ x: parseFloat(r.age), y: r.prediction, z: r.score }));

    setData({
      rows: processed,
      stats: {
        total: processed.length,
        avgScore: avgScore.toFixed(1),
        avgPrediction: avgPrediction.toFixed(1)
      },
      riskDistribution,
      scatterData
    });
    setLoading(false);
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ 
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx']
    }
  });

  return (
    <motion.div 
      initial={{ opacity: 0 }} animate={{ opacity: 1 }}
      className="min-h-screen pt-24 pb-12 px-6 max-w-7xl mx-auto"
    >
      <button onClick={() => navigate('/')} className="text-teal hover:underline flex items-center gap-2 mb-8">
        <ArrowLeft className="w-4 h-4" /> Back to Home
      </button>

      {!data ? (
        <div className="max-w-2xl mx-auto mt-12">
          <h2 className="text-3xl font-display font-bold mb-4 text-center">Doctor / Cohort Portal</h2>
          <p className="text-gray-400 text-center mb-8">Upload patient datasets (.csv) for batch LifeScore prediction and population health analysis. All processing runs securely in your browser.</p>
          
          <div 
            {...getRootProps()} 
            className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-colors ${
              isDragActive ? 'border-teal bg-teal/5' : 'border-border hover:border-teal/50 hover:bg-surface/50'
            }`}
          >
            <input {...getInputProps()} />
            <UploadCloud className="w-12 h-12 text-teal mx-auto mb-4" />
            {isDragActive ? (
              <p className="text-teal font-medium">Drop the files here ...</p>
            ) : (
              <p className="text-gray-300">Drag & drop a CSV file here, or click to select one</p>
            )}
            <p className="text-xs text-gray-500 mt-2">Expects columns like age, height, weight, smoking, etc.</p>
          </div>

          <div className="mt-8 text-center">
            <span className="text-gray-500 mr-4">or</span>
            <button onClick={loadSampleData} className="btn-secondary text-sm">
              {loading ? 'Processing...' : 'Load Demo Synthetic Cohort'}
            </button>
          </div>
        </div>
      ) : (
        <div>
          <div className="flex justify-between items-end mb-8">
            <div>
              <h2 className="text-3xl font-display font-bold mb-2">Cohort Analysis</h2>
              <p className="text-gray-400">Processed {data.stats.total} patient records.</p>
            </div>
            <button onClick={() => setData(null)} className="btn-secondary text-sm">
              Upload New Dataset
            </button>
          </div>

          {/* KPI Cards */}
          <div className="grid md:grid-cols-3 gap-6 mb-8">
            <div className="glass-panel p-6 flex items-center gap-4">
              <div className="bg-blue-500/20 p-4 rounded-xl"><Users className="text-blue-400 w-6 h-6" /></div>
              <div>
                <div className="text-gray-400 text-sm">Total Patients</div>
                <div className="text-2xl font-bold font-mono">{data.stats.total}</div>
              </div>
            </div>
            <div className="glass-panel p-6 flex items-center gap-4">
              <div className="bg-teal/20 p-4 rounded-xl"><Activity className="text-teal w-6 h-6" /></div>
              <div>
                <div className="text-gray-400 text-sm">Avg LifeScore</div>
                <div className="text-2xl font-bold font-mono">{data.stats.avgScore}</div>
              </div>
            </div>
            <div className="glass-panel p-6 flex items-center gap-4">
              <div className="bg-amber/20 p-4 rounded-xl"><AlertTriangle className="text-amber w-6 h-6" /></div>
              <div>
                <div className="text-gray-400 text-sm">Avg Predicted Lifespan</div>
                <div className="text-2xl font-bold font-mono">{data.stats.avgPrediction} <span className="text-xs">yrs</span></div>
              </div>
            </div>
          </div>

          <div className="grid lg:grid-cols-2 gap-8 mb-8">
            {/* Risk Distribution */}
            <div className="glass-panel p-6">
              <h3 className="font-semibold mb-4 text-gray-300">Risk Distribution</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={data.riskDistribution}
                      cx="50%" cy="50%" innerRadius={60} outerRadius={80}
                      paddingAngle={5} dataKey="value"
                    >
                      {data.riskDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={{ backgroundColor: '#111827', borderColor: '#374151' }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="flex justify-center gap-6 mt-4">
                {data.riskDistribution.map(entry => (
                  <div key={entry.name} className="flex items-center gap-2 text-sm text-gray-300">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: entry.color }}></div>
                    {entry.name} ({entry.value})
                  </div>
                ))}
              </div>
            </div>

            {/* Scatter Plot Age vs Prediction */}
            <div className="glass-panel p-6">
              <h3 className="font-semibold mb-4 text-gray-300">Age vs. Predicted Lifespan</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis type="number" dataKey="x" name="Age" stroke="#9CA3AF" domain={['dataMin - 5', 'dataMax + 5']} />
                    <YAxis type="number" dataKey="y" name="Lifespan" stroke="#9CA3AF" domain={[40, 100]} />
                    <ZAxis type="number" dataKey="z" range={[50, 400]} name="Score" />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#111827', borderColor: '#374151', color: '#fff' }} />
                    <Scatter name="Patients" data={data.scatterData} fill="#00F5D4" opacity={0.6} />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Table */}
          <div className="glass-panel overflow-hidden">
            <div className="p-6 border-b border-border/50">
              <h3 className="font-semibold text-gray-300">Patient Registry</h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="bg-surface text-gray-400">
                  <tr>
                    <th className="p-4 font-medium">ID</th>
                    <th className="p-4 font-medium">Age</th>
                    <th className="p-4 font-medium">Gender</th>
                    <th className="p-4 font-medium">BMI</th>
                    <th className="p-4 font-medium">Smoking</th>
                    <th className="p-4 font-medium">Prediction</th>
                    <th className="p-4 font-medium">Risk</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/50">
                  {data.rows.slice(0, 10).map((row, i) => {
                    const bmi = row.weight && row.height ? (row.weight / Math.pow(row.height/100, 2)).toFixed(1) : '--';
                    return (
                      <tr key={i} className="hover:bg-surface/50 transition-colors">
                        <td className="p-4 text-gray-300">{row.id || `P-${i+1}`}</td>
                        <td className="p-4">{row.age}</td>
                        <td className="p-4 capitalize">{row.gender}</td>
                        <td className="p-4 font-mono">{bmi}</td>
                        <td className="p-4">{row.smoking || 'never'}</td>
                        <td className="p-4 font-mono text-teal">{row.prediction.toFixed(1)}</td>
                        <td className="p-4">
                          <span className={`px-2 py-1 rounded text-xs font-semibold ${row.riskLevel === 'High' ? 'bg-danger/20 text-danger' : row.riskLevel === 'Moderate' ? 'bg-amber/20 text-amber' : 'bg-teal/20 text-teal'}`}>
                            {row.riskLevel}
                          </span>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            <div className="p-4 text-center text-sm text-gray-500 border-t border-border/50">
              Showing top 10 rows. Export to PDF for full report.
            </div>
          </div>
        </div>
      )}
    </motion.div>
  );
}
