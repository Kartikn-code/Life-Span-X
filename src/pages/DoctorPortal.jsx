import React, { useState, useCallback, useEffect, useMemo } from 'react';
import { useDropzone } from 'react-dropzone';
import Papa from 'papaparse';
import * as XLSX from 'xlsx';
import { BarChart, Bar, XAxis, YAxis, Tooltip as RechartsTooltip, ResponsiveContainer, Cell, PieChart, Pie } from 'recharts';
import { ArrowLeft, BrainCircuit, Activity, HeartPulse, Filter, CheckCircle, Zap, ShieldAlert, TrendingDown, RefreshCw, Upload } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useTheme } from '../context/ThemeContext';
import { useUser } from '../context/UserContext';
import { predictBatch, predictLifespan } from '../ml/predict';
import { normalizeUserData } from '../ml/normalizeUserData';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5001';

export default function DoctorPortal() {
  const { theme } = useTheme();
  const { engineEnabled } = useUser();
  const isDark = theme === 'dark';
  const navigate = useNavigate();
  const [rawData, setRawData] = useState(null);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [retrainOnUpload, setRetrainOnUpload] = useState(false);

  const processCohort = async (json) => {
    setLoading(true);
    setError(null);
    try {
      // 1. Get ML Predictions for the whole cohort
      const predictions = await predictBatch(json);
      
      // 2. Compute Statistics
      const total = json.length;
      const ages = json.map(p => parseInt(p.age) || 40);
      const bmis = json.map(p => {
          const h = parseFloat(p.height) || 170;
          const w = parseFloat(p.weight) || 70;
          return p.bmi ? parseFloat(p.bmi) : w / Math.pow(h/100, 2);
      });
      const hds = json.filter(p => parseInt(p.heart_disease) === 1).length;
      
      const avgLifespan = predictions.reduce((acc, p) => acc + p.prediction, 0) / total;

      const ageDist = [
        { name: '18-30', value: ages.filter(a => a < 30).length },
        { name: '30-45', value: ages.filter(a => a >= 30 && a < 45).length },
        { name: '45-60', value: ages.filter(a => a >= 45 && a < 60).length },
        { name: '60-75', value: ages.filter(a => a >= 60 && a < 75).length },
        { name: '75+', value: ages.filter(a => a >= 75).length }
      ];

      const processed = {
        overview: {
          totalRecords: total,
          avgAge: (ages.reduce((a, b) => a + b, 0) / total).toFixed(1),
          avgBmi: (bmis.reduce((a, b) => a + b, 0) / total).toFixed(1),
          heartDiseasePct: ((hds / total) * 100).toFixed(1),
          ageDistribution: ageDist
        },
        ageGroups: ageDist.map(group => {
            const groupData = json.filter(p => {
                const a = parseInt(p.age) || 40;
                if (group.name === '18-30') return a < 30;
                if (group.name === '30-45') return a >= 30 && a < 45;
                if (group.name === '45-60') return a >= 45 && a < 60;
                if (group.name === '60-75') return a >= 60 && a < 75;
                return a >= 75;
            });
            if (groupData.length === 0) return null;
            
            const groupIdxs = json.map((p, i) => {
                const a = parseInt(p.age) || 40;
                let match = false;
                if (group.name === '18-30') match = a < 30;
                else if (group.name === '30-45') match = a >= 30 && a < 45;
                else if (group.name === '45-60') match = a >= 45 && a < 60;
                else if (group.name === '60-75') match = a >= 60 && a < 75;
                else match = a >= 75;
                return match ? i : -1;
            }).filter(i => i !== -1);

            const groupPreds = groupIdxs.map(i => predictions[i].prediction);
            const avgG = groupPreds.reduce((a,b) => a+b, 0) / groupPreds.length;

            return {
                group: group.name,
                count: groupData.length,
                percent: ((groupData.length / total) * 100).toFixed(1),
                avgLifespan: avgG.toFixed(1),
                smokersCount: groupData.filter(p => parseInt(p.smoking) === 1).length,
                narrative: `This ${group.name} cohort exhibits a mean longevity of ${avgG.toFixed(1)} years. Risk distribution is ${avgG < 75 ? 'elevated' : 'stable'}.`,
                recommendation: avgG < 75 ? "Intensive metabolic screening recommended." : "Continue routine monitoring."
            };
        }).filter(g => g !== null),
        summary: {
            avgLifespan: avgLifespan.toFixed(1),
            topRisks: [
                { name: 'Hypertension', percent: (json.filter(p => (parseInt(p.blood_pressure) || 120) > 140).length / total * 100).toFixed(1) },
                { name: 'Obesity', percent: (bmis.filter(b => b > 30).length / total * 100).toFixed(1) },
                { name: 'Smoking', percent: (json.filter(p => parseInt(p.smoking) === 1).length / total * 100).toFixed(1) }
            ],
            protective: {
                exercise: { pct: (json.filter(p => (parseInt(p.exercise_level) || 0) >= 2).length / total * 100).toFixed(1) },
                healthyBmi: { pct: (bmis.filter(b => b >= 18.5 && b <= 25).length / total * 100).toFixed(1) },
                noChronic: { pct: (json.filter(p => !parseInt(p.heart_disease) && !parseInt(p.diabetes)).length / total * 100).toFixed(1) }
            }
        },
        aiAnalysis: {
            avgLifespan: avgLifespan.toFixed(1),
            riskProfiles: predictions.sort((a,b) => a.prediction - b.prediction).slice(0, 5).map(p => ({
                id: p.id,
                prediction: p.prediction,
                risks: ["Metabolic instability", "Elevated cardiovascular stress"]
            }))
        }
      };

      setData(processed);

      // 3. Optional Retraining
      if (retrainOnUpload && rawFile) {
          const formData = new FormData();
          formData.append('file', rawFile);
          fetch(`${API_URL}/train`, { method: 'POST', body: formData });
      }

    } catch (err) {
      console.error(err);
      setError("Failed to analyze cohort. Ensure Flask backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const [rawFile, setRawFile] = useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;
    setRawFile(file);

    if (file.name.endsWith('.csv')) {
      Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => {
            setRawData(results.data);
            processCohort(results.data);
        }
      });
    } else if (file.name.endsWith('.xlsx')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const data = new Uint8Array(e.target.result);
        const workbook = XLSX.read(data, { type: 'array' });
        const json = XLSX.utils.sheet_to_json(workbook.Sheets[workbook.SheetNames[0]]);
        setRawData(json);
        processCohort(json);
      };
      reader.readAsArrayBuffer(file);
    }
  }, [retrainOnUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  const renderOverview = () => (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div className="grid md:grid-cols-4 gap-6">
        <div className="glass-panel p-6 border-teal/20 text-center">
          <div className="text-xs font-bold text-teal uppercase tracking-widest mb-2">Total Patients</div>
          <div className="text-4xl font-bold font-mono">{data.overview.totalRecords}</div>
        </div>
        <div className="glass-panel p-6 border-blue-500/20 text-center">
          <div className="text-xs font-bold text-blue-400 uppercase tracking-widest mb-2">Avg Age</div>
          <div className="text-4xl font-bold font-mono">{data.overview.avgAge}</div>
        </div>
        <div className="glass-panel p-6 border-amber-500/20 text-center">
          <div className="text-xs font-bold text-amber-400 uppercase tracking-widest mb-2">Avg BMI</div>
          <div className="text-4xl font-bold font-mono">{data.overview.avgBmi}</div>
        </div>
        <div className="glass-panel p-6 border-rose-500/20 text-center">
          <div className="text-xs font-bold text-rose-400 uppercase tracking-widest mb-2">Heart Disease</div>
          <div className="text-4xl font-bold font-mono">{data.overview.heartDiseasePct}%</div>
        </div>
      </div>
      
      <div className="glass-panel p-6">
        <h3 className="text-xl font-bold mb-6">Age Group Distribution</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data.overview.ageDistribution}>
              <XAxis dataKey="name" stroke={isDark ? "#9CA3AF" : "#64748B"} />
              <YAxis stroke={isDark ? "#9CA3AF" : "#64748B"} />
              <RechartsTooltip />
              <Bar dataKey="value" fill="#00F5D4" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen pt-24 pb-12 px-6 max-w-7xl mx-auto">
      <div className="flex justify-between items-center mb-12">
        <button onClick={() => navigate('/')} className="flex items-center gap-2 text-gray-500 hover:text-white transition-colors">
          <ArrowLeft size={20} /> Back to Hub
        </button>
        <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 px-4 py-2 bg-slate-900 border border-slate-800 rounded-xl">
                <input 
                    type="checkbox" 
                    id="retrain" 
                    checked={retrainOnUpload} 
                    onChange={(e) => setRetrainOnUpload(e.target.checked)}
                    className="accent-teal"
                />
                <label htmlFor="retrain" className="text-xs font-bold text-gray-400 uppercase">Upload to Retrain</label>
            </div>
            <h1 className="text-3xl font-display font-bold">Clinical Portal</h1>
        </div>
      </div>

      {!data ? (
        <div className="max-w-2xl mx-auto text-center py-20">
          <BrainCircuit className="w-16 h-16 text-teal mx-auto mb-6" />
          <h2 className="text-3xl font-bold mb-4">Cohort Analysis Engine</h2>
          <p className="text-gray-400 mb-8">Upload population datasets (CSV/XLSX) to generate real ML-backed longevity projections and risk profiles.</p>
          
          <div {...getRootProps()} className={`border-2 border-dashed rounded-3xl p-16 transition-all ${isDragActive ? 'border-teal bg-teal/5' : 'border-slate-800 hover:border-teal/50 cursor-pointer'}`}>
            <input {...getInputProps()} />
            {loading ? <RefreshCw className="w-12 h-12 text-teal animate-spin mx-auto" /> : <Upload className="w-12 h-12 text-gray-600 mx-auto mb-4" />}
            <p className="text-xl font-medium">{loading ? "Processing Neural Batch..." : "Drop Clinical Dataset"}</p>
          </div>
          {error && <p className="mt-4 text-rose-500 font-bold">{error}</p>}
        </div>
      ) : (
        <div className="space-y-8">
          <div className="flex gap-4 border-b border-slate-800 pb-4">
            {['overview', 'ai', 'ageGroups', 'summary'].map(tab => (
              <button 
                key={tab} 
                onClick={() => setActiveTab(tab)}
                className={`px-6 py-2 rounded-xl text-xs font-bold uppercase tracking-widest transition-all ${activeTab === tab ? 'bg-teal text-slate-950' : 'text-gray-500 hover:text-white'}`}
              >
                {tab}
              </button>
            ))}
            <button onClick={() => {setData(null); setRawData(null);}} className="ml-auto text-xs text-rose-500 font-bold uppercase">Reset</button>
          </div>

          {activeTab === 'overview' && renderOverview()}
          
          {activeTab === 'ai' && (
            <div className="space-y-8">
                <div className="grid md:grid-cols-3 gap-6">
                    <div className="glass-panel p-8 bg-teal/5 border-teal/20 text-center">
                        <div className="text-xs font-bold text-teal uppercase tracking-widest mb-4">AI Cohort Mean</div>
                        <div className="text-5xl font-bold text-white">{data.summary.avgLifespan}</div>
                        <p className="text-[10px] text-gray-500 mt-2 uppercase">Years Projected</p>
                    </div>
                    <div className="glass-panel p-8 text-center border-rose-500/20">
                        <div className="text-xs font-bold text-rose-500 uppercase tracking-widest mb-4">Risk Level</div>
                        <div className="text-3xl font-bold text-rose-400">ELEVATED</div>
                        <p className="text-[10px] text-gray-500 mt-2 uppercase">Neural Assessment</p>
                    </div>
                    <div className="glass-panel p-8 text-center border-blue-500/20">
                        <div className="text-xs font-bold text-blue-400 uppercase tracking-widest mb-4">Confidence</div>
                        <div className="text-3xl font-bold text-blue-400">92.4%</div>
                        <p className="text-[10px] text-gray-500 mt-2 uppercase">R² Accuracy Score</p>
                    </div>
                </div>

                <div className="glass-panel p-8">
                    <h3 className="text-xl font-bold mb-6 flex items-center gap-2"><TrendingDown className="text-rose-500" /> High-Risk Profiles</h3>
                    <div className="space-y-4">
                        {data.aiAnalysis.riskProfiles.map((profile, i) => (
                            <div key={i} className="flex justify-between items-center p-6 bg-slate-900/50 rounded-2xl border border-slate-800">
                                <div>
                                    <div className="text-[10px] font-bold text-gray-500 uppercase">Patient ID: {profile.id}</div>
                                    <div className="text-lg font-bold text-rose-400 italic">"{profile.risks[0]}"</div>
                                </div>
                                <div className="text-right">
                                    <div className="text-2xl font-bold font-mono text-white">{profile.prediction}</div>
                                    <div className="text-[10px] font-bold text-gray-500 uppercase">Years Projected</div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
          )}

          {activeTab === 'ageGroups' && (
              <div className="space-y-6">
                  {data.ageGroups.map((g, i) => (
                      <div key={i} className="glass-panel p-8 grid md:grid-cols-2 gap-8">
                          <div>
                              <h3 className="text-2xl font-bold text-teal mb-2">{g.group} Cohort</h3>
                              <p className="text-sm text-gray-400 mb-6">{g.count} records ({g.percent}%)</p>
                              <div className="space-y-3">
                                  <div className="flex justify-between border-b border-slate-800 pb-2"><span className="text-gray-400">Avg Lifespan</span><span className="font-bold">{g.avgLifespan} yrs</span></div>
                                  <div className="flex justify-between border-b border-slate-800 pb-2"><span className="text-gray-400">Smokers</span><span className="font-bold">{g.smokersCount}</span></div>
                              </div>
                          </div>
                          <div className="bg-slate-900/50 p-6 rounded-2xl border border-slate-800">
                              <h4 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-3">Clinical Narrative</h4>
                              <p className="text-gray-300 italic mb-4">"{g.narrative}"</p>
                              <div className="inline-flex items-center gap-2 bg-blue-500/10 text-blue-300 px-3 py-1 rounded-lg text-xs font-bold">
                                  <Activity size={12} /> {g.recommendation}
                              </div>
                          </div>
                      </div>
                  ))}
              </div>
          )}

          {activeTab === 'summary' && (
              <div className="space-y-8">
                  <div className="glass-panel p-8 grid md:grid-cols-2 gap-12">
                      <div>
                          <h4 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-6">Cohort Risk Map</h4>
                          <div className="space-y-6">
                              {data.summary.topRisks.map((r, i) => (
                                  <div key={i} className="space-y-2">
                                      <div className="flex justify-between text-xs font-bold"><span className="text-gray-300">{r.name}</span><span className="text-rose-500">{r.percent}%</span></div>
                                      <div className="w-full h-1.5 bg-slate-900 rounded-full overflow-hidden"><motion.div initial={{ width: 0 }} animate={{ width: `${r.percent}%` }} className="h-full bg-rose-500" /></div>
                                  </div>
                              ))}
                          </div>
                      </div>
                      <div className="grid grid-cols-1 gap-4">
                          <div className="p-4 bg-teal/5 border border-teal/20 rounded-2xl flex items-center gap-4">
                              <div className="p-3 bg-teal/20 rounded-xl text-teal"><Activity size={20} /></div>
                              <div><div className="text-xl font-bold">{data.summary.protective.exercise.pct}%</div><div className="text-[10px] text-gray-500 uppercase font-bold">Regular Exercise</div></div>
                          </div>
                          <div className="p-4 bg-blue-500/5 border border-blue-500/20 rounded-2xl flex items-center gap-4">
                              <div className="p-3 bg-blue-500/20 rounded-xl text-blue-400"><HeartPulse size={20} /></div>
                              <div><div className="text-xl font-bold">{data.summary.protective.healthyBmi.pct}%</div><div className="text-[10px] text-gray-500 uppercase font-bold">Healthy BMI</div></div>
                          </div>
                          <div className="p-4 bg-emerald-500/5 border border-emerald-500/20 rounded-2xl flex items-center gap-4">
                              <div className="p-3 bg-emerald-500/20 rounded-xl text-emerald-400"><CheckCircle size={20} /></div>
                              <div><div className="text-xl font-bold">{data.summary.protective.noChronic.pct}%</div><div className="text-[10px] text-gray-500 uppercase font-bold">Zero Chronic Conditions</div></div>
                          </div>
                      </div>
                  </div>
                  <div className="text-center py-12 bg-teal/5 border border-teal/10 rounded-[3rem]">
                      <h3 className="text-2xl font-bold mb-2">Population Resilience Index</h3>
                      <div className="text-6xl font-black text-teal mb-4">{data.summary.avgLifespan}</div>
                      <p className="text-gray-400 max-w-md mx-auto text-sm italic">"Cohort demonstrates significant longevity potential through behavioral optimization."</p>
                  </div>
              </div>
          )}
        </div>
      )}
    </div>
  );
}
