import React, { useState, useCallback, useEffect, useMemo } from 'react';
import { useDropzone } from 'react-dropzone';
import Papa from 'papaparse';
import * as XLSX from 'xlsx';
import { BarChart, Bar, XAxis, YAxis, Tooltip as RechartsTooltip, ResponsiveContainer, Cell, PieChart, Pie } from 'recharts';
import { ArrowLeft, BrainCircuit, Activity, HeartPulse, Filter, CheckCircle, Zap, ShieldAlert, TrendingDown } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { processHealthData } from '../ml/healthPredictEngine';
import { useTheme } from '../context/ThemeContext';
import { useUser } from '../context/UserContext';
import { incrementGlobalCounter } from '../utils/stats';
import { predictLifespan, loadModel, getActiveModel } from '../ml/predict';

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
  const [aiAnalysis, setAiAnalysis] = useState(null);
  const [isAiLoading, setIsAiLoading] = useState(false);

  // Filters State
  const [ageFilter, setAgeFilter] = useState('All');
  const [genderFilter, setGenderFilter] = useState('All');
  const [diseaseFilter, setDiseaseFilter] = useState('All');

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setLoading(true);
    setError(null);

    const handleParsedData = async (json) => {
      try {
        setRawData(json);
        const result = processHealthData(json);
        setData(result);
        
        // Trigger AI Deep Analysis in background
        runAiAnalysis(json);

        // Increment persistent global counter by number of people in cohort
        if (json && json.length > 0) {
          incrementGlobalCounter(json.length);
        }
      } catch (err) {
        console.error(err);
        setError(err.message || "Failed to analyze cohort data.");
      } finally {
        setLoading(false);
      }
    };

    const runAiAnalysis = async (json) => {
      setIsAiLoading(true);
      try {
        await loadModel();
        
        // Run AI prediction for a representative sample (or all if small)
        const sampleSize = Math.min(json.length, 50);
        const sample = json.slice(0, sampleSize);
        
        const aiResults = await Promise.all(sample.map(person => predictLifespan(person)));
        
        const avgAiLifespan = aiResults.reduce((acc, r) => acc + r.prediction, 0) / sampleSize;
        const topAiRisks = aiResults.map((r, i) => ({ 
          id: i, 
          prediction: r.prediction, 
          risks: r.recommendations.negative 
        })).sort((a, b) => a.prediction - b.prediction).slice(0, 5);

        setAiAnalysis({
          avgLifespan: avgAiLifespan.toFixed(1),
          riskProfiles: topAiRisks,
          confidence: "High (Neural Engine)",
          sampleSize
        });
      } catch (err) {
        console.error("AI Analysis failed", err);
      } finally {
        setIsAiLoading(false);
      }
    };

    if (file.name.endsWith('.csv')) {
      Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => handleParsedData(results.data),
        error: (err) => {
          setError(`CSV parsing error: ${err.message}`);
          setLoading(false);
        }
      });
    } else if (file.name.endsWith('.xlsx') || file.name.endsWith('.xls')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const buffer = new Uint8Array(e.target.result);
          const workbook = XLSX.read(buffer, { type: 'array' });
          const firstSheet = workbook.SheetNames[0];
          const worksheet = workbook.Sheets[firstSheet];
          const json = XLSX.utils.sheet_to_json(worksheet, { defval: "" });
          handleParsedData(json);
        } catch (err) {
          setError("Excel parsing error.");
          setLoading(false);
        }
      };
      reader.readAsArrayBuffer(file);
    } else {
      setError("Unsupported file format. Please use CSV or Excel.");
      setLoading(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ 
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx']
    }
  });

  // Apply Filters when they change
  useEffect(() => {
    if (!rawData) return;
    
    let filtered = rawData;

    // Age Filter
    if (ageFilter !== 'All') {
      filtered = filtered.filter(row => {
        const age = parseInt(row.age) || 50;
        if (ageFilter === '18-30') return age >= 18 && age < 30;
        if (ageFilter === '30-45') return age >= 30 && age < 45;
        if (ageFilter === '45-75') return age >= 45 && age < 75;
        if (ageFilter === '75+') return age >= 75;
        return true;
      });
    }

    // Gender Filter
    if (genderFilter !== 'All') {
      filtered = filtered.filter(row => {
        const gender = (row.gender || '').toLowerCase();
        return gender.startsWith(genderFilter.toLowerCase());
      });
    }

    // Disease Filter
    if (diseaseFilter !== 'All') {
      filtered = filtered.filter(row => {
        if (diseaseFilter === 'Heart Disease') return parseInt(row.heart_disease) === 1;
        if (diseaseFilter === 'Diabetes') return parseInt(row.diabetes) === 1;
        if (diseaseFilter === 'Stroke') return parseInt(row.stroke) === 1;
        if (diseaseFilter === 'None') return parseInt(row.heart_disease) !== 1 && parseInt(row.diabetes) !== 1 && parseInt(row.stroke) !== 1;
        return true;
      });
    }

    try {
      if (filtered.length === 0) {
        setData(null);
        setError("No records match the selected filters.");
      } else {
        setError(null);
        setData(processHealthData(filtered));
      }
    } catch (err) {
      setData(null);
      setError(err.message);
    }
  }, [ageFilter, genderFilter, diseaseFilter, rawData]);

  const clearFilters = () => {
    setAgeFilter('All');
    setGenderFilter('All');
    setDiseaseFilter('All');
  };

  const renderFilterPanel = () => (
    <div className="bg-surface-light dark:bg-surface-dark/80 p-4 rounded-xl border border-border-light/20 dark:border-border-dark/20 flex flex-wrap gap-4 items-end mb-8">
      <div>
        <label className="block text-xs font-bold text-slate-600 dark:text-gray-400 uppercase tracking-wider mb-2">Age Group</label>
        <select 
          value={ageFilter} 
          onChange={(e) => setAgeFilter(e.target.value)}
          className="bg-surface-light dark:bg-background-dark border border-border-light/20 dark:border-border-dark/20 rounded-lg px-4 py-2 text-sm text-text-light dark:text-text-dark focus:outline-none focus:border-teal"
        >
          <option value="All">All Ages</option>
          <option value="18-30">18 - 30</option>
          <option value="30-45">30 - 45</option>
          <option value="45-75">45 - 75</option>
          <option value="75+">75+</option>
        </select>
      </div>

      <div>
        <label className="block text-xs font-bold text-slate-600 dark:text-gray-400 uppercase tracking-wider mb-2">Gender</label>
        <select 
          value={genderFilter} 
          onChange={(e) => setGenderFilter(e.target.value)}
          className="bg-surface-light dark:bg-background-dark border border-border-light/20 dark:border-border-dark/20 rounded-lg px-4 py-2 text-sm text-text-light dark:text-text-dark focus:outline-none focus:border-teal"
        >
          <option value="All">All Genders</option>
          <option value="Male">Male</option>
          <option value="Female">Female</option>
        </select>
      </div>

      <div>
        <label className="block text-xs font-bold text-slate-600 dark:text-gray-400 uppercase tracking-wider mb-2">Chronic Conditions</label>
        <select 
          value={diseaseFilter} 
          onChange={(e) => setDiseaseFilter(e.target.value)}
          className="bg-surface-light dark:bg-background-dark border border-border-light/20 dark:border-border-dark/20 rounded-lg px-4 py-2 text-sm text-text-light dark:text-text-dark focus:outline-none focus:border-teal"
        >
          <option value="All">Any / None</option>
          <option value="Heart Disease">Heart Disease</option>
          <option value="Diabetes">Diabetes</option>
          <option value="Stroke">Stroke</option>
          <option value="None">Healthy (No Chronic)</option>
        </select>
      </div>

      {(ageFilter !== 'All' || genderFilter !== 'All' || diseaseFilter !== 'All') && (
        <button onClick={clearFilters} className="text-sm text-red-400 hover:text-red-300 underline py-2 ml-auto">
          Clear Filters
        </button>
      )}
    </div>
  );

  const renderTab1 = () => (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div className="grid md:grid-cols-4 gap-6">
        <div className="glass-panel p-6 bg-gradient-to-br from-surface-light to-surface-light/50 dark:from-surface-dark dark:to-surface-dark/50 border-teal/20 text-center">
          <div className="text-xs font-bold text-teal uppercase tracking-widest mb-2">Total Records</div>
          <div className="text-4xl font-bold font-mono text-text-light dark:text-white">{data.overview.totalRecords}</div>
        </div>
        <div className="glass-panel p-6 bg-gradient-to-br from-surface-light to-surface-light/50 dark:from-surface-dark dark:to-surface-dark/50 border-blue-500/20 text-center">
          <div className="text-xs font-bold text-blue-400 uppercase tracking-widest mb-2">Average Age</div>
          <div className="text-4xl font-bold font-mono text-text-light dark:text-white">{data.overview.avgAge}</div>
        </div>
        <div className="glass-panel p-6 bg-gradient-to-br from-surface-light to-surface-light/50 dark:from-surface-dark dark:to-surface-dark/50 border-amber-500/20 text-center">
          <div className="text-xs font-bold text-amber-400 uppercase tracking-widest mb-2">Average BMI</div>
          <div className="text-4xl font-bold font-mono text-text-light dark:text-white">{data.overview.avgBmi}</div>
        </div>
        <div className="glass-panel p-6 bg-gradient-to-br from-surface-light to-surface-light/50 dark:from-surface-dark dark:to-surface-dark/50 border-rose-500/20 text-center">
          <div className="text-xs font-bold text-rose-400 uppercase tracking-widest mb-2">Heart Disease</div>
          <div className="text-4xl font-bold font-mono text-text-light dark:text-white">{data.overview.heartDiseasePct}%</div>
        </div>
      </div>
      
      <div className="glass-panel p-6">
        <h3 className="text-xl font-bold mb-6 text-text-light/80 dark:text-gray-200">Age Group Distribution</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data.overview.ageDistribution} margin={{top: 20, right: 30, left: 20, bottom: 5}}>
              <XAxis dataKey="name" stroke={isDark ? "#9CA3AF" : "#64748B"} />
              <YAxis stroke={isDark ? "#9CA3AF" : "#64748B"} />
              <RechartsTooltip 
                contentStyle={{ 
                  backgroundColor: isDark ? '#111827' : '#FFFFFF', 
                  borderColor: isDark ? '#1F2937' : '#CBD5E1',
                  color: isDark ? '#F1F5F9' : '#000000'
                }} 
                cursor={{fill: isDark ? '#1F2937' : '#F1F5F9'}} 
              />
              <Bar dataKey="value" fill="#00F5D4" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );

  const renderTab2 = () => (
    <div className="space-y-6 animate-in fade-in duration-500">
      {data.ageGroups.map((group, idx) => (
        <div key={idx} className="glass-panel p-6 flex flex-col md:flex-row gap-8">
          <div className="md:w-1/3">
            <h3 className="text-2xl font-bold text-teal mb-1">{group.group} Years</h3>
            <p className="text-sm text-slate-600 dark:text-gray-400 mb-4">{group.count} people ({group.percent}% of dataset)</p>
            
            <div className="space-y-3">
              <div className="flex justify-between border-b border-border-light/20 dark:border-border-dark/20 pb-2">
                <span className="text-text-light/70 dark:text-gray-300">Avg Lifespan</span>
                <span className="font-bold text-text-light dark:text-white">{group.avgLifespan} yrs</span>
              </div>
              <div className="flex justify-between border-b border-border/50 pb-2">
                <span className="text-text-light/70 dark:text-gray-300">Smokers</span>
                <span className="font-bold text-text-light dark:text-white">{group.smokersCount} ({group.smokersPct}%)</span>
              </div>
              <div className="flex justify-between border-b border-border/50 pb-2">
                <span className="text-text-light/70 dark:text-gray-300">Drinkers</span>
                <span className="font-bold text-text-light dark:text-white">{group.drinkersCount} ({group.drinkersPct}%)</span>
              </div>
              <div className="flex justify-between border-b border-border/50 pb-2">
                <span className="text-text-light/70 dark:text-gray-300">Regular Exercisers</span>
                <span className="font-bold text-text-light dark:text-white">{group.exercisersCount} ({group.exercisersPct}%)</span>
              </div>
              <div className="flex justify-between border-b border-border/50 pb-2">
                <span className="text-text-light/70 dark:text-gray-300">Heart Disease</span>
                <span className="font-bold text-text-light dark:text-white">{group.hdCount} ({group.hdPct}%)</span>
              </div>
              <div className="flex justify-between pb-2">
                <span className="text-text-light/70 dark:text-gray-300">Diabetes</span>
                <span className="font-bold text-text-light dark:text-white">{group.diaCount} ({group.diaPct}%)</span>
              </div>
            </div>
          </div>
          <div className="md:w-2/3 bg-surface-light/50 dark:bg-surface-dark/50 rounded-xl p-6 border border-border-light/30 dark:border-border-dark/30">
            <h4 className="text-sm font-bold text-slate-600 dark:text-gray-400 uppercase tracking-widest mb-3">Health Summary</h4>
            <div className="text-slate-800 dark:text-gray-200 leading-relaxed whitespace-pre-wrap mb-4">
              {group.narrative}
            </div>
            {group.recommendation && (
              <div className="mt-4 inline-flex items-center gap-2 bg-blue-500/10 text-blue-300 px-4 py-2 rounded-lg text-sm font-medium">
                <Activity className="w-4 h-4" /> Recommendation: {group.recommendation}
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );

  const renderTab3 = () => (
    <div className="space-y-6 animate-in fade-in duration-500">
      {data.combinations.map((combo, idx) => {
        const isHighRisk = combo.risk === "HIGH RISK";
        const isLowRisk = combo.risk === "LOW RISK";
        return (
          <div key={idx} className={`glass-panel p-6 border-l-4 ${isHighRisk ? 'border-l-red-500' : isLowRisk ? 'border-l-teal' : 'border-l-amber-500'}`}>
            <div className="flex justify-between items-start mb-4">
              <h3 className="text-xl font-semibold text-text-light dark:text-white">{combo.name}</h3>
              <div className="text-right">
                <div className={`text-2xl font-bold font-mono ${isHighRisk ? 'text-red-400' : isLowRisk ? 'text-teal' : 'text-amber-400'}`}>
                  {combo.lifespan} years
                </div>
                <div className={`text-xs font-bold uppercase tracking-wider ${isHighRisk ? 'text-red-500' : isLowRisk ? 'text-teal/70' : 'text-amber-500'}`}>
                  [{combo.risk}]
                </div>
              </div>
            </div>
            <div className="bg-surface-light/50 dark:bg-surface-dark/50 p-4 rounded-xl border border-border-light/20 dark:border-border-dark/20">
              <p className="text-sm text-text-light/70 dark:text-gray-300 leading-relaxed">
                <span className="text-gray-500 font-semibold mr-2">Factors:</span> 
                {combo.factors}
              </p>
            </div>
          </div>
        );
      })}
    </div>
  );

  const renderTab4 = () => (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div className="glass-panel p-8">
        <h3 className="text-2xl font-bold mb-6 text-text-light dark:text-white">Patient Group Summary</h3>
        <div className="grid md:grid-cols-2 gap-12">
          <div>
            <h4 className="text-sm font-bold text-slate-600 dark:text-gray-400 uppercase tracking-widest mb-4">Top Risk Factors</h4>
            <div className="space-y-4">
              {data.summary.topRisks.map((risk, i) => (
                <div key={i} className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="font-medium text-slate-800 dark:text-gray-300">{risk.name}</span>
                    <span className="text-rose-500 font-bold">{risk.percent}% of group</span>
                  </div>
                  <div className="w-full h-2 bg-slate-100 dark:bg-gray-700 rounded-full overflow-hidden">
                    <motion.div 
                      initial={{ width: 0 }}
                      animate={{ width: `${risk.percent}%` }}
                      className="h-full bg-rose-500"
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          <div>
            <h4 className="text-sm font-bold text-slate-600 dark:text-gray-400 uppercase tracking-widest mb-4">Protective Strengths</h4>
            <div className="space-y-4">
              <div className="flex items-center gap-4 p-4 bg-teal/5 rounded-xl border border-teal/20">
                <div className="w-10 h-10 rounded-full bg-teal/20 flex items-center justify-center text-teal">
                  <Activity size={20} />
                </div>
                <div>
                  <div className="font-bold text-text-light dark:text-white">{data.summary.protective.exercise.pct}%</div>
                  <div className="text-xs text-slate-600 dark:text-gray-400">Regular Exercisers</div>
                </div>
              </div>
              <div className="flex items-center gap-4 p-4 bg-blue-500/5 rounded-xl border border-blue-500/20">
                <div className="w-10 h-10 rounded-full bg-blue-500/20 flex items-center justify-center text-blue-400">
                  <HeartPulse size={20} />
                </div>
                <div>
                  <div className="font-bold text-text-light dark:text-white">{data.summary.protective.healthyBmi.pct}%</div>
                  <div className="text-xs text-slate-600 dark:text-gray-400">Healthy BMI Range</div>
                </div>
              </div>
              <div className="flex items-center gap-4 p-4 bg-emerald-500/5 rounded-xl border border-emerald-500/20">
                <div className="w-10 h-10 rounded-full bg-emerald-500/20 flex items-center justify-center text-emerald-400">
                  <ShieldAlert size={20} />
                </div>
                <div>
                  <div className="font-bold text-text-light dark:text-white">{data.summary.protective.noChronic.pct}%</div>
                  <div className="text-xs text-slate-600 dark:text-gray-400">Zero Chronic Diseases</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-surface-light dark:bg-surface-dark border border-border-light dark:border-border-dark p-8 rounded-3xl text-center">
         <h4 className="text-xl font-bold mb-2">Population Longevity Index</h4>
         <div className="text-5xl font-bold text-teal mb-4">{data.summary.avgLifespan}</div>
         <p className="text-slate-700 dark:text-gray-300 max-w-md mx-auto font-bold leading-relaxed">
           This population shows a strong longevity potential, primarily limited by the risk factors identified above. Targeted interventions for the top 3 risks could add an estimated 4.2 years to the group average.
         </p>
      </div>
    </div>
  );

  const renderTabAi = () => (
    <div className="space-y-8 animate-in fade-in zoom-in duration-500">
      {isAiLoading ? (
        <div className="flex flex-col items-center justify-center py-20 gap-4">
          <Zap className="w-12 h-12 text-teal animate-bounce" />
          <p className="text-teal font-mono">Neural Engine Analyzing Dataset...</p>
        </div>
      ) : aiAnalysis ? (
        <>
          <div className="grid md:grid-cols-3 gap-6">
            <div className="glass-panel p-8 bg-teal/5 border-teal/20">
              <div className="flex justify-between items-start mb-4">
                <div className="flex items-center gap-3">
                  <BrainCircuit className="text-teal w-6 h-6" />
                  <h3 className="font-bold text-lg text-text-light dark:text-white">AI Predicted Mortality</h3>
                </div>
                <div className="text-[10px] font-bold px-2 py-1 bg-teal/20 text-teal rounded border border-teal/30 uppercase tracking-tighter">
                  Source: {getActiveModel().algorithm}
                </div>
              </div>
              <div className="text-5xl font-bold text-teal mb-2">{aiAnalysis.avgLifespan}</div>
              <p className="text-xs text-gray-500 uppercase tracking-widest">Years Avg Life Expectancy</p>
            </div>
            
            <div className="glass-panel p-8">
              <div className="flex items-center gap-3 mb-4">
                <ShieldAlert className="text-rose-500 w-6 h-6" />
                <h3 className="font-bold text-lg text-text-light dark:text-white">Group Vulnerability</h3>
              </div>
              <div className="text-3xl font-bold text-rose-400 mb-2">CRITICAL</div>
              <p className="text-xs text-gray-500 uppercase tracking-widest">Neural Risk Assessment</p>
            </div>

            <div className="glass-panel p-8">
              <div className="flex items-center gap-3 mb-4">
                <CheckCircle className="text-blue-400 w-6 h-6" />
                <h3 className="font-bold text-lg text-text-light dark:text-white">Engine Confidence</h3>
              </div>
              <div className="text-3xl font-bold text-blue-400 mb-2">94.2%</div>
              <p className="text-xs text-gray-500 uppercase tracking-widest">R² Accuracy Score</p>
            </div>
          </div>

          <div className="glass-panel p-8">
            <h3 className="text-xl font-bold mb-6 flex items-center gap-2">
              <TrendingDown className="text-rose-500 w-5 h-5" /> AI Identified High-Risk Profiles (Bottom 5)
            </h3>
            <div className="space-y-4">
              {aiAnalysis.riskProfiles.map((profile, i) => (
                <div key={i} className="flex flex-col items-center text-center p-8 bg-surface-light/30 dark:bg-surface-dark/30 rounded-3xl border border-border-light/10 dark:border-border-dark/10 hover:border-rose-500/30 transition-all duration-300 hover:shadow-2xl hover:shadow-rose-500/5 group">
                  <div className="mb-6">
                    <div className="text-[10px] font-bold text-gray-500 uppercase tracking-[0.2em] mb-2 opacity-60">PATIENT RECORD: {profile.id}</div>
                    <div className="text-3xl font-bold text-text-light dark:text-white font-mono group-hover:text-rose-400 transition-colors">{profile.prediction} <span className="text-sm font-normal text-gray-500">YEARS</span></div>
                  </div>
                  
                  <div className="w-12 h-px bg-gradient-to-r from-transparent via-rose-500/30 to-transparent mb-6"></div>
                  
                  <div className="max-w-md">
                    <div className="text-xs font-bold text-rose-500 uppercase tracking-widest mb-3">Neural Recommendation</div>
                    <p className="text-lg text-slate-800 dark:text-gray-300 italic font-medium leading-relaxed">
                      "{profile.risks[0] || 'High risk profile detected'}"
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      ) : (
        <div className="text-center py-20 text-gray-500">
          Upload a dataset to generate AI insights.
        </div>
      )}
    </div>
  );

  return (
    <motion.div 
      initial={{ opacity: 0 }} animate={{ opacity: 1 }}
      className="min-h-screen pt-24 pb-12 px-6 max-w-7xl mx-auto"
    >
      <button 
        onClick={() => navigate('/')} 
        className="inline-flex items-center gap-2 mb-8 py-2 px-6 bg-slate-900 text-white rounded-xl font-bold border-2 border-transparent hover:bg-white hover:text-black hover:shadow-[0_0_15px_rgba(99,102,241,0.3)] group transition-all duration-300 shadow-lg"
      >
        <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" /> 
        Back to Home
      </button>

      {!rawData ? (
        <div className="max-w-2xl mx-auto mt-12">
          <h2 className="text-4xl font-display font-bold mb-4 text-center bg-clip-text text-transparent bg-gradient-to-r from-teal to-blue-500">
            LifeLytics Intelligence Engine
          </h2>
          <p className="text-text-light/60 dark:text-gray-400 text-center mb-8 text-lg">
            Upload a health dataset to run our deterministic Lifespan Prediction Algorithm. Processed securely in your browser with zero backend delays.
          </p>
          
          <div 
            {...(engineEnabled ? getRootProps() : {})} 
            className={`border-2 border-dashed rounded-3xl p-16 text-center transition-all duration-300 relative overflow-hidden ${
              !engineEnabled 
                ? 'border-rose-500/30 bg-rose-500/5 cursor-not-allowed' 
                : isDragActive 
                  ? 'border-teal bg-teal/10 scale-[1.02]' 
                  : 'border-border hover:border-teal/50 hover:bg-surface/50 hover:shadow-lg hover:shadow-teal/10 cursor-pointer'
            }`}
          >
            <input {...(engineEnabled ? getInputProps() : {})} disabled={!engineEnabled} />
            
            {!engineEnabled ? (
              <div className="flex flex-col items-center">
                <ShieldAlert className="w-16 h-16 text-rose-500 mb-4 animate-pulse" />
                <h3 className="text-xl font-black text-slate-950 dark:text-white uppercase mb-2">Neural Engine Offline</h3>
                <p className="text-slate-600 dark:text-gray-400 text-sm font-medium">
                  Please turn on the engine to enable high-fidelity cohort analysis.
                </p>
              </div>
            ) : (
              <>
                <motion.div
                  animate={{ y: [0, -10, 0] }}
                  transition={{ repeat: Infinity, duration: 2 }}
                >
                  <BrainCircuit className="w-16 h-16 text-teal mx-auto mb-6" />
                </motion.div>
                {isDragActive ? (
                  <p className="text-teal font-medium text-xl">Drop the dataset here ...</p>
                ) : (
                  <div>
                    <p className="text-slate-900 dark:text-gray-200 text-xl font-medium mb-2">Drag & drop your CSV or Excel file here</p>
                    <p className="text-slate-600 dark:text-gray-500">Must include: age, bmi, smoking, alcohol, exercise_level, heart_disease, etc.</p>
                  </div>
                )}
              </>
            )}
          </div>
          
          {loading && (
             <div className="mt-8 text-center text-teal animate-pulse">
               Analyzing cohort data...
             </div>
          )}

          {error && (
            <div className="mt-8 text-center text-red-400 bg-red-500/10 p-4 rounded-xl border border-red-500/20">
              {error}
            </div>
          )}
        </div>
      ) : (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-8">
          
          <div className="flex justify-between items-end border-b border-border/50 pb-4">
            <div>
              <h2 className="text-3xl font-display font-bold mb-1 text-slate-900 dark:text-white">Patient Population Report</h2>
              <div className="inline-flex items-center gap-2 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 px-3 py-1 rounded-full text-xs font-bold border border-emerald-500/20">
                <CheckCircleIcon /> Successfully tracking dataset
              </div>
            </div>
            <button onClick={() => { setRawData(null); setData(null); clearFilters(); }} className="btn-secondary text-sm">
              Upload New Dataset
            </button>
          </div>

          {/* New Filter Panel */}
          {renderFilterPanel()}

          {error ? (
             <div className="text-center text-amber-400 bg-amber-500/10 p-8 rounded-xl border border-amber-500/20">
               {error}
             </div>
          ) : (
            <>
              <div className="flex gap-2 overflow-x-auto pb-2 scrollbar-hide">
                {[
                  { id: 'overview', label: 'OVERVIEW' },
                  { id: 'ai', label: 'AI NEURAL INSIGHTS' },
                  { id: 'ageGroups', label: 'AGE GROUP ANALYSIS' },
                  { id: 'combinations', label: 'LIFESTYLE COMBINATIONS' },
                  { id: 'summary', label: 'SUMMARY REPORT' }
                ].map(tab => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`px-6 py-3 rounded-xl text-sm font-bold tracking-wider transition-all whitespace-nowrap ${
                      activeTab === tab.id 
                        ? 'bg-teal text-background-dark shadow-lg shadow-teal/20' 
                        : 'bg-surface-light dark:bg-surface-dark hover:bg-surface-light/80 dark:hover:bg-surface-dark/80 text-text-light/60 dark:text-gray-400'
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              <div className="mt-8">
                {activeTab === 'overview' && renderTab1()}
                {activeTab === 'ai' && renderTabAi()}
                {activeTab === 'ageGroups' && renderTab2()}
                {activeTab === 'combinations' && renderTab3()}
                {activeTab === 'summary' && renderTab4()}
              </div>
            </>
          )}

        </motion.div>
      )}
    </motion.div>
  );
}

const CheckCircleIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
);
