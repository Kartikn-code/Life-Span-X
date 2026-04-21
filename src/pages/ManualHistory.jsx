import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Download, FileSpreadsheet, Trash2 } from 'lucide-react';
import { motion } from 'framer-motion';
import Papa from 'papaparse';
import * as XLSX from 'xlsx';
import { useUser } from '../context/UserContext';
import { supabase } from '../lib/supabase';

export default function ManualHistory() {
  const navigate = useNavigate();
  const { updateUserData } = useUser();
  const [history, setHistory] = useState([]);

  const [fetchError, setFetchError] = useState(null);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        console.log("Fetching from Supabase...");
        const { data, error } = await supabase
          .from('patient_records')
          .select('*')
          .order('id', { ascending: true });
          
        console.log("Supabase response:", { data, error });
        
        if (error) {
          setFetchError(error.message);
          return;
        }
        
        if (data) {
          const formattedData = data.map(row => ({
            ...row,
            userData: row.userdata || {}
          }));
          setHistory(formattedData);
        }
      } catch (err) {
        console.error("Fetch Exception:", err);
        setFetchError(err.message);
      }
    };
    fetchHistory();
  }, []);

  const clearHistory = async () => {
    if (window.confirm("Are you sure you want to clear ALL patient records globally?")) {
      const { error } = await supabase
        .from('patient_records')
        .delete()
        .neq('id', '0'); // deletes everything
        
      if (!error) {
        setHistory([]);
      }
    }
  };

  const deleteSingleRecord = async (id) => {
    if (window.confirm("Delete this specific record?")) {
      const { error } = await supabase
        .from('patient_records')
        .delete()
        .eq('id', id);
        
      if (!error) {
        setHistory(prev => prev.filter(record => record.id !== id));
      } else {
        setFetchError("Failed to delete record: " + error.message);
      }
    }
  };

  const loadRecord = (record) => {
    updateUserData(record.userData);
    navigate('/dashboard');
  };

  const getExportData = () => {
    return history.map(row => ({
      age: row.userData.age || 50,
      gender: row.userData.gender || 'male',
      bmi: row.userData.bmi || 25,
      exercise_level: row.userData.exercise_level || '0',
      smoking: row.userData.smoking || '0',
      alcohol: row.userData.alcohol || '0',
      blood_pressure: row.userData.blood_pressure || 120,
      cholesterol: row.userData.cholesterol || 200,
      glucose: row.userData.glucose || 100,
      fatigue: row.userData.fatigue || '0',
      chest_pain: row.userData.chest_pain || '0',
      dizziness: row.userData.dizziness || '0',
      heart_disease: row.userData.heart_disease || '0',
      diabetes: row.userData.diabetes || '0',
      stroke: row.userData.stroke || '0',
      health_risk_score: row.score ? Math.round(row.score) : 50
    }));
  };

  const exportCSV = () => {
    const data = getExportData();
    const csv = Papa.unparse(data);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `LifeLytics_Patient_Records_${new Date().toISOString().split('T')[0]}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const exportExcel = () => {
    const data = getExportData();
    const worksheet = XLSX.utils.json_to_sheet(data);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, "Patient Records");
    XLSX.writeFile(workbook, `LifeLytics_Patient_Records_${new Date().toISOString().split('T')[0]}.xlsx`);
  };

  return (
    <motion.div 
      initial={{ opacity: 0 }} animate={{ opacity: 1 }}
      className="min-h-screen pt-24 pb-12 px-6 max-w-7xl mx-auto"
    >
      <div className="flex justify-between items-center mb-8">
        <div>
          <button onClick={() => navigate('/dashboard')} className="text-teal hover:underline flex items-center gap-2 mb-4">
            <ArrowLeft className="w-4 h-4" /> Back to Dashboard
          </button>
          <h1 className="text-3xl font-display font-bold">Patient Records</h1>
          <p className="text-gray-400 mt-2">Historical manual predictions and cohort entries.</p>
        </div>

        <div className="flex gap-4">
          <button onClick={exportCSV} className="btn-secondary flex items-center gap-2 py-2 text-sm" disabled={history.length === 0}>
            <Download className="w-4 h-4" /> Export CSV
          </button>
          <button onClick={exportExcel} className="btn-primary flex items-center gap-2 py-2 text-sm" disabled={history.length === 0}>
            <FileSpreadsheet className="w-4 h-4" /> Export Excel
          </button>
          <button onClick={clearHistory} className="bg-danger/10 text-danger border border-danger/30 hover:bg-danger/20 px-4 py-2 rounded-full transition-colors flex items-center gap-2" disabled={history.length === 0}>
            <Trash2 className="w-4 h-4" /> Clear
          </button>
        </div>
      </div>

      <div className="glass-panel overflow-hidden border border-border">
        {fetchError && (
          <div className="p-4 bg-danger/10 border-b border-danger/30 text-danger text-center">
            Database Connection Error: {fetchError}
          </div>
        )}
        
        {history.length === 0 ? (
          <div className="p-12 text-center text-gray-400">
            No manual patient records found. Complete a manual entry to start saving history.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-surface/50 border-b border-border/50 text-xs uppercase tracking-wider text-gray-400">
                  <th className="p-4 font-semibold">Date</th>
                  <th className="p-4 font-semibold">Demographics</th>
                  <th className="p-4 font-semibold">Vitals</th>
                  <th className="p-4 font-semibold">Lifestyle</th>
                  <th className="p-4 font-semibold">Predicted Lifespan</th>
                  <th className="p-4 font-semibold">Risk Score</th>
                  <th className="p-4 font-semibold text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/30">
                {history.map((record) => {
                  const bmi = record.userData.weight && record.userData.height 
                    ? (record.userData.weight / Math.pow(record.userData.height/100, 2)).toFixed(1) 
                    : '--';
                  return (
                    <tr key={record.id} className="hover:bg-surface/30 transition-colors">
                      <td className="p-4 text-sm text-gray-300">
                        {new Date(record.date).toLocaleDateString()} <br/>
                        <span className="text-xs text-gray-500">{new Date(record.date).toLocaleTimeString()}</span>
                      </td>
                      <td className="p-4">
                        <div className="text-white font-medium capitalize">{record.userData.age} yrs, {record.userData.gender}</div>
                        <div className="text-xs text-gray-400">BMI: {bmi}</div>
                      </td>
                      <td className="p-4 text-sm text-gray-300">
                        BP: {record.userData.blood_pressure || '--'} <br/>
                        Chol: {record.userData.cholesterol || '--'}
                      </td>
                      <td className="p-4 text-sm text-gray-300">
                        Ex Level: {record.userData.exercise_level} <br/>
                        Smoke: {record.userData.smoking === '1' ? 'Yes' : 'No'} | Alc: {record.userData.alcohol === '1' ? 'Yes' : 'No'}
                      </td>
                      <td className="p-4">
                        <span className="text-xl font-bold font-mono text-teal">{record.prediction}</span>
                        <span className="text-xs text-gray-400 ml-1">yrs</span>
                      </td>
                      <td className="p-4">
                        <div className="w-full bg-surface rounded-full h-2 mb-1">
                          <div className={`h-2 rounded-full ${record.score < 50 ? 'bg-danger' : record.score < 75 ? 'bg-amber' : 'bg-teal'}`} style={{ width: `${record.score}%` }}></div>
                        </div>
                        <span className="text-xs text-gray-400">{record.score.toFixed(1)} / 100</span>
                      </td>
                      <td className="p-4 text-right">
                        <div className="flex flex-col items-end gap-3">
                          <button onClick={() => loadRecord(record)} className="text-teal hover:underline text-sm font-medium">
                            Load Profile
                          </button>
                          <button onClick={() => deleteSingleRecord(record.id)} className="text-danger hover:text-red-400 text-xs flex items-center gap-1 transition-colors">
                            <Trash2 className="w-3 h-3" /> Delete
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </motion.div>
  );
}
