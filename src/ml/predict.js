import { calculateLifespan } from './healthPredictEngine';
import { predictLife, explain } from './gbdtEngine';
import { predictLinear } from './linearModel';
import defaultModelData from './model.json';

let activeModel = defaultModelData;

export const loadModel = async (onProgress) => {
  try {
    const savedModel = localStorage.getItem('lifespanx_custom_model');
    if (savedModel) {
      activeModel = JSON.parse(savedModel);
      console.log("Loaded custom model from storage:", activeModel.version);
    } else {
      activeModel = defaultModelData;
    }
  } catch (e) {
    console.error("Failed to load custom model", e);
    activeModel = defaultModelData;
  }
  
  if (onProgress) onProgress({ 
    model: activeModel.algorithm || 'AI Engine (GBDT)', 
    current: 100, 
    total: 100 
  });
  return true;
};

export const getActiveModel = () => activeModel;

const clamp = (val, min, max) => Math.min(max, Math.max(min, val));

/**
 * Fast, deterministic baseline prediction (Instant)
 */
export const predictLifespanFast = (userData) => {
  const prediction = predictLinear(userData);
  const currentAge = parseFloat(userData.age) || 30;
  
  return {
    prediction: Math.round(prediction * 10) / 10,
    biologicalAge: parseFloat((currentAge + 80 - prediction).toFixed(1)),
    base: userData.gender === 'female' ? 82 : 79,
    featureImportance: [],
    recommendations: { positive: ["AI Insights Loading..."], negative: ["Calibrating Engine..."] },
    survivalCurve: [],
    confidenceInterval: [prediction - 4, prediction + 4],
    modelUsed: 'Linear ML Baseline',
    isBaseline: true
  };
};

export const predictLifespan = async (userData) => {
  try {
    // 1. Try to fetch from Flask API
    const response = await fetch('http://localhost:5001/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(userData)
    });

    if (response.ok) {
      const apiResult = await response.json();
      const prediction = apiResult.prediction;
      const currentAge = parseFloat(userData.age) || 30;

      return {
        prediction,
        biologicalAge: parseFloat((currentAge + 80 - prediction).toFixed(1)),
        base: userData.gender === 'female' ? 82 : 79,
        featureImportance: [], // Could be fetched from API
        recommendations: { positive: ["AI Insights from Cloud Engine"], negative: ["Advanced Analysis Complete"] },
        survivalCurve: [],
        confidenceInterval: apiResult.confidence_interval,
        modelUsed: 'Advanced Cloud ML (RF)',
        isApi: true
      };
    }
    
    // 2. Fallback to Local GBDT Engine if API fails
    const result = predictLife(activeModel, userData);
    const prediction = result.predicted_lifespan;
    const explanation = explain(userData, prediction);
    const currentAge = parseFloat(userData.age) || 30;

    return { 
      prediction,
      biologicalAge: parseFloat((currentAge + 80 - prediction).toFixed(1)),
      base: userData.gender === 'female' ? 82 : 79,
      featureImportance: [
        ...explanation.risk_factors.map(r => ({ name: r.factor, impact: parseFloat(r.impact) })),
        ...explanation.protective_factors.map(p => ({ name: p.factor, impact: parseFloat(p.benefit) }))
      ],
      explanation, 
      recommendations: { 
        positive: explanation.protective_factors.map(f => `${f.factor}: ${f.benefit}`),
        negative: explanation.improvements.map(i => `${i.action} (+${i.gain_years} yrs)`)
      },
      survivalCurve: [],
      confidenceInterval: [result.confidence_low, result.confidence_high],
      modelUsed: 'Local GBDT Hybrid' 
    };
  } catch (e) {
    console.warn("API/Hybrid inference failed, falling back to Linear Baseline:", e);
    return predictLifespanFast(userData);
  }
};
