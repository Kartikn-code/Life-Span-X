import { normalizeUserData } from './normalizeUserData';
import { predictLife } from './trainer';
import fallbackModel from './model.json';

// Dynamic API URL for production deployment
const ML_API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5001';
const FEATURES_JSON_URL = '/features.json'; 

let customModel = null;

/**
 * Loads the custom model from localStorage if it exists.
 * This model is trained via AILab.
 */
export const loadModel = async () => {
  try {
    const savedModel = localStorage.getItem('lifespanx_custom_model');
    if (savedModel) {
      customModel = JSON.parse(savedModel);
      console.log("Loaded custom model from storage:", customModel.version);
    }
  } catch (e) {
    console.error("Failed to load custom model", e);
  }
  return true;
};

/**
 * Built-in GBDT fallback using model.json.
 */
export const getFallbackPrediction = (userData) => {
  const norm = normalizeUserData(userData);
  if (!norm) return null;

  const result = predictLife(fallbackModel, norm);
  if (!result) return predictLifespanHeuristic(userData);

  return {
    prediction: result.predicted_lifespan,
    biologicalAge: Math.round((norm.age + (78 - result.predicted_lifespan)) * 10) / 10,
    featureImportance: Object.entries(fallbackModel.importances || {}).map(([name, impact]) => ({ 
      name: name.charAt(0).toUpperCase() + name.slice(1).replace('_', ' '), 
      impact 
    })),
    confidenceInterval: [result.confidence_low, result.confidence_high],
    modelUsed: 'Built-in GBDT (Fallback)',
    isOffline: true
  };
};

/**
 * Last-resort heuristic if GBDT fails.
 */
const predictLifespanHeuristic = (userData) => {
  const norm = normalizeUserData(userData);
  let prediction = norm.gender === 1 ? 78 : 82;
  prediction -= (norm.bmi - 22).clip(0, 20) * 0.4;
  prediction -= norm.smoking * 10;
  prediction += norm.exercise_level * 2;
  prediction -= (norm.blood_pressure - 120).clip(0, 60) * 0.1;
  
  return {
    prediction: Math.round(prediction * 10) / 10,
    biologicalAge: Math.round((norm.age + (78 - prediction)) * 10) / 10,
    featureImportance: {},
    confidenceInterval: [prediction - 5, prediction + 5],
    modelUsed: 'Heuristic Baseline',
    isOffline: true
  };
};

/**
 * Fast, instant prediction for UI responsiveness.
 */
export const predictLifespanFast = (userData) => {
  // 1. Try custom model
  if (customModel) {
    const norm = normalizeUserData(userData);
    const result = predictLife(customModel, norm);
    if (result) {
      return {
        prediction: result.predicted_lifespan,
        biologicalAge: Math.round((norm.age + (78 - result.predicted_lifespan)) * 10) / 10,
        featureImportance: Object.entries(customModel.importances || {}).map(([name, impact]) => ({ 
          name: name.charAt(0).toUpperCase() + name.slice(1).replace('_', ' '), 
          impact 
        })),
        confidenceInterval: [result.confidence_low, result.confidence_high],
        modelUsed: 'Local Custom Model (GBDT)',
        isOffline: true
      };
    }
  }
  
  // 2. Try built-in fallback
  return getFallbackPrediction(userData);
};

// Helper for heuristic
Number.prototype.clip = function(min, max) {
  return Math.min(max, Math.max(min, this));
};

/**
 * Batch prediction for cohorts: Flask API -> Parallel Local Inference
 */
export const predictBatch = async (users) => {
  const normalizedUsers = users.map(u => normalizeUserData(u));

  // 1. Try Flask API /predict-batch
  try {
    const response = await fetch(`${API_URL}/predict-batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(normalizedUsers)
    });

    if (response.ok) {
      return await response.json();
    }
  } catch (e) {
    console.warn("Batch API unavailable, falling back to parallel local inference", e);
  }

  // 2. Parallel Local Inference
  return Promise.all(users.map(u => predictLifespan(u)));
};

/**
 * Generates a survival probability curve based on the predicted lifespan.
 * Uses a sigmoid-style decay for realistic visualization.
 */
const generateSurvivalCurve = (currentAge, predictedLifespan) => {
  const curve = [];
  const startAge = Math.floor(currentAge / 5) * 5;
  const endAge = 105;
  
  for (let age = startAge; age <= endAge; age += 5) {
    // Probability starts at 100% and drops as we approach and pass predictedLifespan
    // Sigmoid: 1 / (1 + exp(k * (age - predictedLifespan)))
    const k = 0.2; // Steepness of the curve
    const probability = 100 / (1 + Math.exp(k * (age - predictedLifespan)));
    curve.push({ age, probability: Math.round(probability) });
  }
  return curve;
};

/**
 * Main prediction pipeline: Flask API -> Local Custom Model -> Heuristic Fallback
 */
export const predictLifespan = async (userData) => {
  const normalizedData = normalizeUserData(userData);
  const currentAge = normalizedData.age;
  
  // 1. Try Flask API
  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(normalizedData)
    });

    if (response.ok) {
      const result = await response.json();
      return {
        ...result,
        featureImportance: Object.entries(result.featureImportance || {}).map(([name, impact]) => ({ 
          name: name.charAt(0).toUpperCase() + name.slice(1).replace('_', ' '), 
          impact 
        })),
        survivalCurve: generateSurvivalCurve(currentAge, result.prediction),
        isApi: true
      };
    }
  } catch (e) {
    console.warn("Flask API unavailable, trying local models...", e);
  }

  // 2. Try Local Custom Model (from localStorage)
  if (customModel) {
    const result = predictLife(customModel, normalizedData);
    if (result) {
      const pred = result.predicted_lifespan;
      return {
        prediction: pred,
        biologicalAge: Math.round((currentAge + (78 - pred)) * 10) / 10,
        featureImportance: customModel.importances || {},
        confidenceInterval: [result.confidence_low, result.confidence_high],
        survivalCurve: generateSurvivalCurve(currentAge, pred),
        modelUsed: 'Local Custom Model (GBDT)',
        isOffline: true
      };
    }
  }

  // 3. Last Resort: Heuristic Fast Prediction
  const fastResult = predictLifespanFast(userData);
  return {
    ...fastResult,
    survivalCurve: generateSurvivalCurve(currentAge, fastResult.prediction)
  };
};
