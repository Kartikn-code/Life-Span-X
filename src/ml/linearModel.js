/**
 * Phase 1: Baseline Linear Regression Model
 * Ported from Python scikit-learn training
 */

const WEIGHTS = {
  intercept: 90.1042,
  coefficients: {
    age: 0.2024,
    gender: -1.9113,
    bmi: -0.2885,
    exercise: 2.2686,
    smoking: -7.3728,
    alcohol: -3.1649,
    systolic: -0.0656,
    cholesterol: -0.0230,
    glucose: -0.0350
  }
};

/**
 * Predict lifespan using the learned linear equation
 */
export function predictLinear(userData) {
  // Normalize inputs
  const x = {
    age: parseFloat(userData.age) || 40,
    gender: (userData.gender === 'male' || userData.gender === 'Male' || userData.gender === 1) ? 1 : 0,
    bmi: parseFloat(userData.bmi) || 25,
    exercise: parseInt(userData.exercise_level || userData.exercise || 0),
    smoking: (userData.smoking === 'yes' || userData.smoking === 1 || userData.smoking === true) ? 1 : 0,
    alcohol: (userData.alcohol === 'yes' || userData.alcohol === 1 || userData.alcohol === true) ? 1 : 0,
    systolic: parseInt(userData.blood_pressure || userData.systolic || 120),
    cholesterol: parseInt(userData.cholesterol || 200),
    glucose: parseInt(userData.glucose || 90)
  };

  let prediction = WEIGHTS.intercept;
  
  for (const [feature, weight] of Object.entries(WEIGHTS.coefficients)) {
    prediction += (x[feature] || 0) * weight;
  }

  // Safety clamp
  const currentAge = parseFloat(userData.age) || 30;
  return Math.max(currentAge + 1, Math.min(100, prediction));
}

export const getLinearWeights = () => WEIGHTS.coefficients;
