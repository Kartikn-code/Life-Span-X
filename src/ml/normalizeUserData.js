/**
 * Maps Onboarding/UI form fields to backend-compatible 14-feature vector.
 * Canonical mapping for LifeSpanX ML Engine.
 */
export function normalizeUserData(userData) {
  if (!userData) return null;

  // Smoking: Handle both Onboarding codes and Simulation strings ('yes', 'never')
  const smokingBinary = (['1-10', '11-20', '20+'].includes(userData.smoking) || userData.smoking === 'yes' || userData.smoking === 1) ? 1 : 0;

  // Exercise: Onboarding uses exercise_freq (days/week 0-7), backend uses level 0-3
  const exDays = parseInt(userData.exercise_freq || userData.exercise_level || 0);
  const exerciseLevel = exDays === 0 ? 0 : exDays <= 2 ? 1 : exDays <= 4 ? 2 : 3;

  // BMI: compute from height/weight if bmi not directly available
  const height = parseFloat(userData.height) || 170;
  const weight = parseFloat(userData.weight) || 70;
  const bmi = userData.bmi ? parseFloat(userData.bmi) : weight / Math.pow(height / 100, 2);

  // Alcohol: Handle both Onboarding codes and Simulation strings ('yes', 'no')
  const alcoholBinary = (['8-14', '15-21', '21+'].includes(userData.alcohol) || userData.alcohol === 'yes' || userData.alcohol === 1) ? 1 : 0;

  // Conditions: check from conditions array OR direct binary fields
  const conditions = Array.isArray(userData.conditions) ? userData.conditions : [];
  const heartDisease = conditions.includes('heart_disease') || userData.heart_disease === 1 || userData.heart_disease === '1' ? 1 : 0;
  const diabetes = conditions.includes('diabetes') || userData.diabetes === 1 || userData.diabetes === '1' ? 1 : 0;
  const stroke = conditions.includes('stroke') || userData.stroke === 1 || userData.stroke === '1' ? 1 : 0;

  // Stress: map to 1-5
  const stress = parseInt(userData.stress || userData.stress_level || 3);

  return {
    age: parseFloat(userData.age) || 40,
    gender: userData.gender === 'male' || userData.gender === 1 ? 1 : 0,
    bmi: parseFloat(bmi.toFixed(1)),
    exercise_level: exerciseLevel,
    smoking: smokingBinary,
    alcohol: alcoholBinary,
    blood_pressure: parseFloat(userData.blood_pressure || userData.systolic || 120),
    cholesterol: parseFloat(userData.cholesterol || 200),
    glucose: parseFloat(userData.glucose || 90),
    heart_disease: heartDisease,
    diabetes: diabetes,
    stroke: stroke,
    sleep_hours: parseFloat(userData.sleep_hours || 7),
    stress_level: stress
  };
}
