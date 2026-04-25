/**
 * Maps Onboarding/UI form fields to backend-compatible 14-feature vector.
 * Canonical mapping for LifeSpanX ML Engine.
 */
export function normalizeUserData(userData) {
  if (!userData) return null;

  // Smoking: Handle codes (0-3), strings, or direct intensity (0.0-1.0)
  let smoking = 0;
  if (userData.smoking === '1-10') smoking = 0.3;
  else if (userData.smoking === '11-20') smoking = 0.6;
  else if (userData.smoking === '20+') smoking = 1.0;
  else if (userData.smoking === 'yes' || userData.smoking === true) smoking = 1.0;
  else if (!isNaN(parseFloat(userData.smoking))) smoking = parseFloat(userData.smoking);

  // Exercise: Handle 0-3 codes OR 0.0-1.0 intensity
  let exLevel = 0;
  if (!isNaN(parseFloat(userData.exercise_level || userData.exercise_freq))) {
      exLevel = parseFloat(userData.exercise_level || userData.exercise_freq);
      if (exLevel > 1) exLevel = exLevel / 7; // Normalize if it's raw days
  }

  // BMI: compute or parse
  const height = parseFloat(userData.height) || 170;
  const weight = parseFloat(userData.weight) || 70;
  const bmi = userData.bmi ? parseFloat(userData.bmi) : weight / Math.pow(height / 100, 2);

  // Alcohol: Handle codes (0-3), strings, or direct intensity (0.0-1.0)
  let alcohol = 0;
  if (userData.alcohol === '8-14') alcohol = 0.3;
  else if (userData.alcohol === '15-21') alcohol = 0.6;
  else if (userData.alcohol === '21+') alcohol = 1.0;
  else if (userData.alcohol === 'yes' || userData.alcohol === true) alcohol = 1.0;
  else if (!isNaN(parseFloat(userData.alcohol))) alcohol = parseFloat(userData.alcohol);

  // Conditions: check from conditions array OR direct binary fields
  const conditions = Array.isArray(userData.conditions) ? userData.conditions : [];
  const heartDisease = conditions.includes('heart_disease') || userData.heart_disease === 1 || userData.heart_disease === '1' ? 1 : 0;
  const diabetes = conditions.includes('diabetes') || userData.diabetes === 1 || userData.diabetes === '1' ? 1 : 0;
  const stroke = conditions.includes('stroke') || userData.stroke === 1 || userData.stroke === '1' ? 1 : 0;

  // Stress: map to 0.0-1.0 float
  let stress = 0.3; // Default
  if (!isNaN(parseFloat(userData.stress || userData.stress_level))) {
      stress = parseFloat(userData.stress || userData.stress_level);
      if (stress > 1) stress = (stress - 1) / 9; // Normalize if it's 1-10
  }

  return {
    age: parseFloat(userData.age) || 40,
    gender: userData.gender === 'male' || userData.gender === 1 ? 1 : 0,
    bmi: parseFloat(bmi.toFixed(1)),
    exercise_level: exLevel,
    smoking: smoking,
    alcohol: alcohol,
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
