export const calculateActuarialLifespan = (userData) => {
  let base = 70.2;
  let modifiers = {};

  // Age calculation and basic mapping
  const age = parseFloat(userData.age) || 30;
  
  // BMI calculation
  const height = parseFloat(userData.height) || 170;
  const weight = parseFloat(userData.weight) || 70;
  const bmi = weight / Math.pow(height / 100, 2);

  // Smoking
  if (userData.smoking === '1-10') modifiers.smoking = -3.5;
  else if (userData.smoking === '11-20') modifiers.smoking = -6.8;
  else if (userData.smoking === '20+') modifiers.smoking = -10.2;
  else if (userData.smoking === 'ex') modifiers.smoking = -1.8;
  else modifiers.smoking = 0;

  // Alcohol
  if (userData.alcohol === '8-14') modifiers.alcohol = -2.1;
  else if (userData.alcohol === '15-21') modifiers.alcohol = -4.9;
  else if (userData.alcohol === '21+') modifiers.alcohol = -8.3;
  else modifiers.alcohol = 0;

  // BMI
  if (bmi >= 35) modifiers.bmi = -5.4;
  else if (bmi >= 30) modifiers.bmi = -2.8;
  else if (bmi >= 25) modifiers.bmi = -1.2;
  else modifiers.bmi = 0;

  // Exercise
  const exDays = parseInt(userData.exercise_freq) || 0;
  if (exDays === 0) modifiers.exercise = -3.1;
  else if (exDays <= 2) modifiers.exercise = -0.9;
  else if (exDays >= 6) modifiers.exercise = 2.6;
  else if (exDays >= 4) modifiers.exercise = 1.8;
  else modifiers.exercise = 0;

  // Sleep
  const sleep = parseFloat(userData.sleep_hours) || 7;
  if (sleep < 6) modifiers.sleep = -2.4;
  else if (sleep > 9) modifiers.sleep = -1.1;
  else if (sleep >= 7 && sleep <= 8) modifiers.sleep = 0.9;
  else modifiers.sleep = 0;

  // Stress
  const stress = parseInt(userData.stress) || 5;
  if (stress >= 8) modifiers.stress = -2.7;
  else modifiers.stress = 0;

  // Social
  const social = parseInt(userData.social) || 3;
  if (social < 3) modifiers.social = -1.9;
  else modifiers.social = 0;

  // Mental Health (PHQ-2)
  const phq2 = (parseInt(userData.phq2_1) || 0) + (parseInt(userData.phq2_2) || 0);
  if (phq2 >= 3) modifiers.mental = -1.8;
  else modifiers.mental = 0;

  // Chronic Conditions
  const conditions = userData.conditions || [];
  let chronicMod = 0;
  if (conditions.includes('diabetes')) chronicMod -= 4.2;
  if (conditions.includes('hypertension')) chronicMod -= 2.6;
  if (conditions.includes('heart_disease')) chronicMod -= 5.8;
  modifiers.chronic = chronicMod;

  // Diet
  let dietMod = 0;
  if (userData.diet === 'vegetarian') dietMod += 1.4;
  else if (userData.diet === 'vegan') dietMod += 1.8;
  
  if (parseFloat(userData.fruit_veg) >= 5) dietMod += 1.1;
  if (parseFloat(userData.water) >= 2.5) dietMod += 0.6;
  modifiers.diet = dietMod;

  // Family History & Genetics
  let geneticMod = 0;
  const gpAvg = parseFloat(userData.grandparent_avg) || 70.2;
  geneticMod += (gpAvg - 70.2) * 0.15; // Family history bonus/penalty

  const familyConds = userData.family_conditions || [];
  if (familyConds.includes('heart_disease') && userData.gender === 'male') geneticMod -= 1.8; // Father heart disease approximation
  if (familyConds.includes('diabetes') && userData.gender === 'female') geneticMod -= 2.1; // Mother diabetes approx
  
  const pAge1 = parseFloat(userData.father_age) || 0;
  const pAge2 = parseFloat(userData.mother_age) || 0;
  if (pAge1 >= 80 && pAge2 >= 80) geneticMod += 2.4;

  modifiers.genetics = geneticMod;

  // Income
  if (userData.income === 'low') modifiers.income = -1.2;
  else if (userData.income === 'high') modifiers.income = 0.8;
  else modifiers.income = 0;

  // Sum all modifiers
  let totalModifier = Object.values(modifiers).reduce((acc, val) => acc + val, 0);

  // Age correction: if you are already older, you have survived some risks
  // Simple heuristic: add back a portion of the risk if you've lived past 60
  let ageCorrection = 0;
  if (age > 60 && totalModifier < 0) {
    ageCorrection = Math.abs(totalModifier) * ((age - 60) / 40); 
  }

  let finalPrediction = base + totalModifier + ageCorrection;

  // Clamp
  finalPrediction = Math.max(age + 1, Math.max(40, Math.min(100, finalPrediction)));

  return {
    prediction: parseFloat(finalPrediction.toFixed(1)),
    base,
    modifiers,
    totalModifier: parseFloat(totalModifier.toFixed(1))
  };
};

// SHAP-style weights for feature importance (visualization)
export const featureWeights = {
  smoking: 0.18, 
  alcohol: 0.14, 
  exercise: 0.12,
  bmi: 0.11, 
  stress: 0.09, 
  sleep: 0.08,
  diet: 0.07, 
  chronic: 0.09, 
  genetics: 0.06,
  mental: 0.06
};
