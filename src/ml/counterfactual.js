import { calculateActuarialLifespan } from './actuarial';

export const runSimulation = (baseUserData, simulationChanges) => {
  // Merge base user data with the simulation changes
  // Note: For complex fields like smoking/alcohol which use string bands in Onboarding
  // but sliders in Simulate, we need to map the numeric slider back to the band string 
  // or handle it in actuarial.js. Since actuarial.js expects the strings from Onboarding,
  // we will map the slider values here before calculating.

  const simData = { ...baseUserData };

  // Map Cigarettes per day slider (0-40) to bands
  if (simulationChanges.cigarettes !== undefined) {
    const cigs = simulationChanges.cigarettes;
    if (cigs === 0) simData.smoking = baseUserData.smoking === 'never' ? 'never' : 'ex';
    else if (cigs <= 10) simData.smoking = '1-10';
    else if (cigs <= 20) simData.smoking = '11-20';
    else simData.smoking = '20+';
  }

  // Map Alcohol units (0-35)
  if (simulationChanges.alcohol_units !== undefined) {
    const alc = simulationChanges.alcohol_units;
    if (alc === 0) simData.alcohol = '0';
    else if (alc <= 7) simData.alcohol = '1-7';
    else if (alc <= 14) simData.alcohol = '8-14';
    else if (alc <= 21) simData.alcohol = '15-21';
    else simData.alcohol = '21+';
  }

  // Exercise days (0-7)
  if (simulationChanges.exercise_days !== undefined) {
    simData.exercise_freq = simulationChanges.exercise_days.toString();
  }

  // Sleep hours (4-10)
  if (simulationChanges.sleep !== undefined) {
    simData.sleep_hours = simulationChanges.sleep.toString();
  }

  // Stress (1-10)
  if (simulationChanges.stress !== undefined) {
    simData.stress = simulationChanges.stress.toString();
  }

  // Fruit & Veg (0-10)
  if (simulationChanges.fruit_veg !== undefined) {
    simData.fruit_veg = simulationChanges.fruit_veg.toString();
  }

  // Water (0-4)
  if (simulationChanges.water !== undefined) {
    simData.water = simulationChanges.water.toString();
  }

  // BMI (15-45) -> we overwrite height/weight to hit the target BMI
  if (simulationChanges.bmi !== undefined) {
    const targetBmi = simulationChanges.bmi;
    const heightM = (parseFloat(simData.height) || 170) / 100;
    simData.weight = (targetBmi * (heightM * heightM)).toString();
  }

  const result = calculateActuarialLifespan(simData);
  return result;
};
