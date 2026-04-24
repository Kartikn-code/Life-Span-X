import { predictLifespan } from './predict';

/**
 * Runs a "what-if" simulation by applying changes to base user data
 * and calling the real ML prediction engine.
 */
export async function runSimulation(baseUserData, simulationChanges) {
  // Merge changes into base data
  const simulatedData = {
    ...baseUserData,
    ...simulationChanges
  };

  // Call the main prediction pipeline (API -> Local -> Heuristic)
  const result = await predictLifespan(simulatedData);
  
  return {
    ...result,
    isSimulation: true
  };
}
