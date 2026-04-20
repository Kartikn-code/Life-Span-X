import { calculateActuarialLifespan } from './actuarial';
import * as tf from '@tensorflow/tfjs';

let model = null;

export const loadModel = async () => {
  try {
    // For demo purposes, we catch the error gracefully if the model doesn't exist
    model = await tf.loadLayersModel('/model/model.json');
    console.log("TensorFlow.js model loaded successfully.");
  } catch (error) {
    console.warn("TensorFlow.js model not found at /model/model.json. Falling back to Actuarial Engine.");
    model = null;
  }
};

export const predictLifespan = async (userData) => {
  const actuarialResult = calculateActuarialLifespan(userData);
  
  if (model) {
    try {
      // In a real scenario, map userData to the correct feature tensor
      // This is a placeholder for the actual tensor creation
      // const features = [userData.age, userData.bmi, ...];
      // const tensor = tf.tensor2d([features]);
      // const prediction = model.predict(tensor);
      // const tfResult = prediction.dataSync()[0];
      
      // return { ...actuarialResult, prediction: tfResult, modelUsed: 'TensorFlow Neural Net' };
    } catch (e) {
      console.error("Inference error", e);
    }
  }
  
  return { ...actuarialResult, modelUsed: 'Actuarial Formula' };
};
