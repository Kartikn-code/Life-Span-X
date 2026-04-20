import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { ChevronRight, ChevronLeft, Check } from 'lucide-react';
import { useUser } from '../context/UserContext';

const steps = [
  { id: 1, title: 'Basic Info' },
  { id: 2, title: 'Physical Health' },
  { id: 3, title: 'Lifestyle' },
  { id: 4, title: 'Diet & Mental Health' },
  { id: 5, title: 'Family History' }
];

export default function Onboarding() {
  const navigate = useNavigate();
  const { userData, updateUserData } = useUser();
  const [currentStep, setCurrentStep] = useState(1);
  const [formData, setFormData] = useState({
    // Step 1
    age: userData.age || '',
    gender: userData.gender || 'male',
    location: userData.location || '',
    occupation: userData.occupation || 'office',
    income: userData.income || 'mid',
    education: userData.education || 'bachelors',
    // Step 2
    height: userData.height || '',
    weight: userData.weight || '',
    resting_hr: userData.resting_hr || '',
    systolic_bp: userData.systolic_bp || '',
    conditions: userData.conditions || [],
    // Step 3
    alcohol: userData.alcohol || '0',
    smoking: userData.smoking || 'never',
    exercise_type: userData.exercise_type || 'none',
    exercise_freq: userData.exercise_freq || '0',
    exercise_duration: userData.exercise_duration || '0',
    sleep_hours: userData.sleep_hours || '7',
    sleep_quality: userData.sleep_quality || '3',
    // Step 4
    diet: userData.diet || 'omnivore',
    fruit_veg: userData.fruit_veg || '3',
    water: userData.water || '2',
    processed_food: userData.processed_food || 'few week',
    phq2_1: userData.phq2_1 || '0',
    phq2_2: userData.phq2_2 || '0',
    stress: userData.stress || '5',
    social: userData.social || '3',
    // Step 5
    father_status: userData.father_status || 'alive',
    father_age: userData.father_age || '',
    mother_status: userData.mother_status || 'alive',
    mother_age: userData.mother_age || '',
    grandparent_avg: userData.grandparent_avg || '75',
    family_conditions: userData.family_conditions || []
  });

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    if (type === 'checkbox') {
      const currentList = formData[name];
      if (checked) {
        setFormData({ ...formData, [name]: [...currentList, value] });
      } else {
        setFormData({ ...formData, [name]: currentList.filter(item => item !== value) });
      }
    } else {
      setFormData({ ...formData, [name]: value });
    }
  };

  const nextStep = () => {
    updateUserData(formData);
    if (currentStep < steps.length) {
      setCurrentStep(currentStep + 1);
    } else {
      // Analyze data and redirect
      // In a real app we might want a loading state here
      navigate('/dashboard');
    }
  };

  const prevStep = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  };

  // Live BMI calculation
  const bmi = formData.height && formData.weight 
    ? (formData.weight / Math.pow(formData.height / 100, 2)).toFixed(1) 
    : '--';

  const renderStep = () => {
    switch(currentStep) {
      case 1:
        return (
          <div className="space-y-6">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-400 mb-1">Age</label>
                <input type="number" name="age" value={formData.age} onChange={handleChange} className="w-full bg-surface border border-border rounded-lg p-3 text-white focus:border-teal outline-none" placeholder="e.g. 35" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-400 mb-1">Gender</label>
                <select name="gender" value={formData.gender} onChange={handleChange} className="w-full bg-surface border border-border rounded-lg p-3 text-white focus:border-teal outline-none">
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                  <option value="other">Other</option>
                </select>
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-1">State / Location</label>
              <input type="text" name="location" value={formData.location} onChange={handleChange} className="w-full bg-surface border border-border rounded-lg p-3 text-white focus:border-teal outline-none" placeholder="e.g. Maharashtra" />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-400 mb-1">Occupation Type</label>
                <select name="occupation" value={formData.occupation} onChange={handleChange} className="w-full bg-surface border border-border rounded-lg p-3 text-white focus:border-teal outline-none">
                  <option value="office">Office / Sedentary</option>
                  <option value="active">Active / Field</option>
                  <option value="manual">Heavy Manual Labor</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-400 mb-1">Income Band</label>
                <select name="income" value={formData.income} onChange={handleChange} className="w-full bg-surface border border-border rounded-lg p-3 text-white focus:border-teal outline-none">
                  <option value="low">Lower</option>
                  <option value="mid">Middle</option>
                  <option value="high">Upper</option>
                </select>
              </div>
            </div>
          </div>
        );
      case 2:
        return (
          <div className="space-y-6">
            <div className="grid grid-cols-2 gap-4 items-end">
              <div>
                <label className="block text-sm font-medium text-gray-400 mb-1">Height (cm)</label>
                <input type="number" name="height" value={formData.height} onChange={handleChange} className="w-full bg-surface border border-border rounded-lg p-3 text-white focus:border-teal outline-none" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-400 mb-1">Weight (kg)</label>
                <input type="number" name="weight" value={formData.weight} onChange={handleChange} className="w-full bg-surface border border-border rounded-lg p-3 text-white focus:border-teal outline-none" />
              </div>
            </div>
            <div className="bg-navy/50 p-4 rounded-lg flex justify-between items-center border border-border/50">
              <span className="text-gray-300">Live BMI Calculation:</span>
              <span className={`text-2xl font-bold ${bmi > 25 ? 'text-amber' : bmi > 30 ? 'text-danger' : 'text-teal'}`}>{bmi}</span>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-400 mb-1">Resting HR (bpm)</label>
                <input type="number" name="resting_hr" value={formData.resting_hr} onChange={handleChange} className="w-full bg-surface border border-border rounded-lg p-3 text-white focus:border-teal outline-none" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-400 mb-1">Systolic BP (optional)</label>
                <input type="number" name="systolic_bp" value={formData.systolic_bp} onChange={handleChange} className="w-full bg-surface border border-border rounded-lg p-3 text-white focus:border-teal outline-none" placeholder="e.g. 120" />
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-3">Chronic Conditions (Check all that apply)</label>
              <div className="grid grid-cols-2 gap-2">
                {['diabetes', 'hypertension', 'asthma', 'heart_disease', 'none'].map(cond => (
                  <label key={cond} className="flex items-center space-x-2 bg-surface p-3 rounded-lg border border-border cursor-pointer hover:border-teal/50">
                    <input type="checkbox" name="conditions" value={cond} checked={formData.conditions.includes(cond)} onChange={handleChange} className="accent-teal w-4 h-4" />
                    <span className="capitalize">{cond.replace('_', ' ')}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>
        );
      case 3:
        return (
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-1">Alcohol (units/week)</label>
              <select name="alcohol" value={formData.alcohol} onChange={handleChange} className="w-full bg-surface border border-border rounded-lg p-3 text-white focus:border-teal outline-none">
                <option value="0">0 (Teetotaler)</option>
                <option value="1-7">1-7 (Light)</option>
                <option value="8-14">8-14 (Moderate)</option>
                <option value="15-21">15-21 (Heavy)</option>
                <option value="21+">21+ (Very Heavy)</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-1">Smoking</label>
              <select name="smoking" value={formData.smoking} onChange={handleChange} className="w-full bg-surface border border-border rounded-lg p-3 text-white focus:border-teal outline-none">
                <option value="never">Never</option>
                <option value="ex">Ex-smoker</option>
                <option value="1-10">1-10 / day</option>
                <option value="11-20">11-20 / day</option>
                <option value="20+">20+ / day</option>
              </select>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-400 mb-1">Exercise Type</label>
                <select name="exercise_type" value={formData.exercise_type} onChange={handleChange} className="w-full bg-surface border border-border rounded-lg p-3 text-white focus:border-teal outline-none">
                  <option value="none">None</option>
                  <option value="walking">Walking</option>
                  <option value="gym">Gym / Weights</option>
                  <option value="sports">Sports</option>
                  <option value="mixed">Mixed</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-400 mb-1">Freq (days/wk)</label>
                <input type="number" name="exercise_freq" min="0" max="7" value={formData.exercise_freq} onChange={handleChange} className="w-full bg-surface border border-border rounded-lg p-3 text-white focus:border-teal outline-none" />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-400 mb-1">Sleep (hrs/night)</label>
                <input type="number" name="sleep_hours" step="0.5" value={formData.sleep_hours} onChange={handleChange} className="w-full bg-surface border border-border rounded-lg p-3 text-white focus:border-teal outline-none" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-400 mb-1">Sleep Quality (1-5)</label>
                <input type="range" name="sleep_quality" min="1" max="5" value={formData.sleep_quality} onChange={handleChange} className="w-full mt-2 accent-teal" />
                <div className="text-center text-sm mt-1">{formData.sleep_quality}</div>
              </div>
            </div>
          </div>
        );
      case 4:
        return (
          <div className="space-y-6">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-400 mb-1">Diet Type</label>
                <select name="diet" value={formData.diet} onChange={handleChange} className="w-full bg-surface border border-border rounded-lg p-3 text-white focus:border-teal outline-none">
                  <option value="omnivore">Omnivore</option>
                  <option value="vegetarian">Vegetarian</option>
                  <option value="vegan">Vegan</option>
                  <option value="keto">Keto</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-400 mb-1">Water (L/day)</label>
                <input type="number" name="water" step="0.5" value={formData.water} onChange={handleChange} className="w-full bg-surface border border-border rounded-lg p-3 text-white focus:border-teal outline-none" />
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-1">Stress Level (1-10)</label>
              <input type="range" name="stress" min="1" max="10" value={formData.stress} onChange={handleChange} className="w-full mt-2 accent-teal" />
              <div className="text-center text-sm mt-1">{formData.stress}</div>
            </div>
            <div className="bg-navy/50 p-4 rounded-lg border border-border/50">
              <h4 className="text-white mb-2 font-medium">PHQ-2 Mental Health Screen</h4>
              <p className="text-xs text-gray-400 mb-4">Over the last 2 weeks, how often have you been bothered by:</p>
              
              <label className="block text-sm font-medium text-gray-300 mb-1">1. Little interest or pleasure in doing things</label>
              <select name="phq2_1" value={formData.phq2_1} onChange={handleChange} className="w-full bg-surface border border-border rounded-lg p-2 mb-3 text-white text-sm focus:border-teal outline-none">
                <option value="0">Not at all</option>
                <option value="1">Several days</option>
                <option value="2">More than half the days</option>
                <option value="3">Nearly every day</option>
              </select>

              <label className="block text-sm font-medium text-gray-300 mb-1">2. Feeling down, depressed, or hopeless</label>
              <select name="phq2_2" value={formData.phq2_2} onChange={handleChange} className="w-full bg-surface border border-border rounded-lg p-2 text-white text-sm focus:border-teal outline-none">
                <option value="0">Not at all</option>
                <option value="1">Several days</option>
                <option value="2">More than half the days</option>
                <option value="3">Nearly every day</option>
              </select>
            </div>
          </div>
        );
      case 5:
        return (
          <div className="space-y-6">
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-navy/50 p-4 rounded-lg border border-border/50">
                <label className="block text-sm font-medium text-gray-300 mb-1">Father</label>
                <select name="father_status" value={formData.father_status} onChange={handleChange} className="w-full bg-surface border border-border rounded-lg p-2 mb-2 text-white text-sm focus:border-teal outline-none">
                  <option value="alive">Alive</option>
                  <option value="deceased">Deceased</option>
                </select>
                <input type="number" name="father_age" placeholder={formData.father_status === 'alive' ? "Current Age" : "Age at death"} value={formData.father_age} onChange={handleChange} className="w-full bg-surface border border-border rounded-lg p-2 text-white text-sm focus:border-teal outline-none" />
              </div>
              <div className="bg-navy/50 p-4 rounded-lg border border-border/50">
                <label className="block text-sm font-medium text-gray-300 mb-1">Mother</label>
                <select name="mother_status" value={formData.mother_status} onChange={handleChange} className="w-full bg-surface border border-border rounded-lg p-2 mb-2 text-white text-sm focus:border-teal outline-none">
                  <option value="alive">Alive</option>
                  <option value="deceased">Deceased</option>
                </select>
                <input type="number" name="mother_age" placeholder={formData.mother_status === 'alive' ? "Current Age" : "Age at death"} value={formData.mother_age} onChange={handleChange} className="w-full bg-surface border border-border rounded-lg p-2 text-white text-sm focus:border-teal outline-none" />
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-1">Grandparents Average Lifespan (approx)</label>
              <input type="number" name="grandparent_avg" value={formData.grandparent_avg} onChange={handleChange} className="w-full bg-surface border border-border rounded-lg p-3 text-white focus:border-teal outline-none" />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-400 mb-3">Family History (Parents/Siblings)</label>
              <div className="grid grid-cols-2 gap-2">
                {['diabetes', 'heart_disease', 'cancer', 'stroke'].map(cond => (
                  <label key={cond} className="flex items-center space-x-2 bg-surface p-3 rounded-lg border border-border cursor-pointer hover:border-teal/50">
                    <input type="checkbox" name="family_conditions" value={cond} checked={formData.family_conditions.includes(cond)} onChange={handleChange} className="accent-teal w-4 h-4" />
                    <span className="capitalize">{cond.replace('_', ' ')}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen pt-24 pb-12 px-6 flex items-center justify-center relative">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-teal/5 via-navy to-navy"></div>
      
      <div className="w-full max-w-2xl z-10">
        <div className="mb-8 flex justify-between items-center relative">
          <div className="absolute left-0 top-1/2 -translate-y-1/2 w-full h-1 bg-surface z-0"></div>
          <div className="absolute left-0 top-1/2 -translate-y-1/2 h-1 bg-teal transition-all duration-500 z-0" style={{ width: `${((currentStep - 1) / (steps.length - 1)) * 100}%` }}></div>
          
          {steps.map(step => (
            <div key={step.id} className="relative z-10 flex flex-col items-center">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold border-2 transition-colors duration-300 ${
                currentStep > step.id 
                  ? 'bg-teal border-teal text-navy' 
                  : currentStep === step.id 
                    ? 'bg-surface border-teal text-teal'
                    : 'bg-surface border-border text-gray-500'
              }`}>
                {currentStep > step.id ? <Check className="w-4 h-4" /> : step.id}
              </div>
              <span className={`absolute top-10 text-xs whitespace-nowrap hidden md:block ${currentStep === step.id ? 'text-teal' : 'text-gray-500'}`}>
                {step.title}
              </span>
            </div>
          ))}
        </div>

        <div className="glass-panel p-8 md:p-10">
          <h2 className="text-3xl font-display font-bold mb-6 text-white">{steps[currentStep-1].title}</h2>
          
          <AnimatePresence mode="wait">
            <motion.div
              key={currentStep}
              initial={{ x: 20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: -20, opacity: 0 }}
              transition={{ duration: 0.3 }}
            >
              {renderStep()}
            </motion.div>
          </AnimatePresence>

          <div className="mt-10 flex justify-between">
            <button 
              onClick={prevStep}
              className={`btn-secondary flex items-center gap-2 ${currentStep === 1 ? 'invisible' : ''}`}
            >
              <ChevronLeft className="w-5 h-5" /> Back
            </button>
            
            <button 
              onClick={nextStep}
              className="btn-primary flex items-center gap-2"
            >
              {currentStep === steps.length ? 'Analyze Data' : 'Next'} <ChevronRight className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
