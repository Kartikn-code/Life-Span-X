import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { Send, User, Bot, Loader2, Camera, ShieldAlert } from 'lucide-react';
import { useUser } from '../context/UserContext';
// import BioSelfieScan from '../components/BioSelfieScan';

const CHAT_QUESTIONS = [
  { key: 'age', text: "Welcome to LifeLytics AI. To begin your assessment, what is your age?", type: 'number', placeholder: 'e.g. 35', min: 18, max: 120 },
  { key: 'gender', text: "What is your biological gender?", type: 'select', options: [{ label: 'Male', value: 'male' }, { label: 'Female', value: 'female' }] },
  {
    key: 'country', text: "What country do you live in? (Used for dietary and geographic baselines)", type: 'select', options: [
      { label: 'United States', value: 'USA' }, { label: 'United Kingdom', value: 'UK' },
      { label: 'India', value: 'India' }, { label: 'Canada', value: 'Canada' },
      { label: 'Australia', value: 'Australia' }, { label: 'Other', value: 'Other' }
    ]
  },
  { key: 'height', text: "What is your height in centimeters?", type: 'number', placeholder: 'e.g. 175', min: 50, max: 250 },
  { key: 'weight', text: "What is your weight in kilograms?", type: 'number', placeholder: 'e.g. 70', min: 20, max: 300 },
  { key: 'blood_pressure', text: "What is your typical systolic blood pressure? (If unsure, enter 120)", type: 'number', placeholder: 'e.g. 120', min: 70, max: 250 },
  { key: 'cholesterol', text: "What is your total cholesterol level? (If unsure, enter 180)", type: 'number', placeholder: 'e.g. 180', min: 100, max: 500 },
  { key: 'glucose', text: "What is your fasting glucose level? (If unsure, enter 90)", type: 'number', placeholder: 'e.g. 90', min: 40, max: 400 },
  { key: 'exercise_level', text: "How often do you exercise?", type: 'select', options: [{ label: 'Sedentary (No exercise)', value: '0' }, { label: 'Light (1-2x/wk)', value: '1' }, { label: 'Moderate (3-4x/wk)', value: '2' }, { label: 'High (5+x/wk)', value: '3' }] },
  { key: 'smoking', text: "Do you currently smoke?", type: 'boolean' },
  { key: 'alcohol', text: "Do you regularly consume alcohol?", type: 'boolean' },
  { key: 'sleep_hours', text: "On average, how many hours do you sleep per night?", type: 'number', placeholder: 'e.g. 7', min: 2, max: 15 },
  { key: 'stress_level', text: "How would you rate your daily stress level? (1-5)", type: 'select', options: [{ label: '1 - Minimal', value: '1' }, { label: '2 - Low', value: '2' }, { label: '3 - Moderate', value: '3' }, { label: '4 - High', value: '4' }, { label: '5 - Extreme', value: '5' }] },
  { key: 'fruit_intake', text: "How many servings of fruit do you eat daily?", type: 'number', placeholder: 'e.g. 2', min: 0, max: 10 },
  { key: 'vegetable_intake', text: "How many servings of vegetables do you eat daily?", type: 'number', placeholder: 'e.g. 3', min: 0, max: 12 },
  { key: 'processed_food', text: "How often do you consume processed/fast food?", type: 'select', options: [{ label: 'Rarely/Never', value: '0' }, { label: 'Occasionally (1-2x/wk)', value: '1' }, { label: 'Frequently (3+x/wk)', value: '2' }] },
  { key: 'heart_disease', text: "Have you ever been diagnosed with heart disease?", type: 'boolean' },
  { key: 'diabetes', text: "Have you been diagnosed with diabetes?", type: 'boolean' },
  { key: 'stroke', text: "Have you ever suffered a stroke?", type: 'boolean' }
];

export default function Onboarding() {
  const navigate = useNavigate();
  const { updateUserData, engineEnabled } = useUser();

  const [messages, setMessages] = useState(() => {
    const saved = localStorage.getItem('onboarding_messages');
    return saved ? JSON.parse(saved) : [];
  });
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(() => {
    const saved = localStorage.getItem('onboarding_index');
    return saved ? parseInt(saved) : 0;
  });
  const [formData, setFormData] = useState(() => {
    const saved = localStorage.getItem('onboarding_form');
    return saved ? JSON.parse(saved) : {};
  });
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [showBioScan, setShowBioScan] = useState(false);
  const messagesEndRef = useRef(null);

  // Initial greeting or resume
  useEffect(() => {
    if (messages.length === 0) {
      setIsTyping(true);
      setTimeout(() => {
        setMessages([{ sender: 'bot', text: CHAT_QUESTIONS[0].text }]);
        setIsTyping(false);
      }, 1000);
    }
  }, []);

  // Save progress
  useEffect(() => {
    localStorage.setItem('onboarding_messages', JSON.stringify(messages));
    localStorage.setItem('onboarding_index', currentQuestionIndex.toString());
    localStorage.setItem('onboarding_form', JSON.stringify(formData));
  }, [messages, currentQuestionIndex, formData]);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  const handleNextQuestion = (value, displayValue) => {
    const currentQ = CHAT_QUESTIONS[currentQuestionIndex];

    // Validation Logic
    if (currentQ.type === 'number') {
      const num = parseFloat(value);
      if (isNaN(num)) {
        setMessages(prev => [...prev, { sender: 'user', text: displayValue }, { sender: 'bot', text: "Please enter a valid number." }]);
        setInputValue('');
        return;
      }
      if ((currentQ.min !== undefined && num < currentQ.min) || (currentQ.max !== undefined && num > currentQ.max)) {
        setMessages(prev => [
          ...prev,
          { sender: 'user', text: displayValue },
          { sender: 'bot', text: `That value seems out of range. Please enter a value between ${currentQ.min} and ${currentQ.max}.` }
        ]);
        setInputValue('');
        return;
      }
    }

    // Add user message
    setMessages(prev => [...prev, { sender: 'user', text: displayValue }]);

    // Save data
    const newFormData = { ...formData, [currentQ.key]: value };
    setFormData(newFormData);
    setInputValue('');
    setIsTyping(true);

    const nextIndex = currentQuestionIndex + 1;

    setTimeout(() => {
      /* 
      // Trigger Bio Scan after weight is entered
      if (currentQ.key === 'weight') {
        setShowBioScan(true);
        return;
      }
      */

      if (nextIndex < CHAT_QUESTIONS.length) {
        setMessages(prev => [...prev, { sender: 'bot', text: CHAT_QUESTIONS[nextIndex].text }]);
        setCurrentQuestionIndex(nextIndex);
        setIsTyping(false);
      } else {
        setMessages(prev => [...prev, { sender: 'bot', text: "Thank you. Analysing your health markers now..." }]);

        // Calculate BMI and finalize
        setTimeout(() => {
          const bmi = newFormData.height && newFormData.weight
            ? (newFormData.weight / Math.pow(newFormData.height / 100, 2)).toFixed(1)
            : '25.0';

          const finalData = {
            ...newFormData,
            bmi: parseFloat(bmi),
            isNewEntry: true // FLAG FOR DASHBOARD
          };

          // Clear onboarding persistence
          localStorage.removeItem('onboarding_messages');
          localStorage.removeItem('onboarding_index');
          localStorage.removeItem('onboarding_form');

          updateUserData(finalData);
          navigate('/dashboard');
        }, 2000);
      }
    }, 800);
  };

  /*
  const handleBioScanComplete = (estimatedAge) => {
    setShowBioScan(false);
    setMessages(prev => [...prev, { sender: 'bot', text: `AI analysis complete. Estimated biological age: ${estimatedAge}. I've refined your profile with this biomarker.` }]);
    
    // Save the bio age and continue
    const newFormData = { ...formData, biological_age: estimatedAge };
    setFormData(newFormData);
    
    setIsTyping(true);
    const nextIndex = currentQuestionIndex + 1;
    setTimeout(() => {
      setMessages(prev => [...prev, { sender: 'bot', text: CHAT_QUESTIONS[nextIndex].text }]);
      setCurrentQuestionIndex(nextIndex);
      setIsTyping(false);
    }, 1000);
  };
  */

  const handleInputSubmit = (e) => {
    e.preventDefault();
    if (!inputValue.trim()) return;
    handleNextQuestion(inputValue, inputValue);
  };

  const currentQ = CHAT_QUESTIONS[currentQuestionIndex];

  return (
    <div className="min-h-screen pt-24 pb-12 px-4 md:px-6 max-w-3xl mx-auto flex flex-col">
      {/* 
      {showBioScan && (
        <BioSelfieScan 
          onScanComplete={handleBioScanComplete} 
          onCancel={() => {
            setShowBioScan(false);
            const nextIndex = currentQuestionIndex + 1;
            setMessages(prev => [...prev, { sender: 'bot', text: "No problem, we'll continue with standard data collection." }]);
            setIsTyping(true);
            setTimeout(() => {
              setMessages(prev => [...prev, { sender: 'bot', text: CHAT_QUESTIONS[nextIndex].text }]);
            ))}
            
            {isTyping && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex gap-4">
                <div className="w-8 h-8 rounded-full bg-blue-500/20 text-blue-400 flex items-center justify-center flex-shrink-0">
                  <Bot size={16} />
                </div>
                <div className="bg-surface-light dark:bg-surface-dark border border-border-light/50 dark:border-border-dark/50 rounded-2xl rounded-tl-none px-4 py-3 flex gap-1 items-center">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
      <div className="p-4 border-t border-border-light dark:border-border-dark/50 bg-surface-light/50 dark:bg-surface-dark/50 relative">
        {!engineEnabled ? (
          <div className="flex flex-col items-center justify-center py-4 text-center">
            <ShieldAlert className="w-8 h-8 text-rose-500 mb-2 animate-pulse" />
            <p className="text-sm font-bold text-slate-900 dark:text-white uppercase tracking-tight">Neural Engine Offline</p>
            <p className="text-[10px] text-slate-600 dark:text-gray-400 font-medium">Please turn on the engine to continue your assessment.</p>
          </div>
        ) : (
          !isTyping && currentQ && currentQuestionIndex < CHAT_QUESTIONS.length && (
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>

              {currentQ.type === 'boolean' && (
                <div className="flex gap-4 justify-end">
                  <button onClick={() => handleNextQuestion('1', 'Yes')} className="btn-secondary flex-1 border-teal/50 text-teal hover:bg-teal/10">Yes</button>
                  <button onClick={() => handleNextQuestion('0', 'No')} className="btn-secondary flex-1 border-gray-600 hover:bg-white/5">No</button>
                </div>
              )}

              {currentQ.type === 'select' && (
                <div className="flex flex-col gap-2">
                  {currentQ.options.map(opt => (
                    <button key={opt.value} onClick={() => handleNextQuestion(opt.value, opt.label)} className="btn-secondary w-full text-left justify-start">
                      {opt.label}
                    </button>
                  ))}
                </div>
              )}

              {currentQ.type === 'number' && (
                <form onSubmit={handleInputSubmit} className="flex gap-2">
                  <input
                    type="number"
                    autoFocus
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    placeholder={currentQ.placeholder}
                    min={currentQ.min}
                    max={currentQ.max}
                    className="flex-1 bg-surface-light dark:bg-surface-dark border border-border-light dark:border-border-dark rounded-xl px-4 py-3 text-text-light dark:text-text-dark focus:border-teal outline-none"
                  />
                  <button type="submit" disabled={!inputValue.trim()} className="bg-teal text-background-dark p-3 rounded-xl hover:bg-teal/90 disabled:opacity-50 disabled:cursor-not-allowed transition-all">
                    <Send size={20} />
                  </button>
                </form>
              )}
            </motion.div>
          )
        )}
      </div>
    </div>

  )
}
