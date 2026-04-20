import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useUser } from '../context/UserContext';
import { generatePlan } from '../ml/recommend';
import { ArrowLeft, CheckCircle2, Circle } from 'lucide-react';
import { motion } from 'framer-motion';

export default function ActionPlan() {
  const navigate = useNavigate();
  const { userData, predictions } = useUser();
  const [plan, setPlan] = useState(null);
  const [progress, setProgress] = useState({});

  useEffect(() => {
    if (!userData || !predictions) {
      navigate('/dashboard');
      return;
    }

    const generated = generatePlan(predictions.modifiers, userData.age, "Stressed Professional");
    setPlan(generated);

    // Load progress from localStorage
    const savedProgress = localStorage.getItem('lifespan_plan_progress');
    if (savedProgress) {
      setProgress(JSON.parse(savedProgress));
    }
  }, [userData, predictions, navigate]);

  const toggleTask = (taskId) => {
    const newProgress = { ...progress, [taskId]: !progress[taskId] };
    setProgress(newProgress);
    localStorage.setItem('lifespan_plan_progress', JSON.stringify(newProgress));
  };

  if (!plan) return null;

  const totalTasks = Object.values(plan).reduce((acc, week) => acc + week.tasks.length, 0);
  const completedTasks = Object.values(progress).filter(Boolean).length;
  const completionPercentage = totalTasks > 0 ? Math.round((completedTasks / totalTasks) * 100) : 0;

  return (
    <motion.div 
      initial={{ opacity: 0 }} animate={{ opacity: 1 }}
      className="min-h-screen pt-24 pb-12 px-6 max-w-4xl mx-auto"
    >
      <button onClick={() => navigate('/dashboard')} className="text-teal hover:underline flex items-center gap-2 mb-8">
        <ArrowLeft className="w-4 h-4" /> Back to Dashboard
      </button>

      <div className="flex flex-col md:flex-row justify-between items-end mb-12 gap-6">
        <div>
          <h2 className="text-4xl font-display font-bold mb-2 text-white">Your 12-Week Protocol</h2>
          <p className="text-gray-400">Personalised habit roadmap based on your top modifiable risks.</p>
        </div>
        
        <div className="glass-panel p-4 flex items-center gap-6 min-w-[200px]">
          <div>
            <div className="text-sm text-gray-400">Progress</div>
            <div className="text-2xl font-bold text-teal">{completionPercentage}%</div>
          </div>
          <div className="flex-1 h-2 bg-surface rounded-full overflow-hidden">
            <motion.div 
              initial={{ width: 0 }} 
              animate={{ width: `${completionPercentage}%` }} 
              className="h-full bg-teal"
            ></motion.div>
          </div>
        </div>
      </div>

      <div className="space-y-8 relative before:absolute before:inset-0 before:ml-[1.2rem] before:-translate-x-px md:before:mx-auto md:before:translate-x-0 before:h-full before:w-0.5 before:bg-gradient-to-b before:from-teal/50 before:via-border before:to-transparent">
        {Object.entries(plan).map(([weekSpan, details], index) => (
          <div key={weekSpan} className="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group is-active">
            
            <div className="flex items-center justify-center w-10 h-10 rounded-full border-4 border-navy bg-teal text-navy font-bold shrink-0 md:order-1 md:group-odd:-translate-x-1/2 md:group-even:translate-x-1/2 shadow-[0_0_0_4px_rgba(0,245,212,0.2)]">
              {index + 1}
            </div>
            
            <div className="w-[calc(100%-4rem)] md:w-[calc(50%-2.5rem)] glass-panel p-6">
              <h3 className="text-teal font-bold mb-1">{weekSpan}</h3>
              <h4 className="text-xl font-semibold text-white mb-4">{details.title}</h4>
              
              <ul className="space-y-3">
                {details.tasks.map(task => (
                  <li key={task.id} className="flex items-start gap-3 cursor-pointer group/task" onClick={() => toggleTask(task.id)}>
                    <div className="mt-1">
                      {progress[task.id] ? (
                        <CheckCircle2 className="w-5 h-5 text-teal" />
                      ) : (
                        <Circle className="w-5 h-5 text-gray-500 group-hover/task:text-teal transition-colors" />
                      )}
                    </div>
                    <span className={`text-sm leading-relaxed ${progress[task.id] ? 'text-gray-500 line-through' : 'text-gray-300'}`}>
                      {task.text}
                    </span>
                  </li>
                ))}
              </ul>
            </div>

          </div>
        ))}
      </div>
      
    </motion.div>
  );
}
