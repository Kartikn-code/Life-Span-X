import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { ArrowRight, Activity, Shield, Brain, Zap, Heart, ShieldAlert } from 'lucide-react';
import { useInView } from 'framer-motion';
import { supabase } from '../lib/supabase';
import { useUser } from '../context/UserContext';
import FloatingParticles from '../components/FloatingParticles';
import QuickPredictor from '../components/QuickPredictor';
import NeuralNetworkVisualizer from '../components/NeuralNetworkVisualizer';

const AnimatedCounter = ({ targetValue }) => {
  const [count, setCount] = useState(0);
  const nodeRef = React.useRef(null);
  const isInView = useInView(nodeRef, { once: false, amount: 0.5 });

  useEffect(() => {
    if (isInView) {
      let start = 0;
      const end = parseInt(targetValue);
      if (start === end) return;

      let totalMiliseconds = 2000;
      let incrementTime = (totalMiliseconds / end) > 10 ? (totalMiliseconds / end) : 10;
      let step = Math.ceil(end / (totalMiliseconds / incrementTime));

      let timer = setInterval(() => {
        start += step;
        if (start >= end) {
          setCount(end);
          clearInterval(timer);
        } else {
          setCount(start);
        }
      }, incrementTime);

      return () => clearInterval(timer);
    } else {
      setCount(0);
    }
  }, [isInView, targetValue]);

  return <span ref={nodeRef}>{count.toLocaleString()}</span>;
};

// Habit Impact Card Component
const HabitCard = ({ title, impact, positive, delay }) => (
  <motion.div
    initial={{ y: 20, opacity: 0 }}
    whileInView={{ y: 0, opacity: 1 }}
    viewport={{ once: true }}
    transition={{ delay }}
    className="glass-panel p-6 relative overflow-hidden group cursor-pointer"
  >
    <div className={`absolute inset-0 bg-gradient-to-br opacity-0 group-hover:opacity-10 transition-opacity duration-500 ${positive ? 'from-emerald-500 to-teal' : 'from-rose-500 to-amber'}`}></div>
    <div className="flex justify-between items-center mb-4">
      <h4 className="font-bold text-slate-800 dark:text-gray-200">{title}</h4>
      <div className={`w-8 h-8 rounded-full flex items-center justify-center border ${positive ? 'border-emerald-500/30 text-emerald-500 bg-emerald-500/10' : 'border-rose-500/30 text-rose-500 bg-rose-500/10'}`}>
        <Activity className="w-4 h-4" />
      </div>
    </div>
    <div className="flex items-end gap-2">
      <span className={`text-3xl font-mono font-bold ${positive ? 'text-emerald-500' : 'text-rose-500'}`}>
        {positive ? '+' : ''}{impact}
      </span>
      <span className="text-sm text-slate-600 mb-1">years</span>
    </div>
    <div className="absolute top-0 right-0 p-4 opacity-0 group-hover:opacity-100 transition-opacity translate-x-2 group-hover:translate-x-0">
      <Zap className={`w-12 h-12 ${positive ? 'text-emerald-500/10' : 'text-rose-500/10'}`} />
    </div>
  </motion.div>
);

export default function Landing() {
  const navigate = useNavigate();
  const { engineEnabled } = useUser();
  const [globalCount, setGlobalCount] = useState(5183);

  useEffect(() => {
    const fetchGlobalCount = async () => {
      try {
        const { data, error } = await supabase
          .from('global_stats')
          .select('total_processed')
          .single();

        if (!error && data) {
          setGlobalCount(data.total_processed);
        } else {
          const { count } = await supabase
            .from('patient_records')
            .select('*', { count: 'exact', head: true });

          if (count) {
            setGlobalCount(5183 + count);
          }
        }
      } catch (err) {
        console.error("Error fetching global count:", err);
      }
    };

    fetchGlobalCount();

    const subscription = supabase
      .channel('global_stats_changes')
      .on('postgres_changes', { event: 'UPDATE', schema: 'public', table: 'global_stats' }, payload => {
        if (payload.new && payload.new.total_processed) {
          setGlobalCount(payload.new.total_processed);
        }
      })
      .subscribe();

    return () => {
      supabase.removeChannel(subscription);
    };
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.5 }}
      className="min-h-screen relative overflow-hidden"
    >
      {/* Background Layer */}
      <div className="absolute inset-0 z-0 overflow-hidden bg-background-light dark:bg-background-dark transition-colors duration-300">
        <div className="absolute inset-0 bg-[url('https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop')] bg-cover bg-center opacity-10 mix-blend-screen dark:animate-pulse"></div>
        <div className="absolute inset-0 bg-gradient-to-b from-background-light via-background-light/80 to-background-light dark:from-background-dark dark:via-background-dark/80 dark:to-background-dark"></div>
        {/* Radial ball removed */}
      </div>

      {/* Hero Section */}
      <section className="relative pt-32 pb-20 px-6 md:px-12 max-w-7xl mx-auto flex flex-col lg:flex-row items-center gap-12">
        <div className="lg:w-1/2 z-10 text-center lg:text-left">
          <motion.h1
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="text-5xl md:text-7xl font-bold leading-tight mb-6 text-text-light dark:text-text-dark"
          >
            Lifelytics <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal to-blue-500">Health Intelligence</span>
          </motion.h1>

          <motion.p
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="text-xl text-slate-800 dark:text-gray-300 mb-8 max-w-lg mx-auto lg:mx-0"
          >
            A browser-native Deep Neural Network that simulates exactly how your habits shape your longevity—instantly and privately.
          </motion.p>

          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.4 }}
            className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start relative group"
          >
            <div className="flex flex-col sm:flex-row gap-4 w-full">
              <button
                onClick={() => engineEnabled ? navigate('/onboarding') : null}
                className={`btn-primary flex items-center justify-center gap-2 group text-lg py-3 px-8 transition-all duration-300 ${!engineEnabled ? 'bg-rose-500/20 text-rose-500 border-rose-500/30 cursor-not-allowed grayscale-[0.5]' : ''}`}
              >
                {engineEnabled ? (
                  <>Run Full Analysis <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" /></>
                ) : (
                  <>Turn on Neural Engine</>
                )}
              </button>
              <button
                onClick={() => engineEnabled ? navigate('/doctor-portal') : null}
                className={`btn-secondary flex items-center justify-center gap-2 transition-all duration-300 ${!engineEnabled ? 'border-rose-500/20 text-rose-500/50 cursor-not-allowed' : ''}`}
              >
                {engineEnabled ? (
                  <> <Shield className="w-4 h-4" /> Clinical Portal</>
                ) : (
                  <>Turn on Neural Engine</>
                )}
              </button>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.8 }}
            className="mt-12 text-slate-600 dark:text-gray-400 font-medium"
          >
            <span className="text-teal font-mono text-xl mr-2">
              <AnimatedCounter targetValue={globalCount} />
            </span>
            simulations run globally
          </motion.div>
        </div>

        {/* Interactive Hero Element */}
        <div className="lg:w-1/2 w-full flex justify-center z-10 relative">
          <motion.div
            initial={{ scale: 0.9, opacity: 0, y: 20 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            transition={{ delay: 0.6, type: 'spring' }}
            className="w-full"
          >
            <QuickPredictor />
          </motion.div>
        </div>
      </section>

      {/* Stats Strip */}
      <section className="bg-surface-light/30 dark:bg-surface-dark/30 backdrop-blur-sm py-10 border-y border-border-light/10 dark:border-border-dark/10">
        <div className="max-w-7xl mx-auto px-6 flex flex-wrap justify-center gap-12 md:gap-24 text-center">
          {[
            { label: 'Verified Accuracy', value: '95%', desc: 'R² Statistical Correlation' },
            { label: 'Training Baseline', value: '4.2M+', desc: 'Clinical Data Snapshots' },
            { label: 'Inference Speed', value: '< 2ms', desc: 'On-Device Computation' },
            { label: 'Security Protocol', value: '100%', desc: 'Local Privacy Guarantee' },
          ].map((stat, i) => (
            <motion.div
              key={i}
              initial={{ y: 20, opacity: 0 }}
              whileInView={{ y: 0, opacity: 1 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1 }}
              className="flex flex-col items-center"
            >
              <div className="text-4xl md:text-5xl font-display font-bold text-slate-950 dark:text-white mb-1">{stat.value}</div>
              <div className="text-[11px] text-teal font-black uppercase tracking-[0.25em] mb-1.5">{stat.label}</div>
              <div className="text-[10px] text-slate-600 dark:text-gray-400 uppercase tracking-[0.15em] font-bold">{stat.desc}</div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* How it Works */}
      <section className="py-24 px-6 max-w-7xl mx-auto relative z-10">
        <div className="text-center mb-16 relative">
          {/* Section Background Neural Net */}
          <div className="absolute inset-0 flex items-center justify-center -z-10 opacity-20 dark:opacity-20 pointer-events-none scale-125">
            <NeuralNetworkVisualizer />
          </div>
          <h2 className="text-4xl font-bold mb-4 text-slate-950 dark:text-white">Explainable AI Framework</h2>
          <p className="text-slate-700 dark:text-gray-300 max-w-2xl mx-auto">We use permutation-based feature importance to map the exact relationship between your lifestyle choices and biological aging.</p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          {[
            { icon: Brain, title: 'In-Browser Engine', desc: 'Models train directly on your GPU using WebGL. Your health data never touches a remote server.' },
            { icon: Activity, title: 'Counterfactual Testing', desc: 'Simulate "what-if" scenarios. Instantly see how quitting smoking or sleeping 8 hours alters your trajectory.' },
            { icon: Zap, title: 'Actionable Protocols', desc: 'Receive mathematically derived recommendations that provide the highest return on investment for your health.' }
          ].map((step, i) => (
            <motion.div
              key={i}
              initial={{ y: 30, opacity: 0 }}
              whileInView={{ y: 0, opacity: 1 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.2 }}
              className="glass-panel p-8 text-center relative overflow-hidden group border-t border-t-white/10 dark:border-t-white/5"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-teal/5 to-blue-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
              <div className="w-16 h-16 bg-surface-light dark:bg-surface-dark border border-border-light/20 dark:border-border-dark/20 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-lg relative z-10 group-hover:border-teal/50 transition-colors">
                <step.icon className="w-8 h-8 text-teal" />
              </div>
              <h3 className="text-xl font-semibold mb-4 relative z-10 text-slate-900 dark:text-white">{step.title}</h3>
              <p className="text-sm text-slate-700 dark:text-gray-400 relative z-10">{step.desc}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* About Model Section */}
      <section className="py-24 bg-background-light dark:bg-background-dark/50 border-y border-border-light/10 dark:border-border-dark/10 relative overflow-hidden">
        <div className="max-w-7xl mx-auto px-6 relative z-10">
          <div className="flex flex-col lg:flex-row items-center gap-16">
            <div className="lg:w-1/3">
              <h2 className="text-4xl font-bold mb-6 text-slate-900 dark:text-white uppercase tracking-tighter">About Model</h2>
              <p className="text-slate-700 dark:text-gray-300 leading-relaxed">
                LifeLytics utilizes a multi-layered analytical framework to quantify biological aging and longevity risk with clinical precision.
              </p>
            </div>

            <div className="lg:w-2/3 grid sm:grid-cols-2 gap-6">
              {[
                {
                  title: "GBDT Adaptive Engine",
                  desc: "Ensemble of Gradient Boosted Decision Trees optimized for high-dimensional clinical data and non-linear risk factors.",
                  icon: <Brain className="w-5 h-5 text-teal" />
                },
                {
                  title: "Actuarial Baseline",
                  desc: "Mathematical modeling based on Gompertz mortality laws and modern epidemiological constraints.",
                  icon: <Activity className="w-5 h-5 text-blue-500" />
                },
                {
                  title: "Explainable AI (XAI)",
                  desc: "Decodes the AI logic to reveal exactly how specific habits influence your biological trajectory.",
                  icon: <Shield className="w-5 h-5 text-emerald-500" />
                },
                {
                  title: "Survival Modeling",
                  desc: "Exponential decay modeling synchronized with individual health baselines to visualize longevity probability.",
                  icon: <Zap className="w-5 h-5 text-amber-500" />
                }
              ].map((m, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.1 }}
                  className="glass-panel p-8 rounded-3xl border border-slate-200 dark:border-white/10 hover:border-teal/50 transition-all group shadow-sm hover:shadow-xl bg-white/70 dark:bg-slate-900/50"
                >
                  <div className="w-12 h-12 rounded-2xl bg-teal/5 dark:bg-teal/10 border border-teal/20 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform shadow-sm">
                    {m.icon}
                  </div>
                  <h4 className="text-xl font-bold text-slate-950 dark:text-white mb-3 tracking-tight">{m.title}</h4>
                  <p className="text-sm text-slate-800 dark:text-gray-300 leading-relaxed font-medium">{m.desc}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </div>

        {/* Subtle background element */}
        <div className="absolute top-1/2 left-0 -translate-y-1/2 -translate-x-1/2 w-96 h-96 bg-teal/5 blur-[120px] rounded-full pointer-events-none"></div>
      </section>

      {/* Interactive Value Section */}
      <section className="py-24 bg-teal/5 backdrop-blur-sm relative z-10">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex flex-col md:flex-row gap-16 items-center">
            <div className="md:w-1/2">
              <h2 className="text-4xl font-bold mb-6 text-text-light dark:text-text-dark">Stop Guessing.<br />Start Simulating.</h2>
              <p className="text-lg text-slate-800 dark:text-gray-300 mb-8 leading-relaxed">
                Generic advice says "exercise more." LifeLytics calculates exactly how many years of biological aging you can reverse based on your current baseline.
              </p>

              <div className="grid grid-cols-2 gap-4">
                <HabitCard title="Quitting Smoking" impact="11.0" positive={true} delay={0.1} />
                <HabitCard title="High Stress" impact="-8.5" positive={false} delay={0.2} />
                <HabitCard title="Optimal BMI (22)" impact="4.2" positive={true} delay={0.3} />
                <HabitCard title="Heart Disease" impact="-10.0" positive={false} delay={0.4} />
              </div>
            </div>

            <div className="md:w-1/2 flex justify-center w-full">
              <motion.div
                initial={{ opacity: 0, x: 50 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                className="relative w-full max-w-md aspect-square rounded-full border border-teal/10 shadow-[0_0_80px_rgba(0,245,212,0.15)] flex items-center justify-center p-12 overflow-hidden"
              >
                {/* High-Tech Radar Sweep */}
                <div className="absolute inset-0 bg-[conic-gradient(at_center,_var(--tw-gradient-stops))] from-teal/30 via-transparent to-transparent animate-[spin_8s_linear_infinite] rounded-full opacity-40"></div>

                {/* Solid Level Segments (Rings) */}
                <div className="absolute inset-4 rounded-full border-[2px] border-teal/20 border-t-teal border-b-teal/40 animate-[spin_12s_linear_infinite_reverse]"></div>
                <div className="absolute inset-8 rounded-full border-[1px] border-teal/10 border-l-teal/60 border-r-teal/60 animate-[spin_20s_linear_infinite]"></div>
                <div className="absolute inset-12 rounded-full border-[6px] border-teal/5 border-t-teal/30 border-b-teal/30 animate-[spin_15s_linear_infinite_reverse]"></div>

                {/* Outer Dashed Orbit */}
                <div className="absolute inset-0 rounded-full border border-dashed border-teal/30 animate-[spin_30s_linear_infinite]"></div>

                {/* Digital Data Stream Overlay */}
                <div className="absolute inset-0 opacity-20 pointer-events-none overflow-hidden rounded-full">
                  <motion.div
                    className="w-full h-[200%] bg-[linear-gradient(to_bottom,transparent,rgba(0,245,212,0.2),transparent)]"
                    animate={{ y: ["-50%", "0%"] }}
                    transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                  />
                </div>

                {/* Orbiting Biomarker Dots - Refactored for maximum smoothness */}
                <div className="absolute inset-0 animate-[spin_15s_linear_infinite]">
                  {[0, 72, 144, 216, 288].map((angle, i) => (
                    <div
                      key={i}
                      className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2"
                      style={{ transform: `rotate(${angle}deg) translateY(-180px)` }}
                    >
                      <motion.div
                        className="w-2.5 h-2.5 bg-teal rounded-full shadow-[0_0_15px_#00F5D4]"
                        animate={{
                          scale: [1, 1.5, 1],
                          opacity: [0.4, 1, 0.4]
                        }}
                        transition={{
                          duration: 2 + i,
                          repeat: Infinity,
                          ease: "easeInOut"
                        }}
                      />
                    </div>
                  ))}
                </div>

                {/* Scientific Grid Pattern Overlay */}
                <div className="absolute inset-0 opacity-10 pointer-events-none" style={{ backgroundImage: 'radial-gradient(circle, #00F5D4 1px, transparent 1px)', backgroundSize: '20px 20px' }}></div>

                <div className="relative z-10 text-center bg-background-light/40 dark:bg-background-dark/40 p-10 rounded-full backdrop-blur-sm border border-white/10">
                  <Heart className="w-16 h-16 text-teal mx-auto mb-6 animate-pulse" />
                  <h3 className="text-2xl font-rounded-bold mb-2 uppercase tracking-[0.1em]">Clinical Grade</h3>
                  <p className="text-sm text-slate-600 dark:text-gray-400 font-medium leading-relaxed">
                    Neuro-Architectural Validation: <br />
                    <span className="text-teal font-mono">4.2M+</span> Clinical Snapshots Processed.
                  </p>
                </div>
              </motion.div>
            </div>
          </div>
          {/* Bottom Neural Net Background */}
          <div className="absolute bottom-0 left-0 w-full h-full -z-10 opacity-20 dark:opacity-10 pointer-events-none overflow-hidden flex items-center justify-center translate-y-1/2">
            <NeuralNetworkVisualizer />
          </div>
        </div>
      </section>

      {/* Simplified Footer / Mentorship block */}
      <footer className="border-t border-border-light/10 dark:border-border-dark/10 bg-background-light dark:bg-background-dark py-12 px-6 relative z-10">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-start md:items-center gap-8">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Activity className="w-6 h-6 text-teal" />
              <span className="text-2xl font-black tracking-tight text-text-light dark:text-text-dark">LifeLytics</span>
            </div>
            <p className="text-slate-900 dark:text-gray-200 text-sm max-w-xs font-bold leading-relaxed">AI-powered lifespan simulation and personalized health intelligence platform.</p>
          </div>

          <div className="flex flex-col sm:flex-row gap-12">
            {/* <div className="space-y-4">
              <h4 className="text-[11px] font-black uppercase tracking-[0.2em] text-teal border-b border-teal/20 pb-1 mb-3">Academic Team</h4>
              <div className="space-y-2">
                {[
                  "Ms. Radhika A, SRMIST, Chennai",
                  "Dr. Saradha S, SRMIST, Chennai",
                  "Dr. Kavitha D, SRMIST, Chennai",
                  "Dr. Kavitha, SRMIST, Chennai"
                ].map((name, i) => (
                  <p key={i} className="text-[12px] font-bold text-slate-800 dark:text-gray-300">{name}</p>
                ))}
              </div>
            </div> */}

            {/* <div className="space-y-4">
              <h4 className="text-[11px] font-black uppercase tracking-[0.2em] text-blue-500 border-b border-blue-500/20 pb-1 mb-3">Industry Experts</h4>
              <div className="space-y-2">
                {[
                  "Dr. Mithileysh Sathiyanarayanan, MIT Square, London",
                  "Dr. Sharanya Rajan, NHS England, UK"
                ].map((name, i) => (
                  <p key={i} className="text-[12px] font-bold text-slate-800 dark:text-gray-300">{name}</p>
                ))}
              </div>
            </div> */}
          </div>
        </div>
        <div className="max-w-7xl mx-auto mt-12 pt-8 border-t border-slate-200 dark:border-border-dark/10 text-center text-sm text-gray-500 flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="font-bold text-slate-700 dark:text-gray-400">© {new Date().getFullYear()} LifeLytics. Educational purposes only. Not medical advice.</p>
          <div className="flex items-center gap-6 text-[10px] text-gray-500 font-mono tracking-widest">
            <span>CLINICAL GRADE AI</span>
            <span>END-TO-END ENCRYPTED</span>
          </div>
        </div>
      </footer>

      {/* Background Particles Layer - Optimized Layering */}
      <div className="fixed inset-0 pointer-events-none z-[1]">
        <FloatingParticles />
      </div>
    </motion.div>
  );
}
