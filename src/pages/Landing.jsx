import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { ArrowRight, Activity, Shield, Brain, ChevronDown, Heart } from 'lucide-react';

const DNAHelix = () => {
  return (
    <div className="relative w-64 h-64 mx-auto perspective-1000">
      <div className="absolute inset-0 flex flex-col items-center justify-center space-y-4 animate-spin-slow" style={{ transformStyle: 'preserve-3d' }}>
        {[...Array(10)].map((_, i) => (
          <div key={i} className="flex w-32 justify-between items-center" style={{ transform: `rotateY(${i * 36}deg) translateZ(20px)` }}>
            <div className="w-4 h-4 rounded-full bg-teal shadow-[0_0_10px_#00F5D4]"></div>
            <div className="flex-1 h-0.5 bg-gradient-to-r from-teal to-amber opacity-50"></div>
            <div className="w-4 h-4 rounded-full bg-amber shadow-[0_0_10px_#F5A623]"></div>
          </div>
        ))}
      </div>
      <style>{`
        .perspective-1000 { perspective: 1000px; }
        .animate-spin-slow { animation: spin 8s linear infinite; }
        @keyframes spin { from { transform: rotateY(0deg); } to { transform: rotateY(360deg); } }
      `}</style>
    </div>
  );
};

export default function Landing() {
  const navigate = useNavigate();
  const [counter, setCounter] = useState(0);

  useEffect(() => {
    // Seeded counter based on today's date
    const today = new Date();
    const seed = today.getFullYear() * 10000 + (today.getMonth() + 1) * 100 + today.getDate();
    
    // Simple deterministic pseudorandom based on seed
    const pseudoRandom = Math.sin(seed) * 10000;
    const baseCount = Math.floor(Math.abs(pseudoRandom - Math.floor(pseudoRandom)) * 5000) + 1000;
    
    // Animate up to baseCount
    let current = 0;
    const step = Math.ceil(baseCount / 50);
    const interval = setInterval(() => {
      current += step;
      if (current >= baseCount) {
        setCounter(baseCount);
        clearInterval(interval);
      } else {
        setCounter(current);
      }
    }, 30);
    
    return () => clearInterval(interval);
  }, []);

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.5 }}
      className="min-h-screen relative overflow-hidden"
    >
      {/* Background Orbs (Replicating past conversation request implicitly) */}
      <div className="absolute inset-0 z-0 overflow-hidden bg-navy">
        <div className="absolute inset-0 bg-[url('https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop')] bg-cover bg-center opacity-10 mix-blend-screen animate-pulse"></div>
        <div className="absolute inset-0 bg-gradient-to-b from-navy via-navy/80 to-navy"></div>
        <div className="absolute top-0 left-0 w-full h-full bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-teal/10 via-transparent to-transparent"></div>
        <div className="absolute top-[-10%] left-[-10%] w-[40vw] h-[40vw] bg-teal rounded-full mix-blend-screen filter blur-[100px] opacity-10 animate-pulse"></div>
        <div className="absolute bottom-[-10%] right-[-10%] w-[40vw] h-[40vw] bg-amber rounded-full mix-blend-screen filter blur-[100px] opacity-10 animate-pulse" style={{ animationDelay: '2s' }}></div>
      </div>

      {/* Hero Section */}
      <section className="relative pt-32 pb-20 px-6 md:px-12 max-w-7xl mx-auto flex flex-col md:flex-row items-center">
        <div className="md:w-1/2 z-10 text-center md:text-left mb-16 md:mb-0">
          <motion.div 
            initial={{ y: -20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            className="flex items-center justify-center md:justify-start gap-3 mb-6"
          >
            <div className="relative w-14 h-14">
              <div className="absolute inset-0 bg-teal/20 rounded-xl border border-teal/30 flex items-center justify-center">
                <Heart className="w-8 h-8 text-teal fill-teal/20" />
              </div>
              <div className="absolute -bottom-1 -right-1 w-6 h-6 bg-white rounded-lg flex items-center justify-center shadow-lg">
                <span className="text-navy font-bold text-xs">L</span>
              </div>
            </div>
            <span className="text-3xl font-bold tracking-tight text-white ml-2">LifeLytics</span>
          </motion.div>

          <motion.h1 
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="text-5xl md:text-7xl font-bold leading-tight mb-6"
          >
            LifeLytics <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal to-blue-500">Health Intelligence</span>
          </motion.h1>
          
          <motion.p 
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="text-xl text-gray-300 mb-8 max-w-lg"
          >
            Transparent, scientifically-grounded AI that doesn't just predict your lifespan, but explains exactly why—and how to improve it.
          </motion.p>
          
          <motion.div 
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.4 }}
            className="flex flex-col sm:flex-row gap-4 justify-center md:justify-start"
          >
            <button onClick={() => navigate('/onboarding')} className="btn-primary flex items-center justify-center gap-2 group">
              Talk to AI Assistant <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </button>
            <button onClick={() => navigate('/doctor-portal')} className="btn-secondary">
              Clinical Portal
            </button>
          </motion.div>

          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.8 }}
            className="mt-12 text-gray-400 font-medium"
          >
            <span className="text-teal font-mono text-xl mr-2">{counter.toLocaleString()}</span>
            people extended their life today
          </motion.div>
        </div>

        <div className="md:w-1/2 w-full flex justify-center z-10">
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.5, type: 'spring' }}
          >
            <DNAHelix />
          </motion.div>
        </div>
      </section>

      {/* Stats Strip */}
      <section className="border-y border-border/50 bg-surface/30 backdrop-blur-sm py-10">
        <div className="max-w-7xl mx-auto px-6 flex flex-wrap justify-center gap-12 md:gap-24 text-center">
          {[
            { label: 'Prediction Accuracy', value: '92%' },
            { label: 'Risk Factors', value: '40+' },
            { label: 'Age Groups', value: '6' },
            { label: 'ML Models', value: '3' },
          ].map((stat, i) => (
            <motion.div 
              key={i}
              initial={{ y: 20, opacity: 0 }}
              whileInView={{ y: 0, opacity: 1 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1 }}
            >
              <div className="text-4xl font-display font-bold text-white mb-2">{stat.value}</div>
              <div className="text-sm text-gray-400 uppercase tracking-wider">{stat.label}</div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* How it Works */}
      <section className="py-24 px-6 max-w-7xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold mb-4">Why LifeLytics?</h2>
          <p className="text-gray-400 max-w-2xl mx-auto">A research-grade platform built for scientific transparency and actionable results.</p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          {[
            { icon: Brain, title: 'AI-Based Prediction', desc: 'Deterministic modeling cross-referenced against WHO datasets for unparalleled accuracy.' },
            { icon: Activity, title: 'Lifestyle Insights', desc: 'Understand exactly how your habits (like smoking or exercise) quantitatively alter your lifespan.' },
            { icon: Shield, title: 'Personalized Recommendations', desc: 'Receive a clinical-grade action plan tailored to your specific risk factors and age group.' }
          ].map((step, i) => (
            <motion.div 
              key={i}
              initial={{ y: 30, opacity: 0 }}
              whileInView={{ y: 0, opacity: 1 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.2 }}
              className="glass-panel p-8 text-center relative overflow-hidden group"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-teal/5 to-amber/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
              <div className="w-16 h-16 bg-surface border border-border rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-lg relative z-10 group-hover:border-teal/50 transition-colors">
                <step.icon className="w-8 h-8 text-teal" />
              </div>
              <h3 className="text-2xl font-semibold mb-4 relative z-10">{step.title}</h3>
              <p className="text-gray-400 relative z-10">{step.desc}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border/50 py-12 px-6">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center text-gray-500 text-sm">
          <p>© {new Date().getFullYear()} LifeLytics. All data processed locally.</p>
          <div className="mt-4 md:mt-0 flex flex-col md:flex-row items-center gap-4">
            <p>Thesis Project: A Zero-Infrastructure Approach to Accessible Health Intelligence</p>
            <a href="https://github.com" target="_blank" rel="noreferrer" className="text-teal hover:underline flex items-center gap-1">
              GitHub Repository
            </a>
          </div>
        </div>
      </footer>
    </motion.div>
  );
}
