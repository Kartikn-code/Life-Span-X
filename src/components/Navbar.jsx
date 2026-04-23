import React from 'react';
import { motion } from 'framer-motion';
import { useNavigate, useLocation } from 'react-router-dom';
import { Sun, Moon, BrainCircuit } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';
import { useUser } from '../context/UserContext';
import Logo from './Logo';

const Navbar = () => {
  const { theme, toggleTheme } = useTheme();
  const { engineEnabled, toggleEngine } = useUser();
  const navigate = useNavigate();
  const location = useLocation();
  const [isScrolled, setIsScrolled] = React.useState(false);

  React.useEffect(() => {
    const handleScroll = () => setIsScrolled(window.scrollY > 20);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <nav className={`fixed top-0 left-0 w-full z-50 px-8 py-4 flex justify-between items-center transition-all duration-500 pointer-events-none ${
      isScrolled ? 'bg-background-light/80 dark:bg-background-dark/80 backdrop-blur-xl shadow-2xl' : 'bg-transparent py-6'
    }`}>
      {/* Logo at extreme left */}
      <div className="cursor-pointer pointer-events-auto" onClick={() => navigate('/')}>
        <Logo />
      </div>

      {/* Central AI Status & Ticker (Toggle Switch) */}
      <div className="hidden lg:flex items-center gap-8 pointer-events-auto">
        <motion.button 
          onClick={toggleEngine}
          animate={!engineEnabled ? { 
            borderColor: ['rgba(255, 0, 0, 0.3)', 'rgba(255, 0, 0, 1)', 'rgba(255, 0, 0, 0.3)'],
            borderWidth: ['1px', '2px', '1px'],
            boxShadow: [
              '0 0 0px rgba(255, 0, 0, 0)', 
              '0 0 20px rgba(255, 0, 0, 0.6)', 
              '0 0 0px rgba(255, 0, 0, 0)'
            ]
          } : { 
            borderColor: 'rgba(20, 184, 166, 0.3)', 
            borderWidth: '1px',
            boxShadow: '0 0 15px rgba(20,184,166,0.2)' 
          }}
          transition={!engineEnabled ? { repeat: Infinity, duration: 1.2, ease: "easeInOut" } : {}}
          className={`group flex items-center gap-4 px-6 py-3 rounded-full border backdrop-blur-md transition-all duration-300 ${
            engineEnabled 
              ? 'bg-teal/10 border-teal/40 dark:bg-teal/5 dark:border-teal/30 hover:bg-teal/20 shadow-md dark:shadow-[0_0_20px_rgba(0,245,212,0.1)]' 
              : 'bg-rose-500/10 border-rose-500/40 dark:bg-rose-500/5 dark:border-rose-500/20 hover:bg-rose-500/20 grayscale'
          }`}
        >
          <div className="relative flex h-3 w-3">
            {engineEnabled && <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-teal opacity-75"></span>}
            <span className={`relative inline-flex rounded-full h-3 w-3 ${engineEnabled ? 'bg-teal' : 'bg-rose-500'}`}></span>
          </div>
          <span className={`text-[11px] font-rounded-bold tracking-[0.25em] uppercase transition-colors ${
            engineEnabled 
              ? 'text-slate-900 dark:text-teal' 
              : 'text-rose-600 dark:text-white'
          }`}>
            Neural Engine: {engineEnabled ? 'Active' : 'Offline'}
          </span>
        </motion.button>
      </div>

      <div className="flex items-center gap-6 pointer-events-auto">
        {/* Theme Toggle */}
        <button
          onClick={toggleTheme}
          className="p-3 rounded-2xl text-text-light dark:text-text-dark hover:scale-110 transition-all duration-300"
          aria-label="Toggle Theme"
        >
          {theme === 'dark' ? <Sun className="w-8 h-8 text-amber" /> : <Moon className="w-8 h-8 text-indigo-600" />}
        </button>

        {/* AI Lab Link (Admin) */}
        <button
          onClick={() => navigate('/ai-lab')}
          className={`p-3 rounded-2xl transition-all duration-300 ${
            location.pathname === '/ai-lab' ? 'bg-teal text-slate-950' : 'text-gray-400 hover:text-teal'
          }`}
          title="AI Research Lab"
        >
          <BrainCircuit className="w-8 h-8" />
        </button>
      </div>
    </nav>
  );
};

export default Navbar;
