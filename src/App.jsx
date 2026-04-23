import { HashRouter as Router, Routes, Route } from 'react-router-dom';
import Landing from './pages/Landing';
import Onboarding from './pages/Onboarding';
import Dashboard from './pages/Dashboard';
import Simulate from './pages/Simulate';
import Upload from './pages/Upload';
import DoctorPortal from './pages/DoctorPortal';
import Timeline from './pages/Timeline';
import ActionPlan from './pages/ActionPlan';
import ManualHistory from './pages/ManualHistory';
import Science from './pages/Science';
import { ThemeProvider } from './context/ThemeContext';
import Navbar from './components/Navbar';
import { UserProvider } from './context/UserContext'; // Assuming UserProvider exists based on Dashboard.jsx

function App() {
  return (
    <ThemeProvider>
      <UserProvider>
        <Router>
          <div className="min-h-screen bg-background-light dark:bg-background-dark text-text-light dark:text-text-dark transition-colors duration-300">
            <Navbar />
            <div className="pt-16"> {/* Balanced spacer for navbar height */}
              <Routes>
                <Route path="/" element={<Landing />} />
                <Route path="/onboarding" element={<Onboarding />} />
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/simulate" element={<Simulate />} />
                <Route path="/upload" element={<Upload />} />
                <Route path="/doctor-portal" element={<DoctorPortal />} />
                <Route path="/timeline" element={<Timeline />} />
                <Route path="/action-plan" element={<ActionPlan />} />
                <Route path="/history" element={<ManualHistory />} />
                <Route path="/science" element={<Science />} />
              </Routes>
            </div>
            <footer className="w-full text-center py-4 bg-transparent backdrop-blur-sm border-t border-border-light/10 dark:border-border-dark/10 text-xs text-gray-500 fixed bottom-0 z-50 print:hidden transition-colors duration-300">
              This tool is for educational purposes only and not medical advice.
            </footer>
          </div>
        </Router>
      </UserProvider>
    </ThemeProvider>
  );
}

export default App;
