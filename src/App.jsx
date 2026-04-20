import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Landing from './pages/Landing';
import Onboarding from './pages/Onboarding';
import Dashboard from './pages/Dashboard';
import Simulate from './pages/Simulate';
import Upload from './pages/Upload';
import DoctorPortal from './pages/DoctorPortal';
import Timeline from './pages/Timeline';
import ActionPlan from './pages/ActionPlan';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-navy text-white">
        {/* Navigation could go here later if needed globally */}
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/onboarding" element={<Onboarding />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/simulate" element={<Simulate />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/doctor-portal" element={<DoctorPortal />} />
          <Route path="/timeline" element={<Timeline />} />
          <Route path="/action-plan" element={<ActionPlan />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
