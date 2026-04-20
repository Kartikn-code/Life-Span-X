// Rule-based engine for personalized action plan

export const generatePlan = (modifiers, age, persona) => {
  // Identify top modifiable risks
  const sorted = Object.entries(modifiers).sort((a, b) => a[1] - b[1]);
  const topRisks = sorted.filter(m => m[1] < 0 && m[0] !== 'genetics' && m[0] !== 'income').map(m => m[0]);
  
  const plan = {
    "Week 1-2": {
      title: "Foundation & Baseline",
      tasks: [
        { id: 'w1_1', text: "Track daily water intake (Goal: 2.5L+)", done: false },
        { id: 'w1_2', text: "Establish consistent 11 PM bedtime", done: false },
        { id: 'w1_3', text: "10-minute morning stretching routine", done: false }
      ]
    },
    "Week 3-4": {
      title: "Targeted Habit Shift",
      tasks: []
    },
    "Week 5-6": {
      title: "Nutritional Overhaul",
      tasks: [
        { id: 'w5_1', text: "Add 2 servings of greens to daily meals", done: false },
        { id: 'w5_2', text: "Replace processed snacks with whole fruits", done: false }
      ]
    },
    "Week 7-8": {
      title: "Stress & Cognitive Load",
      tasks: [
        { id: 'w7_1', text: "5 minutes of box breathing daily", done: false }
      ]
    },
    "Week 9-10": {
      title: "Social & Connection Goals",
      tasks: [
        { id: 'w9_1', text: "Schedule one meaningful social activity per week", done: false },
        { id: 'w9_2', text: "Join a local club or fitness group", done: false }
      ]
    },
    "Week 11-12": {
      title: "Consolidation",
      tasks: [
        { id: 'w11_1', text: "Re-take the LifeSpan assessment", done: false },
        { id: 'w11_2', text: "Identify next micro-habits to implement", done: false }
      ]
    }
  };

  // Inject specific tasks based on risks
  if (topRisks.includes('smoking')) {
    plan["Week 3-4"].tasks.push({ id: 'w3_s1', text: "Set a quit date and inform family", done: false });
    plan["Week 3-4"].tasks.push({ id: 'w3_s2', text: "Switch to NRT or drastically reduce by 50%", done: false });
  }
  if (topRisks.includes('alcohol')) {
    plan["Week 3-4"].tasks.push({ id: 'w3_a1', text: "Have 4 alcohol-free days per week", done: false });
  }
  if (topRisks.includes('exercise')) {
    plan["Week 3-4"].tasks.push({ id: 'w3_e1', text: "30-minute brisk walk, 3 times a week", done: false });
    plan["Week 5-6"].tasks.push({ id: 'w5_e1', text: "Introduce light resistance training twice a week", done: false });
  }
  if (topRisks.includes('stress') || topRisks.includes('mental')) {
    plan["Week 7-8"].tasks.push({ id: 'w7_m1', text: "Complete 10-min guided meditation daily", done: false });
    plan["Week 7-8"].tasks.push({ id: 'w7_m2', text: "Digital detox: no screens 1 hour before bed", done: false });
  }
  if (topRisks.includes('bmi')) {
    plan["Week 5-6"].tasks.push({ id: 'w5_b1', text: "Track caloric intake for 3 days to establish baseline", done: false });
  }

  // Ensure Week 3-4 has generic tasks if no specific risks found
  if (plan["Week 3-4"].tasks.length === 0) {
    plan["Week 3-4"].tasks.push({ id: 'w3_g1', text: "Try one new physical activity (e.g. yoga, swimming)", done: false });
  }

  return plan;
};
