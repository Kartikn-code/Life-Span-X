import React, { createContext, useContext, useState, useEffect } from 'react';

const UserContext = createContext();

export const useUser = () => useContext(UserContext);

export const UserProvider = ({ children }) => {
  const [userData, setUserData] = useState(() => {
    const saved = localStorage.getItem('lifespan_user_data');
    return saved ? JSON.parse(saved) : {};
  });

  const [predictions, setPredictions] = useState(() => {
    const saved = localStorage.getItem('lifespan_predictions');
    return saved ? JSON.parse(saved) : null;
  });

  useEffect(() => {
    localStorage.setItem('lifespan_user_data', JSON.stringify(userData));
  }, [userData]);

  useEffect(() => {
    if (predictions) {
      localStorage.setItem('lifespan_predictions', JSON.stringify(predictions));
    }
  }, [predictions]);

  const updateUserData = (stepData) => {
    setUserData(prev => ({ ...prev, ...stepData }));
  };

  const clearData = () => {
    setUserData({});
    setPredictions(null);
    localStorage.removeItem('lifespan_user_data');
    localStorage.removeItem('lifespan_predictions');
  };

  return (
    <UserContext.Provider value={{ userData, updateUserData, predictions, setPredictions, clearData }}>
      {children}
    </UserContext.Provider>
  );
};
