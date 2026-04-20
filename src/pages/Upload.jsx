import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

export default function Upload() {
  const navigate = useNavigate();

  useEffect(() => {
    navigate('/doctor-portal', { replace: true });
  }, [navigate]);

  return null;
}
