import React from 'react';
import { useNavigate } from 'react-router-dom';
import './WelcomePage.css';
import background from './background.jpg';
import logo from './logo.png';

const WelcomePage = () => {
  const navigate = useNavigate();

  const handleEnterKeynes = () => {
    navigate('/chat');
  };

  return (
    <div
      className="welcome-container"
      style={{ backgroundImage: `url(${background})` }}
    >
      <div className="logo-container">
        <img src={logo} alt="Keynes Logo" className="logo-image" />
        <span className="logo-text">Keynes Investments</span>
      </div>

      <div className="welcome-content">
        <h1 className="welcome-title">Welcome to Keynes</h1>
        <p className="welcome-subtitle">Your personal value investing AI assistant</p>
        <button onClick={handleEnterKeynes} className="enter-button">
          Enter Keynes
        </button>
      </div>
    </div>
  );
};

export default WelcomePage;
