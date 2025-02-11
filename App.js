import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import WelcomePage from './components/WelcomePage';
import RegisterPage from './components/RegisterPage';
import LoginPage from './components/LoginPage';
import AIAssistant from './components/AIAssistant';
 import DisplayData from './components/DisplayData';
import ForgotPasswordPage from './components/ForgotPasswordPage'; 
import ResetPasswordPage from './components/ResetPasswordPage'; // Import new component
import ClaimCreation from './components/ClaimCreation';
import SearchClaim from './components/SearchClaim';
import ClaimDetails from './components/ClaimDetails';
import MainPage from './MainPage';
import ClaimDetailsPage from './components/ClaimDetailsPage';
// import PrivateRoute from './components/PrivateRoute';


const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<WelcomePage />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/forgot-password" element={<ForgotPasswordPage />} />
        <Route path="/reset-password" element={<ResetPasswordPage />} />
        <Route path="/claim-creation" element={<ClaimCreation />} />
        <Route path="/ai-assistant" element={<AIAssistant />} />
        <Route path="/display-data" element={<DisplayData />} />
        <Route path="/search-claim" element={<SearchClaim />} />
        <Route path="/claim-details" element={<ClaimDetails />} />
        <Route path='/main-page' element={<MainPage />} />
        <Route path='/claim-details-page' element={<ClaimDetailsPage />} />
      </Routes>
    </Router>
  );
};

export default App;

