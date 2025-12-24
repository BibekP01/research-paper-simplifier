import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Home } from './pages/Home';
import { Chat } from './pages/Chat';
import './App.css';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/chat/:documentId" element={<Chat />} />
      </Routes>
    </Router>
  );
}

export default App;
