import { HashRouter, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import NewInvestigation from "./pages/NewInvestigation";
import Samples from "./pages/Samples";
import History from "./pages/History";
import InvestigationDetail from "./pages/InvestigationDetail";

function App() {
  return (
    <HashRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/investigate" element={<NewInvestigation />} />
          <Route path="/samples" element={<Samples />} />
          <Route path="/history" element={<History />} />
          <Route path="/investigations/:id" element={<InvestigationDetail />} />
        </Routes>
      </Layout>
    </HashRouter>
  );
}

export default App;
