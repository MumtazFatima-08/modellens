import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Search, FlaskConical, FolderClock, ArrowRight } from "lucide-react";
import { Card, EmptyState } from "../components/ui";
import { listSamples, loadSample, createInvestigation } from "../api/client";
import { getHistory } from "../lib/history";

export default function Dashboard() {
  const navigate = useNavigate();
  const [sampleCount, setSampleCount] = useState<number | null>(null);
  const [loadingDemo, setLoadingDemo] = useState(false);
  const history = getHistory();

  useEffect(() => {
    listSamples()
      .then((r) => setSampleCount(r.cases.filter((c: any) => c.ready).length))
      .catch(() => setSampleCount(0));
  }, []);

  async function tryDemo() {
    setLoadingDemo(true);
    try {
      const sample = await loadSample("error_prone");
      const inv = await createInvestigation({
        model_id: sample.model_id,
        dataset_id: sample.dataset_id,
        target_column: sample.target_column,
        reference_dataset_id: sample.reference_dataset_id || undefined,
      });
      navigate(`/investigations/${inv.investigation_id}`);
    } catch (e: any) {
      alert(`Could not start demo investigation: ${e.message}`);
    } finally {
      setLoadingDemo(false);
    }
  }

  return (
    <div className="space-y-10">
      <section className="glass-card relative overflow-hidden p-10 md:p-14">
        <p className="font-mono text-xs uppercase tracking-[0.2em] text-violet-300/80">
          ML model investigation platform
        </p>
        <h1 className="font-display mt-3 max-w-2xl text-4xl font-extrabold leading-tight text-white md:text-5xl">
          Your model predicts.
          <br />
          <span className="bg-gradient-to-r from-violet-300 via-fuchsia-200 to-amber-300 bg-clip-text text-transparent">
            ModelLens investigates.
          </span>
        </h1>
        <p className="mt-4 max-w-xl text-sm leading-relaxed text-white/60">
          Upload a trained model and an evaluation dataset. ModelLens runs a real statistical
          investigation — error analysis, slice discovery, drift detection, calibration — and
          reports evidence-based findings, not a single accuracy number.
        </p>
        <div className="mt-8 flex flex-wrap gap-3">
          <button
            onClick={() => navigate("/investigate")}
            className="inline-flex items-center gap-2 rounded-full bg-gradient-to-r from-violet-600 to-amber-500 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-violet-900/30 transition-transform hover:scale-[1.02]"
          >
            <Search size={16} /> Start an investigation
          </button>
          <button
            onClick={tryDemo}
            disabled={loadingDemo}
            className="inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/5 px-5 py-3 text-sm font-semibold text-white/90 transition-colors hover:bg-white/10 disabled:opacity-50"
          >
            <FlaskConical size={16} /> {loadingDemo ? "Loading demo…" : "Try demo investigation"}
          </button>
        </div>
      </section>

      <section>
        <div className="mb-4 flex items-center justify-between">
          <h2 className="font-display text-lg font-bold text-white">Recent investigations</h2>
          <button onClick={() => navigate("/history")} className="text-xs font-medium text-violet-300 hover:text-violet-200">
            View all
          </button>
        </div>
        {history.length === 0 ? (
          <EmptyState
            icon={<FolderClock size={22} />}
            title="No investigations yet"
            description="Run your first investigation with a sample model, or upload your own model and dataset to get started."
            action={
              <button
                onClick={() => navigate("/samples")}
                className="mt-2 inline-flex items-center gap-1.5 rounded-full bg-white/10 px-4 py-2 text-xs font-semibold text-white hover:bg-white/15"
              >
                Browse sample cases <ArrowRight size={14} />
              </button>
            }
          />
        ) : (
          <div className="grid gap-3 md:grid-cols-2">
            {history.slice(0, 4).map((h) => (
              <Card
                key={h.id}
                className="cursor-pointer transition-colors hover:bg-white/5"
                
              >
                <div onClick={() => navigate(`/investigations/${h.id}`)}>
                  <p className="font-display text-sm font-semibold text-white">{h.label}</p>
                  <p className="mt-1 text-xs text-white/40">{new Date(h.createdAt).toLocaleString()}</p>
                </div>
              </Card>
            ))}
          </div>
        )}
      </section>

      <section className="grid gap-4 md:grid-cols-3">
        <Card>
          <p className="font-display text-sm font-semibold text-white">{sampleCount ?? "…"} sample cases ready</p>
          <p className="mt-1 text-xs text-white/50">Healthy, weak, imbalanced, drifted, error-prone, and regression models — all real, trained artifacts.</p>
        </Card>
        <Card>
          <p className="font-display text-sm font-semibold text-white">Bring your own model</p>
          <p className="mt-1 text-xs text-white/50">Upload a .pkl / .joblib scikit-learn-compatible estimator and a CSV evaluation set.</p>
        </Card>
        <Card>
          <p className="font-display text-sm font-semibold text-white">Evidence, not vibes</p>
          <p className="mt-1 text-xs text-white/50">Every finding traces back to a computed statistic — no hardcoded dashboard numbers.</p>
        </Card>
      </section>
    </div>
  );
}
