import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { FlaskConical, Play } from "lucide-react";
import { Card, LoadingState, ErrorState, Pill, SectionTitle } from "../components/ui";
import { listSamples, loadSample, createInvestigation } from "../api/client";
import { addHistory } from "../lib/history";

const DESCRIPTIONS: Record<string, string> = {
  healthy_classification: "A reasonably well-performing binary classifier — a useful baseline for what 'no major findings' looks like.",
  weak_classification: "A weak classifier with high label noise and low class separability. Expect low-confidence overall performance.",
  imbalanced_classification: "95/5 class imbalance where overall accuracy is misleading — check the per-class metrics and PR-AUC.",
  distribution_shift: "Trained on a reference period; the current dataset has real, deliberate feature drift. Compare reference vs. current.",
  error_prone: "A linear model applied to data with a genuine non-linear interaction it structurally can't capture — errors cluster by design.",
  text_classification: "Free-text product reviews, zero numeric features — a real sklearn Pipeline (TF-IDF + LogisticRegression) proving ModelLens investigates text classifiers through the same upload flow, not just tabular models.",
  regression: "A regression model with real, non-trivial residual structure.",
};

export default function Samples() {
  const navigate = useNavigate();
  const [cases, setCases] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [launchingId, setLaunchingId] = useState<string | null>(null);

  useEffect(() => {
    listSamples()
      .then((r) => setCases(r.cases))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  async function launch(caseId: string) {
    setLaunchingId(caseId);
    try {
      const sample = await loadSample(caseId);
      const inv = await createInvestigation({
        model_id: sample.model_id,
        dataset_id: sample.dataset_id,
        target_column: sample.target_column,
        reference_dataset_id: sample.reference_dataset_id || undefined,
      });
      addHistory({ id: inv.investigation_id, label: caseId.replace(/_/g, " "), createdAt: new Date().toISOString() });
      navigate(`/investigations/${inv.investigation_id}`);
    } catch (e: any) {
      alert(`Could not launch: ${e.message}`);
    } finally {
      setLaunchingId(null);
    }
  }

  if (loading) return <LoadingState message="Loading sample cases…" />;
  if (error) return <ErrorState message={error} />;

  return (
    <div>
      <SectionTitle sublabel="Real trained models and real generated datasets, ready to investigate immediately. Run scripts/generate_sample_models.py to regenerate them.">
        Sample investigation cases
      </SectionTitle>
      <div className="grid gap-4 md:grid-cols-2">
        {cases.map((c) => (
          <Card key={c.id} className="flex flex-col">
            <div className="flex items-start justify-between">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-white/5 text-violet-300">
                <FlaskConical size={18} />
              </div>
              <div className="flex gap-1.5">
                <Pill tone="neutral">{c.task}</Pill>
                {c.has_reference && <Pill tone="warn">has drift ref</Pill>}
                {!c.ready && <Pill tone="bad">not generated</Pill>}
              </div>
            </div>
            <h3 className="font-display mt-4 text-base font-semibold text-white">{c.label}</h3>
            <p className="mt-2 flex-1 text-sm text-white/50">{DESCRIPTIONS[c.id]}</p>
            <button
              disabled={!c.ready || launchingId === c.id}
              onClick={() => launch(c.id)}
              className="mt-5 inline-flex items-center justify-center gap-2 rounded-full bg-white/10 px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-white/15 disabled:opacity-40"
            >
              <Play size={14} /> {launchingId === c.id ? "Launching…" : "Investigate this model"}
            </button>
          </Card>
        ))}
      </div>
    </div>
  );
}
