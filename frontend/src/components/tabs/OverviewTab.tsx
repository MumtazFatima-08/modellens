import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { Card, StatCard, SectionTitle, Pill } from "../ui";
import type { InvestigationResult } from "../../types";

export default function OverviewTab({ result }: { result: InvestigationResult }) {
  const ev = result.evaluation;
  const isClf = ev.task_type === "classification";

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        {isClf ? (
          <>
            <StatCard label="Accuracy" value={pct(ev.accuracy)} accent="violet" />
            <StatCard label="F1 (macro)" value={fmt(ev.f1_macro)} accent="amber" />
            <StatCard label="ROC-AUC" value={ev.roc_auc != null ? fmt(ev.roc_auc) : "n/a"} accent="sky" />
            <StatCard label="Samples" value={String(ev.n_samples)} accent="rose" />
          </>
        ) : (
          <>
            <StatCard label="R²" value={fmt(ev.r2)} accent="violet" />
            <StatCard label="RMSE" value={fmt(ev.rmse)} accent="amber" />
            <StatCard label="MAE" value={fmt(ev.mae)} accent="sky" />
            <StatCard label="Samples" value={String(ev.n_samples)} accent="rose" />
          </>
        )}
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <SectionTitle sublabel="Model & dataset used for this investigation">Setup</SectionTitle>
          <dl className="space-y-2 text-sm">
            <Row label="Estimator" value={result.model_info.estimator_class} />
            <Row label="Task type" value={result.model_info.task_type} />
            <Row label="Feature columns used" value={String(result.dataset_info.n_features_used)} />
            <Row label="Target column" value={result.dataset_info.target_column} />
            {result.model_info.classes && <Row label="Classes" value={result.model_info.classes.join(", ")} />}
          </dl>
        </Card>

        {isClf && ev.confusion_matrix ? (
          <Card>
            <SectionTitle sublabel="Rows = true label, columns = predicted label">Confusion matrix</SectionTitle>
            <ConfusionMatrix matrix={ev.confusion_matrix} labels={ev.confusion_matrix_labels || []} />
          </Card>
        ) : ev.residual_histogram ? (
          <Card>
            <SectionTitle sublabel="Distribution of (true - predicted) residuals">Residual distribution</SectionTitle>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={histToChartData(ev.residual_histogram)}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="bin" tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 10 }} />
                <YAxis tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 10 }} />
                <Tooltip contentStyle={tooltipStyle} />
                <Bar dataKey="count" fill="#a78bfa" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        ) : null}
      </div>

      {isClf && ev.per_class && (
        <Card>
          <SectionTitle sublabel="Precision, recall, F1 computed per class from real predictions">Per-class performance</SectionTitle>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="text-xs uppercase tracking-wide text-white/40">
                  <th className="pb-2">Class</th>
                  <th className="pb-2">Precision</th>
                  <th className="pb-2">Recall</th>
                  <th className="pb-2">F1</th>
                  <th className="pb-2">Support</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {ev.per_class.map((c) => (
                  <tr key={c.class}>
                    <td className="py-2 font-medium text-white/80">{c.class}</td>
                    <td className="py-2 text-white/60">{fmt(c.precision)}</td>
                    <td className="py-2 text-white/60">{fmt(c.recall)}</td>
                    <td className="py-2 text-white/60">{fmt(c.f1)}</td>
                    <td className="py-2 text-white/60">{c.support}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {result.reference_evaluation && (
        <Card>
          <SectionTitle sublabel="Reference-period performance vs. current dataset">Performance comparison</SectionTitle>
          <div className="flex items-center gap-6">
            <div>
              <p className="text-xs text-white/40">Reference</p>
              <p className="font-display text-2xl font-bold text-white">
                {isClf ? fmt(result.reference_evaluation.f1_macro) : fmt(result.reference_evaluation.r2)}
              </p>
            </div>
            <div className="text-white/30">→</div>
            <div>
              <p className="text-xs text-white/40">Current</p>
              <p className="font-display text-2xl font-bold text-white">
                {isClf ? fmt(ev.f1_macro) : fmt(ev.r2)}
              </p>
            </div>
            <DeltaPill
              current={isClf ? ev.f1_macro : ev.r2}
              reference={isClf ? result.reference_evaluation.f1_macro : result.reference_evaluation.r2}
            />
          </div>
        </Card>
      )}
    </div>
  );
}

function DeltaPill({ current, reference }: { current?: number; reference?: number }) {
  if (current == null || reference == null || reference === 0) return null;
  const change = ((current - reference) / Math.abs(reference)) * 100;
  const tone = change < -5 ? "bad" : change > 5 ? "good" : "neutral";
  return <Pill tone={tone as any}>{change >= 0 ? "+" : ""}{change.toFixed(1)}%</Pill>;
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between border-b border-white/5 pb-2">
      <dt className="text-white/40">{label}</dt>
      <dd className="font-medium text-white/80">{value}</dd>
    </div>
  );
}

function fmt(v?: number | null) {
  if (v == null || Number.isNaN(v)) return "n/a";
  return v.toFixed(3);
}
function pct(v?: number | null) {
  if (v == null) return "n/a";
  return `${(v * 100).toFixed(1)}%`;
}

function histToChartData(h: { counts: number[]; bin_edges: number[] }) {
  return h.counts.map((count, i) => ({
    bin: `${h.bin_edges[i].toFixed(1)}`,
    count,
  }));
}

const tooltipStyle = {
  background: "#12102f",
  border: "1px solid rgba(255,255,255,0.1)",
  borderRadius: 8,
  fontSize: 12,
};

function ConfusionMatrix({ matrix, labels }: { matrix: number[][]; labels: string[] }) {
  const max = Math.max(...matrix.flat());
  return (
    <div className="overflow-x-auto">
      <table className="text-sm">
        <thead>
          <tr>
            <th></th>
            {labels.map((l) => (
              <th key={l} className="px-2 pb-2 text-xs font-medium text-white/40">{l}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <td className="pr-2 text-xs font-medium text-white/40">{labels[i]}</td>
              {row.map((v, j) => {
                const intensity = max > 0 ? v / max : 0;
                const isDiag = i === j;
                return (
                  <td key={j} className="p-1">
                    <div
                      className="flex h-12 w-12 items-center justify-center rounded-lg text-xs font-semibold"
                      style={{
                        background: isDiag
                          ? `rgba(167,139,250,${0.15 + intensity * 0.6})`
                          : `rgba(251,146,60,${0.1 + intensity * 0.5})`,
                        color: intensity > 0.4 ? "#fff" : "rgba(255,255,255,0.7)",
                      }}
                    >
                      {v}
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
