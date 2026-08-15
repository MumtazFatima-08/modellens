import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from "recharts";
import { Card, SectionTitle, Pill, EmptyState } from "../ui";
import { GitCompareArrows } from "lucide-react";
import type { InvestigationResult } from "../../types";

const tooltipStyle = { background: "#12102f", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, fontSize: 12 };
const statusColor: Record<string, string> = { normal: "#4ade80", drift: "#fbbf24", severe_drift: "#f87171", unknown: "rgba(255,255,255,0.2)" };
const statusTone: Record<string, any> = { normal: "good", drift: "warn", severe_drift: "bad", unknown: "neutral" };

export default function DriftTab({ result }: { result: InvestigationResult }) {
  const drift = result.drift;

  if (!drift) {
    return (
      <EmptyState
        icon={<GitCompareArrows size={22} />}
        title="No reference dataset provided"
        description="Drift analysis compares your evaluation dataset against a reference/training dataset. Re-run the investigation with a reference dataset to enable this view."
      />
    );
  }

  const chartData = drift.map((d) => ({ name: d.feature, psi: +d.psi.toFixed(3), status: d.status }));

  return (
    <div className="space-y-6">
      <Card>
        <SectionTitle sublabel="Population Stability Index (PSI) per feature: <0.1 normal, 0.1–0.25 moderate drift, ≥0.25 severe drift. Drift alone does not mean the model has failed — cross-reference with the Findings and Slices tabs.">
          Feature drift (reference vs. current)
        </SectionTitle>
        <ResponsiveContainer width="100%" height={Math.max(220, drift.length * 34)}>
          <BarChart data={chartData} layout="vertical" margin={{ left: 24 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" horizontal={false} />
            <XAxis type="number" tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 10 }} />
            <YAxis type="category" dataKey="name" tick={{ fill: "rgba(255,255,255,0.6)", fontSize: 11 }} width={120} />
            <ReferenceLine x={0.1} stroke="rgba(251,191,36,0.4)" strokeDasharray="3 3" />
            <ReferenceLine x={0.25} stroke="rgba(248,113,113,0.4)" strokeDasharray="3 3" />
            <Tooltip contentStyle={tooltipStyle} formatter={(v: any) => [v, "PSI"]} />
            <Bar dataKey="psi" radius={[0, 4, 4, 0]}>
              {chartData.map((d, i) => (
                <Cell key={i} fill={statusColor[d.status]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="text-xs uppercase tracking-wide text-white/40">
                <th className="pb-2 pr-4">Feature</th>
                <th className="pb-2 pr-4">Type</th>
                <th className="pb-2 pr-4">PSI</th>
                <th className="pb-2 pr-4">KS / Chi² p-value</th>
                <th className="pb-2">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {drift.map((d) => (
                <tr key={d.feature}>
                  <td className="py-2 pr-4 font-medium text-white/80">{d.feature}</td>
                  <td className="py-2 pr-4 text-white/50">{d.feature_type}</td>
                  <td className="py-2 pr-4 font-mono text-white/60">{d.psi.toFixed(3)}</td>
                  <td className="py-2 pr-4 font-mono text-white/60">
                    {(d.ks_p_value ?? d.chi2_p_value)?.toFixed(4) ?? "n/a"}
                  </td>
                  <td className="py-2">
                    <Pill tone={statusTone[d.status]}>{d.status.replace("_", " ")}</Pill>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {result.ood?.available && (
        <Card>
          <SectionTitle sublabel="IsolationForest fit on the reference distribution, applied to the current dataset. Flags are potential — not certain — out-of-distribution samples.">
            Out-of-distribution samples
          </SectionTitle>
          <div className="flex items-center gap-6">
            <div>
              <p className="font-display text-3xl font-bold text-white">{((result.ood.pct_flagged || 0) * 100).toFixed(1)}%</p>
              <p className="text-xs text-white/40">of current samples flagged as potential OOD</p>
            </div>
            <div className="text-white/30">·</div>
            <div>
              <p className="font-display text-3xl font-bold text-white">{result.ood.n_flagged_potential_ood}</p>
              <p className="text-xs text-white/40">samples flagged</p>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
}
