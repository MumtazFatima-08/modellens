import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { Card, SectionTitle, FindingCard } from "../ui";
import type { InvestigationResult } from "../../types";

const tooltipStyle = { background: "#12102f", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, fontSize: 12 };

export default function FindingsTab({ result }: { result: InvestigationResult }) {
  const findings = result.findings || [];
  const fi = result.feature_importance;

  return (
    <div className="space-y-8">
      <div className="space-y-4">
        {findings.map((f, i) => (
          <FindingCard key={i} finding={f} />
        ))}
      </div>

      {fi?.available && fi.importances && (
        <Card>
          <SectionTitle sublabel={`Global feature importance (${fi.method?.replace(/_/g, " ")}). Explains which features the model relies on most overall — not why any single prediction was made.`}>
            Global feature importance
          </SectionTitle>
          <ResponsiveContainer width="100%" height={Math.max(220, fi.importances.length * 32)}>
            <BarChart data={fi.importances.slice(0, 12).map((d) => ({ name: d.feature, value: +(d.importance_normalized * 100).toFixed(1) }))} layout="vertical" margin={{ left: 24 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" horizontal={false} />
              <XAxis type="number" tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 10 }} unit="%" />
              <YAxis type="category" dataKey="name" tick={{ fill: "rgba(255,255,255,0.6)", fontSize: 11 }} width={120} />
              <Tooltip contentStyle={tooltipStyle} />
              <Bar dataKey="value" fill="#a78bfa" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      )}

      <p className="text-center text-xs text-white/30">
        ModelLens reports statistical associations observed in your data. It does not establish causation,
        does not guarantee real-world significance, and does not certify a model for any particular use case.
      </p>
    </div>
  );
}
