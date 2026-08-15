import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Legend } from "recharts";
import { Card, SectionTitle, StatCard, EmptyState } from "../ui";
import { Gauge } from "lucide-react";
import type { InvestigationResult } from "../../types";

const tooltipStyle = { background: "#12102f", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, fontSize: 12 };

export default function CalibrationTab({ result }: { result: InvestigationResult }) {
  const c = result.confidence;

  if (!c.available) {
    return (
      <EmptyState
        icon={<Gauge size={22} />}
        title="Confidence & calibration analysis unavailable"
        description={c.reason || "This model does not expose predicted probabilities."}
      />
    );
  }

  const reliabilityData =
    c.reliability_curve?.predicted_confidence.map((p, i) => ({
      predicted: +p.toFixed(2),
      observed: +(c.reliability_curve!.observed_accuracy[i]).toFixed(3),
    })) || [];

  const histData =
    c.confidence_histogram?.bin_edges.slice(0, -1).map((edge, i) => ({
      bin: edge.toFixed(2),
      correct: c.confidence_histogram!.correct_counts[i],
      incorrect: c.confidence_histogram!.incorrect_counts[i],
    })) || [];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <StatCard
          label="High-confidence errors"
          value={`${((c.pct_errors_high_confidence || 0) * 100).toFixed(1)}%`}
          sublabel={`of ${c.n_total_errors} total errors ≥ ${((c.high_confidence_threshold || 0.9) * 100).toFixed(0)}% confidence`}
          accent="rose"
        />
        <StatCard label="Expected Calibration Error" value={(c.expected_calibration_error || 0).toFixed(4)} accent="amber" />
        <StatCard label="Mean confidence (correct)" value={`${((c.mean_confidence_correct || 0) * 100).toFixed(1)}%`} accent="violet" />
        <StatCard label="Mean confidence (incorrect)" value={`${((c.mean_confidence_incorrect || 0) * 100).toFixed(1)}%`} accent="sky" />
      </div>

      <Card>
        <SectionTitle sublabel="Perfectly calibrated predictions fall on the diagonal (predicted confidence = observed accuracy). Deviation indicates over- or under-confidence.">
          Reliability diagram
        </SectionTitle>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={reliabilityData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
            <XAxis dataKey="predicted" tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 10 }} label={{ value: "Predicted confidence", position: "insideBottom", offset: -5, fill: "rgba(255,255,255,0.3)", fontSize: 11 }} />
            <YAxis tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 10 }} domain={[0, 1]} />
            <Tooltip contentStyle={tooltipStyle} />
            <Line type="monotone" dataKey="observed" stroke="#a78bfa" strokeWidth={2} dot={{ fill: "#a78bfa", r: 3 }} name="Observed accuracy" />
            <Line
              type="linear"
              dataKey="predicted"
              stroke="rgba(255,255,255,0.2)"
              strokeDasharray="4 4"
              dot={false}
              name="Perfect calibration"
            />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      <Card>
        <SectionTitle sublabel="Distribution of model confidence, split by whether the prediction was correct or incorrect. A cluster of high-confidence incorrect predictions signals overconfidence.">
          Confidence distribution
        </SectionTitle>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={histData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
            <XAxis dataKey="bin" tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 9 }} />
            <YAxis tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 10 }} />
            <Tooltip contentStyle={tooltipStyle} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Bar dataKey="correct" stackId="a" fill="#4ade80" name="Correct" radius={[0, 0, 0, 0]} />
            <Bar dataKey="incorrect" stackId="a" fill="#f87171" name="Incorrect" radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  );
}
