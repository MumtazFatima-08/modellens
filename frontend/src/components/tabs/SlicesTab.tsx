import { Card, SectionTitle, Pill, EmptyState } from "../ui";
import { Layers } from "lucide-react";
import type { InvestigationResult } from "../../types";

export default function SlicesTab({ result }: { result: InvestigationResult }) {
  const slices = result.slices || [];

  if (slices.length === 0) {
    return (
      <EmptyState
        icon={<Layers size={22} />}
        title="No problematic slices discovered"
        description="Recursive partitioning did not find any subgroup that performs meaningfully worse than the overall model at the sample-size and significance thresholds used."
      />
    );
  }

  return (
    <div className="space-y-4">
      <Card>
        <SectionTitle sublabel="Subgroups discovered via decision-tree-based recursive partitioning on the error signal, ranked by (performance gap × sample size). Only subgroups worse than the overall average are shown — this is an observed association within your data, not a proven cause of failure.">
          Discovered problematic slices
        </SectionTitle>
      </Card>

      {slices.map((s, i) => (
        <Card key={i} className="border-l-4 severity-warning">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <p className="font-mono text-sm text-white/90">{s.rule}</p>
              <p className="mt-2 text-xs text-white/40">
                {s.sample_size} samples · p={s.p_value.toFixed(4)}
              </p>
            </div>
            <Pill tone={s.significant ? "bad" : "neutral"}>
              {s.significant ? "statistically significant" : "not significant"}
            </Pill>
          </div>
          <div className="mt-4 grid grid-cols-3 gap-3 text-center">
            <div className="rounded-lg bg-white/5 py-3">
              <p className="text-xs text-white/40">Group error rate</p>
              <p className="font-display mt-1 text-lg font-bold text-rose-300">{(s.group_error_rate * 100).toFixed(1)}%</p>
            </div>
            <div className="rounded-lg bg-white/5 py-3">
              <p className="text-xs text-white/40">Overall error rate</p>
              <p className="font-display mt-1 text-lg font-bold text-white/70">{(s.overall_error_rate * 100).toFixed(1)}%</p>
            </div>
            <div className="rounded-lg bg-white/5 py-3">
              <p className="text-xs text-white/40">Performance gap</p>
              <p className="font-display mt-1 text-lg font-bold text-amber-300">+{(s.performance_gap * 100).toFixed(1)}pp</p>
            </div>
          </div>
        </Card>
      ))}
    </div>
  );
}
