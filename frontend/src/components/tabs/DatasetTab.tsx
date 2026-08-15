import { Card, SectionTitle, StatCard, Pill } from "../ui";
import type { InvestigationResult } from "../../types";

export default function DatasetTab({ result }: { result: InvestigationResult }) {
  const p = result.dataset_profile;
  if (!p) return null;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <StatCard label="Rows" value={String(p.n_rows)} accent="violet" />
        <StatCard label="Columns" value={String(p.n_columns)} accent="sky" />
        <StatCard
          label="Duplicate rows"
          value={String(p.duplicate_rows)}
          sublabel={p.n_rows ? `${((p.duplicate_rows / p.n_rows) * 100).toFixed(1)}% of rows` : undefined}
          accent={p.duplicate_rows > 0 ? "amber" : "violet"}
        />
        <StatCard
          label="Columns with missing values"
          value={String(p.columns.filter((c) => c.n_missing > 0).length)}
          accent={p.columns.some((c) => c.n_missing > 0) ? "rose" : "violet"}
        />
      </div>

      {p.target_summary && (
        <Card>
          <SectionTitle sublabel="Distribution of the target column across this dataset">Target distribution</SectionTitle>
          {p.target_summary.type === "categorical" ? (
            <div>
              <div className="space-y-2">
                {Object.entries(p.target_summary.class_counts).map(([cls, count]) => {
                  const pct = p.target_summary!.type === "categorical" ? p.target_summary!.class_balance_pct[cls] : 0;
                  return (
                    <div key={cls} className="flex items-center gap-3">
                      <span className="w-20 shrink-0 font-mono text-sm text-white/70">{cls}</span>
                      <div className="h-2 flex-1 overflow-hidden rounded-full bg-white/5">
                        <div className="h-full rounded-full bg-gradient-to-r from-violet-500 to-amber-400" style={{ width: `${pct * 100}%` }} />
                      </div>
                      <span className="w-28 shrink-0 text-right text-xs text-white/50">
                        {count} ({(pct * 100).toFixed(1)}%)
                      </span>
                    </div>
                  );
                })}
              </div>
              {p.target_summary.is_imbalanced && (
                <p className="mt-4 text-xs text-amber-300">
                  ⚠ Class imbalance detected (smallest class is under 20% the size of the largest). Accuracy alone
                  will be misleading here — check PR-AUC and per-class metrics on the Overview tab.
                </p>
              )}
            </div>
          ) : (
            <div className="grid grid-cols-4 gap-3 text-center">
              <Stat label="Mean" value={p.target_summary.mean.toFixed(3)} />
              <Stat label="Std dev" value={p.target_summary.std.toFixed(3)} />
              <Stat label="Min" value={p.target_summary.min.toFixed(3)} />
              <Stat label="Max" value={p.target_summary.max.toFixed(3)} />
            </div>
          )}
        </Card>
      )}

      <Card>
        <SectionTitle sublabel="Every column in the uploaded dataset, computed directly from the data — not a schema guess.">
          Columns
        </SectionTitle>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="text-xs uppercase tracking-wide text-white/40">
                <th className="pb-2 pr-4">Column</th>
                <th className="pb-2 pr-4">Type</th>
                <th className="pb-2 pr-4">Missing</th>
                <th className="pb-2 pr-4">Unique</th>
                <th className="pb-2">Summary</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {p.columns.map((c) => (
                <tr key={c.name}>
                  <td className="py-2.5 pr-4 font-medium text-white/80">
                    {c.name} {c.is_target && <Pill tone="warn">target</Pill>}
                  </td>
                  <td className="py-2.5 pr-4 text-white/50">{c.dtype}</td>
                  <td className="py-2.5 pr-4">
                    {c.n_missing > 0 ? (
                      <Pill tone="bad">{c.n_missing} ({(c.pct_missing * 100).toFixed(1)}%)</Pill>
                    ) : (
                      <span className="text-white/30">none</span>
                    )}
                  </td>
                  <td className="py-2.5 pr-4 text-white/50">{c.n_unique}</td>
                  <td className="py-2.5 font-mono text-xs text-white/50">
                    {c.dtype === "numeric"
                      ? `min ${c.min?.toFixed(2)} · mean ${c.mean?.toFixed(2)} · max ${c.max?.toFixed(2)}`
                      : c.top_values?.map((tv) => `${tv.value} (${(tv.pct * 100).toFixed(0)}%)`).join(", ")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-white/5 py-3">
      <p className="text-xs text-white/40">{label}</p>
      <p className="font-display mt-1 text-lg font-bold text-white/80">{value}</p>
    </div>
  );
}
