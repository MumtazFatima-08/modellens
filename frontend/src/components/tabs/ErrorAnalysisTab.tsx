import { Card, SectionTitle, Pill } from "../ui";
import type { InvestigationResult } from "../../types";

export default function ErrorAnalysisTab({ result }: { result: InvestigationResult }) {
  const conc = result.error_concentration || [];
  const examples = result.error_examples || [];
  const isClf = result.model_info.task_type === "classification";

  return (
    <div className="space-y-8">
      <Card>
        <SectionTitle sublabel="Statistical test (two-proportion z-test for numeric top-quartile groups, chi-square for categorical groups) comparing error rate in each candidate group vs. the rest. Observed association, not proven cause.">
          Error concentration by feature
        </SectionTitle>
        {conc.length === 0 ? (
          <p className="text-sm text-white/40">No feature showed a statistically detectable association with errors at the current sample size.</p>
        ) : (
          <div className="space-y-3">
            {conc.slice(0, 10).map((c, i) => (
              <div key={i} className="flex items-center justify-between rounded-xl bg-white/5 px-4 py-3">
                <div>
                  <p className="font-mono text-sm text-white/80">{c.condition}</p>
                  <p className="mt-1 text-xs text-white/40">
                    n={c.group_size} · group error rate {(c.group_error_rate * 100).toFixed(1)}% vs.{" "}
                    {(c.overall_error_rate * 100).toFixed(1)}% overall · p={c.p_value.toFixed(4)}
                  </p>
                </div>
                <Pill tone={c.significant ? (c.lift > 0 ? "bad" : "good") : "neutral"}>
                  {c.significant ? `${c.lift > 0 ? "+" : ""}${(c.lift * 100).toFixed(1)}pp` : "not significant"}
                </Pill>
              </div>
            ))}
          </div>
        )}
      </Card>

      <Card>
        <SectionTitle sublabel={`Top ${examples.length} error examples, sorted by ${isClf ? "confidence (highest first)" : "absolute error (largest first)"}. Each explanation describes which inputs were statistically unusual for that row — not a claim about the model's internal reasoning.`}>
          Error examples
        </SectionTitle>
        <div className="space-y-2">
          {examples.map((e, i) => (
            <div key={i} className="rounded-xl bg-white/5 px-4 py-3">
              <div className="flex flex-wrap items-center gap-2 text-xs">
                <span className="font-semibold text-rose-300">True: {String(e.true_label)}</span>
                <span className="text-white/30">→</span>
                <span className="font-semibold text-white/70">Predicted: {String(e.predicted_label)}</span>
                {isClf && e.confidence != null && (
                  <span className="ml-auto rounded-full bg-white/10 px-2 py-0.5 text-white/50">{(e.confidence * 100).toFixed(1)}% confidence</span>
                )}
                {!isClf && e.abs_error != null && (
                  <span className="ml-auto rounded-full bg-white/10 px-2 py-0.5 text-white/50">abs. error {e.abs_error.toFixed(3)}</span>
                )}
              </div>
              <p className="mt-2 text-xs leading-relaxed text-white/50">{e.explanation}</p>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
