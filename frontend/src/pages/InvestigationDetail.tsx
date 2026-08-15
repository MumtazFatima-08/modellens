import { useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { Download } from "lucide-react";
import { getInvestigationStatus, getInvestigationResults, reportUrl } from "../api/client";
import { ErrorState, LoadingState } from "../components/ui";
import type { InvestigationResult } from "../types";

import OverviewTab from "../components/tabs/OverviewTab";
import DatasetTab from "../components/tabs/DatasetTab";
import ErrorAnalysisTab from "../components/tabs/ErrorAnalysisTab";
import SlicesTab from "../components/tabs/SlicesTab";
import DriftTab from "../components/tabs/DriftTab";
import CalibrationTab from "../components/tabs/CalibrationTab";
import FindingsTab from "../components/tabs/FindingsTab";

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "dataset", label: "Dataset" },
  { id: "errors", label: "Error Analysis" },
  { id: "slices", label: "Slice Discovery" },
  { id: "drift", label: "Drift" },
  { id: "calibration", label: "Confidence & Calibration" },
  { id: "findings", label: "Findings" },
] as const;

type TabId = (typeof TABS)[number]["id"];

export default function InvestigationDetail() {
  const { id } = useParams<{ id: string }>();
  const [status, setStatus] = useState<any>(null);
  const [result, setResult] = useState<InvestigationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<TabId>("overview");
  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    if (!id) return;
    let cancelled = false;

    async function poll() {
      try {
        const s = await getInvestigationStatus(id!);
        if (cancelled) return;
        setStatus(s);
        if (s.status === "completed") {
          const r = await getInvestigationResults(id!);
          if (!cancelled) setResult(r);
        } else if (s.status === "failed") {
          setError(s.error || "Investigation failed.");
        } else {
          pollRef.current = window.setTimeout(poll, 600);
        }
      } catch (e: any) {
        if (!cancelled) setError(e.message);
      }
    }
    poll();
    return () => {
      cancelled = true;
      if (pollRef.current) window.clearTimeout(pollRef.current);
    };
  }, [id]);

  if (error) return <ErrorState message={error} />;
  if (!status || (status.status !== "completed" && status.status !== "failed")) {
    return (
      <div className="mx-auto max-w-xl">
        <LoadingState message={status ? `${status.message || "Working…"} (${status.progress ?? 0}%)` : "Connecting…"} />
        <div className="mt-4 h-1.5 w-full overflow-hidden rounded-full bg-white/5">
          <div
            className="h-full rounded-full bg-gradient-to-r from-violet-500 to-amber-400 transition-all"
            style={{ width: `${status?.progress ?? 5}%` }}
          />
        </div>
      </div>
    );
  }
  if (!result) return <LoadingState message="Loading results…" />;

  return (
    <div>
      <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="font-mono text-xs text-white/40">Investigation {id}</p>
          <h1 className="font-display text-2xl font-bold text-white">
            {result.model_info.estimator_class} · {result.model_info.task_type}
          </h1>
        </div>
        <div className="flex gap-2">
          {(["json", "csv", "pdf"] as const).map((fmt) => (
            <a
              key={fmt}
              href={reportUrl(id!, fmt)}
              className="inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/5 px-3.5 py-2 text-xs font-semibold text-white/80 hover:bg-white/10"
            >
              <Download size={13} /> {fmt.toUpperCase()}
            </a>
          ))}
        </div>
      </div>

      <div className="mb-6 flex gap-1 overflow-x-auto rounded-full border border-white/10 bg-white/[0.03] p-1">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`whitespace-nowrap rounded-full px-4 py-2 text-xs font-semibold transition-colors ${
              tab === t.id ? "bg-white text-midnight-950" : "text-white/60 hover:text-white"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === "overview" && <OverviewTab result={result} />}
      {tab === "dataset" && <DatasetTab result={result} />}
      {tab === "errors" && <ErrorAnalysisTab result={result} />}
      {tab === "slices" && <SlicesTab result={result} />}
      {tab === "drift" && <DriftTab result={result} />}
      {tab === "calibration" && <CalibrationTab result={result} />}
      {tab === "findings" && <FindingsTab result={result} />}
    </div>
  );
}
