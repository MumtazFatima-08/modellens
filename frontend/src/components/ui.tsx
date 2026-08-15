import type { ReactNode } from "react";
import { AlertTriangle, Info, ShieldAlert, Loader2 } from "lucide-react";
import clsx from "clsx";

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={clsx("glass-card p-6", className)}>{children}</div>;
}

export function StatCard({
  label,
  value,
  sublabel,
  accent = "violet",
}: {
  label: string;
  value: string;
  sublabel?: string;
  accent?: "violet" | "amber" | "rose" | "sky";
}) {
  const accentMap = {
    violet: "from-violet-500/20 to-violet-500/0 text-violet-300",
    amber: "from-amber-500/20 to-amber-500/0 text-amber-300",
    rose: "from-rose-500/20 to-rose-500/0 text-rose-300",
    sky: "from-sky-500/20 to-sky-500/0 text-sky-300",
  };
  return (
    <Card className={clsx("bg-gradient-to-br", accentMap[accent])}>
      <p className="text-xs font-medium uppercase tracking-wider text-white/50">{label}</p>
      <p className="font-display mt-2 text-3xl font-bold text-white">{value}</p>
      {sublabel && <p className="mt-1 text-xs text-white/40">{sublabel}</p>}
    </Card>
  );
}

const severityConfig = {
  critical: { icon: ShieldAlert, color: "text-rose-400", bg: "bg-rose-500/10", border: "severity-critical", label: "Critical" },
  warning: { icon: AlertTriangle, color: "text-amber-400", bg: "bg-amber-500/10", border: "severity-warning", label: "Warning" },
  info: { icon: Info, color: "text-sky-400", bg: "bg-sky-500/10", border: "severity-info", label: "Info" },
};

export function SeverityBadge({ severity }: { severity: "critical" | "warning" | "info" }) {
  const cfg = severityConfig[severity];
  const Icon = cfg.icon;
  return (
    <span className={clsx("inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold", cfg.bg, cfg.color)}>
      <Icon size={12} /> {cfg.label}
    </span>
  );
}

export function FindingCard({ finding }: { finding: { severity: "critical" | "warning" | "info"; title: string; evidence: Record<string, any>; interpretation: string; recommendation: string } }) {
  const cfg = severityConfig[finding.severity];
  return (
    <div className={clsx("glass-card border-l-4 p-5", cfg.border)}>
      <div className="flex items-center justify-between gap-3">
        <h4 className="font-display text-base font-semibold text-white">{finding.title}</h4>
        <SeverityBadge severity={finding.severity} />
      </div>
      <p className="mt-3 text-sm leading-relaxed text-white/70">{finding.interpretation}</p>
      {Object.keys(finding.evidence || {}).length > 0 && (
        <div className="mt-3 flex flex-wrap gap-2">
          {Object.entries(finding.evidence).map(([k, v]) => (
            <span key={k} className="font-mono rounded-md bg-white/5 px-2 py-1 text-[11px] text-white/50">
              {k}: {Array.isArray(v) ? v.join(", ") : String(v)}
            </span>
          ))}
        </div>
      )}
      <p className="mt-3 text-sm text-white/50">
        <span className="font-semibold text-white/70">Recommended next step: </span>
        {finding.recommendation}
      </p>
    </div>
  );
}

export function EmptyState({ icon, title, description, action }: { icon: ReactNode; title: string; description: string; action?: ReactNode }) {
  return (
    <Card className="flex flex-col items-center gap-3 py-16 text-center">
      <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-white/5 text-white/40">{icon}</div>
      <h3 className="font-display text-lg font-semibold text-white">{title}</h3>
      <p className="max-w-md text-sm text-white/50">{description}</p>
      {action}
    </Card>
  );
}

export function LoadingState({ message }: { message: string }) {
  return (
    <Card className="flex flex-col items-center gap-3 py-16 text-center">
      <Loader2 className="animate-spin text-violet-400" size={28} />
      <p className="text-sm text-white/60">{message}</p>
    </Card>
  );
}

export function ErrorState({ message }: { message: string }) {
  return (
    <Card className="severity-critical border-l-4 py-8 text-center">
      <p className="text-sm text-rose-300">{message}</p>
    </Card>
  );
}

export function SectionTitle({ children, sublabel }: { children: ReactNode; sublabel?: string }) {
  return (
    <div className="mb-4">
      <h2 className="font-display text-xl font-bold text-white">{children}</h2>
      {sublabel && <p className="mt-1 text-sm text-white/50">{sublabel}</p>}
    </div>
  );
}

export function Pill({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "good" | "bad" | "warn" }) {
  const toneMap = {
    neutral: "bg-white/10 text-white/70",
    good: "bg-emerald-500/15 text-emerald-300",
    bad: "bg-rose-500/15 text-rose-300",
    warn: "bg-amber-500/15 text-amber-300",
  };
  return <span className={clsx("rounded-full px-2.5 py-0.5 text-xs font-medium", toneMap[tone])}>{children}</span>;
}
