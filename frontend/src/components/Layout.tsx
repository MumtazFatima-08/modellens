import { NavLink } from "react-router-dom";
import { Search, LayoutDashboard, FlaskConical, FolderClock } from "lucide-react";
import type { ReactNode } from "react";

const navItems = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard, end: true },
  { to: "/investigate", label: "New Investigation", icon: Search, end: false },
  { to: "/samples", label: "Samples", icon: FlaskConical, end: false },
  { to: "/history", label: "History", icon: FolderClock, end: false },
];

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen">
      <div className="modellens-bg" />
      <header className="sticky top-4 z-40 mx-4 md:mx-8">
        <div className="glass-nav mx-auto flex max-w-6xl items-center justify-between rounded-full px-5 py-3">
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-br from-violet-500 to-amber-400">
              <Search size={16} className="text-midnight-950" strokeWidth={2.5} />
            </div>
            <span className="font-display text-lg font-extrabold tracking-tight">ModelLens</span>
          </div>
          <nav className="hidden items-center gap-1 md:flex">
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  `rounded-full px-4 py-2 text-sm font-medium transition-colors ${
                    isActive
                      ? "bg-white/10 text-white"
                      : "text-white/60 hover:bg-white/5 hover:text-white/90"
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
          <NavLink
            to="/investigate"
            className="rounded-full bg-gradient-to-r from-violet-600 to-amber-500 px-4 py-2 text-sm font-semibold text-white shadow-lg shadow-violet-900/30 transition-transform hover:scale-[1.03]"
          >
            Run Investigation
          </NavLink>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-4 pb-24 pt-10 md:px-8">{children}</main>
      <p className="mx-auto max-w-6xl px-4 pb-10 text-center text-xs text-white/30 md:px-8">
        Your model predicts. ModelLens investigates. — All figures shown are computed by the
        analysis engine from your uploaded model and data.
      </p>
    </div>
  );
}
