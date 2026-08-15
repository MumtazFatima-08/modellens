export interface HistoryEntry {
  id: string;
  label: string;
  createdAt: string;
}

const KEY = "modellens_history_v1";

export function getHistory(): HistoryEntry[] {
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as HistoryEntry[];
    return parsed.sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1));
  } catch {
    return [];
  }
}

export function addHistory(entry: HistoryEntry) {
  const current = getHistory();
  const next = [entry, ...current.filter((h) => h.id !== entry.id)].slice(0, 50);
  window.localStorage.setItem(KEY, JSON.stringify(next));
}
