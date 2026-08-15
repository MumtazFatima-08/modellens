const API_BASE = (import.meta as any).env?.VITE_API_URL || "http://localhost:8000";

async function handle(res: Response) {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      /* noop */
    }
    throw new Error(detail);
  }
  return res.json();
}

export async function uploadModel(file: File) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${API_BASE}/api/models/upload`, { method: "POST", body: fd });
  return handle(res);
}

export async function uploadDataset(file: File) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${API_BASE}/api/datasets/upload`, { method: "POST", body: fd });
  return handle(res);
}

export async function listSamples() {
  const res = await fetch(`${API_BASE}/api/samples`);
  return handle(res);
}

export async function loadSample(caseId: string) {
  const res = await fetch(`${API_BASE}/api/samples/load`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ case_id: caseId }),
  });
  return handle(res);
}

export async function createInvestigation(payload: {
  model_id: string;
  dataset_id: string;
  target_column: string;
  feature_columns?: string[];
  reference_dataset_id?: string;
}) {
  const res = await fetch(`${API_BASE}/api/investigations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handle(res);
}

export async function getInvestigationStatus(id: string) {
  const res = await fetch(`${API_BASE}/api/investigations/${id}`);
  return handle(res);
}

export async function getInvestigationResults(id: string) {
  const res = await fetch(`${API_BASE}/api/investigations/${id}/results`);
  return handle(res);
}

export function reportUrl(id: string, format: "json" | "csv" | "pdf") {
  return `${API_BASE}/api/investigations/${id}/report?format=${format}`;
}

export { API_BASE };
