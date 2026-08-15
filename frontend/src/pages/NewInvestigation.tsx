import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { UploadCloud, CheckCircle2, ArrowRight, Loader2 } from "lucide-react";
import { Card, SectionTitle, ErrorState } from "../components/ui";
import { uploadModel, uploadDataset, createInvestigation } from "../api/client";
import { addHistory } from "../lib/history";

type Step = 1 | 2 | 3;

export default function NewInvestigation() {
  const navigate = useNavigate();
  const [step, setStep] = useState<Step>(1);
  const [error, setError] = useState<string | null>(null);

  const [modelFile, setModelFile] = useState<File | null>(null);
  const [modelInfo, setModelInfo] = useState<any>(null);
  const [uploadingModel, setUploadingModel] = useState(false);

  const [datasetFile, setDatasetFile] = useState<File | null>(null);
  const [datasetInfo, setDatasetInfo] = useState<any>(null);
  const [uploadingDataset, setUploadingDataset] = useState(false);

  const [refFile, setRefFile] = useState<File | null>(null);
  const [refInfo, setRefInfo] = useState<any>(null);
  const [uploadingRef, setUploadingRef] = useState(false);

  const [targetColumn, setTargetColumn] = useState("");
  const [launching, setLaunching] = useState(false);

  async function handleModelUpload(f: File) {
    setModelFile(f);
    setUploadingModel(true);
    setError(null);
    try {
      const info = await uploadModel(f);
      setModelInfo(info);
    } catch (e: any) {
      setError(e.message);
      setModelFile(null);
    } finally {
      setUploadingModel(false);
    }
  }

  async function handleDatasetUpload(f: File) {
    setDatasetFile(f);
    setUploadingDataset(true);
    setError(null);
    try {
      const info = await uploadDataset(f);
      setDatasetInfo(info);
      if (info.columns.length) setTargetColumn(info.columns[info.columns.length - 1]);
    } catch (e: any) {
      setError(e.message);
      setDatasetFile(null);
    } finally {
      setUploadingDataset(false);
    }
  }

  async function handleRefUpload(f: File) {
    setRefFile(f);
    setUploadingRef(true);
    setError(null);
    try {
      const info = await uploadDataset(f);
      setRefInfo(info);
    } catch (e: any) {
      setError(e.message);
      setRefFile(null);
    } finally {
      setUploadingRef(false);
    }
  }

  async function launch() {
    if (!modelInfo || !datasetInfo || !targetColumn) return;
    setLaunching(true);
    setError(null);
    try {
      const inv = await createInvestigation({
        model_id: modelInfo.model_id,
        dataset_id: datasetInfo.dataset_id,
        target_column: targetColumn,
        reference_dataset_id: refInfo?.dataset_id,
      });
      addHistory({ id: inv.investigation_id, label: datasetFile?.name || "Investigation", createdAt: new Date().toISOString() });
      navigate(`/investigations/${inv.investigation_id}`);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLaunching(false);
    }
  }

  return (
    <div className="mx-auto max-w-2xl">
      <SectionTitle sublabel="Model → Dataset → Configure → Run. Every result that follows is computed from what you upload here.">
        New investigation
      </SectionTitle>

      <div className="mb-6 flex items-center gap-2">
        {[1, 2, 3].map((n) => (
          <div key={n} className={`h-1.5 flex-1 rounded-full ${n <= step ? "bg-gradient-to-r from-violet-500 to-amber-400" : "bg-white/10"}`} />
        ))}
      </div>

      {error && <div className="mb-4"><ErrorState message={error} /></div>}

      {step === 1 && (
        <Card>
          <h3 className="font-display text-base font-semibold text-white">1. Upload your model</h3>
          <p className="mt-1 text-sm text-white/50">.pkl or .joblib — a scikit-learn compatible estimator (or Pipeline).</p>
          <Uploader
            accept=".pkl,.joblib"
            file={modelFile}
            uploading={uploadingModel}
            onSelect={handleModelUpload}
            hint="Drop model file here"
          />
          {modelInfo && (
            <div className="mt-4 grid grid-cols-2 gap-2 text-xs text-white/60">
              <InfoRow label="Task type" value={modelInfo.task_type} />
              <InfoRow label="Estimator" value={modelInfo.estimator_class} />
              <InfoRow label="Expects features" value={String(modelInfo.n_features_expected ?? "any")} />
              <InfoRow label="Supports probabilities" value={modelInfo.supports_proba ? "yes" : "no"} />
            </div>
          )}
          <div className="mt-6 flex justify-end">
            <button
              disabled={!modelInfo}
              onClick={() => setStep(2)}
              className="inline-flex items-center gap-1.5 rounded-full bg-gradient-to-r from-violet-600 to-amber-500 px-5 py-2.5 text-sm font-semibold text-white disabled:opacity-40"
            >
              Next <ArrowRight size={14} />
            </button>
          </div>
        </Card>
      )}

      {step === 2 && (
        <Card>
          <h3 className="font-display text-base font-semibold text-white">2. Upload evaluation dataset</h3>
          <p className="mt-1 text-sm text-white/50">.csv or .xlsx including the true label / target column.</p>
          <Uploader accept=".csv,.xlsx,.xls" file={datasetFile} uploading={uploadingDataset} onSelect={handleDatasetUpload} hint="Drop CSV or Excel file here" />
          {datasetInfo && (
            <div className="mt-4 text-xs text-white/60">
              <p>{datasetInfo.n_rows} rows · {datasetInfo.columns.length} columns</p>
            </div>
          )}

          <div className="mt-6 border-t border-white/10 pt-5">
            <h4 className="font-display text-sm font-semibold text-white/90">Optional: reference / training dataset</h4>
            <p className="mt-1 text-xs text-white/40">Enables drift detection and reference-vs-current performance comparison.</p>
            <Uploader accept=".csv,.xlsx,.xls" file={refFile} uploading={uploadingRef} onSelect={handleRefUpload} hint="Drop reference CSV or Excel file here" compact />
          </div>

          <div className="mt-6 flex justify-between">
            <button onClick={() => setStep(1)} className="rounded-full px-4 py-2.5 text-sm font-medium text-white/60 hover:text-white">Back</button>
            <button
              disabled={!datasetInfo}
              onClick={() => setStep(3)}
              className="inline-flex items-center gap-1.5 rounded-full bg-gradient-to-r from-violet-600 to-amber-500 px-5 py-2.5 text-sm font-semibold text-white disabled:opacity-40"
            >
              Next <ArrowRight size={14} />
            </button>
          </div>
        </Card>
      )}

      {step === 3 && datasetInfo && (
        <Card>
          <h3 className="font-display text-base font-semibold text-white">3. Configure target column</h3>
          <p className="mt-1 text-sm text-white/50">Select which column holds the true label / value the model predicts.</p>
          <select
            value={targetColumn}
            onChange={(e) => setTargetColumn(e.target.value)}
            className="mt-4 w-full rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white outline-none focus:border-violet-400"
          >
            {datasetInfo.columns.map((c: string) => (
              <option key={c} value={c} className="bg-midnight-900">{c}</option>
            ))}
          </select>

          <div className="mt-6 flex justify-between">
            <button onClick={() => setStep(2)} className="rounded-full px-4 py-2.5 text-sm font-medium text-white/60 hover:text-white">Back</button>
            <button
              disabled={!targetColumn || launching}
              onClick={launch}
              className="inline-flex items-center gap-1.5 rounded-full bg-gradient-to-r from-violet-600 to-amber-500 px-5 py-2.5 text-sm font-semibold text-white disabled:opacity-40"
            >
              {launching ? <Loader2 size={14} className="animate-spin" /> : <CheckCircle2 size={14} />}
              {launching ? "Starting…" : "Run investigation"}
            </button>
          </div>
        </Card>
      )}
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-white/5 px-3 py-2">
      <p className="text-[10px] uppercase tracking-wide text-white/30">{label}</p>
      <p className="mt-0.5 font-medium text-white/80">{value}</p>
    </div>
  );
}

function Uploader({
  accept,
  file,
  uploading,
  onSelect,
  hint,
  compact = false,
}: {
  accept: string;
  file: File | null;
  uploading: boolean;
  onSelect: (f: File) => void;
  hint: string;
  compact?: boolean;
}) {
  return (
    <label
      className={`mt-3 flex cursor-pointer items-center justify-center gap-3 rounded-xl border border-dashed border-white/15 bg-white/[0.02] text-center transition-colors hover:border-violet-400/60 hover:bg-white/[0.04] ${
        compact ? "px-4 py-4" : "px-4 py-8"
      }`}
    >
      <input
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => e.target.files && onSelect(e.target.files[0])}
      />
      {uploading ? (
        <Loader2 size={18} className="animate-spin text-violet-300" />
      ) : file ? (
        <CheckCircle2 size={18} className="text-emerald-400" />
      ) : (
        <UploadCloud size={18} className="text-white/40" />
      )}
      <span className="text-sm text-white/60">
        {uploading ? "Uploading & validating…" : file ? file.name : hint}
      </span>
    </label>
  );
}
