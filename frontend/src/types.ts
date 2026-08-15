export interface Finding {
  severity: "critical" | "warning" | "info";
  title: string;
  evidence: Record<string, any>;
  interpretation: string;
  recommendation: string;
}

export interface PerClassMetric {
  class: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface Evaluation {
  task_type: "classification" | "regression";
  accuracy?: number;
  precision_macro?: number;
  recall_macro?: number;
  f1_macro?: number;
  roc_auc?: number | null;
  pr_auc?: number | null;
  confusion_matrix?: number[][];
  confusion_matrix_labels?: string[];
  per_class?: PerClassMetric[];
  class_distribution?: Record<string, number>;
  n_samples: number;
  mae?: number;
  mse?: number;
  rmse?: number;
  r2?: number;
  residual_mean?: number;
  residual_std?: number;
  residual_histogram?: { counts: number[]; bin_edges: number[] };
}

export interface Slice {
  rule: string;
  sample_size: number;
  group_error_rate: number;
  overall_error_rate: number;
  performance_gap: number;
  p_value: number;
  significant: boolean;
}

export interface ErrorConcentration {
  feature: string;
  condition: string;
  group_error_rate: number;
  rest_error_rate: number;
  overall_error_rate: number;
  lift: number;
  group_size: number;
  p_value: number;
  significant: boolean;
}

export interface Confidence {
  available: boolean;
  reason?: string;
  mean_confidence_correct?: number;
  mean_confidence_incorrect?: number;
  high_confidence_threshold?: number;
  n_high_confidence_errors?: number;
  n_total_errors?: number;
  pct_errors_high_confidence?: number;
  expected_calibration_error?: number;
  reliability_curve?: { predicted_confidence: number[]; observed_accuracy: number[] };
  confidence_histogram?: { bin_edges: number[]; correct_counts: number[]; incorrect_counts: number[] };
}

export interface DriftEntry {
  feature: string;
  feature_type: "numeric" | "categorical";
  psi: number;
  status: "normal" | "drift" | "severe_drift" | "unknown";
  ks_statistic?: number;
  ks_p_value?: number;
  wasserstein_distance?: number;
  chi2_statistic?: number;
  chi2_p_value?: number;
}

export interface FeatureImportance {
  available: boolean;
  reason?: string;
  method?: string;
  importances?: { feature: string; importance: number; importance_normalized: number }[];
}

export interface DatasetColumnProfile {
  name: string;
  is_target: boolean;
  dtype: "numeric" | "categorical";
  n_missing: number;
  pct_missing: number;
  n_unique: number;
  min?: number;
  max?: number;
  mean?: number;
  median?: number;
  std?: number;
  top_values?: { value: string; count: number; pct: number }[];
}

export interface DatasetProfile {
  n_rows: number;
  n_columns: number;
  columns: DatasetColumnProfile[];
  duplicate_rows: number;
  target_summary:
    | {
        type: "categorical";
        class_counts: Record<string, number>;
        class_balance_pct: Record<string, number>;
        is_imbalanced: boolean;
      }
    | { type: "continuous"; min: number; max: number; mean: number; std: number }
    | null;
}

export interface ErrorExample {
  features: Record<string, any>;
  true_label: any;
  predicted_label: any;
  confidence?: number | null;
  residual?: number;
  abs_error?: number;
  explanation: string;
}

export interface InvestigationResult {
  model_info: { task_type: string; estimator_class: string; supports_proba: boolean; classes: string[] | null };
  dataset_info: { n_samples: number; n_features_used: number; feature_columns: string[]; target_column: string };
  dataset_profile: DatasetProfile;
  evaluation: Evaluation;
  reference_evaluation: Evaluation | null;
  error_examples: ErrorExample[];
  error_concentration: ErrorConcentration[];
  slices: Slice[];
  confidence: Confidence;
  feature_importance: FeatureImportance;
  drift: DriftEntry[] | null;
  ood: { available: boolean; reason?: string; n_flagged_potential_ood?: number; pct_flagged?: number } | null;
  findings: Finding[];
}
