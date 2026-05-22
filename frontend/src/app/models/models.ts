export interface Customer {
  id: string;
  fullName: string;
  dateOfBirth: string;
  gender: 'MALE' | 'FEMALE' | 'OTHER';
  idCardNumber: string;
  phone?: string;
  email?: string;
  address?: string;
  monthlyIncome: number;
  employer?: string;
  occupation?: string;
  isActive: boolean;
  createdAt: string;
  updatedAt: string;
}

export interface CreditApplication {
  id: string;
  customerId: string;
  customer?: Customer;
  requestedAmount: number;
  loanPurpose: string;
  loanTermMonths: number;
  status: ApplicationStatus;
  creditScore?: number;
  riskProbability?: number;
  riskLevel?: RiskLevel;
  aiExplanations?: ShapExplanation[];
  multiModelPayload?: MultiModelPayload;
  rejectionReason?: string;
  submittedAt?: string;
  scoredAt?: string;
  decidedAt?: string;
  decidedBy?: string;
  createdAt: string;
  updatedAt: string;
  // GMSC feature fields
  RevolvingUtilizationOfUnsecuredLines?: number;
  age?: number;
  NumberOfTime30_59DaysPastDueNotWorse?: number;
  DebtRatio?: number;
  MonthlyIncome?: number;
  NumberOfOpenCreditLinesAndLoans?: number;
  NumberOfTimes90DaysLate?: number;
  NumberRealEstateLoansOrLines?: number;
  NumberOfTime60_89DaysPastDueNotWorse?: number;
  NumberOfDependents?: number;
}

export type ApplicationStatus = 'DRAFT' | 'PROCESSING' | 'COMPLETED' | 'FAILED' | 'REJECTED' | 'APPROVED';
export type RiskLevel = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';

export interface ShapExplanation {
  feature: string;
  contribution: number;
  value: number;
  direction: 'POSITIVE' | 'NEGATIVE';
}

export interface ScoringRequest {
  applicationId: string;
  // GMSC dataset column names — sent directly to AI service
  age: number;
  RevolvingUtilizationOfUnsecuredLines: number;
  NumberOfTime30_59DaysPastDueNotWorse: number;
  DebtRatio: number;
  MonthlyIncome: number;
  NumberOfOpenCreditLinesAndLoans: number;
  NumberOfTimes90DaysLate: number;
  NumberRealEstateLoansOrLines: number;
  NumberOfTime60_89DaysPastDueNotWorse: number;
  NumberOfDependents: number;
}

export interface MultiModelPayload {
  models: Record<string, ModelResult>;
  ensemble: EnsembleResult;
  pipeline_metadata?: Record<string, unknown>;
  inference_ms: number;
}

export interface ModelResult {
  credit_score: number;
  risk_probability: number;
  risk_level: string;
  shap_explanations: ShapExplanation[];
  prediction: 'GOOD' | 'DEFAULT';
  probability: number;
  metrics?: Record<string, number>;
  model_name?: string;
  model_type?: string;
}

export interface EnsembleResult {
  credit_score: number;
  risk_probability: number;
  risk_level: string;
  shap_explanations: ShapExplanation[];
  voting: { approved: number; rejected: number };
  weighted_probability?: number;
  best_model_key?: string;
}

export interface PagedResponse<T> {
  content: T[];
  totalElements: number;
  totalPages: number;
  size: number;
  number: number;
}
