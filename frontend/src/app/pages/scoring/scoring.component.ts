import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { ApiService } from '../../services/api.service';
import { ShapExplanation } from '../../models/models';

const FEATURE_LABELS: Record<string, string> = {
  age: 'Tuổi',
  RevolvingUtilizationOfUnsecuredLines: 'Tỷ lệ SD hạn mức tín dụng',
  'NumberOfTime30-59DaysPastDueNotWorse': 'Số lần trễ 30-59 ngày',
  DebtRatio: 'Tỷ lệ Nợ/Thu nhập',
  MonthlyIncome: 'Thu nhập tháng',
  NumberOfOpenCreditLinesAndLoans: 'Số hạn mức tín dụng đang mở',
  NumberOfTimes90DaysLate: 'Số lần trễ ≥90 ngày',
  NumberRealEstateLoansOrLines: 'Số khoản vay bất động sản',
  'NumberOfTime60-89DaysPastDueNotWorse': 'Số lần trễ 60-89 ngày',
  NumberOfDependents: 'Số người phụ thuộc',
};

// ── Sub-components ──────────────────────────────────────────────────────────────

@Component({
  selector: 'app-score-gauge',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div [style.position]="'relative'" [style.width.px]="size" [style.height.px]="size / 1.5" style="margin:0 auto">
      <svg [attr.viewBox]="'0 0 ' + size + ' ' + (size / 1.5)" [style.width]="'100%'" [style.height]="'100%'" [style.overflow]="'visible'">
        <path [attr.d]="bgArc" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="14" stroke-linecap="round"/>
        <path [attr.d]="fgArc" fill="none" [attr.stroke]="color" stroke-width="14" stroke-linecap="round"
          [attr.stroke-dasharray]="arcDash"/>
      </svg>
      <div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center">
        <span [style.font-size.px]="size / 6" [style.font-weight]="800"
          [style.fontFamily]="'JetBrains Mono,monospace'" [style.color]="color">{{ score }}</span>
        <span style="font-size:0.7rem;color:var(--color-text-muted);text-transform:uppercase;letter-spacing:0.05em">Credit Score</span>
      </div>
    </div>
  `
})
export class ScoreGaugeComponent {
  @Input() score = 0;
  @Input() size = 180;

  get pct(): number { return Math.max(0, Math.min(100, (this.score - 300) / 550 * 100)); }
  get color(): string { return this.score >= 700 ? '#22c55e' : this.score >= 580 ? '#f59e0b' : '#ef4444'; }
  get r(): number { return this.size / 2 - 20; }
  get circ(): number { return 2 * Math.PI * this.r; }
  get arcLen(): number { return (this.pct / 100) * this.circ; }
  get arcDash(): string { return this.arcLen + ' ' + this.circ; }
  get bgArc(): string { return `M 20 ${this.size / 1.5 - 10} A ${this.r} ${this.r} 0 0 1 ${this.size - 20} ${this.size / 1.5 - 10}`; }
  get fgArc(): string { return `M 20 ${this.size / 1.5 - 10} A ${this.r} ${this.r} 0 0 1 ${this.size - 20} ${this.size / 1.5 - 10}`; }
}

@Component({
  selector: 'app-shap-bar',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div>
      @for (exp of explanations.slice(0, 8); track exp.feature) {
        <div style="margin-bottom:10px">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px">
            <span class="shap-bar-label">{{ featureLabel(exp.feature) }}</span>
            <span style="display:flex;gap:8px;align-items:center">
              <span style="font-size:0.72rem;color:var(--color-text-muted);font-family:'JetBrains Mono,monospace'">{{ formatValue(exp.value) }}</span>
              <span [style.fontSize]="'0.72rem'" [style.fontFamily]="'JetBrains Mono,monospace'" [style.fontWeight]="700"
                [style.color]="exp.direction === 'POSITIVE' ? 'var(--color-critical)' : 'var(--color-low)'">
                {{ exp.contribution.toFixed(1) }}
              </span>
            </span>
          </div>
          <div class="shap-bar-track">
            <div [class]="'shap-bar-fill ' + (exp.direction === 'POSITIVE' ? 'shap-bar-fill--positive' : 'shap-bar-fill--negative')"
              [style.width.%]="barWidth(exp)"></div>
          </div>
        </div>
      }
    </div>
  `
})
export class ShapBarComponent {
  @Input() explanations: ShapExplanation[] = [];

  featureLabel(f: string): string { return FEATURE_LABELS[f] ?? f; }
  formatValue(v: unknown): string { return typeof v === 'number' ? v.toLocaleString('vi-VN') : String(v); }
  barWidth(exp: ShapExplanation): number {
    const max = Math.max(...this.explanations.map(e => Math.abs(e.contribution)), 0.001);
    return (Math.abs(exp.contribution) / max) * 100;
  }
}

@Component({
  selector: 'app-risk-tag-inline',
  standalone: true,
  imports: [CommonModule],
  template: `
    <span [style.display]="'inline-flex'" [style.alignItems]="'center'" [style.gap.px]="6"
      [style.padding]="'4px 10px'" [style.borderRadius]="'9999px'"
      [style.background]="bgColor" [style.color]="textColor"
      [style.border]="'1px solid ' + borderColor"
      [style.fontWeight]="700" [style.fontSize]="'0.75rem'">
      {{ level }}
    </span>
  `
})
export class RiskTagInlineComponent {
  @Input() level = 'MEDIUM';
  get bgColor(): string {
    return { LOW: 'rgba(34,197,94,0.15)', MEDIUM: 'rgba(245,158,11,0.15)', HIGH: 'rgba(249,115,22,0.15)', CRITICAL: 'rgba(239,68,68,0.15)' }[this.level] ?? 'rgba(245,158,11,0.15)';
  }
  get textColor(): string {
    return { LOW: '#22c55e', MEDIUM: '#f59e0b', HIGH: '#f97316', CRITICAL: '#ef4444' }[this.level] ?? '#f59e0b';
  }
  get borderColor(): string {
    return { LOW: 'rgba(34,197,94,0.3)', MEDIUM: 'rgba(245,158,11,0.3)', HIGH: 'rgba(249,115,22,0.3)', CRITICAL: 'rgba(239,68,68,0.3)' }[this.level] ?? 'rgba(245,158,11,0.3)';
  }
}

// ── Main ScoringComponent ──────────────────────────────────────────────────────

interface ScoreCard {
  key: string;
  score: number;
  prob: number;
  risk: string;
  shapExplanations?: ShapExplanation[];
  voting?: { approved: number; rejected: number };
}

interface MultiResult {
  models: { [key: string]: ScoreCard };
  ensemble: ScoreCard;
}

@Component({
  selector: 'app-scoring',
  standalone: true,
  imports: [CommonModule, FormsModule, ScoreGaugeComponent, ShapBarComponent, RiskTagInlineComponent],
  templateUrl: './scoring.component.html',
  styleUrl: './scoring.component.css',
})
export class ScoringComponent {
  scoring = false;
  errorMsg = '';
  activeModel = 'ensemble';
  multiResult: MultiResult | null = null;

  form = {
    age: 45, MonthlyIncome: 9120, RevolvingUtilizationOfUnsecuredLines: 0.3,
    DebtRatio: 0.5, NumberOfTime30_59DaysPastDueNotWorse: 0,
    NumberOfTime60_89DaysPastDueNotWorse: 0, NumberOfTimes90DaysLate: 0,
    NumberOfOpenCreditLinesAndLoans: 13, NumberRealEstateLoansOrLines: 6,
    NumberOfDependents: 2,
  };

  readonly algorithms = [
    { name: 'Logistic Regression', cat: 'Linear Baseline', icon: '📊', desc: 'Hệ số toán học — dễ hiểu, tuân thủ quy định', color: '#60a5fa' },
    { name: 'XGBoost', cat: 'Gradient Boosting', icon: '🚀', desc: 'Sửa lỗi liên tục — độ chính xác cao nhất (Champion)', color: '#34d399' },
  ];
  readonly tags = ['SHAP giải thích AI', 'Ensemble voting', 'Isotonic Calibration'];

  constructor(private api: ApiService, private router: Router) {}

  get modelKeys(): string[] { return this.multiResult ? Object.keys(this.multiResult.models) : []; }

  get modelCards(): ScoreCard[] {
    if (!this.multiResult) return [];
    return Object.values(this.multiResult.models);
  }

  get activeCard(): ScoreCard | null {
    if (!this.multiResult) return null;
    if (this.activeModel === 'ensemble') return this.multiResult.ensemble;
    return this.multiResult.models?.[this.activeModel] ?? null;
  }

  get displayScore(): number { return this.activeCard?.score ?? 0; }
  get displayProb(): number { return this.activeCard?.prob ?? 0; }
  get displayRisk(): string { return this.activeCard?.risk ?? 'MEDIUM'; }

  get displayShap(): ShapExplanation[] {
    const card = this.activeCard;
    if (!card) return [];
    return card['shapExplanations'] as unknown as ShapExplanation[] ?? [];
  }

  setActiveModel(key: string) { this.activeModel = key; }

  modelLabel(key: string): string {
    return {
      champion_xgb: 'XGBoost (Champion)',
      benchmark_lr:  'Logistic Regression',
      xgb:           'XGBoost (Champion)',
      lr:            'Logistic Regression',
      ensemble:      'Ensemble (TB)',
    }[key] ?? key;
  }

  handleScore() {
    this.scoring = true;
    this.errorMsg = '';
    this.api.score(this.form as Record<string, number>).subscribe({
      next: (raw) => {
        const data = raw as Record<string, unknown>;
        const modelsMap = data['models'] as Record<string, Record<string, unknown>> | undefined;
        const ensemble = data['ensemble'] as Record<string, unknown>;

        // Backend returns models with keys: xgb, lr (mapped by Spring ScoringController)
        // Fallback to raw single-model response if modelsMap is absent
        const xgbData = modelsMap?.['xgb'] ?? modelsMap?.['champion_xgb'] ?? (modelsMap ? undefined : data);
        const lrData  = modelsMap?.['lr']  ?? modelsMap?.['benchmark_lr']  ?? (modelsMap ? undefined : data);

        const toCard = (m: Record<string, unknown>, key: string): ScoreCard => ({
          key,
          score:      (m['creditScore'] ?? m['credit_score'] ?? 0) as number,
          prob:       (m['riskProbability'] ?? m['risk_probability'] ?? 0) as number,
          risk:       (m['riskLevel'] ?? m['risk_level'] ?? 'MEDIUM') as string,
          shapExplanations: (m['shapExplanations'] ?? m['shap_explanations'] ?? []) as ShapExplanation[],
          voting:     (m['voting'] ?? undefined) as { approved: number; rejected: number } | undefined,
        });

        this.multiResult = {
          models: {
            xgb:   xgbData ? toCard(xgbData, 'xgb')   : toCard({}, 'xgb'),
            lr:    lrData  ? toCard(lrData,  'lr')    : toCard({}, 'lr'),
          },
          ensemble: toCard(ensemble as Record<string, unknown>, 'ensemble'),
        };
        this.activeModel = 'ensemble';
        this.scoring = false;
      },
      error: (e: unknown) => {
        const err = (e as { error?: { message?: string } });
        this.errorMsg = err?.error?.message ?? 'Lỗi khi chấm điểm. Đảm bảo AI Service đang chạy.';
        this.scoring = false;
      },
    });
  }
}
