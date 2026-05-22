import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule, RouterLink],
  templateUrl: './dashboard.component.html',
  styleUrl: './dashboard.component.css'
})
export class DashboardComponent {
  readonly tips = [
    'Điểm tín dụng từ 300 – 850 (FICO-like)',
    'Điểm càng cao → rủi ro càng thấp',
    'SHAP giải thích từng yếu tố ảnh hưởng điểm',
    'XGBoost Champion: AUC 0.87, Logistic Regression Baseline: AUC 0.86',
    'Isotonic Calibration — xác suất nợ xấu được hiệu chỉnh chính xác',
  ];
}
