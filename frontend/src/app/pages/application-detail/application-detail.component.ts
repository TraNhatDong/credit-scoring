import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, Router } from '@angular/router';
import { ApiService } from '../../services/api.service';
import { CreditApplication } from '../../models/models';
import { RiskTagComponent } from '../../components/risk-tag/risk-tag.component';

const STATUS_LABELS: Record<string, string> = {
  DRAFT: 'Nháp', PROCESSING: 'Đang xử lý', COMPLETED: 'Hoàn thành',
  FAILED: 'Lỗi', REJECTED: 'Từ chối', APPROVED: 'Đã duyệt',
};

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

type RowPair = [string, string];

@Component({
  selector: 'app-application-detail',
  standalone: true,
  imports: [CommonModule, RiskTagComponent],
  templateUrl: './application-detail.component.html',
  styleUrl: './application-detail.component.css'
})
export class ApplicationDetailComponent implements OnInit {
  app: CreditApplication | null = null;
  loading = true;

  customerRows: RowPair[] = [];
  loanRows: RowPair[] = [];

  constructor(
    private api: ApiService,
    private route: ActivatedRoute,
    private router: Router,
  ) {}

  ngOnInit() {
    const id = this.route.snapshot.paramMap.get('id')!;
    this.api.getApplication(id).subscribe({
      next: app => {
        this.app = app;
        this.buildRows(app);
      },
      error: () => { this.app = null; },
      complete: () => { this.loading = false; }
    });
  }

  private buildRows(app: CreditApplication) {
    const c = app.customer;
    this.customerRows = [
      ['Họ tên', c?.fullName ?? '—'],
      ['Số CCCD', c?.idCardNumber ?? '—'],
      ['Ngày sinh', c?.dateOfBirth ?? '—'],
      ['Giới tính', c?.gender ?? '—'],
      ['Điện thoại', c?.phone ?? '—'],
      ['Email', c?.email ?? '—'],
      ['Thu nhập tháng', c?.monthlyIncome != null ? c.monthlyIncome.toLocaleString('vi-VN') + ' VND' : '—'],
      ['Nghề nghiệp', c?.occupation ?? '—'],
    ];
    this.loanRows = [
      ['Số tiền yêu cầu', app.requestedAmount != null ? app.requestedAmount.toLocaleString('vi-VN') + ' VND' : '—'],
      ['Mục đích', app.loanPurpose],
      ['Thời hạn', app.loanTermMonths + ' tháng'],
      ['Ngày nộp', app.submittedAt ? new Date(app.submittedAt).toLocaleString('vi-VN') : '—'],
      ['Ngày chấm điểm', app.scoredAt ? new Date(app.scoredAt).toLocaleString('vi-VN') : '—'],
      ['Ngày quyết định', app.decidedAt ? new Date(app.decidedAt).toLocaleString('vi-VN') : '—'],
    ];
  }

  statusLabel(s: string): string { return STATUS_LABELS[s] ?? s; }
  featureLabel(f: string): string { return FEATURE_LABELS[f] ?? f; }

  goBack() { this.router.navigate(['/applications']); }
  goToScoring() { if (this.app) this.router.navigate(['/scoring', this.app.id]); }
}
