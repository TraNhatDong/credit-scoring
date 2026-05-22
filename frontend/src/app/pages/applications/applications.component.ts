import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import { CreditApplication } from '../../models/models';
import { RiskTagComponent } from '../../components/risk-tag/risk-tag.component';

const STATUS_LABELS: Record<string, string> = {
  DRAFT: 'Nháp', PROCESSING: 'Đang xử lý', COMPLETED: 'Hoàn thành',
  FAILED: 'Lỗi', REJECTED: 'Từ chối', APPROVED: 'Đã duyệt',
};

@Component({
  selector: 'app-applications',
  standalone: true,
  imports: [CommonModule, RouterLink, FormsModule, RiskTagComponent],
  templateUrl: './applications.component.html',
  styleUrl: './applications.component.css'
})
export class ApplicationsComponent implements OnInit {
  apps: CreditApplication[] = [];
  loading = true;
  page = 0;
  totalPages = 0;
  filter = '';

  readonly skeletonRows = Array(5).fill(0);
  readonly statusList = Object.entries(STATUS_LABELS).map(([key, label]) => ({ key, label }));

  constructor(private api: ApiService) {}

  ngOnInit() { this.fetch(0, ''); }

  fetch(p = 0, status = '') {
    this.loading = true;
    this.api.getApplications(p, 20, status || undefined).subscribe({
      next: data => {
        this.apps = data.content || [];
        this.totalPages = data.totalPages || 1;
      },
      error: () => { this.apps = []; },
      complete: () => { this.loading = false; }
    });
  }

  onFilterChange(status: string) {
    this.page = 0;
    this.filter = status;
    this.fetch(0, status);
  }

  goPage(p: number) {
    this.page = p;
    this.fetch(p, this.filter);
  }

  statusLabel(s: string): string {
    return STATUS_LABELS[s] ?? s;
  }
}
