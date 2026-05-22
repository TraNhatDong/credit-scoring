import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import { Customer } from '../../models/models';

@Component({
  selector: 'app-customers',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:24px">
        <div>
          <h2>Khách hàng</h2>
          <p class="text-muted text-sm" style="margin-top:4px">Quản lý hồ sơ khách hàng</p>
        </div>
        <div style="display:flex;gap:12px">
          <button class="btn btn--secondary" (click)="fetch(page, search)">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-3.5"/>
            </svg>
          </button>
        </div>
      </div>

      <!-- Search -->
      <div class="glass-card" style="padding:12px 16px;margin-bottom:16px">
        <div style="display:flex;gap:12px">
          <input class="glass-input" placeholder="Tìm theo tên…"
            [(ngModel)]="search" (keydown.enter)="doSearch()"
            style="max-width:320px" />
          <button class="btn btn--primary" (click)="doSearch()">Tìm kiếm</button>
          @if (search) {
            <button class="btn btn--secondary" (click)="clearSearch()">Xóa</button>
          }
        </div>
      </div>

      <!-- Table -->
      <div class="glass-card" style="padding:0;overflow:hidden">
        @if (loading) {
          <div style="padding:24px">
            @for (i of skeletonRows; track i) {
              <div class="skeleton" style="height:52px;margin-bottom:8px;border-radius:8px"></div>
            }
          </div>
        } @else if (customers.length === 0) {
          <div style="padding:48px;text-align:center;color:var(--color-text-muted)">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"
              style="margin-bottom:12px;opacity:0.4">
              <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>
              <circle cx="9" cy="7" r="4"/>
              <path d="M23 21v-2a4 4 0 0 0-3-3.87"/>
              <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
            </svg>
            <p>Chưa có khách hàng nào</p>
          </div>
        } @else {
          <table class="glass-table">
            <thead>
              <tr>
                <th>Họ tên</th><th>CCCD</th><th>Ngày sinh</th><th>Giới tính</th>
                <th>Thu nhập tháng</th><th>Nghề nghiệp</th><th>Trạng thái</th><th>Ngày tạo</th>
              </tr>
            </thead>
            <tbody>
              @for (c of customers; track c.id) {
                <tr>
                  <td style="font-weight:600">{{ c.fullName }}</td>
                  <td style="font-family:'JetBrains Mono',monospace;font-size:0.8rem">{{ c.idCardNumber }}</td>
                  <td>{{ c.dateOfBirth }}</td>
                  <td>{{ genderLabel(c.gender) }}</td>
                  <td style="font-family:'JetBrains Mono',monospace;font-size:0.85rem">
                    {{ c.monthlyIncome.toLocaleString('vi-VN') }}
                  </td>
                  <td>{{ c.occupation ?? '—' }}</td>
                  <td>
                    <span style="display:inline-flex;align-items:center;gap:4px;padding:2px 8px;border-radius:9999px;font-size:0.72rem;font-weight:600"
                      [style.background]="c.isActive ? 'var(--color-low-dim)' : 'rgba(100,116,139,0.2)'"
                      [style.color]="c.isActive ? 'var(--color-low)' : '#94a3b8'">
                      {{ c.isActive ? 'Hoạt động' : 'Khóa' }}
                    </span>
                  </td>
                  <td style="font-size:0.8rem;color:var(--color-text-muted)">{{ c.createdAt | date:'dd/MM/yyyy' }}</td>
                </tr>
              }
            </tbody>
          </table>
        }
      </div>

      @if (totalPages > 1) {
        <div style="display:flex;justify-content:center;gap:8px;margin-top:20px">
          <button class="btn btn--secondary btn--sm" [disabled]="page === 0" (click)="goPage(page - 1)">←</button>
          <span style="display:flex;align-items:center;color:var(--color-text-muted);font-size:0.85rem">
            {{ page + 1 }} / {{ totalPages }}
          </span>
          <button class="btn btn--secondary btn--sm" [disabled]="page >= totalPages - 1" (click)="goPage(page + 1)">→</button>
        </div>
      }
    </div>
  `
})
export class CustomersComponent implements OnInit {
  customers: Customer[] = [];
  loading = true;
  page = 0;
  totalPages = 0;
  search = '';
  readonly skeletonRows = Array(5).fill(0);

  constructor(private api: ApiService) {}

  ngOnInit() { this.fetch(0, ''); }

  fetch(p = 0, q = '') {
    this.loading = true;
    const req = q ? this.api.searchCustomers(q, p, 20) : this.api.getCustomers(p, 20);
    req.subscribe({
      next: data => { this.customers = data.content || []; this.totalPages = data.totalPages || 1; },
      error: () => { this.customers = []; },
      complete: () => { this.loading = false; }
    });
  }

  doSearch() { this.page = 0; this.fetch(0, this.search); }
  clearSearch() { this.search = ''; this.page = 0; this.fetch(0, ''); }
  goPage(p: number) { this.page = p; this.fetch(p, this.search); }

  genderLabel(g: string): string {
    return g === 'MALE' ? 'Nam' : g === 'FEMALE' ? 'Nữ' : 'Khác';
  }
}
