import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';

@Component({
  selector: 'app-new-application',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './new-application.component.html',
  styleUrl: './new-application.component.css'
})
export class NewApplicationComponent {
  submitting = false;
  errorMsg = '';
  createCustomerMode = false;
  newCustomerId = '';

  form: {
    customerId: string; fullName: string; dateOfBirth: string; gender: string; idCardNumber: string;
    phone: string; email: string; address: string; monthlyIncome: number | null;
    occupation: string; requestedAmount: number | null; loanPurpose: string; loanTermMonths: number;
  } = {
    customerId: '', fullName: '', dateOfBirth: '', gender: 'MALE', idCardNumber: '',
    phone: '', email: '', address: '', monthlyIncome: null,
    occupation: '', requestedAmount: null, loanPurpose: '', loanTermMonths: 12,
  };

  readonly termOptions = [6, 12, 18, 24, 36, 48, 60, 72, 84, 120];

  constructor(private api: ApiService, private router: Router) {}

  handleCreateCustomer() {
    if (!this.form.fullName || !this.form.idCardNumber || !this.form.dateOfBirth || !this.form.monthlyIncome) {
      this.errorMsg = 'Vui lòng điền đầy đủ thông tin bắt buộc (*)';
      return;
    }
    this.submitting = true;
    this.errorMsg = '';
    this.api.createCustomer({
      fullName: this.form.fullName, dateOfBirth: this.form.dateOfBirth, gender: this.form.gender as any,
      idCardNumber: this.form.idCardNumber, phone: this.form.phone || undefined,
      email: this.form.email || undefined, address: this.form.address || undefined,
      monthlyIncome: this.form.monthlyIncome ?? 0,
      occupation: this.form.occupation || undefined,
    } as any).subscribe({
      next: c => { this.newCustomerId = c.id; this.createCustomerMode = false; },
      error: (e: any) => { this.errorMsg = e?.error?.message ?? 'Lỗi tạo khách hàng'; },
      complete: () => { this.submitting = false; }
    });
  }

  handleCreateApp() {
    const customerId = this.newCustomerId || this.form.customerId;
    if (!customerId) { this.errorMsg = 'Vui lòng chọn hoặc tạo khách hàng'; return; }
    if (!this.form.requestedAmount || !this.form.loanPurpose) { this.errorMsg = 'Vui lòng điền đầy đủ thông tin khoản vay'; return; }

    this.submitting = true;
    this.errorMsg = '';
    this.api.createApplication({
      customerId,
      requestedAmount: this.form.requestedAmount,
      loanPurpose: this.form.loanPurpose,
      loanTermMonths: this.form.loanTermMonths,
    }).subscribe({
      next: app => { this.router.navigate(['/scoring', app.id]); },
      error: (e: any) => { this.errorMsg = e?.error?.message ?? 'Lỗi tạo hồ sơ'; this.submitting = false; },
    });
  }

  navigateToApplications() { this.router.navigate(['/applications']); }
}
