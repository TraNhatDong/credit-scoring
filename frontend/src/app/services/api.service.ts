import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';
import { Customer, CreditApplication, PagedResponse } from '../models/models';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly base = environment.apiBaseUrl;

  constructor(private http: HttpClient) {}

  // ── Standalone Scoring ──────────────────────────────────────────────────────

  /**
   * Standalone scoring — no application record needed.
   * Accepts credit features directly, calls AI service, returns score result.
   */
  score(payload: Record<string, number>): Observable<Record<string, unknown>> {
    return this.http.post<Record<string, unknown>>(`${this.base}/score`, payload);
  }

  // ── Customers ───────────────────────────────────────────────────────────────

  getCustomers(page = 0, size = 20): Observable<PagedResponse<Customer>> {
    return this.http.get<PagedResponse<Customer>>(`${this.base}/customers`, {
      params: new HttpParams().set('page', page).set('size', size),
    });
  }

  getCustomer(id: string): Observable<Customer> {
    return this.http.get<Customer>(`${this.base}/customers/${id}`);
  }

  searchCustomers(name: string, page = 0, size = 20): Observable<PagedResponse<Customer>> {
    return this.http.get<PagedResponse<Customer>>(`${this.base}/customers/search`, {
      params: new HttpParams()
        .set('name', name)
        .set('page', page)
        .set('size', size),
    });
  }

  createCustomer(payload: Record<string, unknown>): Observable<Customer> {
    return this.http.post<Customer>(`${this.base}/customers`, payload);
  }

  // ── Applications ─────────────────────────────────────────────────────────────

  getApplications(
    page = 0,
    size = 20,
    status?: string
  ): Observable<PagedResponse<CreditApplication>> {
    let params = new HttpParams().set('page', page).set('size', size);
    if (status) params = params.set('status', status);
    return this.http.get<PagedResponse<CreditApplication>>(`${this.base}/applications`, { params });
  }

  getApplication(id: string): Observable<CreditApplication> {
    return this.http.get<CreditApplication>(`${this.base}/applications/${id}`);
  }

  getApplicationsByCustomer(
    customerId: string,
    page = 0,
    size = 20
  ): Observable<PagedResponse<CreditApplication>> {
    return this.http.get<PagedResponse<CreditApplication>>(
      `${this.base}/applications/customer/${customerId}`,
      { params: new HttpParams().set('page', page).set('size', size) }
    );
  }

  createApplication(payload: Record<string, unknown>): Observable<CreditApplication> {
    return this.http.post<CreditApplication>(`${this.base}/applications`, payload);
  }
}
