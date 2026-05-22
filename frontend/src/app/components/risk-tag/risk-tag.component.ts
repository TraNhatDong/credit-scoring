import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-risk-tag',
  standalone: true,
  imports: [CommonModule],
  template: `
    <span class="risk-tag" [style.background]="bgColor" [style.color]="textColor" [style.border-color]="borderColor">
      <span class="dot"></span>
      {{ level }}
    </span>
  `,
  styles: [`
    .risk-tag {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 12px;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      border: 1px solid;
    }
    .dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: currentColor;
    }
  `]
})
export class RiskTagComponent {
  @Input() level = 'MEDIUM';

  get bgColor(): string {
    const map: Record<string, string> = {
      LOW: 'rgba(34,197,94,0.15)', MEDIUM: 'rgba(245,158,11,0.15)',
      HIGH: 'rgba(249,115,22,0.15)', CRITICAL: 'rgba(239,68,68,0.15)'
    };
    return map[this.level] ?? 'rgba(245,158,11,0.15)';
  }

  get textColor(): string {
    const map: Record<string, string> = {
      LOW: '#22c55e', MEDIUM: '#f59e0b', HIGH: '#f97316', CRITICAL: '#ef4444'
    };
    return map[this.level] ?? '#f59e0b';
  }

  get borderColor(): string {
    const map: Record<string, string> = {
      LOW: 'rgba(34,197,94,0.3)', MEDIUM: 'rgba(245,158,11,0.3)',
      HIGH: 'rgba(249,115,22,0.3)', CRITICAL: 'rgba(239,68,68,0.3)'
    };
    return map[this.level] ?? 'rgba(245,158,11,0.3)';
  }
}
