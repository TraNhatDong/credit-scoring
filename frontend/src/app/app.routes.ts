import { Routes } from '@angular/router';
import { LayoutComponent } from './components/layout/layout.component';

export const routes: Routes = [
  {
    path: '',
    component: LayoutComponent,
    children: [
      { path: '', redirectTo: 'dashboard', pathMatch: 'full' },
      {
        path: 'dashboard',
        loadComponent: () => import('./pages/dashboard/dashboard.component').then(m => m.DashboardComponent)
      },
      {
        path: 'scoring',
        loadComponent: () => import('./pages/scoring/scoring.component').then(m => m.ScoringComponent)
      },
      { path: '**', redirectTo: 'dashboard' }
    ]
  }
];
