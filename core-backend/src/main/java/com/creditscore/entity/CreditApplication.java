package com.creditscore.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.annotations.UpdateTimestamp;
import org.hibernate.type.SqlTypes;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@Entity
@Table(name = "credit_applications")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class CreditApplication {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    @Column(name = "id", updatable = false, nullable = false)
    private UUID id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "customer_id", nullable = false)
    private Customer customer;

    @Column(name = "requested_amount", nullable = false, precision = 15, scale = 2)
    private BigDecimal requestedAmount;

    @Column(name = "loan_purpose", nullable = false, length = 100)
    private String loanPurpose;

    @Column(name = "loan_term_months", nullable = false)
    private Integer loanTermMonths;

    // GMSC Pre-scoring feature fields — column names match GMSC dataset exactly
    @Column(name = "RevolvingUtilizationOfUnsecuredLines", precision = 8, scale = 6)
    private BigDecimal RevolvingUtilizationOfUnsecuredLines;

    @Column(name = "age")
    private Integer age;

    @Column(name = "NumberOfTime30_59DaysPastDueNotWorse")
    @Builder.Default
    private Integer NumberOfTime30_59DaysPastDueNotWorse = 0;

    @Column(name = "DebtRatio", precision = 10, scale = 6)
    private BigDecimal DebtRatio;

    @Column(name = "MonthlyIncome", precision = 15, scale = 2)
    private BigDecimal MonthlyIncome;

    @Column(name = "NumberOfOpenCreditLinesAndLoans")
    @Builder.Default
    private Integer NumberOfOpenCreditLinesAndLoans = 0;

    @Column(name = "NumberOfTimes90DaysLate")
    @Builder.Default
    private Integer NumberOfTimes90DaysLate = 0;

    @Column(name = "NumberRealEstateLoansOrLines")
    @Builder.Default
    private Integer NumberRealEstateLoansOrLines = 0;

    @Column(name = "NumberOfTime60_89DaysPastDueNotWorse")
    @Builder.Default
    private Integer NumberOfTime60_89DaysPastDueNotWorse = 0;

    @Column(name = "NumberOfDependents", precision = 5, scale = 2)
    @Builder.Default
    private BigDecimal NumberOfDependents = BigDecimal.ZERO;

    // AI Scoring results
    @Column(name = "credit_score")
    private Integer creditScore;

    @Column(name = "risk_probability", precision = 5, scale = 4)
    private BigDecimal riskProbability;

    @Enumerated(EnumType.STRING)
    @Column(name = "risk_level")
    private RiskLevel riskLevel;

    // Per-model scores from /score/multi — champion XGBoost
    @Column(name = "champion_score")
    private Integer championScore;

    @Column(name = "champion_risk_probability", precision = 5, scale = 4)
    private BigDecimal championRiskProbability;

    // Per-model scores from /score/multi — benchmark Logistic Regression
    @Column(name = "challenger_score")
    private Integer challengerScore;

    @Column(name = "challenger_risk_probability", precision = 5, scale = 4)
    private BigDecimal challengerRiskProbability;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "ai_explanations", columnDefinition = "jsonb")
    private List<Map<String, Object>> aiExplanations;

    // Full multi-model ensemble result from /score/multi
    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "multi_model_payload", columnDefinition = "jsonb")
    private Map<String, Object> multiModelPayload;

    // Audit trail (JSONB)
    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "ai_request_payload", columnDefinition = "jsonb")
    private Map<String, Object> aiRequestPayload;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "ai_response_payload", columnDefinition = "jsonb")
    private Map<String, Object> aiResponsePayload;

    @Enumerated(EnumType.STRING)
    @Column(name = "status", nullable = false)
    @Builder.Default
    private ApplicationStatus status = ApplicationStatus.DRAFT;

    @Column(name = "rejection_reason", columnDefinition = "TEXT")
    private String rejectionReason;

    @Column(name = "submitted_at")
    private LocalDateTime submittedAt;

    @Column(name = "scored_at")
    private LocalDateTime scoredAt;

    @Column(name = "decided_at")
    private LocalDateTime decidedAt;

    @Column(name = "decided_by", length = 100)
    private String decidedBy;

    @CreationTimestamp
    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    @UpdateTimestamp
    @Column(name = "updated_at", nullable = false)
    private LocalDateTime updatedAt;

    public enum ApplicationStatus {
        DRAFT, PROCESSING, COMPLETED, FAILED, REJECTED, APPROVED
    }

    public enum RiskLevel {
        LOW, MEDIUM, HIGH, CRITICAL
    }
}
