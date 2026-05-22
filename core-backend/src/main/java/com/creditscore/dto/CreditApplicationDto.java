package com.creditscore.dto;

import com.creditscore.entity.CreditApplication;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.*;
import lombok.*;

import java.math.BigDecimal;
import java.util.List;
import java.util.Map;
import java.util.UUID;

public class CreditApplicationDto {

    // ── Create / Update request ────────────────────────────────
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class CreateRequest {
        @NotNull(message = "Customer ID is required")
        private UUID customerId;

        @NotNull(message = "Requested amount is required")
        @DecimalMin(value = "1000000", message = "Minimum loan amount is 1,000,000 VND")
        private BigDecimal requestedAmount;

        @NotBlank(message = "Loan purpose is required")
        @Size(max = 100)
        private String loanPurpose;

        @NotNull(message = "Loan term is required")
        @Min(value = 1, message = "Loan term minimum is 1 month")
        @Max(value = 360, message = "Loan term maximum is 360 months")
        private Integer loanTermMonths;
    }

    // ── Submit for scoring ────────────────────────────────────
    // All fields use GMSC (Give Me Some Credit) dataset column names exactly.
    // This payload is sent directly to the AI service /api/v1/score endpoint.
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class SubmitRequest {
        @NotNull(message = "Application ID is required")
        private UUID applicationId;

        // GMSC features — column names match Python Pydantic schema exactly
        @NotNull
        @Min(0) @Max(120)
        @JsonProperty("age")
        private Integer age;

        @NotNull @DecimalMin(value = "0.0", inclusive = false)
        @JsonProperty("RevolvingUtilizationOfUnsecuredLines")
        private BigDecimal RevolvingUtilizationOfUnsecuredLines;

        @NotNull @Min(0)
        @JsonProperty("NumberOfTime30_59DaysPastDueNotWorse")
        private Integer NumberOfTime30_59DaysPastDueNotWorse;

        @NotNull @DecimalMin("0.0")
        @JsonProperty("DebtRatio")
        private BigDecimal DebtRatio;

        @NotNull @DecimalMin(value = "0.0", inclusive = false)
        @JsonProperty("MonthlyIncome")
        private BigDecimal MonthlyIncome;

        @NotNull @Min(0)
        @JsonProperty("NumberOfOpenCreditLinesAndLoans")
        private Integer NumberOfOpenCreditLinesAndLoans;

        @NotNull @Min(0)
        @JsonProperty("NumberOfTimes90DaysLate")
        private Integer NumberOfTimes90DaysLate;

        @NotNull @Min(0)
        @JsonProperty("NumberRealEstateLoansOrLines")
        private Integer NumberRealEstateLoansOrLines;

        @NotNull @Min(0)
        @JsonProperty("NumberOfTime60_89DaysPastDueNotWorse")
        private Integer NumberOfTime60_89DaysPastDueNotWorse;

        @NotNull @DecimalMin("0.0")
        @JsonProperty("NumberOfDependents")
        private BigDecimal NumberOfDependents;
    }

    // ── Response ───────────────────────────────────────────────
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public static class Response {
        private String id;
        private String customerId;
        private CustomerDto.Response customer;
        private BigDecimal requestedAmount;
        private String loanPurpose;
        private Integer loanTermMonths;

        // GMSC feature fields
        private BigDecimal RevolvingUtilizationOfUnsecuredLines;
        private Integer age;
        private Integer NumberOfTime30_59DaysPastDueNotWorse;
        private BigDecimal DebtRatio;
        private BigDecimal MonthlyIncome;
        private Integer NumberOfOpenCreditLinesAndLoans;
        private Integer NumberOfTimes90DaysLate;
        private Integer NumberRealEstateLoansOrLines;
        private Integer NumberOfTime60_89DaysPastDueNotWorse;
        private BigDecimal NumberOfDependents;

        // AI results
        private Integer creditScore;
        private BigDecimal riskProbability;
        private String riskLevel;

        // Per-model scores from /score/multi
        private Integer championScore;
        private BigDecimal championRiskProbability;
        private Integer challengerScore;
        private BigDecimal challengerRiskProbability;

        private List<Map<String, Object>> aiExplanations;
        // Full multi-model result from /score/multi (for frontend LR/RF/XGB tabs)
        private Map<String, Object> multiModelPayload;

        private String status;
        private String rejectionReason;
        private String submittedAt;
        private String scoredAt;
        private String decidedAt;
        private String decidedBy;
        private String createdAt;
        private String updatedAt;

        public static Response from(CreditApplication entity) {
            return Response.builder()
                    .id(entity.getId().toString())
                    .customerId(entity.getCustomer().getId().toString())
                    .customer(CustomerDto.Response.from(entity.getCustomer()))
                    .requestedAmount(entity.getRequestedAmount())
                    .loanPurpose(entity.getLoanPurpose())
                    .loanTermMonths(entity.getLoanTermMonths())
                    .RevolvingUtilizationOfUnsecuredLines(entity.getRevolvingUtilizationOfUnsecuredLines())
                    .age(entity.getAge())
                    .NumberOfTime30_59DaysPastDueNotWorse(entity.getNumberOfTime30_59DaysPastDueNotWorse())
                    .DebtRatio(entity.getDebtRatio())
                    .MonthlyIncome(entity.getMonthlyIncome())
                    .NumberOfOpenCreditLinesAndLoans(entity.getNumberOfOpenCreditLinesAndLoans())
                    .NumberOfTimes90DaysLate(entity.getNumberOfTimes90DaysLate())
                    .NumberRealEstateLoansOrLines(entity.getNumberRealEstateLoansOrLines())
                    .NumberOfTime60_89DaysPastDueNotWorse(entity.getNumberOfTime60_89DaysPastDueNotWorse())
                    .NumberOfDependents(entity.getNumberOfDependents())
                    .creditScore(entity.getCreditScore())
                    .riskProbability(entity.getRiskProbability())
                    .riskLevel(entity.getRiskLevel() != null ? entity.getRiskLevel().name() : null)
                    .championScore(entity.getChampionScore())
                    .championRiskProbability(entity.getChampionRiskProbability())
                    .challengerScore(entity.getChallengerScore())
                    .challengerRiskProbability(entity.getChallengerRiskProbability())
                    .aiExplanations(entity.getAiExplanations())
                    .multiModelPayload(entity.getMultiModelPayload())
                    .status(entity.getStatus().name())
                    .rejectionReason(entity.getRejectionReason())
                    .submittedAt(entity.getSubmittedAt() != null ? entity.getSubmittedAt().toString() : null)
                    .scoredAt(entity.getScoredAt() != null ? entity.getScoredAt().toString() : null)
                    .decidedAt(entity.getDecidedAt() != null ? entity.getDecidedAt().toString() : null)
                    .decidedBy(entity.getDecidedBy())
                    .createdAt(entity.getCreatedAt() != null ? entity.getCreatedAt().toString() : null)
                    .updatedAt(entity.getUpdatedAt() != null ? entity.getUpdatedAt().toString() : null)
                    .build();
        }
    }

    // ── Decide (approve / reject) ─────────────────────────────
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class DecideRequest {
        @NotNull
        private UUID applicationId;

        @NotNull
        private Decision decision;

        private String reason;

        public enum Decision {
            APPROVE, REJECT
        }
    }
}
