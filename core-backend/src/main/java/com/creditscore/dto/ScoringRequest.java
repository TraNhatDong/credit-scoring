package com.creditscore.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.*;
import lombok.*;

import java.math.BigDecimal;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ScoringRequest {
    // GMSC features — column names match Python Pydantic schema exactly
    @NotNull
    @Min(0) @Max(120)
    @JsonProperty("age")
    private Integer age;

    @NotNull
    @DecimalMin(value = "0.0", inclusive = false)
    @JsonProperty("RevolvingUtilizationOfUnsecuredLines")
    private BigDecimal RevolvingUtilizationOfUnsecuredLines;

    @NotNull @Min(0)
    @JsonProperty("NumberOfTime30_59DaysPastDueNotWorse")
    private Integer NumberOfTime30_59DaysPastDueNotWorse;

    @NotNull
    @DecimalMin("0.0")
    @JsonProperty("DebtRatio")
    private BigDecimal DebtRatio;

    @NotNull
    @DecimalMin(value = "0.0", inclusive = false)
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

    @NotNull
    @DecimalMin("0.0")
    @JsonProperty("NumberOfDependents")
    private BigDecimal NumberOfDependents;
}
