package com.creditscore.controller;

import com.creditscore.dto.ScoringRequest;
import com.creditscore.service.AIServiceClient;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/score")
@RequiredArgsConstructor
@Slf4j
public class ScoringController {

    private final AIServiceClient aiServiceClient;

    /**
     * Standalone scoring — no application record needed.
     * Accepts credit features directly, calls AI service, returns result.
     */
    @PostMapping
    public ResponseEntity <Map<String, Object>> score(
            @Valid @RequestBody ScoringRequest request
    ) {
        log.info("Standalone scoring: age={}, debtRatio={}",
                request.getAge(), request.getDebtRatio());

        // Build a mock SubmitRequest to reuse AIServiceClient.callScoringMulti
        com.creditscore.dto.CreditApplicationDto.SubmitRequest submitReq =
                com.creditscore.dto.CreditApplicationDto.SubmitRequest.builder()
                        .age(request.getAge())
                        .RevolvingUtilizationOfUnsecuredLines(request.getRevolvingUtilizationOfUnsecuredLines())
                        .NumberOfTime30_59DaysPastDueNotWorse(request.getNumberOfTime30_59DaysPastDueNotWorse())
                        .DebtRatio(request.getDebtRatio())
                        .MonthlyIncome(request.getMonthlyIncome())
                        .NumberOfOpenCreditLinesAndLoans(request.getNumberOfOpenCreditLinesAndLoans())
                        .NumberOfTimes90DaysLate(request.getNumberOfTimes90DaysLate())
                        .NumberRealEstateLoansOrLines(request.getNumberRealEstateLoansOrLines())
                        .NumberOfTime60_89DaysPastDueNotWorse(request.getNumberOfTime60_89DaysPastDueNotWorse())
                        .NumberOfDependents(request.getNumberOfDependents())
                        .build();

        Map<String, Object> aiResponse = aiServiceClient.callScoringMulti(submitReq);

        @SuppressWarnings("unchecked")
        Map<String, Object> ensemble = (Map<String, Object>) aiResponse.get("ensemble");
        @SuppressWarnings("unchecked")
        Map<String, Object> championResult = (Map<String, Object>)
                ((Map<String, Object>) aiResponse.get("models")).get("champion_xgb");
        @SuppressWarnings("unchecked")
        Map<String, Object> challengerResult = (Map<String, Object>)
                ((Map<String, Object>) aiResponse.get("models")).get("benchmark_lr");

        return ResponseEntity.ok(Map.of(
                "creditScore", ((Number) ensemble.get("credit_score")).intValue(),
                "riskProbability", ((Number) ensemble.get("risk_probability")).doubleValue(),
                "riskLevel", ensemble.get("risk_level"),
                "shapExplanations", ensemble.get("shap_explanations"),
                "models", Map.of(
                        "xgb", Map.of(
                                "creditScore", championResult != null
                                        ? ((Number) championResult.get("credit_score")).intValue() : null,
                                "riskProbability", championResult != null
                                        ? ((Number) championResult.get("risk_probability")).doubleValue() : null,
                                "riskLevel", championResult != null ? championResult.get("risk_level") : null,
                                "shapExplanations", championResult != null
                                        ? championResult.get("shap_explanations") : null
                        ),
                        "lr", Map.of(
                                "creditScore", challengerResult != null
                                        ? ((Number) challengerResult.get("credit_score")).intValue() : null,
                                "riskProbability", challengerResult != null
                                        ? ((Number) challengerResult.get("risk_probability")).doubleValue() : null,
                                "riskLevel", challengerResult != null ? challengerResult.get("risk_level") : null,
                                "shapExplanations", challengerResult != null
                                        ? challengerResult.get("shap_explanations") : null
                        )
                ),
                "ensemble", Map.of(
                        "creditScore", ensemble.get("credit_score"),
                        "riskProbability", ensemble.get("risk_probability"),
                        "riskLevel", ensemble.get("risk_level"),
                        "voting", ensemble.get("voting"),
                        "shapExplanations", ensemble.get("shap_explanations")
                ),
                "inferenceMs", aiResponse.get("inference_ms") != null
                        ? ((Number) aiResponse.get("inference_ms")).intValue() : 0
        ));
    }
}
