package com.creditscore.service;

import com.creditscore.dto.CreditApplicationDto;
import com.creditscore.dto.CreditApplicationDto.DecideRequest;
import com.creditscore.dto.CreditApplicationDto.SubmitRequest;
import com.creditscore.entity.AiAuditLog;
import com.creditscore.entity.CreditApplication;
import com.creditscore.entity.CreditApplication.ApplicationStatus;
import com.creditscore.entity.CreditApplication.RiskLevel;
import com.creditscore.entity.Customer;
import com.creditscore.exception.BusinessRuleException;
import com.creditscore.exception.ResourceNotFoundException;
import com.creditscore.repository.AiAuditLogRepository;
import com.creditscore.repository.CreditApplicationRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@Service
@RequiredArgsConstructor
@Slf4j
@Transactional
public class CreditApplicationService {

    private final CreditApplicationRepository applicationRepository;
    private final CustomerService customerService;
    private final AIServiceClient aiServiceClient;
    private final AiAuditLogRepository auditLogRepository;

    // Auto-approve threshold: LOW risk AND score >= 700
    private static final int AUTO_APPROVE_SCORE_THRESHOLD = 700;

    @Transactional(readOnly = true)
    public Page<CreditApplicationDto.Response> findAll(Pageable pageable) {
        return applicationRepository.findAll(pageable)
                .map(app -> {
                    // Force load customer to avoid lazy-init issues
                    app.getCustomer().getFullName();
                    return CreditApplicationDto.Response.from(app);
                });
    }

    @Transactional(readOnly = true)
    public CreditApplicationDto.Response findById(UUID id) {
        CreditApplication app = applicationRepository.findByIdWithCustomer(id);
        if (app == null) {
            throw new ResourceNotFoundException("CreditApplication", "id", id);
        }
        return CreditApplicationDto.Response.from(app);
    }

    @Transactional(readOnly = true)
    public Page<CreditApplicationDto.Response> findByStatus(String status, Pageable pageable) {
        ApplicationStatus appStatus = ApplicationStatus.valueOf(status.toUpperCase());
        return applicationRepository.findByStatus(appStatus, pageable)
                .map(app -> {
                    app.getCustomer().getFullName();
                    return CreditApplicationDto.Response.from(app);
                });
    }

    @Transactional(readOnly = true)
    public Page<CreditApplicationDto.Response> findByCustomer(UUID customerId, Pageable pageable) {
        // Verify customer exists
        customerService.findEntityById(customerId);
        return applicationRepository.findByCustomerId(customerId, pageable)
                .map(CreditApplicationDto.Response::from);
    }

    public CreditApplicationDto.Response create(CreditApplicationDto.CreateRequest request) {
        Customer customer = customerService.findEntityById(request.getCustomerId());

        CreditApplication app = CreditApplication.builder()
                .customer(customer)
                .requestedAmount(request.getRequestedAmount())
                .loanPurpose(request.getLoanPurpose())
                .loanTermMonths(request.getLoanTermMonths())
                .status(ApplicationStatus.DRAFT)
                .build();

        app = applicationRepository.save(app);
        log.info("Created credit application: {} for customer: {}", app.getId(), customer.getId());
        return CreditApplicationDto.Response.from(app);
    }

    @SuppressWarnings("unchecked")
    public CreditApplicationDto.Response submitForScoring(SubmitRequest request) {
        CreditApplication app = applicationRepository.findByIdWithCustomer(request.getApplicationId());
        if (app == null) {
            throw new ResourceNotFoundException("CreditApplication", "id", request.getApplicationId());
        }

        if (app.getStatus() != ApplicationStatus.DRAFT) {
            throw new BusinessRuleException(
                    "Application " + app.getId() + " is not in DRAFT status. Current: " + app.getStatus());
        }

        app.setAge(request.getAge());
        app.setRevolvingUtilizationOfUnsecuredLines(request.getRevolvingUtilizationOfUnsecuredLines());
        app.setNumberOfTime30_59DaysPastDueNotWorse(request.getNumberOfTime30_59DaysPastDueNotWorse());
        app.setDebtRatio(request.getDebtRatio());
        app.setMonthlyIncome(request.getMonthlyIncome());
        app.setNumberOfOpenCreditLinesAndLoans(request.getNumberOfOpenCreditLinesAndLoans());
        app.setNumberOfTimes90DaysLate(request.getNumberOfTimes90DaysLate());
        app.setNumberRealEstateLoansOrLines(request.getNumberRealEstateLoansOrLines());
        app.setNumberOfTime60_89DaysPastDueNotWorse(request.getNumberOfTime60_89DaysPastDueNotWorse());
        app.setNumberOfDependents(request.getNumberOfDependents());
        app.setStatus(ApplicationStatus.PROCESSING);
        app.setSubmittedAt(LocalDateTime.now());
        applicationRepository.save(app);

        Map<String, Object> aiRequest = Map.ofEntries(
                Map.entry("age",                                        request.getAge()),
                Map.entry("RevolvingUtilizationOfUnsecuredLines",      request.getRevolvingUtilizationOfUnsecuredLines()),
                Map.entry("NumberOfTime30-59DaysPastDueNotWorse",      request.getNumberOfTime30_59DaysPastDueNotWorse()),
                Map.entry("DebtRatio",                                  request.getDebtRatio()),
                Map.entry("MonthlyIncome",                              request.getMonthlyIncome()),
                Map.entry("NumberOfOpenCreditLinesAndLoans",           request.getNumberOfOpenCreditLinesAndLoans()),
                Map.entry("NumberOfTimes90DaysLate",                   request.getNumberOfTimes90DaysLate()),
                Map.entry("NumberRealEstateLoansOrLines",               request.getNumberRealEstateLoansOrLines()),
                Map.entry("NumberOfTime60-89DaysPastDueNotWorse",      request.getNumberOfTime60_89DaysPastDueNotWorse()),
                Map.entry("NumberOfDependents",                         request.getNumberOfDependents())
        );

        // ── Call multi-model endpoint (LR + RF + XGBoost + K-Means) ──────────
        Map<String, Object> aiResponse;
        String endpointCalled = "/api/v1/score/multi";
        try {
            aiResponse = aiServiceClient.callScoringMulti(request);
        } catch (Exception ex) {
            app.setStatus(ApplicationStatus.FAILED);
            applicationRepository.save(app);
            writeAuditLog(app.getId(), aiRequest, null, "unknown", 0,
                    ex.getMessage(), endpointCalled, null);
            throw ex;
        }

        // ── Parse ensemble result from /score/multi ───────────────────────────
        Map<String, Object> ensemble = (Map<String, Object>) aiResponse.get("ensemble");
        Integer creditScore    = ((Number) ensemble.get("credit_score")).intValue();
        BigDecimal riskProb    = BigDecimal.valueOf(
                ((Number) ensemble.get("risk_probability")).doubleValue());
        String riskLevelStr    = (String) ensemble.get("risk_level");
        List<Map<String, Object>> shapExplanations =
                (List<Map<String, Object>>) ensemble.get("shap_explanations");
        Map<String, Object> pipelineMetadata =
                (Map<String, Object>) aiResponse.get("pipeline_metadata");
        Integer inferenceMs    = aiResponse.get("inference_ms") != null
                ? ((Number) aiResponse.get("inference_ms")).intValue() : 0;

        // ── Parse per-model scores from the models map ─────────────────────────
        Map<String, Object> championModelResult = null;
        Map<String, Object> challengerModelResult = null;
        Map<String, Object> models = (Map<String, Object>) aiResponse.get("models");
        if (models != null) {
            championModelResult   = (Map<String, Object>) models.get("champion_xgb");
            challengerModelResult = (Map<String, Object>) models.get("benchmark_lr");
        }

        Integer championScore = championModelResult != null
                ? ((Number) championModelResult.get("credit_score")).intValue()
                : null;
        BigDecimal championRiskProb = championModelResult != null
                ? BigDecimal.valueOf(((Number) championModelResult.get("risk_probability")).doubleValue())
                : null;

        Integer challengerScore = challengerModelResult != null
                ? ((Number) challengerModelResult.get("credit_score")).intValue()
                : null;
        BigDecimal challengerRiskProb = challengerModelResult != null
                ? BigDecimal.valueOf(((Number) challengerModelResult.get("risk_probability")).doubleValue())
                : null;

        // pipeline_metadata.best_model does not exist in the actual AI service response.
        // Read best_model_key (e.g. "xgb") and map to display name.
        String modelVersion = "unknown";
        if (pipelineMetadata != null) {
            Object bestKeyObj = pipelineMetadata.get("best_model_key");
            if (bestKeyObj != null) {
                String bestKey = bestKeyObj.toString();
                // Map pipeline key -> human-readable model name
                modelVersion = "xgb".equals(bestKey) ? "XGBoost (Champion)"
                        : "lr".equals(bestKey) ? "Logistic Regression (Benchmark)"
                        : bestKey;
            }
            Object version = pipelineMetadata.get("pipeline_version");
            if (version != null && "unknown".equals(modelVersion)) {
                modelVersion = version.toString();
            }
        }

        // ── Persist ───────────────────────────────────────────────────────────
        app.setCreditScore(creditScore);
        app.setRiskProbability(riskProb);
        app.setRiskLevel(RiskLevel.valueOf(riskLevelStr));
        // Per-model scores extracted from /score/multi
        app.setChampionScore(championScore);
        app.setChampionRiskProbability(championRiskProb);
        app.setChallengerScore(challengerScore);
        app.setChallengerRiskProbability(challengerRiskProb);
        app.setAiExplanations(shapExplanations);
        app.setAiRequestPayload(aiRequest);
        app.setAiResponsePayload(Map.of(
                "credit_score",       creditScore,
                "risk_probability",   riskProb,
                "risk_level",        riskLevelStr,
                "model_version",     modelVersion,
                "inference_ms",      inferenceMs,
                "shap_explanations", shapExplanations
        ));
        // Store full multi-model result so frontend can display LR/RF/XGB tabs
        app.setMultiModelPayload(aiResponse);
        app.setScoredAt(LocalDateTime.now());
        app.setStatus(ApplicationStatus.COMPLETED);

        if (creditScore >= AUTO_APPROVE_SCORE_THRESHOLD
                && app.getRiskLevel() == RiskLevel.LOW) {
            app.setStatus(ApplicationStatus.APPROVED);
            app.setDecidedAt(LocalDateTime.now());
            app.setDecidedBy("SYSTEM_AUTO");
            log.info("Auto-approved application {}: score={}, risk={}",
                    app.getId(), creditScore, riskLevelStr);
        }

        app = applicationRepository.save(app);

        // ── Write audit log (success) ────────────────────────────────────────
        writeAuditLog(app.getId(), aiRequest, aiResponse, modelVersion,
                inferenceMs, null, endpointCalled, aiResponse);

        log.info("Application {} scored (multi-model): score={}, risk={}, status={}",
                app.getId(), creditScore, riskLevelStr, app.getStatus());

        return CreditApplicationDto.Response.from(app);
    }

    private void writeAuditLog(UUID applicationId, Map<String, Object> requestPayload,
            Map<String, Object> responsePayload, String modelVersion,
            int inferenceMs, String errorMessage, String endpointCalled,
            Map<String, Object> multiModelPayload) {
        try {
            AiAuditLog audit = AiAuditLog.builder()
                    .applicationId(applicationId)
                    .requestPayload(requestPayload)
                    .responsePayload(responsePayload != null ? responsePayload : Map.of())
                    .modelVersion(modelVersion)
                    .inferenceMs(inferenceMs)
                    .errorMessage(errorMessage)
                    .endpointCalled(endpointCalled)
                    .multiModelPayload(multiModelPayload)
                    .build();
            auditLogRepository.save(audit);
            log.debug("Audit log written for application {}", applicationId);
        } catch (Exception ex) {
            log.error("Failed to write audit log for application {}: {}",
                    applicationId, ex.getMessage());
        }
    }

    public CreditApplicationDto.Response decide(DecideRequest request) {
        CreditApplication app = applicationRepository.findByIdWithCustomer(request.getApplicationId());
        if (app == null) {
            throw new ResourceNotFoundException("CreditApplication", "id", request.getApplicationId());
        }

        if (app.getStatus() != ApplicationStatus.COMPLETED) {
            throw new BusinessRuleException(
                    "Application must be in COMPLETED status to decide. Current: " + app.getStatus());
        }

        if (request.getDecision() == DecideRequest.Decision.APPROVE) {
            app.setStatus(ApplicationStatus.APPROVED);
        } else {
            app.setStatus(ApplicationStatus.REJECTED);
            app.setRejectionReason(request.getReason());
        }

        app.setDecidedAt(LocalDateTime.now());
        app.setDecidedBy(request.getDecision().name());

        app = applicationRepository.save(app);
        log.info("Application {} {} by officer",
                app.getId(), request.getDecision());

        return CreditApplicationDto.Response.from(app);
    }
}
