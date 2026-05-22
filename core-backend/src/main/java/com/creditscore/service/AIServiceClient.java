package com.creditscore.service;

import com.creditscore.dto.CreditApplicationDto.SubmitRequest;
import com.creditscore.exception.AiServiceException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatusCode;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import reactor.core.publisher.Mono;
import reactor.util.retry.Retry;

import java.time.Duration;
import java.util.Map;

@Service
@RequiredArgsConstructor
@Slf4j
public class AIServiceClient {

    private final WebClient aiServiceWebClient;

    @Value("${ai-service.timeout-seconds:30}")
    private int timeoutSeconds;

    @Value("${ai-service.retry-attempts:3}")
    private int retryAttempts;

    /**
     * Call the single-model AI scoring endpoint (/api/v1/score).
     * Returns the raw AI response Map (deserialised JSON).
     *
     * @param request the scoring request payload
     * @return AI response (creditScore, riskProbability, riskLevel, shapExplanations)
     * @throws AiServiceException on failure
     */
    @SuppressWarnings("unchecked")
    public Map<String, Object> callScoring(SubmitRequest request) {
        log.info("Calling AI Service /score (single model): age={}, debtRatio={}",
                request.getAge(), request.getDebtRatio());

        Map<String, Object> body = Map.ofEntries(
                Map.entry("age",                                        request.getAge()),
                Map.entry("revolving_utilization_of_unsecured_lines", request.getRevolvingUtilizationOfUnsecuredLines()),
                Map.entry("number_of_time30_59days_past_due_not_worse", request.getNumberOfTime30_59DaysPastDueNotWorse()),
                Map.entry("debt_ratio",                                 request.getDebtRatio()),
                Map.entry("monthly_income",                             request.getMonthlyIncome()),
                Map.entry("number_of_open_credit_lines_and_loans",     request.getNumberOfOpenCreditLinesAndLoans()),
                Map.entry("number_of_times90days_late",                request.getNumberOfTimes90DaysLate()),
                Map.entry("number_real_estate_loans_or_lines",         request.getNumberRealEstateLoansOrLines()),
                Map.entry("number_of_time60_89days_past_due_not_worse", request.getNumberOfTime60_89DaysPastDueNotWorse()),
                Map.entry("number_of_dependents",                      request.getNumberOfDependents())
        );

        try {
            Map<String, Object> response = aiServiceWebClient
                    .post()
                    .uri("/api/v1/score")
                    .bodyValue(body)
                    .retrieve()
                    .onStatus(HttpStatusCode::isError, clientResponse ->
                            clientResponse.bodyToMono(Map.class)
                                    .flatMap(errorBody -> {
                                        log.error("AI service /score returned error: {}", errorBody);
                                        return Mono.error(
                                                new AiServiceException(
                                                        "AI service error: " + errorBody.get("message"),
                                                        clientResponse.statusCode().value()
                                                )
                                        );
                                    })
                    )
                    .bodyToMono(Map.class)
                    .retryWhen(Retry.backoff(retryAttempts, Duration.ofMillis(500))
                            .filter(this::isRetryable)
                            .doBeforeRetry(signal ->
                                    log.warn("Retrying AI /score call (attempt {})",
                                            signal.totalRetries() + 1)))
                    .timeout(Duration.ofSeconds(timeoutSeconds))
                    .block();

            if (response == null) {
                throw new AiServiceException("AI service /score returned empty response");
            }

            log.info("AI /score responded: score={}, prob={}, risk={}",
                    response.get("credit_score"),
                    response.get("risk_probability"),
                    response.get("risk_level"));
            return response;

        } catch (WebClientResponseException e) {
            throw new AiServiceException(
                    "AI service responded with " + e.getStatusCode() + ": " + e.getMessage(),
                    e.getStatusCode().value()
            );
        } catch (Exception e) {
            if (e instanceof AiServiceException) throw (AiServiceException) e;
            throw new AiServiceException("Failed to call AI service /score: " + e.getMessage(), e);
        }
    }

    /**
     * Call the multi-model scoring endpoint (/api/v1/score/multi).
     * Runs all 3 classification models (LR, RF, XGBoost) + K-Means clustering.
     * Returns the full response including per-model scores, ensemble, and cluster info.
     */
    @SuppressWarnings("unchecked")
    public Map<String, Object> callScoringMulti(SubmitRequest request) {
        log.info("Calling AI Service /score/multi (multi-model): age={}, debtRatio={}",
                request.getAge(), request.getDebtRatio());

        Map<String, Object> body = Map.ofEntries(
                Map.entry("age",                                        request.getAge()),
                Map.entry("revolving_utilization_of_unsecured_lines", request.getRevolvingUtilizationOfUnsecuredLines()),
                Map.entry("number_of_time30_59days_past_due_not_worse", request.getNumberOfTime30_59DaysPastDueNotWorse()),
                Map.entry("debt_ratio",                                 request.getDebtRatio()),
                Map.entry("monthly_income",                             request.getMonthlyIncome()),
                Map.entry("number_of_open_credit_lines_and_loans",     request.getNumberOfOpenCreditLinesAndLoans()),
                Map.entry("number_of_times90days_late",               request.getNumberOfTimes90DaysLate()),
                Map.entry("number_real_estate_loans_or_lines",        request.getNumberRealEstateLoansOrLines()),
                Map.entry("number_of_time60_89days_past_due_not_worse", request.getNumberOfTime60_89DaysPastDueNotWorse()),
                Map.entry("number_of_dependents",                     request.getNumberOfDependents())
        );

        try {
            Map<String, Object> response = aiServiceWebClient
                    .post()
                    .uri("/api/v1/score/multi")
                    .bodyValue(body)
                    .retrieve()
                    .onStatus(HttpStatusCode::isError, clientResponse ->
                            clientResponse.bodyToMono(Map.class)
                                    .flatMap(errorBody -> {
                                        log.error("AI service /score/multi returned error: {}",
                                                errorBody);
                                        return Mono.error(
                                                new AiServiceException(
                                                        "AI service error: " + errorBody.get("message"),
                                                        clientResponse.statusCode().value()
                                                )
                                        );
                                    })
                    )
                    .bodyToMono(Map.class)
                    .retryWhen(Retry.backoff(retryAttempts, Duration.ofMillis(500))
                            .filter(this::isRetryable)
                            .doBeforeRetry(signal ->
                                    log.warn("Retrying AI /score/multi call (attempt {})",
                                            signal.totalRetries() + 1)))
                    .timeout(Duration.ofSeconds(timeoutSeconds))
                    .block();

            if (response == null) {
                throw new AiServiceException(
                        "AI service /score/multi returned empty response");
            }

            // Log ensemble summary
            Map<String, Object> ensemble = (Map<String, Object>) response.get("ensemble");
            if (ensemble != null) {
                log.info("AI /score/multi: ensemble score={}, prob={}, risk={}, voting={}",
                        ensemble.get("credit_score"),
                        ensemble.get("risk_probability"),
                        ensemble.get("risk_level"),
                        ensemble.get("voting"));
            }
            return response;

        } catch (WebClientResponseException e) {
            throw new AiServiceException(
                    "AI service responded with " + e.getStatusCode() + ": " + e.getMessage(),
                    e.getStatusCode().value()
            );
        } catch (Exception e) {
            if (e instanceof AiServiceException) throw (AiServiceException) e;
            throw new AiServiceException(
                    "Failed to call AI multi-model service: " + e.getMessage(), e);
        }
    }

    private boolean isRetryable(Throwable throwable) {
        return throwable instanceof java.net.ConnectException
                || throwable instanceof java.net.SocketTimeoutException
                || throwable instanceof java.io.IOException;
    }
}
