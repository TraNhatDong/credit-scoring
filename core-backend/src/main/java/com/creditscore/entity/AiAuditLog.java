package com.creditscore.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

import java.time.LocalDateTime;
import java.util.Map;
import java.util.UUID;

/**
 * Audit trail for every AI scoring call.
 * Maps to the partitioned ai_audit_log table in PostgreSQL.
 */
@Entity
@Table(name = "ai_audit_log")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class AiAuditLog {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "application_id")
    private UUID applicationId;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "request_payload", columnDefinition = "jsonb", nullable = false)
    private Map<String, Object> requestPayload;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "response_payload", columnDefinition = "jsonb", nullable = false)
    private Map<String, Object> responsePayload;

    @Column(name = "model_version", length = 50, nullable = false)
    private String modelVersion;

    @Column(name = "inference_ms", nullable = false)
    private Integer inferenceMs;

    @Column(name = "error_message", columnDefinition = "TEXT")
    private String errorMessage;

    @CreationTimestamp
    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "multi_model_payload", columnDefinition = "jsonb")
    private Map<String, Object> multiModelPayload;

    @Column(name = "endpoint_called", length = 50)
    private String endpointCalled;
}
