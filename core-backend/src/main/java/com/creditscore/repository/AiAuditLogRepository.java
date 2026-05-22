package com.creditscore.repository;

import com.creditscore.entity.AiAuditLog;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.UUID;

@Repository
public interface AiAuditLogRepository extends JpaRepository<AiAuditLog, Long> {

    Page<AiAuditLog> findByApplicationId(UUID applicationId, Pageable pageable);

    List<AiAuditLog> findAllByApplicationIdOrderByCreatedAtDesc(UUID applicationId);

    @Query("SELECT a FROM AiAuditLog a WHERE a.applicationId = :appId ORDER BY a.createdAt DESC")
    List<AiAuditLog> findLatestByApplicationId(@Param("appId") UUID applicationId);

    void deleteByApplicationId(UUID applicationId);
}
