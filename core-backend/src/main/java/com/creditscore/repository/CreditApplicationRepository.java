package com.creditscore.repository;

import com.creditscore.entity.CreditApplication;
import com.creditscore.entity.CreditApplication.ApplicationStatus;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.UUID;

@Repository
public interface CreditApplicationRepository extends JpaRepository<CreditApplication, UUID> {

    Page<CreditApplication> findByCustomerId(UUID customerId, Pageable pageable);

    Page<CreditApplication> findByStatus(ApplicationStatus status, Pageable pageable);

    Page<CreditApplication> findByStatusIn(List<ApplicationStatus> statuses, Pageable pageable);

    @Query("SELECT ca FROM CreditApplication ca JOIN FETCH ca.customer WHERE ca.id = :id")
    CreditApplication findByIdWithCustomer(@Param("id") UUID id);

    @Query("SELECT ca FROM CreditApplication ca JOIN FETCH ca.customer WHERE ca.customer.id = :customerId")
    List<CreditApplication> findAllByCustomerIdWithCustomer(@Param("customerId") UUID customerId);

    @Query(value = """
        SELECT ca.*, c.full_name as cust_name, ca.credit_score as score
        FROM credit_applications ca
        JOIN customers c ON ca.customer_id = c.id
        WHERE ca.status = :status
        ORDER BY ca.created_at DESC
        """,
        countQuery = "SELECT count(*) FROM credit_applications WHERE status = :status",
        nativeQuery = true)
    Page<CreditApplication> findByStatusOrderByCreatedAtDesc(
            @Param("status") String status, Pageable pageable);

    @Query("SELECT COUNT(ca) FROM CreditApplication ca WHERE ca.customer.id = :customerId AND ca.status NOT IN ('DRAFT')")
    long countActiveApplicationsByCustomer(@Param("customerId") UUID customerId);
}
