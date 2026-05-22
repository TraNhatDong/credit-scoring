package com.creditscore.repository;

import com.creditscore.entity.Customer;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;
import java.util.UUID;

@Repository
public interface CustomerRepository extends JpaRepository<Customer, UUID> {

    Optional<Customer> findByIdCardNumber(String idCardNumber);

    boolean existsByIdCardNumber(String idCardNumber);

    Page<Customer> findByIsActiveTrue(Pageable pageable);

    Page<Customer> findByFullNameContainingIgnoreCase(String name, Pageable pageable);
}
