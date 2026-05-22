package com.creditscore.dto;

import com.creditscore.entity.Customer;
import jakarta.validation.constraints.*;
import lombok.*;

import java.math.BigDecimal;
import java.time.LocalDate;

public class CustomerDto {

    // ── Create / Update request ────────────────────────────────
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class CreateRequest {
        @NotBlank(message = "Full name is required")
        @Size(max = 100)
        private String fullName;

        @NotNull(message = "Date of birth is required")
        @Past(message = "Date of birth must be in the past")
        private LocalDate dateOfBirth;

        @NotNull(message = "Gender is required")
        private Customer.Gender gender;

        @NotBlank(message = "ID card number is required")
        @Size(max = 20)
        private String idCardNumber;

        @Size(max = 20)
        private String phone;

        @Email
        @Size(max = 100)
        private String email;

        private String address;

        @NotNull(message = "Monthly income is required")
        @DecimalMin(value = "0.0", inclusive = false, message = "Income must be positive")
        private BigDecimal monthlyIncome;

        @Size(max = 200)
        private String employer;

        @Size(max = 100)
        private String occupation;
    }

    // ── Response ───────────────────────────────────────────────
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Response {
        private String id;
        private String fullName;
        private String dateOfBirth;
        private String gender;
        private String idCardNumber;
        private String phone;
        private String email;
        private String address;
        private BigDecimal monthlyIncome;
        private String employer;
        private String occupation;
        private Boolean isActive;
        private String createdAt;
        private String updatedAt;

        public static Response from(Customer entity) {
            return Response.builder()
                    .id(entity.getId().toString())
                    .fullName(entity.getFullName())
                    .dateOfBirth(entity.getDateOfBirth().toString())
                    .gender(entity.getGender().name())
                    .idCardNumber(entity.getIdCardNumber())
                    .phone(entity.getPhone())
                    .email(entity.getEmail())
                    .address(entity.getAddress())
                    .monthlyIncome(entity.getMonthlyIncome())
                    .employer(entity.getEmployer())
                    .occupation(entity.getOccupation())
                    .isActive(entity.getIsActive())
                    .createdAt(entity.getCreatedAt() != null ? entity.getCreatedAt().toString() : null)
                    .updatedAt(entity.getUpdatedAt() != null ? entity.getUpdatedAt().toString() : null)
                    .build();
        }
    }
}
