package com.creditscore.controller;

import com.creditscore.dto.CreditApplicationDto;
import com.creditscore.dto.CreditApplicationDto.DecideRequest;
import com.creditscore.dto.CreditApplicationDto.SubmitRequest;
import com.creditscore.service.CreditApplicationService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.data.web.PageableDefault;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.UUID;

@RestController
@RequestMapping("/applications")
@RequiredArgsConstructor
public class CreditApplicationController {

    private final CreditApplicationService applicationService;

    @GetMapping
    public ResponseEntity<Page<CreditApplicationDto.Response>> getAll(
            @RequestParam(required = false) String status,
            @PageableDefault(size = 20, sort = "createdAt", direction = Sort.Direction.DESC)
            Pageable pageable
    ) {
        if (status != null && !status.isBlank()) {
            return ResponseEntity.ok(applicationService.findByStatus(status.toUpperCase(), pageable));
        }
        return ResponseEntity.ok(applicationService.findAll(pageable));
    }

    @GetMapping("/{id}")
    public ResponseEntity<CreditApplicationDto.Response> getById(@PathVariable UUID id) {
        return ResponseEntity.ok(applicationService.findById(id));
    }

    @GetMapping("/customer/{customerId}")
    public ResponseEntity<Page<CreditApplicationDto.Response>> getByCustomer(
            @PathVariable UUID customerId,
            @PageableDefault(size = 20, sort = "createdAt", direction = Sort.Direction.DESC)
            Pageable pageable
    ) {
        return ResponseEntity.ok(applicationService.findByCustomer(customerId, pageable));
    }

    @PostMapping
    public ResponseEntity<CreditApplicationDto.Response> create(
            @Valid @RequestBody CreditApplicationDto.CreateRequest request
    ) {
        CreditApplicationDto.Response created = applicationService.create(request);
        return ResponseEntity.status(HttpStatus.CREATED).body(created);
    }

    @PostMapping("/submit")
    public ResponseEntity<CreditApplicationDto.Response> submit(
            @Valid @RequestBody SubmitRequest request
    ) {
        return ResponseEntity.ok(applicationService.submitForScoring(request));
    }

    @PostMapping("/decide")
    public ResponseEntity<CreditApplicationDto.Response> decide(
            @Valid @RequestBody DecideRequest request
    ) {
        return ResponseEntity.ok(applicationService.decide(request));
    }
}
