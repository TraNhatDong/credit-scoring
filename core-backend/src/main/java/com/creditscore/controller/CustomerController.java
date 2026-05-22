package com.creditscore.controller;

import com.creditscore.dto.CustomerDto;
import com.creditscore.service.CustomerService;
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
@RequestMapping("/customers")
@RequiredArgsConstructor
public class CustomerController {

    private final CustomerService customerService;

    @GetMapping
    public ResponseEntity<Page<CustomerDto.Response>> getAll(
            @PageableDefault(size = 20, sort = "createdAt", direction = Sort.Direction.DESC)
            Pageable pageable
    ) {
        return ResponseEntity.ok(customerService.findAll(pageable));
    }

    @GetMapping("/{id}")
    public ResponseEntity<CustomerDto.Response> getById(@PathVariable UUID id) {
        return ResponseEntity.ok(customerService.findById(id));
    }

    @GetMapping("/search")
    public ResponseEntity<Page<CustomerDto.Response>> search(
            @RequestParam String name,
            @PageableDefault(size = 20) Pageable pageable
    ) {
        return ResponseEntity.ok(customerService.searchByName(name, pageable));
    }

    @PostMapping
    public ResponseEntity<CustomerDto.Response> create(
            @Valid @RequestBody CustomerDto.CreateRequest request
    ) {
        CustomerDto.Response created = customerService.create(request);
        return ResponseEntity.status(HttpStatus.CREATED).body(created);
    }

    @PatchMapping("/{id}/deactivate")
    public ResponseEntity<CustomerDto.Response> deactivate(@PathVariable UUID id) {
        return ResponseEntity.ok(customerService.deactivate(id));
    }
}
