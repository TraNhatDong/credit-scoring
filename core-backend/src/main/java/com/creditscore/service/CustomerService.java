package com.creditscore.service;

import com.creditscore.dto.CustomerDto;
import com.creditscore.entity.Customer;
import com.creditscore.exception.DuplicateResourceException;
import com.creditscore.exception.ResourceNotFoundException;
import com.creditscore.repository.CustomerRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.UUID;

@Service
@RequiredArgsConstructor
@Slf4j
@Transactional
public class CustomerService {

    private final CustomerRepository customerRepository;

    @Transactional(readOnly = true)
    public Page<CustomerDto.Response> findAll(Pageable pageable) {
        return customerRepository.findAll(pageable)
                .map(CustomerDto.Response::from);
    }

    @Transactional(readOnly = true)
    public CustomerDto.Response findById(UUID id) {
        return customerRepository.findById(id)
                .map(CustomerDto.Response::from)
                .orElseThrow(() -> new ResourceNotFoundException("Customer", "id", id));
    }

    @Transactional(readOnly = true)
    public Customer findEntityById(UUID id) {
        return customerRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Customer", "id", id));
    }

    public CustomerDto.Response create(CustomerDto.CreateRequest request) {
        if (customerRepository.existsByIdCardNumber(request.getIdCardNumber())) {
            throw new DuplicateResourceException(
                    "Customer with ID card " + request.getIdCardNumber() + " already exists");
        }

        Customer customer = Customer.builder()
                .fullName(request.getFullName())
                .dateOfBirth(request.getDateOfBirth())
                .gender(request.getGender())
                .idCardNumber(request.getIdCardNumber())
                .phone(request.getPhone())
                .email(request.getEmail())
                .address(request.getAddress())
                .monthlyIncome(request.getMonthlyIncome())
                .employer(request.getEmployer())
                .occupation(request.getOccupation())
                .isActive(true)
                .build();

        customer = customerRepository.save(customer);
        log.info("Created customer: {} [{}]", customer.getFullName(), customer.getId());
        return CustomerDto.Response.from(customer);
    }

    @Transactional(readOnly = true)
    public Page<CustomerDto.Response> searchByName(String name, Pageable pageable) {
        return customerRepository.findByFullNameContainingIgnoreCase(name, pageable)
                .map(CustomerDto.Response::from);
    }

    public CustomerDto.Response deactivate(UUID id) {
        Customer customer = customerRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Customer", "id", id));
        customer.setIsActive(false);
        customer = customerRepository.save(customer);
        log.info("Deactivated customer: {}", id);
        return CustomerDto.Response.from(customer);
    }
}
