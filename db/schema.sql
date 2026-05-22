-- ============================================================
-- schema.sql
-- Consolidated Credit Scoring Database Schema
-- Standard: 3NF + JSONB for AI audit trail
-- ============================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Enum Types ──────────────────────────────────────────────
CREATE TYPE application_status AS ENUM (
    'DRAFT',       -- Ban ghi nhap, chua gui AI
    'PROCESSING',  -- Dang cho AI xu ly
    'COMPLETED',   -- Hoan thanh, co ket qua
    'FAILED',      -- Loi khi goi AI
    'REJECTED',    -- Bi tu choi (no xau cao)
    'APPROVED'     -- Duoc phe duyet
);

CREATE TYPE risk_level AS ENUM ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL');

-- ── Table: customers ────────────────────────────────────────
CREATE TABLE customers (
    id            UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    full_name     VARCHAR(100) NOT NULL,
    date_of_birth DATE        NOT NULL,
    gender        VARCHAR(20) NOT NULL DEFAULT 'MALE',
    id_card_number VARCHAR(20) UNIQUE NOT NULL,
    phone         VARCHAR(20),
    email         VARCHAR(100),
    address       TEXT,
    monthly_income DECIMAL(15,2)  NOT NULL DEFAULT 0,
    employer      VARCHAR(200),
    occupation    VARCHAR(100),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_active     BOOLEAN     NOT NULL DEFAULT TRUE,

    CONSTRAINT chk_age CHECK (
        DATE_PART('year', AGE(date_of_birth)) BETWEEN 18 AND 70
    ),
    CONSTRAINT chk_income CHECK (monthly_income >= 0),
    CONSTRAINT chk_gender CHECK (gender IN ('MALE', 'FEMALE', 'OTHER'))
);

CREATE INDEX idx_customers_id_card   ON customers(id_card_number);
CREATE INDEX idx_customers_phone     ON customers(phone);
CREATE INDEX idx_customers_is_active ON customers(is_active);

-- ── Table: credit_applications ──────────────────────────────
-- GMSC dataset feature fields — column names match GMSC dataset exactly
CREATE TABLE credit_applications (
    id                UUID              PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id       UUID              NOT NULL REFERENCES customers(id),
    requested_amount  DECIMAL(15,2)    NOT NULL,
    loan_purpose      VARCHAR(100)      NOT NULL,
    loan_term_months  INTEGER           NOT NULL CHECK (loan_term_months BETWEEN 1 AND 360),

    -- GMSC dataset feature fields
    RevolvingUtilizationOfUnsecuredLines      DECIMAL(10,6),
    age                                        INTEGER,
    NumberOfTime30_59DaysPastDueNotWorse      INTEGER  DEFAULT 0,
    DebtRatio                                 DECIMAL(10,6),
    MonthlyIncome                             DECIMAL(15,2),
    NumberOfOpenCreditLinesAndLoans          INTEGER  DEFAULT 0,
    NumberOfTimes90DaysLate                   INTEGER  DEFAULT 0,
    NumberRealEstateLoansOrLines              INTEGER  DEFAULT 0,
    NumberOfTime60_89DaysPastDueNotWorse      INTEGER  DEFAULT 0,
    NumberOfDependents                        DECIMAL(5,2) DEFAULT 0,

    -- AI Scoring results
    credit_score      INTEGER,
    risk_probability  DECIMAL(5,4),
    risk_level        risk_level,

    -- Per-model scores (from /score/multi)
    champion_score             INTEGER,
    champion_risk_probability  DECIMAL(5,4),
    challenger_score           INTEGER,
    challenger_risk_probability DECIMAL(5,4),

    ai_explanations   JSONB,

    -- Request/Response audit trail
    ai_request_payload  JSONB,
    ai_response_payload JSONB,

    -- Full multi-model ensemble result from /score/multi
    multi_model_payload JSONB,

    status            application_status NOT NULL DEFAULT 'DRAFT',
    rejection_reason  TEXT,

    submitted_at      TIMESTAMPTZ,
    scored_at         TIMESTAMPTZ,
    decided_at        TIMESTAMPTZ,
    decided_by        VARCHAR(100),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_applications_customer         ON credit_applications(customer_id);
CREATE INDEX idx_applications_status          ON credit_applications(status);
CREATE INDEX idx_applications_created          ON credit_applications(created_at DESC);
CREATE INDEX idx_applications_score            ON credit_applications(credit_score);
CREATE INDEX idx_applications_champion_score   ON credit_applications(champion_score);
CREATE INDEX idx_applications_challenger_score ON credit_applications(challenger_score);

-- ── Table: ai_audit_log ─────────────────────────────────────
-- Partitioned by month for high-volume audit trail
CREATE TABLE ai_audit_log (
    id                BIGSERIAL,
    application_id    UUID,
    request_payload   JSONB    NOT NULL,
    response_payload  JSONB    NOT NULL,
    model_version     VARCHAR(50) NOT NULL,
    inference_ms      INTEGER  NOT NULL,
    error_message     TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Create monthly partitions (rolling 12 months)
CREATE TABLE ai_audit_log_2026_04 PARTITION OF ai_audit_log
    FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE ai_audit_log_2026_05 PARTITION OF ai_audit_log
    FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE ai_audit_log_2026_06 PARTITION OF ai_audit_log
    FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
CREATE TABLE ai_audit_log_2026_07 PARTITION OF ai_audit_log
    FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');
CREATE TABLE ai_audit_log_2026_08 PARTITION OF ai_audit_log
    FOR VALUES FROM ('2026-08-01') TO ('2026-09-01');
CREATE TABLE ai_audit_log_2026_09 PARTITION OF ai_audit_log
    FOR VALUES FROM ('2026-09-01') TO ('2026-10-01');
CREATE TABLE ai_audit_log_2026_10 PARTITION OF ai_audit_log
    FOR VALUES FROM ('2026-10-01') TO ('2026-11-01');
CREATE TABLE ai_audit_log_2026_11 PARTITION OF ai_audit_log
    FOR VALUES FROM ('2026-11-01') TO ('2026-12-01');
CREATE TABLE ai_audit_log_2026_12 PARTITION OF ai_audit_log
    FOR VALUES FROM ('2026-12-01') TO ('2027-01-01');
CREATE TABLE ai_audit_log_2027_01 PARTITION OF ai_audit_log
    FOR VALUES FROM ('2027-01-01') TO ('2027-02-01');
CREATE TABLE ai_audit_log_2027_02 PARTITION OF ai_audit_log
    FOR VALUES FROM ('2027-02-01') TO ('2027-03-01');
CREATE TABLE ai_audit_log_2027_03 PARTITION OF ai_audit_log
    FOR VALUES FROM ('2027-03-01') TO ('2027-04-01');

CREATE INDEX idx_audit_application ON ai_audit_log(application_id);
CREATE INDEX idx_audit_created     ON ai_audit_log(created_at DESC);

-- ── Trigger: auto-update updated_at ─────────────────────────
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_customers_updated_at
    BEFORE UPDATE ON customers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trg_applications_updated_at
    BEFORE UPDATE ON credit_applications
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
