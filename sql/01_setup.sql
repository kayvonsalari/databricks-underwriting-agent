-- Mortgage underwriting agent: governed data and rules
-- Run in the Databricks SQL Editor, top to bottom.
-- Everything lives in one schema so data, rules and model share one set of permissions.

CREATE SCHEMA IF NOT EXISTS workspace.underwriting;

-- 1. Applicant figures. Rows for A001-A003 are seed data; A004 is written by the agent.
CREATE TABLE IF NOT EXISTS workspace.underwriting.applicants (
  applicant_id STRING,
  annual_income DOUBLE,
  monthly_debts DOUBLE,
  monthly_mortgage_payment DOUBLE,
  loan_amount DOUBLE,
  property_value DOUBLE,
  credit_score INT
);

INSERT INTO workspace.underwriting.applicants VALUES
  ('A001', 96000, 600, 2000, 300000, 400000, 740),   -- passes all three rules
  ('A002', 60000, 900, 1800, 285000, 300000, 590),   -- fails all three
  ('A003', 84000, 1100, 1900, 340000, 400000, 700);  -- fails one

-- 2. Lending policy. Limits live in one place so a change here flows through
--    every decision and every case note.
CREATE OR REPLACE TABLE workspace.underwriting.policy AS
SELECT 0.43 AS max_affordability, 0.80 AS max_loan_to_value, 620 AS min_credit_score;

-- 3. Documents folder. Applicant paperwork, governed like the tables.
CREATE VOLUME IF NOT EXISTS workspace.underwriting.documents;

-- 4. The rules. One function returns every rule result with its limit and the
--    decision. The agent calls this; it never calculates or decides itself.
CREATE OR REPLACE FUNCTION workspace.underwriting.assess_applicant(applicant STRING)
RETURNS TABLE (
  applicant_id STRING,
  affordability_pct DOUBLE, affordability_limit_pct DOUBLE, affordability_passed BOOLEAN,
  loan_to_value_pct DOUBLE, loan_to_value_limit_pct DOUBLE, loan_to_value_passed BOOLEAN,
  credit_score INT, credit_score_minimum INT, credit_passed BOOLEAN,
  rules_failed INT, decision STRING
)
COMMENT 'Full underwriting assessment for one applicant. Reads limits from the policy table. Returns each rule result with its limit, plus the decision: APPROVE if none fail, REFER to a human underwriter if one fails, DECLINE if two or more fail. Returns no rows if the applicant does not exist.'
RETURN
  WITH c AS (
    SELECT a.applicant_id,
           (a.monthly_debts + a.monthly_mortgage_payment) / (a.annual_income / 12) AS aff,
           a.loan_amount / a.property_value AS ltv,
           CAST(a.credit_score AS INT) AS cs,
           p.max_affordability, p.max_loan_to_value, p.min_credit_score
    FROM workspace.underwriting.applicants a
    CROSS JOIN workspace.underwriting.policy p
    WHERE a.applicant_id = applicant
  ),
  r AS (
    SELECT *, aff <= max_affordability AS ap, ltv <= max_loan_to_value AS lp, cs >= min_credit_score AS cp
    FROM c
  ),
  f AS (
    SELECT *, CAST((CASE WHEN ap THEN 0 ELSE 1 END) + (CASE WHEN lp THEN 0 ELSE 1 END)
                 + (CASE WHEN cp THEN 0 ELSE 1 END) AS INT) AS failed
    FROM r
  )
  SELECT applicant_id,
         ROUND(aff * 100, 1), CAST(max_affordability * 100 AS DOUBLE), ap,
         ROUND(ltv * 100, 1), CAST(max_loan_to_value * 100 AS DOUBLE), lp,
         cs, CAST(min_credit_score AS INT), cp,
         failed,
         CASE WHEN failed = 0 THEN 'APPROVE' WHEN failed = 1 THEN 'REFER' ELSE 'DECLINE' END
  FROM f;

-- Check
SELECT * FROM workspace.underwriting.assess_applicant('A003');  -- expect REFER
