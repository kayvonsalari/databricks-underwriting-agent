# Databricks notebook source
# MAGIC %md
# MAGIC # Mortgage underwriting agent
# MAGIC
# MAGIC Documents in, decision out. The model reads and writes; exact code decides.
# MAGIC
# MAGIC Run `sql/01_setup.sql` first, and upload the sample documents to the
# MAGIC `workspace.underwriting.documents` volume.

# COMMAND ----------

# MAGIC %pip install -U openai mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Connection. The model is reached through Unity Gateway, which applies
# permissions and counts usage. Identity comes from the current notebook
# session, so no access token is created or stored anywhere.

import json
import mlflow
from openai import OpenAI

mlflow.openai.autolog()

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
client = OpenAI(
    api_key=token,
    base_url="https://dbc-061a351f-b289.cloud.databricks.com/ai-gateway/mlflow/v1",
)
MODEL = "system.ai.llama-4-maverick"

print(client.responses.create(model=MODEL, input="Reply with the single word: ready").output_text)

# COMMAND ----------

# Collect one applicant's documents from the governed volume.
# A new applicant needs new files, not new code.

FOLDER = "/Volumes/workspace/underwriting/documents"

def load_documents(applicant_id):
    docs = {}
    for f in dbutils.fs.ls(FOLDER):
        if applicant_id in f.name:
            docs[f.name] = open(f"{FOLDER}/{f.name}").read()
    return docs

print(list(load_documents("A004").keys()))

# COMMAND ----------

# Extraction. The model copies figures and never calculates. Each figure carries
# the file it came from. Contradictions and applicant-only claims are flagged.

EXTRACTION_PROMPT = """You are a mortgage document checker. You will receive an applicant's documents. Copy figures exactly as written in the documents; never calculate anything.

Rules:
- Evidence beats claims: payslips, bank statements and credit reports outrank what the applicant says in an email or letter.
- gross_monthly_income: the gross monthly salary shown on the payslip.
- stated_annual_income: the annual income the applicant claims, if any, so it can be compared.
- debts: every regular monthly debt repayment (loans, credit cards). List each distinct debt once, even if it appears in several documents.
- excluded_payments: regular payments that are not debts (for example rent, utilities, phone), with the reason they are excluded.
- discrepancies: every place where documents disagree, or where the applicant left something out.
- unverified_figures: every figure whose only source is the applicant's own statement (an email or letter from the applicant), with no independent document supporting it.
- If a figure is missing, use null. Never guess.

Reply with JSON only, no other text, in exactly this shape:
{
  "gross_monthly_income": {"value": number or null, "source": "file name"},
  "stated_annual_income": {"value": number or null, "source": "file name"},
  "monthly_mortgage_payment": {"value": number or null, "source": "file name"},
  "loan_amount": {"value": number or null, "source": "file name"},
  "property_value": {"value": number or null, "source": "file name"},
  "credit_score": {"value": number or null, "source": "file name"},
  "debts": [{"name": "text", "monthly_amount": number, "sources": ["file names"]}],
  "excluded_payments": [{"name": "text", "monthly_amount": number, "reason": "text"}],
  "discrepancies": ["text"],
  "unverified_figures": [{"figure": "text", "only_source": "file name"}]
}"""

def parse_json(text):
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])

@mlflow.trace(span_type="PARSER")
def extract_figures(applicant_id):
    docs = load_documents(applicant_id)
    bundle = "\n\n".join(f"=== FILE: {name} ===\n{text}" for name, text in docs.items())
    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": bundle},
        ],
    )
    return parse_json(resp.output_text)

# COMMAND ----------

# Test set for applicant A004. Checks look for substance, not for where the
# model filed something: an earlier version counted items in "discrepancies"
# and broke when the model moved one item to "unverified_figures".

def check_extraction(x):
    monthly_debts = sum(d["monthly_amount"] for d in x["debts"])
    flagged_text = " ".join(
        x.get("discrepancies", [])
        + [f"{u.get('figure','')} {u.get('only_source','')}" for u in x.get("unverified_figures", [])]
    ).lower()
    return {
        "Income taken from payslip, not email": x["gross_monthly_income"]["value"] == 5800,
        "Car loan found": any("car" in d["name"].lower() for d in x["debts"]),
        "Rent not counted as a debt": not any("rent" in d["name"].lower() for d in x["debts"]),
        "Total monthly debts are 750": monthly_debts == 750,
        "Mortgage payment 1,950": x["monthly_mortgage_payment"]["value"] == 1950,
        "Loan amount 320,000": x["loan_amount"]["value"] == 320000,
        "Property value 400,000": x["property_value"]["value"] == 400000,
        "Credit score 680": x["credit_score"]["value"] == 680,
        "Income claim flagged somewhere": "90,000" in flagged_text or "90000" in flagged_text,
        "Undeclared car loan flagged somewhere": "car" in flagged_text,
        "Property value flagged as unverified": any(
            "property" in u.get("figure", "").lower() for u in x.get("unverified_figures", [])
        ),
    }

# COMMAND ----------

# Consistency. One good run proves nothing; models vary between runs.
# A crash counts as a failure rather than stopping the loop.

RUNS = 5
results = []
for i in range(RUNS):
    try:
        r = check_extraction(extract_figures("A004"))
    except Exception as e:
        print(f"Run {i + 1}: crashed ({e})")
        r = {"Run completed without crashing": False}
    results.append(r)
    print(f"Run {i + 1}: {sum(r.values())} of {len(r)} checks passed")

print("\nPer check, across all runs:")
for name in dict.fromkeys(n for r in results for n in r):
    print(f"{sum(r.get(name, False) for r in results)}/{RUNS}  {name}")

# COMMAND ----------

# Save. Code refuses to save an incomplete application, does the arithmetic
# itself, and passes values separately from the SQL text so nothing the model
# produced can alter the query.

REQUIRED = ["gross_monthly_income", "monthly_mortgage_payment", "loan_amount", "property_value", "credit_score"]

def save_applicant(applicant_id, x):
    missing = [f for f in REQUIRED if x[f]["value"] is None]
    if missing:
        raise ValueError(f"Not saved: missing figures {missing}")

    annual_income = int(x["gross_monthly_income"]["value"] * 12)
    monthly_debts = int(sum(d["monthly_amount"] for d in x["debts"]))

    spark.sql(
        "DELETE FROM workspace.underwriting.applicants WHERE applicant_id = :id",
        args={"id": applicant_id},
    )
    spark.sql(
        """INSERT INTO workspace.underwriting.applicants
           (applicant_id, annual_income, monthly_debts, monthly_mortgage_payment,
            loan_amount, property_value, credit_score)
           VALUES (:id, :inc, :debts, :pay, :loan, :value, :score)""",
        args={
            "id": applicant_id,
            "inc": annual_income,
            "debts": monthly_debts,
            "pay": int(x["monthly_mortgage_payment"]["value"]),
            "loan": int(x["loan_amount"]["value"]),
            "value": int(x["property_value"]["value"]),
            "score": int(x["credit_score"]["value"]),
        },
    )
    print(f"Saved {applicant_id}: income {annual_income}, debts {monthly_debts}")

# COMMAND ----------

# Case note for the human underwriter. Rule results are fetched fresh so the
# note cannot drift from the official decision. The evidence checklist is the
# underwriter's domain knowledge, written down; without it the suggestions are
# generic. The model never recommends approve or decline.

CASE_NOTE_PROMPT = """You write case notes for a human mortgage underwriter. You receive the official rule results (each with its limit) and the figures extracted from the applicant's documents, including sources, discrepancies and unverified figures.

Write a short note, under 250 words, in plain English, with these headings: Outcome, Why, Discrepancies, Unverified figures, Suggested next steps.

Rules:
- Use only the numbers given to you. Never calculate new numbers: no differences, percentages or totals.
- State which rule failed, quoting the figure and its limit exactly as given.
- For every discrepancy, name the documents involved.
- Never recommend approving or declining. That decision belongs to the underwriter.

Evidence checklist for Suggested next steps:
- Claimed income higher than the payslip: this may be a bonus, overtime or a thirteenth-month salary. Request the annual income tax statement (Lohnsteuerbescheinigung) or the last twelve payslips. Say that if higher income is proven, the affordability result may change.
- Property value or loan amount supported only by the applicant: request an independent valuation report and the purchase contract.
- Mortgage payment supported only by the applicant: check it against the lender's own offer.
- A debt proven by documents but not declared by the applicant: do not ask the applicant to confirm it, because the documents already prove it. Flag it as a disclosure issue for the underwriter to weigh.
- Only suggest evidence that addresses a problem actually present in this case."""

@mlflow.trace(span_type="AGENT")
def write_case_note(applicant_id, x):
    rules = spark.sql(
        "SELECT * FROM workspace.underwriting.assess_applicant(:id)",
        args={"id": applicant_id},
    ).collect()[0].asDict()

    if rules["decision"] == "APPROVE":
        return f"{applicant_id}: approved by the rules. No case note needed."

    payload = json.dumps(
        {"applicant_id": applicant_id, "rule_results": rules, "extracted_figures": x},
        indent=2,
        default=str,
    )
    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": CASE_NOTE_PROMPT},
            {"role": "user", "content": payload},
        ],
    )
    return resp.output_text

# COMMAND ----------

# End to end for one applicant.

x = extract_figures("A004")

for name, ok in check_extraction(x).items():
    print(("PASS  " if ok else "FAIL  ") + name)

print()
save_applicant("A004", x)
display(spark.sql("SELECT * FROM workspace.underwriting.assess_applicant('A004')"))

print("\n" + write_case_note("A004", x))
