# Mortgage underwriting agent on Databricks

A small, governed agent that reads an applicant's paperwork, extracts the figures a lender needs, flags anything that does not add up, and hands the decision to exact code rather than to the model.

Built in one working day on Databricks Free Edition, partly to learn the platform and partly to test a claim I keep hearing: that agents are transforming document-heavy back office work. The findings below are as useful as the demo.

## The idea in one line

The model reads and writes. Exact code decides. A human handles judgement.

## What it does

An applicant sends in four documents: a covering email, a payslip, a bank statement and a credit report. The agent then:

1. Reads all four and extracts the six figures the lending rules need, recording which document each figure came from.
2. Flags contradictions between documents, and figures that rest only on the applicant's own word.
3. Hands the figures to code, which does the arithmetic and writes one row to a governed table.
4. Calls an unchanged SQL function that applies three lending rules and returns APPROVE, REFER or DECLINE.
5. On anything other than APPROVE, drafts a case note for a human underwriter: what failed, which documents disagree, and what evidence would resolve it.

Every step is recorded in MLflow, so each figure can be traced back to the document it came from.

## The rules

Illustrative, not any real lender's policy. They live in a `policy` table, so changing a limit changes every decision and every case note.

| Rule | Limit |
|---|---|
| Affordability: debts plus mortgage payment as a share of gross income | 43% maximum |
| Loan to value: loan as a share of property value | 80% maximum |
| Credit score | 620 minimum |

No failures: APPROVE. One failure: REFER to a human. Two or more: DECLINE.

## Architecture

```
Documents (Unity Catalog volume)
        |
        v
Agent code (notebook)  <---->  Llama 4 Maverick via Unity Gateway
        |                       (permissions, token counting, budgets)
        v
applicants table  -->  assess_applicant()  -->  decision
        |                    ^
        |                    |
        |              policy table (the limits)
        v
MLflow tracing (every step recorded)
```

Data, rules, documents and the model service all sit in one Unity Catalog schema, so they share one set of permissions and one audit trail.

## The test case

Applicant A004 has three traps built in, each one a mistake a careless reader would make:

1. The email claims a salary of EUR 90,000. The payslip shows EUR 5,800 a month, which is EUR 69,600 a year.
2. The email declares only a credit card. The bank statement and credit report both show an undeclared car loan of EUR 450 a month.
3. The bank statement shows EUR 1,300 of rent, which is not a debt, because it stops when the applicant buys the flat.

Fall for trap 1 or 2 and the case is an APPROVE. Read the evidence properly and affordability is 46.6% against a 43% limit, so the correct answer is REFER with both discrepancies flagged.

## Results

Extraction, eleven checks, five consecutive runs: all passed every time. The model took the payslip figure over the claimed one, found the undeclared car loan, excluded the rent, listed the credit card once despite it appearing in three documents, and flagged all four figures that rest only on the applicant's email.

That is a real result on a narrow task. It is not evidence that the approach survives messy PDFs, applicants with two jobs or bonus structures, or documents that contradict each other more subtly.

## What I learned, including what went wrong

**Most agent demos are a database query with a chatbot on top.** The first version of this was exactly that: a table, a rule function and a model that fetched a row. Removing the model would have made it faster, cheaper and more reliable. The model only earns its place once the input is messy, which is why the demo was rebuilt around documents. It is a useful test to apply to any agent proposal: take the model out and see what is lost.

**Tracing is not evaluation.** MLflow faithfully recorded a run where the model broke one of its own instructions, and said nothing, because nothing had told it what correct looks like. Tracing records what happened. Evaluation says whether it was right. Only a person can say what "right" means, and that is where the domain expert earns their fee.

**Tests that check the shape of an output break; tests that check substance survive.** A check counting items in a `discrepancies` list failed the moment the model filed one of them under a new field instead. Nothing was lost, but the test said otherwise. Rewriting it to look for the substance, wherever the model put it, fixed it.

**The model's gaps were mostly our gaps.** Its first case note said a figure was "above the limit" without quoting the limit, because the rule function never returned it. Its suggested next steps were generic because nobody had told it what a German underwriter knows: that a gap between claimed and payslip income often means a bonus or a thirteenth-month salary, and that the document which settles it is the annual tax statement. The fix was not a bigger model. It was writing the domain knowledge down.

**Keep the model away from the arithmetic.** Told never to calculate, it calculated anyway, correctly, because it judged that helpful. Harmless here. But models predict what an answer looks like rather than computing it, so the one time it is wrong it will be wrong confidently and silently. All arithmetic in this build is done by code.

**The platform was mid-migration and its own showcase was broken.** Databricks was moving model serving under a new Unity Gateway. The Playground offered no models, the featured-model links pointed at endpoints that had been switched off, and attaching tools to one hosted model produced "rate limited" errors every time. The tool list in the Playground also silently dropped between page loads, so an agent that appeared to be calling tools was inventing plausible tool names and writing them out as text. It produced a correct-looking answer for an applicant that does not exist, having looked at nothing. Right by accident is the most dangerous failure mode there is, and no output-only review would have caught it.

**Failed calls still cost tokens.** After a morning of nothing but errors, the usage dashboard showed 7,100 tokens consumed on the model that never once succeeded. Retries are billed.

**Notebooks lie about their own state.** Two versions of the same function in one notebook, the older one run last, produced test output that had nothing to do with the code on screen. Real systems belong in files with tests, not in notebook cells.

## Honest limitations

- One applicant, four documents, plain text rather than real PDFs.
- No human approval step is implemented; the case note is produced, not routed.
- The three rules are illustrative and simplified. Real affordability assessment is considerably more involved.
- Free Edition, one mid-range model. A stronger model would likely slip less, though none slip never.
- And the obvious one: for four documents a human underwriter would do this better in five minutes. The case for building it is volume, consistency and auditability, not quality on a single file. Below a certain caseload, it is not worth doing at all.

## Next steps, if taken further

- A second applicant whose documents should produce APPROVE, so the failure case is not the only thing tested.
- Real PDFs instead of text files.
- Model judges in MLflow evaluation for the case note, which no code check can grade.
- A human approval queue, with telemetry on how often the human simply agrees. A rubber-stamp approval step is worse than none, because it looks like control.
- Compare models on the same test set. Reliability on tool use varied noticeably between the hosted models.

## Repository contents

```
sql/01_setup.sql                 Schema, tables, policy, documents volume, rule function
notebook/underwriting_agent.py   The pipeline, as a Databricks notebook
documents/A004/                  The four sample documents
```

## Running it

1. Create a Databricks workspace (Free Edition is enough) and run `sql/01_setup.sql` in the SQL Editor.
2. Upload the four files in `documents/A004/` to the `workspace.underwriting.documents` volume.
3. Import `notebook/underwriting_agent.py` as a notebook, set `base_url` to your own workspace address, and run the cells in order.

---

Built by Kayvon Salari. Enterprise, data and AI architecture.
