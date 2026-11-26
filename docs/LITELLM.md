# LiteLLM: controlling what the agent spends

Monitoring an agent is four separate jobs. This document covers the third.

| Job | Question it answers | Tool here |
|---|---|---|
| Record | What happened? | MLflow tracing |
| Check | Was it right? | Promptfoo |
| **Control** | **What did it cost, and can it be stopped?** | **LiteLLM** |
| Protect | Was the input or output unsafe? | NeMo Guardrails (not built yet) |

## What LiteLLM does here, and why

LiteLLM is a checkpoint between the agent and the model. In version 4 of the notebook (`notebook/underwriting_agent_v4.py`) nothing calls the model directly. Every call goes through one function, `ask_model`, which hands it to LiteLLM. That gives three things:

- **A cost record.** After each call, LiteLLM's price list turns the tokens used into a cost in US dollars. The notebook prints it, adds it to a total for the applicant and for the run, and writes it onto the MLflow trace for that call.
- **A spending limit.** Before each call, the notebook compares what the run has spent so far with the limit. Once the limit is reached, no further call is made.
- **An easy model switch.** The model name lives in one config cell, so trying a cheaper model is a one-line change.

The model still only reads and writes. The SQL rule function still makes every decision, and this change does not touch it.

### A note on the dollar figures

Databricks Free Edition does not bill for these calls. The costs shown are what the same tokens would cost at the list price LiteLLM holds for this model: USD 0.50 per million input tokens and USD 1.50 per million output tokens (its entry `databricks/databricks-llama-4-maverick`, taken from the Databricks pricing page). The figures are for comparing runs and models and for exercising the limit, not an invoice.

## How it was set up, in order

### 1. An access token

I created a Databricks access token named `CC_LiteLLM`, valid for 30 days, with no auto-scope, in **Settings > Developer > Access tokens**. It expires on 26 December 2026.

The token is only for the command line on my PC: uploading the notebook, starting test runs and reading their results. The notebook itself never sees it. Inside Databricks the notebook uses the identity of its own session.

### 2. Storing the token outside the repository

The Databricks CLI (version 1.28.0) was installed with `winget install Databricks.DatabricksCLI`. I then stored the token myself:

```powershell
databricks configure --host https://<your-workspace>.cloud.databricks.com
```

The command asks for the token with hidden input and writes it to `~/.databrickscfg` in my user folder. It is not in this repository and never was.

Two things went wrong here that are worth knowing about:

- **The paste added invisible characters.** Every command failed with `invalid header field value for "Authorization"`. The stored token had seven invisible null characters in front of it, left by the paste into the hidden prompt. Removing them from the file fixed it. If you see that error, retype or re-paste the token.
- **Proving which token is in use.** A workspace can hold several tokens, and the CLI does not say which one it is using. `databricks tokens list` shows each token's name and ID but never its value. The ID turned out to be the SHA-256 hash of the token, so hashing the stored token and comparing it with the list confirmed that `CC_LiteLLM` was the one in use, without displaying it.

### 3. What changed in the notebook

Version 4 is version 3 with these changes:

- **Install.** The first cell installs `litellm` (with `mlflow` and `databricks-sdk`).
- **One config cell (cell 2).** It holds the model name (`MODEL`), the name LiteLLM prices it under (`PRICE_KEY`), the spending limit for one run (`RUN_BUDGET_USD`, default 0.50) and which parts of the notebook to run (`SECTIONS`). A job can override these with parameters of the same names in lower case (`model`, `price_key`, `run_budget_usd`, `sections`).
- **No hard-coded address.** The workspace address and the session token are read from the notebook's own context. Version 3 had the address typed in.
- **One way to call the model.** `ask_model` checks the limit, calls `litellm.completion` against the same gateway address as before (`<workspace>/ai-gateway/mlflow/v1`), records tokens and cost, and writes them onto an MLflow span named `model_call: <step>`. Extraction, the case note, the approval record, the connection check and the servable class all use it.
- **A clean stop.** When the limit is reached, the call is refused with a clear message. The case is marked `not processed: budget reached`, and anything saved for that applicant during the run is removed from the `applicants` table, so the table never holds a half-finished case. Later cases in the same run are refused before any call is made.
- **A model with no price is rejected at the start.** A model LiteLLM cannot price would cost "0.00" forever and the limit would never block anything, so the notebook refuses to start instead. Prices can be set by hand in the config cell (`PRICE_OVERRIDE_USD_PER_MILLION`).
- **Totals.** The last cells print the cost per applicant and for the run, log it to MLflow as a run named `underwriting_cost` (metrics plus a `cost_per_call.json` file), and return the same summary when the notebook is run as a job.

One difference in how the model is called: version 3 used the OpenAI "Responses" style of call; LiteLLM uses the "chat completions" style, which is the style the Promptfoo suite already used against this gateway.

### 4. How the tests were run, and the results

All runs were on 26 November 2026. The notebook was uploaded to a workspace folder (`/Users/<me>/underwriting-agent-litellm/`, with `notebook/` and `prompts/` inside it) and started from the command line as one-time serverless jobs with `databricks jobs submit`. Free Edition allowed this.

**Local check, no model calls.** The cost-control cell was run on my PC against LiteLLM's built-in fake responses. It confirmed that a call is priced at the list price, the cost reaches the MLflow span, the limit refuses the next call, the case is marked, a later case is refused with no call made, and a model with no price is rejected.

**Test a: one sample applicant, limit USD 0.50.** Applicant A005 (Marek Sobczak), from documents to decision.

| Call | Step | Tokens in | Tokens out | Cost (USD) |
|---|---|---|---|---|
| 1 | connection check | 17 | 2 | 0.000012 |
| 2 | A005, extract figures | 1,499 | 364 | 0.001296 |
| 3 | A005, write approval record | 890 | 241 | 0.000807 |
| | **Run total** | | | **0.002114** |

The rule function returned APPROVE, as it should. All six figures were extracted correctly. One of the eleven extraction checks failed (`'loan' flagged`): the model listed `loan_amount` as resting only on the applicant's email, and the check looks for the separate word "loan", which `loan_amount` does not match. That is the kind of test-code fault the README describes, not a fault in cost control, and it was left alone.

**Test b: the same applicant, limit USD 0.0001.** The limit was chosen so that the run would stop part way through a case.

| Call | Step | Cost (USD) | Run total (USD) |
|---|---|---|---|
| 1 | connection check | 0.000012 | 0.000012 |
| 2 | A005, extract figures | 0.001291 | 0.001303 |
| 3 | A005, write approval record | refused | 0.001303 |

The notebook printed:

```
STOPPED  Spending limit reached: USD 0.001303 spent of USD 0.000100 allowed for this run. The call for 'write approval record' (A005) was not made.
STATUS   A005: not processed: budget reached. No decision was recorded for this case.
```

The job itself still finished successfully: reaching the limit is a normal outcome, not a crash. A query afterwards showed no A005 row in the `applicants` table. The sample was then run once more at USD 0.50 (three calls, USD 0.002100) to put A005 back.

All three runs appear in MLflow as `underwriting_cost` runs with their totals, and the second one shows one call blocked.

**Test c: Promptfoo.** `npm run eval` gave the same result as before the change: A004, A005 and A007 pass, and A006 (the injection attack) fails with income 12,000 instead of 7,200, as expected.

**Not run.** The four attacks, registering a new model version and calling the deployed endpoint (`SECTIONS = "all"`) were not run as part of this work.

## Its limits

- **The limit only counts calls within one run.** Each run starts at zero. Ten runs at USD 0.50 can spend USD 5.00. A limit shared across runs, users or days needs the LiteLLM proxy server, which keeps the count in one place. That is not built.
- **A run can pass the limit by one call.** A call's cost is only known after it returns, so the check happens before each call. With the default limit that overshoot is a fraction of a cent.
- **Failed calls are not counted.** If a call errors, LiteLLM returns no token count and nothing is added. The README notes that Databricks can still charge for failed calls.
- **The cost is an estimate at list price.** It depends on LiteLLM's price list being right and current for the model named in `PRICE_KEY`. If you change `MODEL`, change `PRICE_KEY` to match.
- **The deployed endpoint is not limited.** The servable class calls the model through LiteLLM, but a serving endpoint handles each request separately and has no run to count against. The endpoint also keeps running the old code until a version registered from the version 4 notebook is deployed in the Serving UI.
- **Promptfoo calls the gateway directly.** Its calls do not go through the notebook, so they are not on any meter.

## Rebuilding it on another machine

1. Install the Databricks CLI: `winget install Databricks.DatabricksCLI` on Windows (see the Databricks documentation for other systems). Open a new terminal afterwards.
2. In the workspace, create an access token under **Settings > Developer > Access tokens**.
3. Store it with `databricks configure --host https://<your-workspace>.cloud.databricks.com`, pasting the token when asked.
4. Check the connection with a command that only reads: `databricks current-user me`.
5. Make sure `sql/01_setup.sql` has been run and the documents are in the `workspace.underwriting.documents` volume (see the README).
6. Upload the notebook and the prompt, keeping the two folders side by side, because the notebook reads `../prompts/extraction_prompt.txt`:

   ```powershell
   $W = "/Users/<you>/underwriting-agent-litellm"
   databricks workspace mkdirs "$W/notebook"
   databricks workspace mkdirs "$W/prompts"
   databricks workspace import "$W/notebook/underwriting_agent_v4" --file notebook\underwriting_agent_v4.py --format SOURCE --language PYTHON --overwrite
   databricks workspace import "$W/prompts/extraction_prompt.txt" --file prompts\extraction_prompt.txt --format AUTO --overwrite
   ```

   A Databricks Git folder pointing at this repository works as well.
7. Open the notebook in the workspace and use **Run all**. To run only the sample applicant, set `SECTIONS = "sample"` in the config cell. To see the limit block a run, set `RUN_BUDGET_USD = 0.0001`.

## Revoking the token

In the workspace: **Settings > Developer > Access tokens**, find `CC_LiteLLM`, and choose **Revoke**.

From the command line: `databricks tokens list` shows the token's ID, and `databricks tokens delete <token_id>` revokes it. Then delete `~/.databrickscfg` from the PC, or run `databricks configure` again with a new token.

The token expires by itself on 26 December 2026. The notebook does not depend on it, so runs started inside the workspace keep working after it is gone.
