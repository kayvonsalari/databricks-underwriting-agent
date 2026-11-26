# Tracing: recording what the agent did, one case at a time

Monitoring an agent is four separate jobs. This document covers the first.

| Job | Question it answers | Tool here |
|---|---|---|
| **Record** | **What happened?** | **MLflow tracing** |
| Check | Was it right? | Promptfoo |
| Control | What did it cost, and can it be stopped? | LiteLLM |
| Protect | Was the input or output unsafe? | NeMo Guardrails, plus one plain-code check |

## What tracing does here, and why

Version 6 of the notebook (`notebook/underwriting_agent_v6.py`) records each applicant's case as one MLflow trace, named after the applicant. Inside it, every step of the pipeline appears in the order it ran:

1. **Load documents.**
2. **Input guard**, one step per document, each ending `passed` or `blocked`.
3. **Extract figures** (a model call).
4. **Figures check**, ending `passed` or `stopped`.
5. **Rule function**: the SQL decision, with the figures it was given and the decision and key ratios it returned.
6. **Write approval record** or **write case note** (a model call).
7. **Output guard**, ending `passed` or `withheld`.

Each step shows its input, its output and how long it took. Each model call sits inside the step that made it and also shows tokens in and out and the cost LiteLLM worked out for it.

The trace as a whole carries the applicant ID, the status, the decision, the reason if the case was stopped, the number of model calls, the total cost, and the time of the slowest call. These are tags, so the trace list in MLflow can be filtered and sorted by them.

Earlier versions recorded something, but it was hard to read. Three things were wrong, and version 6 fixes each:

- **Guard checks appeared as separate, unnamed traces.** The guards run in their own thread, and the trace did not follow them there. Now each guard check sits inside the applicant's trace, under the step that asked for it.
- **A case stopped by a guard looked like a failure.** Stopping a poisoned application is the correct result. Such a trace now ends in state OK, labelled `not processed` with the reason, and has a final step `mark as not processed`. Only a real error, such as a crash or a failed model call, shows as Error.
- **Slow calls were invisible.** Every model call is now timed on its own. A call of 20 seconds or more (`SLOW_CALL_SECONDS`) is marked slow on its step, counted on the trace and listed in the run summary.

Tracing only records. It does not check whether an answer was right, it does not stop anything, and it does not change a decision. Every model call still goes through LiteLLM and the per-run spending limit (see [LITELLM.md](LITELLM.md)), the guards still work and can still be switched off with `GUARDRAILS` (see [GUARDRAILS.md](GUARDRAILS.md)), and the SQL rule function still makes every decision.

## How it was set up, in order

### 1. What changed in the notebook

Version 6 is version 5 with these changes:

- **One trace per case.** `run_case` opens the top step, named `applicant <ID>`, and everything the case does happens inside it. A stop by a guard or by the spending limit is caught inside that step, so the trace closes normally.
- **One step per pipeline stage.** A small helper, `traced_step`, wraps each stage, records its input and output, and records a stop as a result (`blocked`, `stopped`) and not as an error.
- **Model calls timed and priced.** `ask_model` still checks the spending limit and calls LiteLLM. It now also times the call and writes tokens, cost, seconds and the slow mark onto the call's step, using the field names MLflow shows as token usage and cost.
- **MLflow's automatic LiteLLM tracing is switched off.** With it on, every model call would appear twice.
- **Guards inside the trace.** The guard code passes the trace context into its thread, so the rail's model call lands under its `input guard` or `output guard` step.
- **The pipeline in one function.** `process_application` runs the seven stages in order. The sample run and the attacks all use it.
- **Traces are read back and checked.** After the run, each trace is fetched from MLflow and checked: state OK, one top step named after the case, ID, status and cost present, every step OK with an input and an output, every model call inside the trace with cost and time. The result is printed and included in the job's return value.
- **Two new run modes.** `SECTIONS = "free"` is now a full dry run: made-up applications go through the whole pipeline with fake model replies supplied by LiteLLM, so no call is made and nothing is spent, while the rule function and MLflow run for real. `SECTIONS = "tracing"` is the short paid test described below.
- **Summary.** The last cells list the slow calls and add them, with the trace checks, to the MLflow cost run.

### 2. How the tests were run, and the results

Both runs were on 26 November 2026, started from the command line as one-time serverless jobs, the same way as the earlier tests.

**Free dry run, no model calls (`SECTIONS = "free"`).** 25 of 25 tests as expected: the figures check, stopped cases staying on record, and three made-up applications (a clean one, one with hidden instructions, and one whose explanation is unfair). All 10 traces the run made were read back from MLflow and passed every check.

**Paid run (`SECTIONS = "tracing"`, limit USD 0.10).** 21 model calls, USD 0.006157 in total. All four traces were read back and passed every check.

| Test | Result | Trace | Model calls | Cost (USD) |
|---|---|---|---|---|
| Connection check | Answered | OK | 1 | 0.000012 |
| Clean sample applicant, start to finish | Five documents passed, figures passed, rule function returned APPROVE, approval record passed the fairness check | OK, `processed`, APPROVE, explanation shown, 40 s | 8 | 0.003054 |
| Output guard blocks: the clean documents under a made-up ID, with a made-up unfair explanation in place of the model's | Decision recorded, explanation withheld | OK, `processed`, APPROVE, explanation `withheld by the fairness check`, 75 s | 7 | 0.002211 |
| Hidden-instructions applicant | The poisoned payslip was blocked, the other four documents passed, the case stopped before extraction | OK, `not processed: hidden instructions found`, reason recorded, 67 s | 5 | 0.000881 |

The clean applicant's trace reads like this in MLflow:

```
applicant <ID>        OK   processed | decision: APPROVE | explanation: shown   40.1 s
   load documents                                        5.0 s
   input guard: <bank statement>        passed           1.1 s
      model call: input rail            362 in, 10 out   0.6 s
   ... one input guard step for each of the other four documents ...
   extract figures                                       6.8 s
      model call: extract figures       1,499 in, 345 out
   figures check                        passed           0.0 s
   rule function                                        18.0 s
   write approval record                                 4.8 s
      model call: write approval record   871 in, 219 out
   output guard: approval record        passed           0.6 s
      model call: output rail           338 in, 10 out   0.5 s
```

**Slow calls.** Two of the 21 calls took about 60 seconds each (60.4 and 60.5). The other 19 took between 0.3 and 7 seconds. Both slow calls were input-rail checks that returned ten tokens, and the same kind of call on the same document took 0.4 seconds in another case, so the delay is not caused by the size of the request. It looks like a wait on the model service that ends after about a minute. The trace shows exactly which step waited.

The longest step that is not a model call was the rule function in the first case (18 seconds, against 6 in the next). That is the SQL warehouse starting up.

One existing extraction check still fails on the clean sample (`'loan' flagged`), as in the earlier rounds. The check is too literal. It was reported and left alone.

After the run, the applicants table again holds a row for the hidden-instructions applicant, marked `not processed` with the reason and with no figures. The made-up test row was removed.

## Its limits

- **Only notebook runs are traced.** The servable class is unchanged in version 6: it has no tracing steps and no guards. Tracing a deployed endpoint is separate work and was not done or tested. Whether Free Edition allows an endpoint to write traces is not known.
- **Tracing records, it does not judge.** A trace in state OK means the pipeline ran as designed. It does not mean the figures or the decision were right. That is the job of the checks.
- **Slow calls are shown, not prevented.** There is no timeout and no retry. A call that waits a minute is marked slow and the run carries on.
- **The slow threshold is a fixed number** (20 seconds), chosen from the handful of timings seen so far.
- **Documents are stored in the trace.** Step inputs include document text, and for a blocked case that includes the hidden instruction itself. Anyone who can read the MLflow experiment can read them. The applicants table still holds nothing from a blocked case.
- **Costs in a trace are estimates at list price**, as everywhere else (see [LITELLM.md](LITELLM.md)).
- **A failed model call has no cost on its step.** LiteLLM returns no token count for a call that errors.
- **Not tested with tracing:** a referred applicant with a real case note, the forged-letter and pushy-broker attacks, a figures-check stop on real documents, a real spending-limit stop, a real error, a run with `GUARDRAILS = "off"`, and a full run (`SECTIONS = "all"`). The stops by the figures check and the spending limit were exercised in the free dry run only.
- **The dry run uses fake replies.** It proves the code and the trace structure, not the model's behaviour.

## Rebuilding it on another machine

1. Follow steps 1 to 6 of "Rebuilding it on another machine" in [LITELLM.md](LITELLM.md), uploading `notebook/underwriting_agent_v6.py` in place of version 4.
2. Run it once with `SECTIONS = "free"`. This makes no model call. It adds the two status columns to the `applicants` table if they are missing, runs the made-up applications and checks their traces.
3. Run it with `SECTIONS = "tracing"` to see real traces: about 21 model calls and less than one cent at list price.
4. Open the notebook's MLflow experiment and choose the **Traces** tab. Each case is one row, named after the applicant. Open a row to see its steps.
5. To compare with the unguarded behaviour, set `GUARDRAILS = "off"`. The guard steps then appear once each as `skipped`.
