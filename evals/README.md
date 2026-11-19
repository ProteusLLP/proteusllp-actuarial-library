# PAL coding-assistant evaluation set

The tasks in [pal_modeling_tasks.json](pal_modeling_tasks.json) measure whether
an AI coding assistant can construct small, correct actuarial models using
an **installed PAL package**. They are not training data and do not require
a model provider, network access or a particular assistant.

## Suggested protocol

1. Install a fixed PAL version in a disposable environment with its standard
   test dependencies. Record the Python and PAL versions.
2. For each prompt, start a fresh assistant conversation. Permit access to
   the installed package and the public PAL documentation, but do not reveal
   the answer recipe or a previous attempt. Record whether the PAL skill was
   available and whether recipe search was available.
3. Save the first generated Python program. Test it in an isolated environment,
   not on a production machine. Generated code must be treated as untrusted.
4. Score each task independently: `runs` (imports and runs), `public_api`
   (documented supported APIs), `numerical` (mathematically sensible and
   independently checked), `semantics` (correct actuarial interpretation),
   and `coupling` (stochastic dependencies and axes, where applicable).
   For tasks without dependence or stochastic parameters, record `coupling`
   as true if the code neither misuses nor breaks PAL coupling.
5. Use the same task set and judgement criteria to compare a baseline agent,
   an agent with the PAL skill, and an agent with the skill and recipe API.

Each record in an evaluation-results JSONL file uses:
```json
{"task_id":"PAL-001","runs":true,"public_api":true,"numerical":true,"semantics":true,"coupling":true}
```

Compute a reproducible aggregate report using:
```bash
python scripts/score_ai_evals.py evals/results.jsonl
```

A task passes only if all five criteria are true. Also report each criterion
separately to identify which part of the library is hard for assistants.

**No agent-generated answers or baseline scores are claimed in this repository.**
The packaged PAL recipe modules are tested in CI, but that is a different
measurement from whether an unfamiliar assistant can discover and use them.
