# Six Failure Modes — Supplementary Material

This bundle contains the audit data, evaluator, and ledger needed to
reproduce the six failure-mode audits that motivate the methodology
contribution of the paper.

## Layout

```
code/
  src/                    architecture (referenced by the audits)
  tools/eval_callback.py  callback-aware evaluator (Failure Mode 1
                          dilution audit hinges on this vs eval_chain.py)
audits/                   the six audit reports + raw JSONs:
  audit_a1_window_leakage.{md,json}
  audit_a2_redaction.md, audit_a2_v15a.json, audit_a2_v15b.json
  audit_a3_base_prior.md, audit_a3_base_prior_1p7b_cloud.md
  audit_a3_data.json, audit_a3_template_prior.{md,cloud.json}
  audit_a_corpus_prior.md
  audit_b_literature.md
ledgers/
  NEURIPS_NUMBERS.md      v27b/v28 numbers cited as the surviving recipe
  runs.md                 active run ledger (post-flip)
eval_jsons/               per-checkpoint evaluator output for v14k / v15a /
                          v15b / v15e / v15f (the failure-mode audit
                          subjects) + v27b / v28 (the surviving recipe)
```

## Reproducing the audits

Each audit doc in `audits/` documents its own command line and inputs.
The `audit_a1_window_leakage.json` and `audit_a3_data.json` files are
self-contained (no external corpus reads).
