# Reproduction boundaries

## What is included

PDFs, editable LaTeX, figures, saved aggregate results, feature definitions, computational Python modules and three research entry points. Public exhibit scripts need no observations or credentials. The public checker does not repeat the historical model-equivalence suite.

## What is needed for the research pipeline

~~~bash
export USDJPY_RESEARCH_ROOT=/path/to/data_review
export USDJPY_PRIVATE_DIR=/private/path/to/jquants
export USDJPY_PRIVATE_SCRATCH=/private/path/to/new_scratch
export USDJPY_OUTPUT_DIR=/path/to/new_outputs
~~~

Use the recorded Python/library versions in requirements.txt. The research root needs original raw caches, observation/aligned tables, calendars, manifests and result records. A provider subscription alone does not recreate that folder. The private directory holds licensed TOPIX, Banks and Electric Appliances & Precision histories and derived tables. Access and retention remain subject to provider terms.

| Entry point | Requirements and limits |
|---|---|
| scripts/rebuild_aligned_tables.py | Original caches, metadata and calendars; full build needs private sector histories. Public-only mode still needs public-source caches and does not reproduce the locked matrix. |
| scripts/score_frozen_models.py | Original frozen models/manifests and required feature/evaluation artifacts. Not runnable from this ZIP alone. |
| scripts/train_final_procedure.py | Full research inputs, pinned environment and explicit fitting opt-in. A new fit is not the original frozen evaluation. |

Historical A/B/C and control drivers are not repackaged as if this archive could replay them independently. The experiments module records their settings. The historical refactor report is evidence, not a promise private-reference tests run here.

Acquisition is off by default. Do not put credentials or licensed caches in this repository. Secure credential retrieval on another operating system is an explicit setup task. Do not substitute missing inputs or change settings silently.

Exact reproduction depends on snapshots and stochastic state: development elastic net was unseeded; only the final fit used seed zero. Those settings are not revised here. Fresh downloads may contain revisions. Original frozen models, rather than new fits, are necessary to reproduce the original probabilities.

## Public evidence and document builds

The exhibit builder reads results/ and writes figures and LaTeX extracts. It summarizes existing aggregates without prediction recomputation. The report builder invokes Tectonic in temporary directories and writes reports/. Set CODEX_TECTONIC_PATH if needed.

No licensed rows, exam/lecture PDFs, résumé, signed declaration, credentials or model binaries belong in this public study. For guidance contact Sai Reddivari at reddivari.sai95@gmail.com; this is not a promise to redistribute licensed data.
