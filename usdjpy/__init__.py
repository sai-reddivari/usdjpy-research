"""usdjpy: the USD/JPY hurdle-exceedance study as a small, readable package.

Importing any module makes no network request, fits no model and writes no
file. Modules, in pipeline order:

- config       folders (environment-overridable), windows, clocks, protected-folder guards
- sources      cache readers and parsers; acquisition functions, off by default
- timing       FMP cleaning, calendars, availability rules, as-of join
- target       17:00 ET decision rows, sigma_20 hurdle, label
- features     the 39 candidates on native calendars, aligned to decision rows
- splits       quarters, outer/inner folds, purging, the final freeze fold, reserved rows
- selection    stages A-D of the selection funnel
- experiments  Run A, Run B, Run C (no wrapper) and the locked final procedure, as explicit settings
- models       boosters, baselines, the final fits, frozen-model load/verify/predict
- evaluation   quarterly metrics, paired differences, summary and display tables

The code is relocated from data_review/exam3 and the run scripts. Scientific
behaviour is unchanged; see README.md for what was verified and how.
"""
