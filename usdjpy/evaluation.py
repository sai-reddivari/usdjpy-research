"""Evaluation of outer-quarter predictions (protocol v2.1 sections 5 and 8; clarification 5).

Relocated from data_review/exam3/evaluate.py, plus table builders from
data_review/summarise_run_a.py and reserved_year_evaluate.py and the report
display tables formerly inside submission/scripts/build_results_display.py.
tests/test_summary_tables.py rebuilds the saved tables from saved predictions.

Per quarter: ROC-AUC, Hanley-McNeil standard error (reference only; ignores
temporal dependence), confusion matrix and classification report at the
threshold chosen on the inner folds (maximum balanced accuracy), calibration
bins (five quantile bins of the predicted probability: count, mean predicted,
observed rate). Across quarters: mean quarterly AUC (primary), pooled AUC
(secondary), range and standard deviation labelled approximate, paired
per-quarter differences between procedures. SHAP: TreeExplainer on each
outer-fold model applied to its own quarter, summarised as mean |SHAP| per
feature; aggregation across folds happens later, never one model on all data.

Nothing here returns feature values: only predictions, labels, metrics and
per-feature SHAP summaries.
"""
from __future__ import annotations

import json

import numpy
import pandas
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score

N_CALIBRATION_BINS = 5


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley & McNeil (1982) standard error of an AUC; assumes independent observations."""
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return float(numpy.sqrt(max(var, 0.0)))


def choose_threshold(p: numpy.ndarray, y: numpy.ndarray) -> dict:
    """Threshold maximising balanced accuracy over the midpoints of the sorted predicted probabilities."""
    p, y = numpy.asarray(p, dtype=float), numpy.asarray(y, dtype=int)
    ok = numpy.isfinite(p)
    p, y = p[ok], y[ok]
    u = numpy.unique(p)
    if len(u) < 2:
        return {'threshold': 0.5, 'inner_balanced_accuracy': 0.5, 'n_candidates': 0}
    cands = (u[:-1] + u[1:]) / 2
    scores = [balanced_accuracy_score(y, (p >= t).astype(int)) for t in cands]
    i = int(numpy.argmax(scores))                                     # first maximum
    return {'threshold': float(cands[i]), 'inner_balanced_accuracy': float(scores[i]), 'n_candidates': int(len(cands))}


def calibration_bins(p: numpy.ndarray, y: numpy.ndarray, n_bins: int = N_CALIBRATION_BINS) -> list[dict]:
    p, y = numpy.asarray(p, dtype=float), numpy.asarray(y, dtype=int)
    if len(numpy.unique(p)) < 2:
        return [{'bin': 0, 'n': int(len(p)), 'mean_predicted': float(p.mean()), 'observed_rate': float(y.mean())}]
    edges = numpy.unique(numpy.quantile(p, numpy.linspace(0, 1, n_bins + 1)))
    idx = numpy.clip(numpy.searchsorted(edges, p, side='right') - 1, 0, len(edges) - 2)
    out = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            out.append({'bin': b, 'lower': float(edges[b]), 'upper': float(edges[b + 1]), 'n': int(m.sum()),
                        'mean_predicted': float(p[m].mean()), 'observed_rate': float(y[m].mean())})
    return out


def quarter_metrics(y: numpy.ndarray, p: numpy.ndarray, threshold: float) -> dict:
    y, p = numpy.asarray(y, dtype=int), numpy.asarray(p, dtype=float)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    auc = float(roc_auc_score(y, p)) if 0 < n_pos < len(y) else float('nan')
    yhat = (p >= threshold).astype(int)
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    rep = classification_report(y, yhat, labels=[0, 1], output_dict=True, zero_division=0)
    return {'n': int(len(y)), 'n_positive': n_pos, 'n_negative': n_neg, 'auc': auc,
            'hanley_mcneil_se_reference_only': hanley_mcneil_se(auc, n_pos, n_neg) if numpy.isfinite(auc) else float('nan'),
            'threshold': float(threshold), 'balanced_accuracy': float(balanced_accuracy_score(y, yhat)),
            'confusion_matrix': {'tn': int(cm[0, 0]), 'fp': int(cm[0, 1]), 'fn': int(cm[1, 0]), 'tp': int(cm[1, 1])},
            'classification_report': rep, 'calibration_bins': calibration_bins(p, y),
            'mean_predicted': float(p.mean()), 'positive_rate': float(y.mean())}


def summarise_quarters(per_quarter_auc: dict[int, float], pooled_y: numpy.ndarray | None = None,
                       pooled_p: numpy.ndarray | None = None) -> dict:
    vals = numpy.array([v for v in per_quarter_auc.values() if numpy.isfinite(v)])
    out = {'n_quarters': int(len(vals)), 'mean_quarterly_auc': float(vals.mean()) if len(vals) else float('nan'),
           'range_approx': [float(vals.min()), float(vals.max())] if len(vals) else None,
           'sd_approx': float(vals.std(ddof=1)) if len(vals) > 1 else None,
           'note': 'range and sd are approximate; quarters are not independent'}
    if pooled_y is not None and pooled_p is not None:
        out['pooled_auc'] = float(roc_auc_score(pooled_y, pooled_p))
    return out


def paired_differences(per_quarter: dict[str, dict[int, float]], a: str, b: str) -> dict:
    """Per-quarter AUC(a) - AUC(b) with mean, sd and range (approximate)."""
    qs = sorted(set(per_quarter[a]) & set(per_quarter[b]))
    d = {q: per_quarter[a][q] - per_quarter[b][q] for q in qs}
    vals = numpy.array(list(d.values()))
    return {'a': a, 'b': b, 'per_quarter': {str(q): float(v) for q, v in d.items()},
            'mean': float(vals.mean()) if len(vals) else float('nan'),
            'sd_approx': float(vals.std(ddof=1)) if len(vals) > 1 else None,
            'range_approx': [float(vals.min()), float(vals.max())] if len(vals) else None,
            'quarters_favouring_a': int((vals > 0).sum())}


def shap_mean_abs(model, X: pandas.DataFrame, name: str) -> dict:
    """Mean |SHAP| per feature for one outer-fold model on its own quarter (TreeExplainer)."""
    import shap
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):                                       # older binary output: [class0, class1]
        sv = sv[-1]
    sv = numpy.asarray(sv)
    if sv.ndim == 3:                                               # (n, features, classes)
        sv = sv[:, :, -1]
    assert sv.shape == X.shape, (sv.shape, X.shape)
    m = numpy.abs(sv).mean(axis=0)
    order = numpy.argsort(-m)
    return {'model': name, 'n_rows': int(len(X)),
            'mean_abs_shap': {X.columns[i]: float(m[i]) for i in order},
            'expected_value': float(numpy.ravel(explainer.expected_value)[-1])}


def shap_values(model, X: pandas.DataFrame) -> tuple[numpy.ndarray, float]:
    """Per-row SHAP matrix (n, features) of the positive class for one outer-fold model on its own quarter,
    plus the explainer's expected value. Same TreeExplainer and output handling as shap_mean_abs; added for
    run A so the per-row values (public features only) can be saved without computing SHAP twice."""
    import shap
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):                                       # older binary output: [class0, class1]
        sv = sv[-1]
    sv = numpy.asarray(sv)
    if sv.ndim == 3:                                               # (n, features, classes)
        sv = sv[:, :, -1]
    assert sv.shape == X.shape, (sv.shape, X.shape)
    return sv, float(numpy.ravel(explainer.expected_value)[-1])


def shap_summary_from_values(sv: numpy.ndarray, columns: list[str], name: str, expected_value: float) -> dict:
    """The shap_mean_abs dictionary computed from an already obtained SHAP matrix."""
    m = numpy.abs(sv).mean(axis=0)
    order = numpy.argsort(-m)
    return {'model': name, 'n_rows': int(sv.shape[0]),
            'mean_abs_shap': {columns[i]: float(m[i]) for i in order}, 'expected_value': float(expected_value)}


# ----------------------------------------------------------------------------
# Saved predictions -> summary tables (from data_review/summarise_run_a.py and reserved_year_evaluate.py)
# ----------------------------------------------------------------------------
DEVELOPMENT_QUARTERS = list(range(9, 17))
LAST_FOUR_QUARTERS = [13, 14, 15, 16]
RESERVED_QUARTER_LABELS = ['R1', 'R2', 'R3', 'R4']


def development_predictions(results_dir, model, quarters=DEVELOPMENT_QUARTERS) -> pandas.DataFrame:
    """decision_date, y, p, flagged and quarter for one model, all outer quarters, from Q<q>_<model>.csv files."""
    frames = [pandas.read_csv(results_dir / 'predictions' / f'Q{q}_{model}.csv').assign(quarter=q) for q in quarters]
    return pandas.concat(frames, ignore_index=True)


def reserved_predictions(results_dir, model) -> pandas.DataFrame:
    """decision_date, quarter (R1..R4), y, p and flagged for one model, from predictions/<model>.csv."""
    return pandas.read_csv(results_dir / 'predictions' / f'{model}.csv')


def quarter_aucs(predictions: pandas.DataFrame, quarters) -> dict:
    """{quarter: ROC-AUC} recomputed from saved predictions."""
    return {q: float(roc_auc_score(predictions.loc[predictions['quarter'] == q, 'y'],
                                   predictions.loc[predictions['quarter'] == q, 'p'])) for q in quarters}


def development_auc_table(predictions_by_model: dict) -> pandas.DataFrame:
    """The auc_per_quarter_main.csv layout of runs A and B, recomputed from saved predictions.

    Mean and sample SD (ddof 1) are over the eight quarterly AUCs; the SD is
    descriptive, not a standard error. Pooled AUC ranks all outer predictions
    together. The last-four columns use Q13-Q16.
    """
    rows = []
    for model, predictions in predictions_by_model.items():
        per_quarter = quarter_aucs(predictions, DEVELOPMENT_QUARTERS)
        values = numpy.array([per_quarter[q] for q in DEVELOPMENT_QUARTERS])
        last_four = predictions[predictions['quarter'].isin(LAST_FOUR_QUARTERS)]
        rows.append({'model': model, **{f'Q{q}': per_quarter[q] for q in DEVELOPMENT_QUARTERS},
                     'mean_quarterly_auc': values.mean(), 'sd_approx': values.std(ddof=1),
                     'min': values.min(), 'max': values.max(),
                     'pooled_auc': roc_auc_score(predictions['y'], predictions['p']),
                     'last4_mean_quarterly_auc': values[-4:].mean(),
                     'last4_pooled_auc': roc_auc_score(last_four['y'], last_four['p']),
                     'quarters_above_0.5': int((values > 0.5).sum())})
    return pandas.DataFrame(rows).set_index('model')


def development_paired_table(per_quarter_auc: dict, pairs) -> pandas.DataFrame:
    """The paired_differences.csv layout of runs A and B: AUC(a) - AUC(b) per quarter, mean, SD, count positive."""
    rows = []
    for first, second in pairs:
        difference = paired_differences(per_quarter_auc, first, second)
        rows.append({'comparison': f'{first} - {second}',
                     **{f'Q{q}': difference['per_quarter'][str(q)] for q in DEVELOPMENT_QUARTERS},
                     'mean': difference['mean'], 'sd_approx': difference['sd_approx'],
                     'quarters_positive': difference['quarters_favouring_a'],
                     'last4_mean': float(numpy.mean([difference['per_quarter'][str(q)] for q in LAST_FOUR_QUARTERS]))})
    return pandas.DataFrame(rows).set_index('comparison')


def reserved_auc_table(predictions_by_model: dict) -> pandas.DataFrame:
    """The reserved auc_per_quarter.csv layout (R1..R4, mean, SD, pooled, flagged-row sensitivity, count above 0.5)."""
    rows = []
    for model, predictions in predictions_by_model.items():
        per_quarter = quarter_aucs(predictions, RESERVED_QUARTER_LABELS)
        values = numpy.array([per_quarter[q] for q in RESERVED_QUARTER_LABELS])
        kept = predictions[~predictions['flagged'].astype(bool)]
        kept_per_quarter = quarter_aucs(kept, RESERVED_QUARTER_LABELS)
        rows.append({'model': model, **per_quarter, 'mean_quarterly_auc': float(values.mean()),
                     'sd_approx': float(values.std(ddof=1)),
                     'pooled_auc': float(roc_auc_score(predictions['y'], predictions['p'])),
                     'mean_quarterly_auc_excl_flagged': float(numpy.mean(list(kept_per_quarter.values()))),
                     'pooled_auc_excl_flagged': float(roc_auc_score(kept['y'], kept['p'])),
                     'quarters_above_0.5': int(sum(v > 0.5 for v in per_quarter.values()))})
    return pandas.DataFrame(rows).set_index('model')


def reserved_paired_table(auc_table: pandas.DataFrame, pairs) -> pandas.DataFrame:
    """The reserved paired_differences.csv layout, from a reserved_auc_table."""
    per_quarter = {model: {q: auc_table.loc[model, q] for q in RESERVED_QUARTER_LABELS} for model in auc_table.index}
    rows = []
    for first, second in pairs:
        differences = {q: per_quarter[first][q] - per_quarter[second][q] for q in RESERVED_QUARTER_LABELS}
        values = numpy.array(list(differences.values()))
        rows.append({'comparison': f'{first} - {second}', **differences, 'mean': float(values.mean()),
                     'sd_approx': float(values.std(ddof=1)), 'quarters_positive': int((values > 0).sum()),
                     'quarters_negative': int((values < 0).sum()),
                     'pooled_auc_difference': float(auc_table.loc[first, 'pooled_auc'] - auc_table.loc[second, 'pooled_auc'])})
    return pandas.DataFrame(rows).set_index('comparison')


def confusion_counts(y, p, threshold) -> dict:
    """tn, fp, fn, tp at a threshold, with the p >= threshold -> 1 rule of quarter_metrics."""
    predicted = (numpy.asarray(p, dtype=float) >= threshold).astype(int)
    matrix = confusion_matrix(numpy.asarray(y, dtype=int), predicted, labels=[0, 1])
    return {'tn': int(matrix[0, 0]), 'fp': int(matrix[0, 1]), 'fn': int(matrix[1, 0]), 'tp': int(matrix[1, 1])}


# ----------------------------------------------------------------------------
# Report display tables (moved from submission/scripts/build_results_display.py; logic unchanged)
# ----------------------------------------------------------------------------
DISPLAY_BOOSTERS = ('lightgbm', 'xgboost', 'catboost')
DISPLAY_NAMES = {'lightgbm': 'LightGBM', 'xgboost': 'XGBoost', 'catboost': 'CatBoost'}
DISPLAY_SPECS = {
    'run_a': ('Run A', 'run_a/summary/auc_per_quarter_main.csv', ''),
    'run_b': ('Run B', 'run_b/summary/auc_per_quarter_main.csv', ''),
    'no_wrapper': ('No-wrapper replay', 'no_wrapper_development/summary/auc_per_quarter_comparison.csv', 'nw_'),
    'reserved': ('Reserved: original 229 rows', 'reserved_year/summary/auc_per_quarter.csv', 'nw_'),
}


def display_model_label(model, stage):
    """Keep unmatched baselines and the reserved nominee visibly identified."""
    if model == 'majority':
        return 'Constant prevalence'
    if model == 'logistic_own_market':
        return 'Logistic: own price'
    if model == 'lgb_own_market':
        return 'LightGBM: own price (unmatched)'
    if model.endswith('_own_market'):
        return DISPLAY_NAMES[model.removesuffix('_own_market')] + ': own price'
    name = DISPLAY_NAMES[model.removeprefix('nw_')]
    description = 'selected' if stage in ('run_a', 'run_b') else 'shortlist'
    suffix = ' (nominee)' if stage == 'reserved' and model == 'nw_catboost' else ''
    return f'{name}: {description}{suffix}'


def display_iteration_tables(results_root):
    """Headline models per stage with the mean and sample SD (ddof 1) of quarterly AUC; returns (tables, sources)."""
    tables = {}
    sources = []
    for stage, (_title, relative_path, prefix) in DISPLAY_SPECS.items():
        path = results_root / relative_path
        source = pandas.read_csv(path).set_index('model')
        sources.append(path)
        quarters = [f'R{i}' for i in range(1, 5)] if stage == 'reserved' else [f'Q{i}' for i in range(9, 17)]
        models = ['logistic_own_market'] + [prefix + model for model in DISPLAY_BOOSTERS]
        models += ['lgb_own_market'] if stage == 'run_a' else [model + '_own_market' for model in DISPLAY_BOOSTERS]
        models += ['majority']
        table = source.loc[models, quarters + ['mean_quarterly_auc', 'pooled_auc']].copy()
        quarterly_scores = table[quarters]
        assert numpy.isfinite(quarterly_scores.to_numpy()).all(), f'{stage}: non-finite quarterly AUC'
        numpy.testing.assert_allclose(quarterly_scores.mean(axis=1), table['mean_quarterly_auc'], atol=1e-12, rtol=0)
        table['quarterly_sd'] = quarterly_scores.std(axis=1, ddof=1)
        table['quarters_above_chance'] = quarterly_scores.gt(0.5).sum(axis=1)
        table['quarter_count'] = len(quarters)
        if stage != 'reserved':
            table['last4_mean'] = table[quarters[-4:]].mean(axis=1)
        table.insert(0, 'label', [display_model_label(model, stage) for model in models])
        tables[stage] = table
    return tables, sources


def display_paired_comparisons(tables):
    """Paired quarter differences of each booster against its references, without a significance claim."""
    rows = []
    for stage, table in tables.items():
        quarters = [column for column in table if column.startswith(('Q', 'R')) and column[1:].isdigit()]
        prefix = DISPLAY_SPECS[stage][2]
        for model in DISPLAY_BOOSTERS:
            selected = table.loc[prefix + model, quarters].astype(float)
            comparisons = [('own-price logistic', table.loc['logistic_own_market', quarters])]
            if stage != 'run_a':
                comparisons.append(('matched own-price ' + DISPLAY_NAMES[model], table.loc[model + '_own_market', quarters]))
            if stage == 'no_wrapper':
                comparisons.append(('Run B with wrapper', tables['run_b'].loc[model, quarters]))
            for reference, values in comparisons:
                differences = selected - values.astype(float)
                row = {'iteration': stage, 'booster': DISPLAY_NAMES[model], 'reference': reference,
                       'mean_difference': differences.mean(), 'sd_quarterly_difference': differences.std(ddof=1),
                       'positive_quarters': int(differences.gt(0).sum()), 'quarter_count': len(quarters)}
                row.update(differences.to_dict())
                rows.append(row)
    return pandas.DataFrame(rows)


def display_control_summary(results_root):
    """Matched real/shifted-label diagnostics of Run B; returns (diagnostics, summary table, sources)."""
    chain_path = results_root / 'run_b/negative_controls/negative_controls_summary.csv'
    logistic_path = results_root / 'run_b/negative_controls/logistic_own_market_controls.json'
    chain = pandas.read_csv(chain_path).set_index('chain')
    logistic = json.loads(logistic_path.read_text())['chains']
    logistic_scores = pandas.Series({name: row['mean_quarterly_auc'] for name, row in logistic.items()})
    diagnostics = {
        'B cheap chain: outer mean AUC': chain['mean_quarterly_auc'],
        'Own-price logistic: outer mean AUC': logistic_scores,
        'B wrapper: inner AUC gain': chain['wrapper_inner_gain_mean'],
    }
    summary = []
    for label, scores in diagnostics.items():
        shifted = scores.drop('real')
        assert len(shifted) == 20, f'{label}: expected twenty shifted runs'
        summary.append({'diagnostic': label, 'real': scores.loc['real'], 'shift_mean': shifted.mean(),
                        'shift_median': shifted.median(), 'shift_sd': shifted.std(ddof=1),
                        'shift_min': shifted.min(), 'shift_max': shifted.max(),
                        'shifts_at_or_above_real': int(shifted.ge(scores.loc['real']).sum()), 'shift_count': len(shifted)})
    return diagnostics, pandas.DataFrame(summary), [chain_path, logistic_path]
