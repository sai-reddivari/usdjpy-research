"""Final boosters, their grids and the baselines (protocol v2.1 section 8; clarification 1).

Relocated from data_review/exam3/models.py, with the final (freeze) fits,
persistence and frozen-model loading of data_review/exam3/reserved.py and the
freeze orchestration of data_review/run_reserved_year.py appended. The fitting
code is unchanged; tests/test_synthetic_differential.py compares it with the
historical code on synthetic data, and tests/test_frozen_predictions.py checks
the 13 frozen models exactly.

Outer prediction procedure for every booster:
 1. For each of the eight grid configurations and each inner fold, fit with
    early stopping on that inner fold's own validation rows (cap 1000,
    patience 50) and record its AUC and its own iteration count.
 2. Choose the configuration by mean inner AUC (ties -> first in grid order).
 3. Iteration count for the refit = the latest inner fold's (fold 3) count
    under the chosen configuration.
 4. Refit on the full purged outer-training window with that count fixed and
    predict the outer quarter. The outer quarter is never used for any choice.
The probability threshold for the confusion matrix is chosen on the pooled
inner-validation predictions of the chosen configuration (maximum balanced
accuracy; evaluate.choose_threshold).

Baselines: majority class (constant probability = training positive rate);
logistic regression on the own-market family (StandardScaler + L2 logistic,
C = 1, lbfgs; no tuning); the Stage C LightGBM on the own-market family with
the iteration count from inner fold 3.

Memory: one model lives at a time; grid search keeps scores and inner
predictions only. Each returned refit model is for SHAP and is discarded by
the caller.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from itertools import product
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy
import pandas
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .features import FEATURES, OWN_MARKET
from .selection import LGB_FIXED, PATIENCE, SELECTOR_CAP, Ledger, lgb_fit_es, timed
from .splits import Dataset, OuterFold

FINAL_CAP = 1000
FINAL_PATIENCE = 50
THREADS = 2

GRIDS: dict[str, list[dict]] = {
    'lightgbm': [dict(num_leaves=nl, min_child_samples=md, learning_rate=lr)
                 for nl, md, lr in product([7, 15], [20, 40], [0.03, 0.1])],
    'xgboost': [dict(max_depth=d, min_child_weight=w, learning_rate=eta)
                for d, w, eta in product([3, 4], [1, 5], [0.03, 0.1])],
    'catboost': [dict(depth=d, l2_leaf_reg=l2, learning_rate=lr)
                 for d, l2, lr in product([4, 6], [3, 10], [0.03, 0.1])],
}

# Run B item 5 (manifests/usdjpy_run_b_spec.md): one extra structural knob per model, two values, the first being the
# library default. The sixteen-combination grid is the original eight (knob at default, listed first and in the original
# order) followed by the same eight with the knob at its second value, so a tie across the two halves keeps the default.
EXTRA_KNOB: dict[str, tuple[str, list]] = {
    'lightgbm': ('min_split_gain', [0.0, 0.1]),      # sklearn name of min_gain_to_split
    'xgboost': ('gamma', [0.0, 1.0]),
    'catboost': ('random_strength', [1, 3]),
}
GRIDS16: dict[str, list[dict]] = {
    name: [dict(cfg, **{knob: v}) for v in values for cfg in GRIDS[name]] for name, (knob, values) in EXTRA_KNOB.items()
}
CAT_REPORT_KEYS = ('boosting_type', 'grow_policy', 'bootstrap_type', 'random_strength', 'subsample', 'depth', 'l2_leaf_reg',
                   'learning_rate', 'iterations', 'has_time', 'loss_function', 'eval_metric')

LGB_FINAL_FIXED = dict(objective='binary', reg_lambda=1.0, colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
                       random_state=0, verbose=-1, n_jobs=THREADS, metric='auc')
XGB_FIXED = dict(objective='binary:logistic', tree_method='hist', reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
                 random_state=0, n_jobs=THREADS, eval_metric='auc')
CAT_FIXED = dict(loss_function='Logloss', eval_metric='AUC', grow_policy='SymmetricTree', bootstrap_type='Bernoulli',
                 subsample=0.8, has_time=True, random_seed=0, verbose=0, thread_count=THREADS, allow_writing_files=False)


def fit_booster(name: str, cfg: dict, Xtr: pandas.DataFrame, ytr: numpy.ndarray, Xva: pandas.DataFrame | None = None,
                yva: numpy.ndarray | None = None, n_iter: int | None = None, cap: int = FINAL_CAP,
                patience: int = FINAL_PATIENCE):
    """Fit one booster. With (Xva, yva): early stopping on it, returns (model, n_trees, p_valid).
    With n_iter: fixed iteration count and no validation, returns (model, n_iter, None)."""
    es = Xva is not None
    if name == 'lightgbm':
        model = lgb.LGBMClassifier(n_estimators=cap if es else n_iter, **LGB_FINAL_FIXED, **cfg)
        if es:
            model.fit(Xtr, ytr, eval_X=Xva, eval_y=yva,
                      callbacks=[lgb.early_stopping(patience, first_metric_only=True, verbose=False)])
            n = int(model.best_iteration_ or cap)
            return model, n, model.predict_proba(Xva, num_iteration=n)[:, 1]
        model.fit(Xtr, ytr)
        return model, n_iter, None
    if name == 'xgboost':
        model = XGBClassifier(n_estimators=cap if es else n_iter, early_stopping_rounds=patience if es else None,
                              **XGB_FIXED, **cfg)
        if es:
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            n = int(model.best_iteration) + 1                      # best_iteration is 0-based
            return model, n, model.predict_proba(Xva, iteration_range=(0, n))[:, 1]
        model.fit(Xtr, ytr)
        return model, n_iter, None
    if name == 'catboost':
        model = CatBoostClassifier(iterations=cap if es else n_iter, **CAT_FIXED, **cfg)
        if es:
            model.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=patience, use_best_model=True)
            n = int(model.get_best_iteration()) + 1                # 0-based; model already shrunk to it
            assert model.tree_count_ == n
            return model, n, model.predict_proba(Xva)[:, 1]
        model.fit(Xtr, ytr)
        return model, n_iter, None
    raise ValueError(name)


def predict(model, name: str, X: pandas.DataFrame, n_iter: int) -> numpy.ndarray:
    if name == 'lightgbm':
        return model.predict_proba(X, num_iteration=n_iter)[:, 1]
    if name == 'xgboost':
        return model.predict_proba(X, iteration_range=(0, n_iter))[:, 1]
    return model.predict_proba(X)[:, 1]


def catboost_effective_params(model) -> dict:
    """Run B item 6: the parameters CatBoost actually used, read back from the fitted model (report, not tune)."""
    P = model.get_all_params()
    return {k: P.get(k) for k in CAT_REPORT_KEYS}


def run_booster(name: str, ds: Dataset, fold: OuterFold, feats: list[str], ledger: Ledger,
                nested: bool = False, label: str | None = None) -> dict:
    """Grid on the inner folds, refit with the latest inner fold's count, predict the outer quarter.

    nested=False (run A): the eight-combination grid GRIDS[name].
    nested=True (run B item 5): the sixteen-combination grid GRIDS16[name]; the returned headline choice is the best
    of the sixteen, and `eight` holds the choice restricted to the eight with the extra knob at its default, refitted
    and predicted the same way (served from the same refit when the two choices coincide).
    `label` names the result (default: `name`), e.g. 'lightgbm_own_market' for the matched own-price version.
    """
    feats = [c for c in FEATURES if c in feats]
    label = label or name
    grid = GRIDS16[name] if nested else GRIDS[name]
    knob = EXTRA_KNOB[name][0] if nested else None
    rows = []
    inner_preds: dict[int, list[numpy.ndarray]] = {}
    with timed(ledger, f'final_{label}_grid'):
        for gi, cfg in enumerate(grid):
            aucs, iters, preds = [], [], []
            for j, (tr, va) in enumerate(fold.inner, start=1):
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        model, n, p = fit_booster(name, cfg, ds.df.iloc[tr][feats], ds.y[tr],
                                                  ds.df.iloc[va][feats], ds.y[va])
                    ledger.warn(f'final_{label} cfg{gi} inner{j}', caught)
                    ledger.fit(f'final_{label}_grid')
                    aucs.append(float(roc_auc_score(ds.y[va], p)))
                    iters.append(n)
                    preds.append(p)
                    del model
                except Exception as e:                              # recorded, never swallowed silently
                    ledger.fail(f'final_{label}', f'cfg{gi} inner{j} {cfg}', e)
                    aucs.append(float('nan'))
                    iters.append(-1)
                    preds.append(numpy.full(len(va), numpy.nan))
            inner_preds[gi] = preds
            row = {'config_index': gi, 'config': cfg, 'inner_auc': [round(a, 5) for a in aucs],
                   'mean_inner_auc': float(numpy.mean(aucs)), 'best_iteration': iters}
            if nested:
                row['knob_at_default'] = cfg[knob] == EXTRA_KNOB[name][1][0]
            rows.append(row)
    ok = [r for r in rows if numpy.isfinite(r['mean_inner_auc'])]
    assert ok, f'{label}: every configuration failed'
    chosen = max(ok, key=lambda r: r['mean_inner_auc'])      # max keeps the first on ties -> grid order
    pooled_inner_y = numpy.concatenate([ds.y[v] for _, v in fold.inner])
    tr, va = fold.train_idx, fold.valid_idx

    def refit(ch: dict, stage: str) -> tuple:
        n_refit = int(ch['best_iteration'][-1])               # latest inner fold (fold 3)
        with timed(ledger, f'final_{label}_refit'):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                m, _, _ = fit_booster(name, ch['config'], ds.df.iloc[tr][feats], ds.y[tr], n_iter=n_refit)
            ledger.warn(f'final_{label} {stage}', caught)
            ledger.fit(f'final_{label}_refit')
            p_out = predict(m, name, ds.df.iloc[va][feats], n_refit)
        return m, n_refit, p_out

    model, n_refit, p_outer = refit(chosen, 'refit')
    out = {'model': model, 'name': label, 'booster': name, 'features': feats, 'grid': rows, 'chosen': chosen,
           'refit_iterations': n_refit, 'p_outer': p_outer,
           'inner_p': numpy.concatenate(inner_preds[chosen['config_index']]), 'inner_y': pooled_inner_y,
           'grid_size': len(grid)}
    if name == 'catboost':
        out['catboost_effective_params'] = catboost_effective_params(model)
    if nested:
        ok8 = [r for r in ok if r['knob_at_default']]
        chosen8 = max(ok8, key=lambda r: r['mean_inner_auc'])
        same = chosen8['config_index'] == chosen['config_index']
        if same:
            model8, n8, p8 = model, n_refit, p_outer
        else:
            model8, n8, p8 = refit(chosen8, 'refit_eight')
        out['eight'] = {'model': model8, 'name': f'{label}_grid8', 'booster': name, 'features': feats, 'chosen': chosen8,
                        'refit_iterations': n8, 'p_outer': p8, 'inner_p': numpy.concatenate(inner_preds[chosen8['config_index']]),
                        'inner_y': pooled_inner_y, 'same_as_sixteen_choice': same, 'grid_size': len(ok8)}
        out['extra_knob'] = {'name': knob, 'values': EXTRA_KNOB[name][1], 'chosen_value': chosen['config'][knob],
                             'non_default_chosen': not chosen['knob_at_default'],
                             'eight_choice_index': chosen8['config_index'], 'sixteen_choice_index': chosen['config_index'],
                             'same_choice': same, 'eight_mean_inner_auc': chosen8['mean_inner_auc'],
                             'sixteen_mean_inner_auc': chosen['mean_inner_auc']}
        if name == 'catboost':
            out['eight']['catboost_effective_params'] = catboost_effective_params(model8)
    return out


# ----------------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------------
def baseline_majority(ds: Dataset, fold: OuterFold, ledger: Ledger) -> dict:
    with timed(ledger, 'baseline_majority'):
        rate = float(ds.y[fold.train_idx].mean())
        p = numpy.full(len(fold.valid_idx), rate)
        inner_p = numpy.concatenate([numpy.full(len(v), float(ds.y[t].mean())) for t, v in fold.inner])
        inner_y = numpy.concatenate([ds.y[v] for _, v in fold.inner])
    return {'model': None, 'name': 'majority', 'features': [], 'train_positive_rate': rate,
            'majority_class': int(rate >= 0.5), 'p_outer': p, 'inner_p': inner_p, 'inner_y': inner_y}


def _logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))


def baseline_logistic_own_market(ds: Dataset, fold: OuterFold, ledger: Ledger) -> dict:
    feats = [c for c in OWN_MARKET if c in ds.features]
    with timed(ledger, 'baseline_logistic'):
        inner_p, inner_auc = [], []
        for j, (tr, va) in enumerate(fold.inner, start=1):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                m = _logistic().fit(ds.df.iloc[tr][feats], ds.y[tr])
            ledger.warn(f'baseline_logistic inner{j}', caught)
            ledger.fit('baseline_logistic')
            p = m.predict_proba(ds.df.iloc[va][feats])[:, 1]
            inner_p.append(p)
            inner_auc.append(float(roc_auc_score(ds.y[va], p)))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model = _logistic().fit(ds.df.iloc[fold.train_idx][feats], ds.y[fold.train_idx])
        ledger.warn('baseline_logistic refit', caught)
        ledger.fit('baseline_logistic')
        p_outer = model.predict_proba(ds.df.iloc[fold.valid_idx][feats])[:, 1]
    return {'model': model, 'name': 'logistic_own_market', 'features': feats, 'inner_auc': [round(a, 5) for a in inner_auc],
            'p_outer': p_outer, 'inner_p': numpy.concatenate(inner_p), 'inner_y': numpy.concatenate([ds.y[v] for _, v in fold.inner])}


def baseline_lgb_own_market(ds: Dataset, fold: OuterFold, ledger: Ledger) -> dict:
    feats = [c for c in OWN_MARKET if c in ds.features]
    with timed(ledger, 'baseline_lgb'):
        inner_p, inner_auc, iters = [], [], []
        for j, (tr, va) in enumerate(fold.inner, start=1):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                m, n, a, p = lgb_fit_es(ds.df.iloc[tr][feats], ds.y[tr], ds.df.iloc[va][feats], ds.y[va],
                                        LGB_FIXED, SELECTOR_CAP, PATIENCE)
            ledger.warn(f'baseline_lgb inner{j}', caught)
            ledger.fit('baseline_lgb')
            inner_p.append(p)
            inner_auc.append(a)
            iters.append(n)
            del m
        n_refit = iters[-1]
        model = lgb.LGBMClassifier(n_estimators=n_refit, **LGB_FIXED)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(ds.df.iloc[fold.train_idx][feats], ds.y[fold.train_idx])
        ledger.warn('baseline_lgb refit', caught)
        ledger.fit('baseline_lgb')
        p_outer = model.predict_proba(ds.df.iloc[fold.valid_idx][feats])[:, 1]
    return {'model': model, 'name': 'lgb_own_market', 'features': feats, 'inner_auc': [round(a, 5) for a in inner_auc],
            'inner_iterations': iters, 'refit_iterations': n_refit, 'p_outer': p_outer,
            'inner_p': numpy.concatenate(inner_p), 'inner_y': numpy.concatenate([ds.y[v] for _, v in fold.inner])}


# ----------------------------------------------------------------------------
# Final (freeze) fits: run_booster without the outer prediction (from data_review/exam3/reserved.py)
# ----------------------------------------------------------------------------
def fit_frozen_booster(name: str, ds: Dataset, fold: OuterFold, feats: list[str], ledger: Ledger,
                       label: str | None = None) -> dict:
    """Sixteen-configuration grid on the inner folds, choice, refit on the final window; no prediction.

    Mirrors exam3.models.run_booster(nested=True) line for line except that no
    outer rows are predicted (the fold has none). The eight-grid by-product is
    recorded as in run B.
    """
    feats = [c for c in FEATURES if c in feats]
    label = label or name
    grid = GRIDS16[name]
    knob = EXTRA_KNOB[name][0]
    rows = []
    inner_preds: dict[int, list[numpy.ndarray]] = {}
    with timed(ledger, f'final_{label}_grid'):
        for gi, cfg in enumerate(grid):
            aucs, iters, preds = [], [], []
            for j, (tr, va) in enumerate(fold.inner, start=1):
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        model, n, p = fit_booster(name, cfg, ds.df.iloc[tr][feats], ds.y[tr],
                                                     ds.df.iloc[va][feats], ds.y[va])
                    ledger.warn(f'final_{label} cfg{gi} inner{j}', caught)
                    ledger.fit(f'final_{label}_grid')
                    aucs.append(float(roc_auc_score(ds.y[va], p)))
                    iters.append(n)
                    preds.append(p)
                    del model
                except Exception as e:                              # recorded, never swallowed silently
                    ledger.fail(f'final_{label}', f'cfg{gi} inner{j} {cfg}', e)
                    aucs.append(float('nan'))
                    iters.append(-1)
                    preds.append(numpy.full(len(va), numpy.nan))
            inner_preds[gi] = preds
            rows.append({'config_index': gi, 'config': cfg, 'inner_auc': [round(a, 5) for a in aucs],
                         'inner_auc_full_precision': aucs,
                         'mean_inner_auc': float(numpy.mean(aucs)), 'best_iteration': iters,
                         'knob_at_default': cfg[knob] == EXTRA_KNOB[name][1][0]})
    ok = [r for r in rows if numpy.isfinite(r['mean_inner_auc'])]
    assert ok, f'{label}: every configuration failed'
    chosen = max(ok, key=lambda r: r['mean_inner_auc'])      # max keeps the first on ties -> grid order
    ties = [r['config_index'] for r in ok if r['mean_inner_auc'] == chosen['mean_inner_auc']]
    pooled_inner_y = numpy.concatenate([ds.y[v] for _, v in fold.inner])
    tr = fold.train_idx

    def refit(ch: dict, stage: str):
        n_refit = int(ch['best_iteration'][-1])               # latest inner fold (fold 3, Q16)
        with timed(ledger, f'final_{label}_refit'):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                m, _, _ = fit_booster(name, ch['config'], ds.df.iloc[tr][feats], ds.y[tr], n_iter=n_refit)
            ledger.warn(f'final_{label} {stage}', caught)
            ledger.fit(f'final_{label}_refit')
        return m, n_refit

    model, n_refit = refit(chosen, 'refit')
    out = {'model': model, 'name': label, 'booster': name, 'features': feats, 'grid': rows, 'chosen': chosen,
           'tied_config_indices_at_max': ties, 'refit_iterations': n_refit,
           'inner_p': numpy.concatenate(inner_preds[chosen['config_index']]), 'inner_y': pooled_inner_y,
           'grid_size': len(grid), 'n_train': int(len(tr))}
    if name == 'catboost':
        out['catboost_effective_params'] = catboost_effective_params(model)
        out['catboost_get_all_params'] = {k: _jsonable(v) for k, v in model.get_all_params().items()}
        out['catboost_constructor_has_time'] = model.get_params().get('has_time')
    ok8 = [r for r in ok if r['knob_at_default']]
    chosen8 = max(ok8, key=lambda r: r['mean_inner_auc'])
    same = chosen8['config_index'] == chosen['config_index']
    if same:
        model8, n8 = model, n_refit
    else:
        model8, n8 = refit(chosen8, 'refit_eight')
    out['eight'] = {'model': model8, 'name': f'{label}_grid8', 'booster': name, 'features': feats, 'chosen': chosen8,
                    'refit_iterations': n8, 'inner_p': numpy.concatenate(inner_preds[chosen8['config_index']]),
                    'inner_y': pooled_inner_y, 'same_as_sixteen_choice': same, 'grid_size': len(ok8), 'n_train': int(len(tr))}
    out['extra_knob'] = {'name': knob, 'values': EXTRA_KNOB[name][1], 'chosen_value': chosen['config'][knob],
                         'non_default_chosen': not chosen['knob_at_default'],
                         'eight_choice_index': chosen8['config_index'], 'sixteen_choice_index': chosen['config_index'],
                         'same_choice': same, 'eight_mean_inner_auc': chosen8['mean_inner_auc'],
                         'sixteen_mean_inner_auc': chosen['mean_inner_auc']}
    if name == 'catboost':
        out['eight']['catboost_effective_params'] = catboost_effective_params(model8)
    return out


def fit_frozen_logistic(ds: Dataset, fold: OuterFold, ledger: Ledger) -> dict:
    """Own-price logistic regression fitted once on the final window; inner predictions for the threshold."""
    feats = [c for c in OWN_MARKET if c in ds.features]
    with timed(ledger, 'baseline_logistic'):
        inner_p, inner_auc = [], []
        for j, (tr, va) in enumerate(fold.inner, start=1):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                m = _logistic().fit(ds.df.iloc[tr][feats], ds.y[tr])
            ledger.warn(f'baseline_logistic inner{j}', caught)
            ledger.fit('baseline_logistic')
            p = m.predict_proba(ds.df.iloc[va][feats])[:, 1]
            inner_p.append(p)
            inner_auc.append(float(roc_auc_score(ds.y[va], p)))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model = _logistic().fit(ds.df.iloc[fold.train_idx][feats], ds.y[fold.train_idx])
        ledger.warn('baseline_logistic refit', caught)
        ledger.fit('baseline_logistic')
    lr = model.named_steps['logisticregression']
    return {'model': model, 'name': 'logistic_own_market', 'features': feats, 'inner_auc': [round(a, 5) for a in inner_auc],
            'inner_p': numpy.concatenate(inner_p), 'inner_y': numpy.concatenate([ds.y[v] for _, v in fold.inner]),
            'n_train': int(len(fold.train_idx)), 'coefficients_standardised': dict(zip(feats, lr.coef_.ravel().tolist())),
            'intercept': float(lr.intercept_[0]), 'n_iter': int(numpy.ravel(lr.n_iter_)[0]),
            'settings': {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 5000, 'scaler': 'StandardScaler on the training partition'}}


def frozen_majority(ds: Dataset, fold: OuterFold) -> dict:
    rate = float(ds.y[fold.train_idx].mean())
    inner_p = numpy.concatenate([numpy.full(len(v), float(ds.y[t].mean())) for t, v in fold.inner])
    inner_y = numpy.concatenate([ds.y[v] for _, v in fold.inner])
    return {'model': None, 'name': 'majority', 'features': [], 'train_positive_rate': rate,
            'majority_class': int(rate >= 0.5), 'inner_p': inner_p, 'inner_y': inner_y, 'n_train': int(len(fold.train_idx))}


# ----------------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------------
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, numpy.integer):
        return int(o)
    if isinstance(o, (numpy.floating, float)):
        return None if not numpy.isfinite(o) else float(o)
    if isinstance(o, (numpy.bool_, bool)):
        return bool(o)
    if isinstance(o, numpy.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (pandas.Timestamp,)):
        return str(o)
    return o


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_model(model, booster: str | None, stem: Path) -> dict:
    """Pickle (used for scoring) plus the library's native format (provenance); returns file names and hashes.

    The historical manifest stored paths relative to the research folder; a new
    freeze stores file names relative to its own frozen folder.
    """
    files = {}
    jp = stem.with_suffix('.joblib')
    joblib.dump(model, jp)
    files['joblib'] = jp
    if booster == 'lightgbm':
        p = stem.with_suffix('.lgb.txt'); model.booster_.save_model(str(p)); files['native'] = p
    elif booster == 'xgboost':
        p = stem.with_suffix('.xgb.json'); model.save_model(str(p)); files['native'] = p
    elif booster == 'catboost':
        p = stem.with_suffix('.cbm'); model.save_model(str(p), format='cbm'); files['native'] = p
    return {k: {'path': v.name, 'sha256': sha256_file(v), 'bytes': v.stat().st_size} for k, v in files.items()}


def load_model(stem: Path):
    """The frozen model object from its joblib file (the native copies are provenance only)."""
    return joblib.load(stem.with_suffix('.joblib'))


def predict_frozen(model, booster: str | None, X: pandas.DataFrame, n_iter: int | None) -> numpy.ndarray:
    """Probability of the positive class from a frozen model (mo.predict for boosters, pipeline for logistic)."""
    if booster in FROZEN_BOOSTERS:
        return predict(model, booster, X, int(n_iter))
    return model.predict_proba(X)[:, 1]


def fingerprint(model, booster: str | None, ds: Dataset, fold: OuterFold, feats: list[str], n_iter: int | None,
                k: int = 25) -> dict:
    """Predictions on the last k training rows (dates and probabilities), to re-assert identity after reloading."""
    idx = fold.train_idx[-k:]
    p = predict_frozen(model, booster, ds.df.iloc[idx][feats], n_iter)
    return {'decision_dates': [str(d.date()) for d in ds.df['decision_date'].iloc[idx]], 'p': [float(v) for v in p]}




# ----------------------------------------------------------------------------
# Frozen reserved-year models: load, verify, predict (from data_review/reserved_year_evaluate.py)
# ----------------------------------------------------------------------------
FROZEN_BOOSTERS = ('lightgbm', 'xgboost', 'catboost')
FROZEN_PROCEDURES = ['nw_lightgbm', 'nw_xgboost', 'nw_catboost', 'lightgbm_own_market', 'xgboost_own_market',
                     'catboost_own_market', 'logistic_own_market', 'majority']
FROZEN_GRID8 = [f'{name}_grid8' for name in FROZEN_PROCEDURES[:6]]


def load_frozen_manifest(frozen_dir: Path) -> dict:
    """frozen_manifest.json of a frozen folder."""
    return json.loads((frozen_dir / 'frozen_manifest.json').read_text())


def load_frozen_models(frozen_dir: Path, manifest: dict | None = None) -> tuple[dict, dict]:
    """(manifest, {name: model or None}); each joblib file is asserted against its recorded sha256 first."""
    manifest = manifest or load_frozen_manifest(frozen_dir)
    frozen = {}
    for name, record in manifest['models'].items():
        if not record['files']:
            frozen[name] = None                                  # the majority baseline has no model file
            continue
        joblib_path = frozen_dir / Path(record['files']['joblib']['path']).name
        if sha256_file(joblib_path) != record['files']['joblib']['sha256']:
            raise ValueError(f'{name}: frozen model file differs from its recorded hash')
        frozen[name] = load_model(frozen_dir / name)
    return manifest, frozen


def frozen_probabilities(manifest: dict, frozen: dict, name: str, X: pandas.DataFrame) -> numpy.ndarray:
    """Positive-class probability of one frozen procedure on rows X (columns taken in the recorded order)."""
    record = manifest['models'][name]
    if name == 'majority':
        return numpy.full(len(X), record['train_positive_rate'])
    return predict_frozen(frozen[name], record['booster'], X[record['features']], record.get('refit_iterations'))


def frozen_classes(manifest: dict, name: str, probabilities: numpy.ndarray) -> numpy.ndarray:
    """Class decisions at the frozen inner-chosen threshold (p >= threshold -> 1), as evaluation.quarter_metrics."""
    threshold = manifest['models'][name]['threshold_choice']['threshold']
    return (numpy.asarray(probabilities, dtype=float) >= threshold).astype(int)


def assert_fingerprints(manifest: dict, frozen: dict, ds: Dataset, fold: OuterFold) -> dict:
    """Re-predict the recorded fingerprint rows (last training rows) and assert exact identity."""
    checked = {}
    for name, record in manifest['models'].items():
        if frozen[name] is None:
            continue
        fp = record['fingerprint']
        idx = fold.train_idx[-len(fp['p']):]
        dates = [str(d.date()) for d in ds.df['decision_date'].iloc[idx]]
        assert dates == fp['decision_dates'], f'{name}: fingerprint rows moved'
        p = predict_frozen(frozen[name], record['booster'], ds.df.iloc[idx][record['features']], record.get('refit_iterations'))
        assert numpy.array_equal(p, numpy.array(fp['p'])), f'{name}: fingerprint mismatch'
        checked[name] = True
    return checked


# ----------------------------------------------------------------------------
# The locked final procedure: freeze orchestration (from data_review/run_reserved_year.py, freeze stage)
# ----------------------------------------------------------------------------
FINGERPRINT_ROWS = 25


def freeze_final_procedure(ds: Dataset, fold: OuterFold, frozen_dir: Path, settings, elastic_net_seed: int,
                           ledger: Ledger) -> dict:
    """Stages A-C, the shortlist, every final fit, thresholds, saved models and fingerprints.

    This is the freeze stage of data_review/run_reserved_year.py with the
    output folder, settings and seed passed in; the steps and assertions are
    unchanged. It fits models: callers must hold explicit authorisation
    (scripts/train_final_procedure.py --allow-fitting). The real-data checks
    (1,018 eligible rows, 1,009 training rows, 39 features) stay in the script,
    so the orchestration can also be exercised on synthetic data in tests.
    Returns {'selection': ..., 'models': ..., 'shortlist': ..., 'own_market_features': ...}.
    """
    from . import evaluation, selection
    from .features import FAMILY

    frozen_dir.mkdir(parents=True, exist_ok=False)            # a fresh folder: never over an existing freeze

    with timed(ledger, 'A'):
        stage_a = selection.stage_a(ds, fold)
    with timed(ledger, 'B'):
        stage_b = selection.stage_b(ds, fold, stage_a['inputs'], ledger)
    with timed(ledger, 'C_linear'):
        linear = selection.stage_c_linear(ds, fold, stage_b['survivors'], ledger, random_state=elastic_net_seed)
    with timed(ledger, 'C_lgb'):
        nonlinear = selection.stage_c_nonlinear(ds, fold, stage_b['survivors'], stage_a['group_of'], ledger,
                                                fixed_iterations=settings.fixed_iterations, params=settings.lgb_params)
    shortlist_record = selection.shortlist_from(linear['candidates'], nonlinear['candidates'], ds.features)
    shortlist = shortlist_record['shortlist']
    assert all(c in shortlist for c in selection.PROTECTED if c in ds.features), 'a protected feature left the shortlist'
    assert nonlinear['best_iteration'] == [settings.fixed_iterations] * 3 and linear['random_state'] == elastic_net_seed
    selection_record = {
        'settings': settings.as_dict(), 'elastic_net_random_state': elastic_net_seed, 'sizes': fold.sizes(),
        'inner_quarters': fold.inner_quarters, 'stage_a': stage_a, 'stage_b': stage_b,
        'stage_c': {'linear': linear, 'nonlinear': nonlinear, **shortlist_record}, 'shortlist': shortlist,
        'shortlist_size': len(shortlist), 'shortlist_families': {c: FAMILY[c] for c in shortlist},
        'stage_d': 'not run (locked: no wrapper)',
    }
    (frozen_dir / 'selection_stages_a_to_c.json').write_text(json.dumps(_jsonable(selection_record), indent=1))

    own = [c for c in OWN_MARKET if c in ds.features]
    frozen: dict[str, dict] = {}

    def record(result: dict, label: str, booster: str | None) -> dict:
        threshold = evaluation.choose_threshold(result['inner_p'], result['inner_y'])
        stem = frozen_dir / label
        files = save_model(result['model'], booster, stem) if result['model'] is not None else {}
        n_iter = result.get('refit_iterations')
        fp = (fingerprint(result['model'], booster, ds, fold, result['features'], n_iter, FINGERPRINT_ROWS)
              if result['model'] is not None else None)
        pooled_inner_auc = (float(roc_auc_score(result['inner_y'], result['inner_p']))
                            if numpy.isfinite(result['inner_p']).all() else None)
        entry = {'name': label, 'booster': booster, 'features': result['features'], 'n_features': len(result['features']),
                 'n_train': result['n_train'], 'threshold_choice': threshold, 'files': files, 'fingerprint': fp,
                 'inner_auc_pooled': pooled_inner_auc, 'inner_n': int(len(result['inner_y'])),
                 'inner_positive': int(result['inner_y'].sum())}
        for key in ('chosen', 'refit_iterations', 'grid', 'grid_size', 'extra_knob', 'same_as_sixteen_choice',
                    'tied_config_indices_at_max', 'catboost_effective_params', 'catboost_get_all_params',
                    'catboost_constructor_has_time', 'inner_auc', 'train_positive_rate', 'majority_class',
                    'coefficients_standardised', 'intercept', 'n_iter', 'settings'):
            if key in result:
                entry[key] = result[key]
        if result['model'] is not None:
            reloaded = load_model(stem)
            rows = ds.df.iloc[fold.train_idx[-FINGERPRINT_ROWS:]][result['features']]
            assert numpy.array_equal(predict_frozen(reloaded, booster, rows, n_iter), numpy.array(fp['p'])), \
                f'{label}: reloaded model does not reproduce the fingerprint'
        frozen[label] = entry
        return entry

    for name in FROZEN_BOOSTERS:
        for label, feats in ((f'nw_{name}', shortlist), (f'{name}_own_market', own)):
            with timed(ledger, f'final_{label}_total'):
                result = fit_frozen_booster(name, ds, fold, feats, ledger, label=label)
            assert result['features'] == [c for c in FEATURES if c in feats]
            assert result['grid_size'] == 16 and [g['config'] for g in result['grid']] == GRIDS16[name]
            if name == 'catboost':
                params = result['catboost_effective_params']
                assert (params['boosting_type'], params['grow_policy'], params['bootstrap_type']) == \
                    ('Plain', 'SymmetricTree', 'Bernoulli'), params
                assert result['catboost_constructor_has_time'] is True and 'has_time' not in result['catboost_get_all_params']
                assert params['iterations'] == result['refit_iterations']
            record(result, label, name)
            eight = result['eight']
            entry8 = record(eight, eight['name'], name)
            entry8['same_as_sixteen_choice'] = eight['same_as_sixteen_choice']
            del result, eight
    with timed(ledger, 'baseline_logistic_total'):
        logistic = fit_frozen_logistic(ds, fold, ledger)
    record(logistic, 'logistic_own_market', None)
    record(frozen_majority(ds, fold), 'majority', None)
    return {'selection': selection_record, 'models': frozen, 'shortlist': shortlist, 'own_market_features': own}
