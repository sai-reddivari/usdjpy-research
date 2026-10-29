"""Feature selection inside one outer training window: stages A to E of protocol v2.1 section 6.

Relocated verbatim from data_review/exam3/selection.py (only numpy/pandas names
changed). tests/test_synthetic_differential.py runs both on the same synthetic
data with the same seeds and requires identical records. Runs A, B and C left
the elastic net unseeded (random_state=None); keep that default.

Everything here sees only the outer training window of the fold it is given
(its three purged inner folds); the outer quarter never enters.

Conventions chosen where the protocol is silent (documented, simplest option):
- Stage A clustering uses every eligible column (constants and duplicates
  removed), including yen_relative_5d; the linear-basis hold-out applies only
  to the elastic-net stage. Within a multi-member cluster the first listed
  representative rule whose two members are both present fires; otherwise the
  member with the highest mean |Spearman| to the rest of the cluster is kept.
- Stage B "bottom quartile" = the n // 4 lowest mutual-information values of
  that inner fold. Removals are processed in increasing order of mean MI so the
  family safeguard ("not the last eligible member of its family") is evaluated
  against the current survivors.
- Elastic net: StandardScaler and LogisticRegression(solver='saga',
  l1_ratio=0.5, max_iter=20000, tol=1e-4); convergence warnings
  are recorded, never suppressed. Ties in mean AUC across C go to the smaller C.
- Early stopping: metric AUC on the inner fold's own validation rows, patience
  50 rounds, cap 600 (selector) — local to each inner fold (clarification 1).
- Grouped permutation importance: the rows of every feature of the group are
  permuted with the same permutation (20 repeats, numpy Generator seed 0);
  the drop is validation AUC at the best iteration minus the permuted AUC.
- Wrapper budget: 400 fits counting each inner-fold fit; a candidate subset is
  evaluated only when three more fits fit in the budget; cache hits are free.
- Stop rule: smallest evaluated subset whose mean inner AUC is within 0.005 of
  the best evaluated subset; ties on size go to the higher score.
- Interaction comparisons: for each declared interaction, background = selected
  subset minus the interaction plus its components (forced in); "without" versus
  "with" the interaction, three inner folds each (18 fits, outside the budget).

Run B (manifests/usdjpy_run_b_spec.md) is reached through SelectionSettings
only: a fixed selector iteration count (no early stopping in stages C-LGB, D
and the interaction comparisons), a different wrapper budget, and an optional
switch to skip the interaction comparisons (the negative-control cheap chain).
Every default reproduces run A exactly; run_a.py passes no settings.

Privacy: this module returns names, scores, ranks, coefficients, importances
and cluster memberships. It never returns or writes feature values.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy
import pandas
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .features import FAMILY, FEATURES
from .splits import Dataset, OuterFold

# Declared interactions and their components (feature manifest sections 5, 7, 9).
INTERACTIONS: dict[str, list[str]] = {
    'yen_relative_5d': ['ret_5d', 'usd_basket_5d'],
    'short_yen_stress': ['cftc_net', 'vol_z'],
    'gapchg_x_vixchg': ['gap2y_chg_5d', 'vix_chg_5d'],
}
COMPONENTS: list[str] = [c for v in INTERACTIONS.values() for c in v]
PROTECTED: set[str] = set(INTERACTIONS) | set(COMPONENTS)
AUXILIARY_COMPONENTS = ['cftc_net', 'vol_z', 'gap2y_chg_5d', 'vix_chg_5d']   # section 6 stage A item 5
LINEAR_HOLDOUT = ['yen_relative_5d']                                       # linear basis {ret_5d, usd_basket_5d}
DISCRETE = ['event_in_window', 'weekend_in_window']

# Conditional representatives (preferred, other); applied only when both share a cluster.
REP_RULES: list[tuple[str, str]] = [
    ('ret_1d', 'tokyo_reference_to_ny_close'), ('vol_5_over_20', 'abs_ret_1d'), ('vol_20', 'vol_z'),
    ('dist_ma20', 'ret_20d'), ('pos_z', 'cftc_net'), ('qqq_minus_spy_5d', 'qqq_minus_spy_20d'),
    ('banks_rel_5d', 'banks_rel_20d'), ('vix_chg_5d', 'vix_chg_1d'),
]
CLUSTER_CUT = 0.10

# Stage C LightGBM: fixed structural parameters (sklearn parameter names; see protocol section 6).
LGB_FIXED = dict(objective='binary', num_leaves=7, max_depth=3, learning_rate=0.03, min_child_samples=40,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=0,
                 verbose=-1, n_jobs=2, metric='auc')
SELECTOR_CAP = 600
PATIENCE = 50
WRAPPER_BUDGET = 400
STOP_TOLERANCE = 0.005
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0]
MI_NEIGHBORS = 3
PERM_REPEATS = 20
SEED = 0


# ----------------------------------------------------------------------------
# Bookkeeping: fit counts, wall times, warnings
# ----------------------------------------------------------------------------
@dataclass
class Ledger:
    fits: dict[str, int] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)
    warnings: list[dict] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)

    def fit(self, stage: str, n: int = 1) -> None:
        self.fits[stage] = self.fits.get(stage, 0) + n

    def warn(self, stage: str, caught: list[warnings.WarningMessage]) -> None:
        for w in caught:
            self.warnings.append({'stage': stage, 'category': w.category.__name__, 'message': str(w.message)[:300]})

    def fail(self, stage: str, what: str, err: Exception) -> None:
        self.failures.append({'stage': stage, 'what': what, 'error': f'{type(err).__name__}: {err}'[:500]})

    def summary(self) -> dict:
        return {'fits': dict(self.fits), 'total_fits': int(sum(self.fits.values())),
                'seconds': {k: round(v, 2) for k, v in self.seconds.items()},
                'total_seconds': round(sum(self.seconds.values()), 2),
                'n_warnings': len(self.warnings), 'n_failures': len(self.failures)}


class timed:
    """with timed(ledger, 'stage'): ... records wall time with time.perf_counter."""

    def __init__(self, ledger: Ledger, stage: str):
        self.ledger, self.stage = ledger, stage

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ledger.seconds[self.stage] = self.ledger.seconds.get(self.stage, 0.0) + time.perf_counter() - self.t0


# ----------------------------------------------------------------------------
# LightGBM selector: one fit with early stopping on the fold's own validation
# ----------------------------------------------------------------------------
def lgb_fit_es(Xtr: pandas.DataFrame, ytr: numpy.ndarray, Xva: pandas.DataFrame, yva: numpy.ndarray,
               params: dict = LGB_FIXED, cap: int = SELECTOR_CAP, patience: int = PATIENCE):
    """Fit with early stopping on (Xva, yva); returns (model, best_iteration, validation AUC, p_valid)."""
    model = lgb.LGBMClassifier(n_estimators=cap, **params)
    model.fit(Xtr, ytr, eval_X=Xva, eval_y=yva,
              callbacks=[lgb.early_stopping(patience, first_metric_only=True, verbose=False)])
    best = int(model.best_iteration_ or cap)
    p = model.predict_proba(Xva, num_iteration=best)[:, 1]
    return model, best, float(roc_auc_score(yva, p)), p


def lgb_fit_fixed(Xtr: pandas.DataFrame, ytr: numpy.ndarray, Xva: pandas.DataFrame, yva: numpy.ndarray,
                  params: dict = LGB_FIXED, n_iter: int = 150):
    """Run B selector: a fixed iteration count, no early stopping; same return shape as lgb_fit_es
    (the 'best iteration' is the fixed count). (Xva, yva) are used for scoring only."""
    model = lgb.LGBMClassifier(n_estimators=n_iter, **params)
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xva, num_iteration=n_iter)[:, 1]
    return model, int(n_iter), float(roc_auc_score(yva, p)), p


@dataclass(frozen=True)
class SelectionSettings:
    """Run-A defaults. fixed_iterations=None means the inner-selected (early-stopped) count of protocol v2.1."""
    fixed_iterations: int | None = None       # run B: 150 (stages C-LGB, D, interactions)
    wrapper_budget: int = WRAPPER_BUDGET      # run B: 2000
    tolerance: float = STOP_TOLERANCE         # unchanged in run B
    interactions: bool = True                 # False for the negative-control cheap chain (stages A-D only)
    selector_threads: int | None = None       # None = LGB_FIXED's n_jobs (run A). Thread count changes wall time only:
                                              # LightGBM predictions were checked bit-identical for 1, 2 and 4 threads.

    @property
    def label(self) -> str:
        return 'early-stopped selector (run A)' if self.fixed_iterations is None else f'fixed {self.fixed_iterations}-tree selector (run B)'

    @property
    def lgb_params(self) -> dict:
        return LGB_FIXED if self.selector_threads is None else {**LGB_FIXED, 'n_jobs': self.selector_threads}

    def as_dict(self) -> dict:
        return {'fixed_iterations': self.fixed_iterations, 'wrapper_budget': self.wrapper_budget, 'tolerance': self.tolerance,
                'interactions': self.interactions, 'selector_threads': self.selector_threads, 'label': self.label}


RUN_A = SelectionSettings()
RUN_B = SelectionSettings(fixed_iterations=150, wrapper_budget=2000, selector_threads=1)


# ----------------------------------------------------------------------------
# Stage A: structural filter
# ----------------------------------------------------------------------------
def stage_a(ds: Dataset, fold: OuterFold) -> dict:
    X = ds.df.iloc[fold.train_idx][ds.features]
    out: dict = {'n_train_rows': int(len(X)), 'n_input': len(ds.features)}
    miss = X.isna().sum()
    out['missing_values'] = {c: int(v) for c, v in miss.items() if v > 0}
    assert not out['missing_values'], f'unexpected missing values in training rows: {out["missing_values"]}'
    out['non_finite'] = {c: int((~numpy.isfinite(X[c])).sum()) for c in X.columns if (~numpy.isfinite(X[c])).any()}
    assert not out['non_finite']
    const = [c for c in X.columns if X[c].nunique() <= 1]
    cols = [c for c in X.columns if c not in const]
    dups: list[list[str]] = []
    seen: dict[bytes, str] = {}
    for c in cols:
        key = numpy.ascontiguousarray(X[c].to_numpy(dtype=float)).tobytes()
        if key in seen:
            dups.append([seen[key], c])
        else:
            seen[key] = c
    dup_drop = [b for _, b in dups]
    cols = [c for c in cols if c not in dup_drop]
    out['dropped_constant'], out['dropped_duplicate'] = const, dups

    rho = X[cols].corr(method='spearman').abs()
    dist = 1.0 - rho.to_numpy()
    numpy.fill_diagonal(dist, 0.0)
    dist = numpy.clip((dist + dist.T) / 2, 0.0, None)
    Z = linkage(squareform(dist, checks=False), method='complete')
    labels = fcluster(Z, t=CLUSTER_CUT, criterion='distance')
    clusters: dict[str, list[str]] = {}
    order = sorted(set(labels), key=lambda l: min(cols.index(c) for c, lab in zip(cols, labels) if lab == l))
    for gi, lab in enumerate(order, start=1):
        clusters[f'g{gi:02d}'] = [c for c, l in zip(cols, labels) if l == lab]
    group_of = {c: g for g, members in clusters.items() for c in members}

    representatives: dict[str, str] = {}
    rules_fired: list[dict] = []
    within_corr: dict[str, dict] = {}
    for g, members in clusters.items():
        if len(members) == 1:
            representatives[g] = members[0]
            continue
        sub = rho.loc[members, members]
        within_corr[g] = {f'{a}|{b}': round(float(sub.loc[a, b]), 4) for i, a in enumerate(members) for b in members[i + 1:]}
        rep = None
        for pref, other in REP_RULES:
            if pref in members and other in members:
                rep = pref
                rules_fired.append({'cluster': g, 'kept': pref, 'over': other, 'members': members})
                break
        if rep is None:
            mean_abs = {m: float((sub.loc[m, [o for o in members if o != m]]).mean()) for m in members}
            rep = max(mean_abs, key=mean_abs.get)
            rules_fired.append({'cluster': g, 'kept': rep, 'rule': 'highest mean |rho| within cluster',
                                'mean_abs_rho': {k: round(v, 4) for k, v in mean_abs.items()}, 'members': members})
        representatives[g] = rep
    inputs = [c for c in cols if c == representatives[group_of[c]]]
    removed = [c for c in cols if c not in inputs]
    out.update({
        'clusters': clusters, 'group_of': group_of, 'representatives': representatives,
        'rules_fired': rules_fired, 'within_cluster_abs_spearman': within_corr,
        'max_abs_spearman_pair': _max_pair(rho),
        'inputs': inputs, 'removed_as_inputs': removed,
        'auxiliary_components_retained': [c for c in AUXILIARY_COMPONENTS if c in removed],
        'linear_holdout': LINEAR_HOLDOUT, 'n_inputs': len(inputs),
    })
    return out


def _max_pair(rho: pandas.DataFrame) -> dict:
    r = rho.to_numpy().copy()
    numpy.fill_diagonal(r, 0)
    i, j = numpy.unravel_index(numpy.argmax(r), r.shape)
    return {'pair': [rho.index[i], rho.columns[j]], 'abs_spearman': round(float(r[i, j]), 4)}


# ----------------------------------------------------------------------------
# Stage B: light supervised screen (mutual information)
# ----------------------------------------------------------------------------
def stage_b(ds: Dataset, fold: OuterFold, inputs: list[str], ledger: Ledger) -> dict:
    n = len(inputs)
    q = n // 4
    mi_table: dict[str, list[float]] = {c: [] for c in inputs}
    rank_table: dict[str, list[int]] = {c: [] for c in inputs}
    discrete = numpy.array([c in DISCRETE for c in inputs])
    assert discrete.sum() == len([c for c in DISCRETE if c in inputs]), 'discrete flags must be passed to MI'
    for j, (tr, _) in enumerate(fold.inner, start=1):
        Xtr, ytr = ds.df.iloc[tr][inputs], ds.y[tr]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            mi = mutual_info_classif(Xtr, ytr, discrete_features=discrete, n_neighbors=MI_NEIGHBORS, random_state=SEED)
        ledger.warn('B', caught)
        ranks = pandas.Series(mi, index=inputs).rank(method='first', ascending=True)
        for c, v, r in zip(inputs, mi, ranks):
            mi_table[c].append(float(v))
            rank_table[c].append(int(r))
    bottom_all = [c for c in inputs if all(r <= q for r in rank_table[c])]
    mean_mi = {c: float(numpy.mean(mi_table[c])) for c in inputs}
    survivors = list(inputs)
    removed, kept_despite = [], []
    for c in sorted(bottom_all, key=mean_mi.get):
        fam = FAMILY[c]
        fam_left = [s for s in survivors if FAMILY[s] == fam]
        if c in PROTECTED:
            kept_despite.append({'feature': c, 'reason': 'declared interaction or component needed to test one'})
        elif fam_left == [c]:
            kept_despite.append({'feature': c, 'reason': f'last eligible member of family {fam}'})
        else:
            removed.append(c)
            survivors.remove(c)
    return {'n_inputs': n, 'bottom_quartile_size': q, 'mi': mi_table, 'rank': rank_table,
            'plain_rule_would_remove': bottom_all, 'removed': removed, 'kept_despite_bottom_quartile': kept_despite,
            'survivors': survivors, 'discrete_features': [c for c in DISCRETE if c in inputs]}


# ----------------------------------------------------------------------------
# Stage C: elastic net and LightGBM grouped permutation importance
# ----------------------------------------------------------------------------
def stage_c_linear(ds: Dataset, fold: OuterFold, feats_in: list[str], ledger: Ledger,
                   random_state: int | None = None) -> dict:
    """Elastic-net stage. ``random_state`` (default None = the unseeded saga shuffle of runs A/B and the replay) is
    passed to LogisticRegression; the reserved-year final fit passes 0 (manifests/usdjpy_locked_procedure.md 3.5)."""
    feats = [c for c in feats_in if c not in LINEAR_HOLDOUT]
    auc: dict[float, list[float]] = {}
    nonzero: dict[float, dict[str, int]] = {}
    coef_store: dict[float, list[numpy.ndarray]] = {}
    n_iter: dict[float, list[int]] = {}
    for C in C_GRID:
        auc[C], nonzero[C], coef_store[C], n_iter[C] = [], {c: 0 for c in feats}, [], []
        for j, (tr, va) in enumerate(fold.inner, start=1):
            sc = StandardScaler().fit(ds.df.iloc[tr][feats])
            Xtr, Xva = sc.transform(ds.df.iloc[tr][feats]), sc.transform(ds.df.iloc[va][feats])
            # sklearn >= 1.8: l1_ratio in (0, 1) with solver saga is the elastic-net penalty (penalty= is deprecated)
            lr = LogisticRegression(solver='saga', l1_ratio=0.5, C=C, max_iter=20000, tol=1e-4, random_state=random_state)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                lr.fit(Xtr, ds.y[tr])
            ledger.warn(f'C_linear C={C} inner{j}', caught)
            ledger.fit('C_linear')
            p = lr.predict_proba(Xva)[:, 1]
            auc[C].append(float(roc_auc_score(ds.y[va], p)))
            coef = lr.coef_.ravel()
            coef_store[C].append(coef)
            n_iter[C].append(int(numpy.ravel(lr.n_iter_)[0]))
            for c, b in zip(feats, coef):
                nonzero[C][c] += int(abs(b) > 0)
    mean_auc = {C: float(numpy.mean(v)) for C, v in auc.items()}
    chosen = max(C_GRID, key=lambda C: (round(mean_auc[C], 12), -C))
    candidates = [c for c in feats if nonzero[chosen][c] >= 2]
    mean_abs_coef = {c: round(float(numpy.mean([abs(k[i]) for k in coef_store[chosen]])), 5) for i, c in enumerate(feats)}
    return {'features_used': feats, 'held_out': [c for c in feats_in if c in LINEAR_HOLDOUT], 'random_state': random_state,
            'auc_by_C': {str(C): [round(a, 4) for a in v] for C, v in auc.items()},
            'mean_auc_by_C': {str(C): round(v, 4) for C, v in mean_auc.items()},
            'n_iter_by_C': {str(C): v for C, v in n_iter.items()},
            'chosen_C': chosen, 'nonzero_folds_at_chosen_C': nonzero[chosen],
            'mean_abs_std_coef_at_chosen_C': mean_abs_coef, 'candidates': candidates}


def grouped_permutation_drop(model, best_iter: int, Xva: pandas.DataFrame, yva: numpy.ndarray, groups: dict[str, list[str]],
                             repeats: int = PERM_REPEATS, seed: int = SEED) -> dict[str, float]:
    rng = numpy.random.default_rng(seed)
    base = roc_auc_score(yva, model.predict_proba(Xva, num_iteration=best_iter)[:, 1])
    drops: dict[str, float] = {}
    n = len(Xva)
    for g, cols in groups.items():
        vals = []
        for _ in range(repeats):
            Xp = Xva.copy()
            perm = rng.permutation(n)
            Xp[cols] = Xva[cols].to_numpy()[perm]
            vals.append(base - roc_auc_score(yva, model.predict_proba(Xp, num_iteration=best_iter)[:, 1]))
        drops[g] = float(numpy.mean(vals))
    return drops


def stage_c_nonlinear(ds: Dataset, fold: OuterFold, feats: list[str], group_of: dict[str, str], ledger: Ledger,
                      fixed_iterations: int | None = None, params: dict = LGB_FIXED) -> dict:
    groups: dict[str, list[str]] = {}
    for c in feats:
        groups.setdefault(group_of[c], []).append(c)
    fold_auc, iters, drops = [], [], []
    for j, (tr, va) in enumerate(fold.inner, start=1):
        Xtr, Xva = ds.df.iloc[tr][feats], ds.df.iloc[va][feats]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            if fixed_iterations is None:
                model, best, a, _ = lgb_fit_es(Xtr, ds.y[tr], Xva, ds.y[va], params)
            else:
                model, best, a, _ = lgb_fit_fixed(Xtr, ds.y[tr], Xva, ds.y[va], params, fixed_iterations)
            d = grouped_permutation_drop(model, best, Xva, ds.y[va], groups)
        ledger.warn(f'C_lgb inner{j}', caught)
        ledger.fit('C_lgb')
        fold_auc.append(a)
        iters.append(best)
        drops.append(d)
        del model
    positive_folds = {g: int(sum(d[g] > 0 for d in drops)) for g in groups}
    cand_groups = [g for g in groups if positive_folds[g] >= 2]
    candidates = [c for g in cand_groups for c in groups[g]]
    return {'features_used': feats, 'groups': groups, 'inner_auc': [round(a, 4) for a in fold_auc],
            'best_iteration': iters,
            'mean_auc_drop_by_fold': {g: [round(d[g], 5) for d in drops] for g in groups},
            'positive_folds': positive_folds, 'candidate_groups': cand_groups, 'candidates': candidates}


def shortlist_from(linear: list[str], nonlinear: list[str], available: list[str]) -> dict:
    """Union + the three interactions + their components, in dictionary order; restricted to available columns."""
    wanted = set(linear) | set(nonlinear) | set(INTERACTIONS) | set(COMPONENTS)
    sl = [c for c in FEATURES if c in wanted and c in available]
    return {'shortlist': sl, 'from_linear_only': sorted(set(linear) - set(nonlinear)),
            'from_nonlinear_only': sorted(set(nonlinear) - set(linear)), 'in_both': sorted(set(linear) & set(nonlinear)),
            'added_as_interaction_or_component': sorted(wanted - set(linear) - set(nonlinear))}


# ----------------------------------------------------------------------------
# Stage D: grouped backward wrapper
# ----------------------------------------------------------------------------
class Wrapper:
    """Scores a feature subset by mean inner AUC (three inner folds, each early-stopped on its own validation)."""

    def __init__(self, ds: Dataset, fold: OuterFold, ledger: Ledger, params: dict = LGB_FIXED,
                 cap: int = SELECTOR_CAP, patience: int = PATIENCE, fixed_iterations: int | None = None):
        self.ds, self.fold, self.ledger, self.params, self.cap, self.patience = ds, fold, ledger, params, cap, patience
        self.fixed_iterations = fixed_iterations          # run B: 150 (no early stopping); None = run A
        self.cache: dict[frozenset, dict] = {}
        self.n_cache_hits = 0

    def score(self, subset: list[str], stage: str = 'D') -> dict:
        key = frozenset(subset)
        if key in self.cache:
            self.n_cache_hits += 1
            return self.cache[key]
        feats = [c for c in FEATURES if c in key]
        folds, iters = [], []
        for j, (tr, va) in enumerate(self.fold.inner, start=1):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                if self.fixed_iterations is None:
                    model, best, a, _ = lgb_fit_es(self.ds.df.iloc[tr][feats], self.ds.y[tr],
                                                   self.ds.df.iloc[va][feats], self.ds.y[va],
                                                   self.params, self.cap, self.patience)
                else:
                    model, best, a, _ = lgb_fit_fixed(self.ds.df.iloc[tr][feats], self.ds.y[tr],
                                                      self.ds.df.iloc[va][feats], self.ds.y[va],
                                                      self.params, self.fixed_iterations)
            self.ledger.warn(f'{stage} inner{j}', caught)
            self.ledger.fit(stage)
            folds.append(a)
            iters.append(best)
            del model
        res = {'features': feats, 'size': len(feats), 'mean_auc': float(numpy.mean(folds)),
               'fold_auc': [round(a, 5) for a in folds], 'spread': round(float(max(folds) - min(folds)), 5),
               'sd': round(float(numpy.std(folds, ddof=1)), 5), 'best_iteration': iters}
        self.cache[key] = res
        return res


def stage_d(wrapper: Wrapper, shortlist: list[str], group_of: dict[str, str], budget: int = WRAPPER_BUDGET,
            tol: float = STOP_TOLERANCE) -> dict:
    ledger = wrapper.ledger
    used0 = ledger.fits.get('D', 0)

    def used() -> int:
        return ledger.fits.get('D', 0) - used0

    evaluated: list[dict] = []
    start = wrapper.score(shortlist)
    evaluated.append(start)
    current = list(start['features'])
    path = [{'step': 0, 'removed': None, 'size': len(current), 'mean_auc': round(start['mean_auc'], 5),
             'fold_auc': start['fold_auc'], 'best_iteration': start['best_iteration']}]
    stopped_by = None
    step = 0
    while len(current) > 1:
        cands: list[tuple[str, list[str]]] = [(f, [c for c in current if c != f]) for f in current]
        groups: dict[str, list[str]] = {}
        for c in current:
            groups.setdefault(group_of[c], []).append(c)
        for g, members in groups.items():
            if len(members) >= 2:
                cands.append((f'group {g} ({"+".join(members)})', [c for c in current if c not in members]))
        results = []
        for name, sub in cands:
            cached = frozenset(sub) in wrapper.cache
            if not cached and used() + len(wrapper.fold.inner) > budget:
                stopped_by = f'budget ({budget} fits) exhausted during step {step + 1}'
                break
            r = wrapper.score(sub)
            if not cached:
                evaluated.append(r)
            results.append((name, r))
        if stopped_by:
            break
        if not results:
            stopped_by = 'no candidates'
            break
        name, best = max(results, key=lambda t: (t[1]['mean_auc'], -t[1]['size']))
        step += 1
        current = list(best['features'])
        path.append({'step': step, 'removed': name, 'size': best['size'], 'mean_auc': round(best['mean_auc'], 5),
                     'fold_auc': best['fold_auc'], 'spread': best['spread'], 'best_iteration': best['best_iteration'],
                     'n_candidates': len(results)})
    if stopped_by is None:
        stopped_by = 'single feature reached'
    best_eval = max(evaluated, key=lambda r: r['mean_auc'])
    within = [r for r in evaluated if r['mean_auc'] >= best_eval['mean_auc'] - tol]
    selected = min(within, key=lambda r: (r['size'], -r['mean_auc']))
    return {'shortlist': shortlist, 'path': path, 'n_evaluated_subsets': len(evaluated), 'fits_used': used(),
            'cache_hits': wrapper.n_cache_hits, 'budget': budget, 'stopped_by': stopped_by,
            'best': best_eval, 'selected': selected, 'tolerance': tol,
            'n_subsets_within_tolerance': len(within),
            'evaluated': sorted(evaluated, key=lambda r: (-r['mean_auc'], r['size']))}


def interaction_comparisons(wrapper: Wrapper, selected: list[str], available: list[str]) -> dict:
    """Clarification 2: each interaction with versus without, components forced into the selected background."""
    out = {}
    fits_before = wrapper.ledger.fits.get('interactions', 0)
    for inter, comps in INTERACTIONS.items():
        if inter not in available:
            out[inter] = {'skipped': 'not available'}
            continue
        background = [c for c in FEATURES if (c in selected and c != inter) or c in comps]
        without = wrapper.score(background, stage='interactions')
        with_ = wrapper.score(background + [inter], stage='interactions')
        diffs = [w - o for w, o in zip(with_['fold_auc'], without['fold_auc'])]
        out[inter] = {'components': comps, 'background': background,
                      'interaction_in_selected': inter in selected,
                      'components_added_to_selected': [c for c in comps if c not in selected],
                      'without': {'mean_auc': round(without['mean_auc'], 5), 'fold_auc': without['fold_auc'],
                                  'best_iteration': without['best_iteration']},
                      'with': {'mean_auc': round(with_['mean_auc'], 5), 'fold_auc': with_['fold_auc'],
                               'best_iteration': with_['best_iteration']},
                      'mean_difference_with_minus_without': round(with_['mean_auc'] - without['mean_auc'], 5),
                      'fold_differences': [round(d, 5) for d in diffs],
                      'folds_favouring_interaction': int(sum(d > 0 for d in diffs))}
    out['_fits'] = wrapper.ledger.fits.get('interactions', 0) - fits_before
    out['_evaluations'] = 2 * len([k for k in out if not k.startswith('_') and 'skipped' not in out[k]]) * len(wrapper.fold.inner)
    return out


# ----------------------------------------------------------------------------
# Driver: stages A to E for one outer fold
# ----------------------------------------------------------------------------
def run_selection(ds: Dataset, fold: OuterFold, ledger: Ledger | None = None,
                  settings: SelectionSettings = RUN_A) -> dict:
    """Stages A-E for one outer fold. The default `settings` (RUN_A) reproduce run A exactly; run B passes RUN_B
    (fixed 150-tree selector in C-LGB, D and the interaction comparisons; 2,000-fit wrapper budget)."""
    ledger = ledger or Ledger()
    rep: dict = {'outer_fold': fold.k, 'quarter': fold.quarter, 'valid_start': str(fold.valid_start.date()),
                 'sizes': fold.sizes(), 'inner_quarters': fold.inner_quarters}
    if settings != RUN_A:
        rep['settings'] = settings.as_dict()
    with timed(ledger, 'A'):
        A = stage_a(ds, fold)
    rep['stage_a'] = A
    with timed(ledger, 'B'):
        B = stage_b(ds, fold, A['inputs'], ledger)
    rep['stage_b'] = B
    with timed(ledger, 'C_linear'):
        CL = stage_c_linear(ds, fold, B['survivors'], ledger)
    with timed(ledger, 'C_lgb'):
        CN = stage_c_nonlinear(ds, fold, B['survivors'], A['group_of'], ledger,
                               fixed_iterations=settings.fixed_iterations, params=settings.lgb_params)
    SL = shortlist_from(CL['candidates'], CN['candidates'], ds.features)
    rep['stage_c'] = {'linear': CL, 'nonlinear': CN, **SL}
    wrapper = Wrapper(ds, fold, ledger, params=settings.lgb_params, fixed_iterations=settings.fixed_iterations)
    with timed(ledger, 'D'):
        D = stage_d(wrapper, SL['shortlist'], A['group_of'], budget=settings.wrapper_budget, tol=settings.tolerance)
    rep['stage_d'] = D
    if settings.interactions:
        with timed(ledger, 'interactions'):
            I = interaction_comparisons(wrapper, D['selected']['features'], ds.features)
    else:
        I = {'_skipped': 'interaction comparisons not run under these settings', '_fits': 0, '_evaluations': 0}
    rep['interactions'] = I
    sel = D['selected']['features']
    rep['stage_e'] = {
        'selected': sel, 'selected_size': len(sel),
        'selected_families': sorted({FAMILY[c] for c in sel}),
        'family_counts': {f: int(sum(FAMILY[c] == f for c in sel)) for f in sorted(set(FAMILY.values()))},
        'cluster_of_selected': {c: A['group_of'][c] for c in sel},
        'clusters': A['clusters'], 'representatives': A['representatives'],
        'stage_a_inputs': A['inputs'], 'stage_b_survivors': B['survivors'],
        'linear_candidates': CL['candidates'], 'nonlinear_candidates': CN['candidates'],
        'shortlist': SL['shortlist'], 'path_removed': [p['removed'] for p in D['path'] if p['step'] > 0],
        'best_mean_auc': round(D['best']['mean_auc'], 5), 'selected_mean_auc': round(D['selected']['mean_auc'], 5),
        'selected_fold_auc': D['selected']['fold_auc'], 'selected_spread': D['selected']['spread'],
        'interaction_mean_differences': {k: v['mean_difference_with_minus_without'] for k, v in I.items()
                                         if not k.startswith('_') and 'skipped' not in v},
        # Run B item 4: the shortlist's own wrapper score (step 0 of the path) beside the selected subset's
        'shortlist_size': len(SL['shortlist']), 'shortlist_mean_auc': round(D['path'][0]['mean_auc'], 5),
        'shortlist_fold_auc': D['path'][0]['fold_auc'],
        'budget_bound': 'budget' in D['stopped_by'],
        'latest_inner_iteration_selected': D['selected']['best_iteration'][-1],
    }
    rep['ledger'] = ledger.summary()
    return rep
