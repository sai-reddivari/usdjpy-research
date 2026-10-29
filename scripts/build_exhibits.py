"""Render saved aggregate evidence. No raw data, network, training or scoring."""

import csv
import html
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("lightgbm", "xgboost", "catboost")
NAMES = {"lightgbm": "LightGBM", "xgboost": "XGBoost", "catboost": "CatBoost"}


def rows(name):
    with (ROOT / "results" / name).open() as stream:
        return list(csv.DictReader(stream))


def escape(value):
    value = str(value)
    return "".join({"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#",
                    "$": r"\$", "{": r"\{", "}": r"\}"}.get(c, c) for c in value)


def table(headers, data, spec):
    out = [r"\begin{center}\small", r"\begin{tabular}{@{}" + spec + r"@{}}\toprule",
           " & ".join(headers) + r"\\\midrule"]
    out += [" & ".join(row) + r"\\" for row in data]
    return "\n".join(out + [r"\bottomrule\end{tabular}\end{center}"]) + "\n"


def formula(value):
    value = value.replace("<sub>", "@SUB@").replace("</sub>", "@END@")
    value = escape(html.unescape(value))
    value = value.replace("@SUB@", r"\textsubscript{").replace("@END@", "}")
    for a, b in {"−": "-", "…": r"\ldots{}", "σ": r"\ensuremath{\sigma}",
                 "ℓ": r"\ensuremath{\ell}", "×": r"\ensuremath{\times}",
                 "𝟙": r"\ensuremath{\mathbf1}", "≥": r"\ensuremath{\ge}"}.items():
        value = value.replace(a, b)
    return value


def build():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#183B48", "text.color": "#183B48",
                         "savefig.bbox": "tight", "pdf.fonttype": 42})
    output = ROOT / "latex" / "generated"
    output.mkdir(parents=True, exist_ok=True)
    figs = ROOT / "figures"
    figs.mkdir(exist_ok=True)
    runs = {k: {r["model"]: r for r in rows(name)} for k, name in
            {"A": "run_a_quarterly.csv", "B": "run_b_quarterly.csv",
             "C": "no_wrapper_quarterly.csv"}.items()}
    reserved = {r["model"]: r for r in rows("reserved_quarterly.csv")}
    posthoc = {(r["cohort"], r["model"]): r for r in rows("posthoc_auc_summary.csv")}
    fmt = lambda value: f"{float(value):.3f}"
    data = []
    for model in MODELS:
        a, b, c = [runs[k]["nw_"+model if k == "C" else model] for k in "ABC"]
        wins = sum(float(c[f"Q{q}"]) > float(b[f"Q{q}"]) for q in range(9, 17))
        data.append([NAMES[model], fmt(a["mean_quarterly_auc"]), fmt(b["mean_quarterly_auc"]),
                     fmt(c["mean_quarterly_auc"]),
                     f'{float(c["mean_quarterly_auc"])-float(b["mean_quarterly_auc"]):+.3f}',
                     f"{wins}/8", fmt(runs["B"][model+"_own_market"]["mean_quarterly_auc"])])
    (output / "development_comparison.tex").write_text(table(
        ["Model", "A", "B", "C", r"$C-B$", "C wins", "Own price"], data, "lrrrrrr"))
    order = ["nw_xgboost", "nw_catboost", "nw_lightgbm", "logistic_own_market",
             "xgboost_own_market", "catboost_own_market", "lightgbm_own_market"]
    labels = {"nw_xgboost": "XGBoost: shortlist", "nw_catboost": "CatBoost: shortlist",
              "nw_lightgbm": "LightGBM: shortlist", "logistic_own_market": "Logistic: own price",
              "xgboost_own_market": "XGBoost: own price", "catboost_own_market": "CatBoost: own price",
              "lightgbm_own_market": "LightGBM: own price", "majority": "Constant prevalence"}
    columns = ["R1", "R2", "R3", "R4", "mean_quarterly_auc", "pooled_auc"]
    (output / "reserved_table.tex").write_text(table(
        ["Procedure", "R1", "R2", "R3", "R4", "Mean", "Pooled"],
        [[labels[m]]+[fmt(reserved[m][c]) for c in columns] for m in order], "lrrrrrr"))
    data = []
    for m in ["xgboost", "catboost", "lightgbm"]:
        row = [NAMES[m]]
        for cohort in ["original229", "companion258"]:
            row += [fmt(posthoc[cohort, key]["mean_quarterly_auc"])
                    for key in ["nw_"+m, m+"_own_market"]]
        data.append(row)
    (output / "coverage_table.tex").write_text(table(
        ["Model", "229: expanded", "229: own", "258: expanded", "258: own"], data, "lrrrr"))

    frozen = json.loads((ROOT/"results/frozen_configuration_summary.json").read_text())
    parts = [r"\begingroup\small\begin{longtable}{@{}p{.22\linewidth}p{.41\linewidth}rrr@{}}\toprule",
             r"Procedure & Configuration & Trees & Inner AUC & $\tau$\\\midrule\endhead"]
    for model, rec in frozen["models"].items():
        cfg = rec["chosen"]["config"]
        if rec["booster"] == "lightgbm":
            desc = f'leaves {cfg["num_leaves"]}; rows {cfg["min_child_samples"]}; rate {cfg["learning_rate"]}; gain {cfg["min_split_gain"]}'
        elif rec["booster"] == "xgboost":
            desc = f'depth {cfg["max_depth"]}; weight {cfg["min_child_weight"]}; rate {cfg["learning_rate"]}; gamma {cfg["gamma"]}'
        else:
            desc = f'depth {cfg["depth"]}; L2 {cfg["l2_leaf_reg"]}; rate {cfg["learning_rate"]}; strength {cfg["random_strength"]}'
        parts += [labels[model]+" & "+desc+" & "+str(rec["refit_iterations"])+" & "+
                  fmt(rec["chosen"]["mean_inner_auc"])+" & "+fmt(rec["threshold_choice"]["threshold"])+r"\\"]
    parts += [r"\bottomrule\end{longtable}\endgroup"]
    (output/"frozen_settings.tex").write_text("\n".join(parts)+"\n")

    fig, ax = plt.subplots(figsize=(7.1, 2.7))
    colors = ["#247B82", "#AD6B32", "#715D91"]
    for m, color in zip(MODELS, colors):
        y = [float(runs[k]["nw_"+m if k == "C" else m]["mean_quarterly_auc"]) for k in "ABC"]
        ax.plot([0, 1], y[:2], "--", color=color, lw=1.5)
        ax.plot([1, 2], y[1:], "-", color=color, lw=2.2, label=NAMES[m])
        ax.scatter([0, 1, 2], y, color=color, s=35, zorder=4)
    baseline = float(runs["A"]["logistic_own_market"]["mean_quarterly_auc"])
    ax.axhline(baseline, color="#394A56", lw=1, ls=":", label="Own-price logistic")
    ax.axhline(.5, color="#CED5D8", lw=1)
    ax.set(xticks=[0, 1, 2], xticklabels=["A: constrained search", "B: expanded search", "C: no wrapper"],
           ylim=(.47, .565), ylabel="Mean quarterly ROC-AUC")
    ax.legend(loc="lower left", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(figs/"selection_comparison.pdf")
    fig.savefig(figs/"selection_comparison.png", dpi=180)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(7.15, 2.25), sharey=True)
    for ax, m in zip(axs, ["xgboost", "catboost", "lightgbm"]):
        for key, label, color, marker in [("nw_"+m, "Shortlist", "#247B82", "o"),
                                         (m+"_own_market", "Own price", "#AD6B32", "s")]:
            ax.plot(range(1, 5), [float(reserved[key][f"R{i}"]) for i in range(1, 5)],
                    color=color, marker=marker, ms=4, lw=1.5, label=label)
        ax.axhline(.5, color="#ADB9BE", lw=1, ls="--")
        ax.set(title=NAMES[m], xticks=range(1, 5), xticklabels=["R1", "R2", "R3", "R4"],
               ylim=(.33, .71))
    axs[0].set_ylabel("ROC-AUC")
    axs[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(figs/"reserved_pairs.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.15, 1.75))
    ax.axvspan(17, 41, color="#E3EFF0")
    ax.plot([0, 41], [0, 0], color="#9DAEB4", lw=1)
    marks = [(1.58, "Tokyo sectors\n01:35 EST", .8),
             (3.08, "BOJ reference\n03:05", -.8),
             (16.083, "US equity close\n16:05", .8),
             (17, "Decision / FX anchor\n17:00 (assumed)", -.8),
             (41, "Next close / outcome\n17:00 Tuesday", .8)]
    for x, label, y in marks:
        ax.plot([x, x], [0, y*.5], color="#247B82", lw=1)
        ax.scatter([x], [0], s=27, color="#247B82", zorder=3)
        ax.text(x, y*.54, label, ha="center", va="bottom" if y>0 else "top", fontsize=8)
    ax.text(29, -.08, "Future return window", ha="center", va="top", fontsize=9)
    ax.set(xlim=(-2, 45), ylim=(-.95, 1.05))
    ax.axis("off");fig.tight_layout()
    fig.savefig(figs/"information_clock.pdf")
    plt.close(fig)

    # Full dictionary: definitions only, no feature observations.
    features = json.loads((ROOT/"results/feature_dictionary.json").read_text())
    parts = []
    for family in dict.fromkeys(f["family"] for f in features):
        parts += [r"\subsection*{"+escape(family)+"}",
                  r"\begingroup\small\begin{longtable}{@{}p{.28\linewidth}p{.66\linewidth}@{}}\toprule",
                  r"Feature / final status & Definition and hypothesis\\\midrule\endhead"]
        for f in features:
            if f["family"] != family: continue
            name = r"\path{"+f["name"]+"}"+r"\newline "+("Retained" if f["selected"] else "Removed by MI")
            value = formula(f["formula_html"])+r"\par "+formula(f["intuition"])+r"\par \emph{Calendar:} "+formula(f["calendar"])
            parts += [name+" & "+value+r"\\[5pt]"]
        parts += [r"\bottomrule\end{longtable}\endgroup"]
    (output/"dictionary.tex").write_text("\n".join(parts)+"\n")

    # Full quarter tables preserve original labels and numbers, without re-scoring.
    parts = []
    for run in "ABC":
        parts += [r"\subsection*{Development "+run+"}",
                  r"\captiontext{Saved eight-quarter results. SD is sample dispersion across quarters, not a confidence interval.}",
                  r"\begingroup\scriptsize\setlength{\tabcolsep}{2.5pt}",
                  r"\begin{longtable}{@{}p{.26\linewidth}rrrrrrrrrrr@{}}\toprule",
                  r"Procedure & Q9 & Q10 & Q11 & Q12 & Q13 & Q14 & Q15 & Q16 & Mean & SD & Pool\\\midrule\endhead"]
        for r in runs[run].values():
            parts += [escape(r["label"])+" & "+" & ".join(fmt(r[c]) for c in
                        [f"Q{i}" for i in range(9, 17)]+["mean_quarterly_auc","quarterly_sd","pooled_auc"])+r"\\"]
        parts += [r"\bottomrule\end{longtable}\endgroup"]
    (output/"quarterly.tex").write_text("\n".join(parts)+"\n")
    print("Rendered saved-result figures and tables; no models or raw data loaded.")


if __name__ == "__main__":
    build()
