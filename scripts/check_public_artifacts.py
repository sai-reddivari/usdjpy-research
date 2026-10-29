"""Read-only checks of public evidence, generated tables and local document links.

This is not the historical private-data/model equivalence suite.
Only Python's standard library is required. No scientific modules are imported.
"""
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def rows(name):
    with (ROOT/"results"/name).open() as stream:
        return list(csv.DictReader(stream))


def main():
    evidence = json.loads((ROOT/"provenance/evidence_manifest.json").read_text())
    hashes = {**evidence["code_unchanged_from_verified_refactor"], **evidence["saved_evidence_sha256"]}
    for name, expected in hashes.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    syntax = 0
    for p in ROOT.rglob("*.py"):
        ast.parse(p.read_text()); syntax += 1
    features = json.loads((ROOT/"results/feature_dictionary.json").read_text())
    assert len(features) == 39 and sum(f["selected"] for f in features) == 34
    frozen = json.loads((ROOT/"results/frozen_configuration_summary.json").read_text())
    assert set(frozen["shortlist"]) == {f["name"] for f in features if f["selected"]}
    assert {f["name"] for f in features if not f["selected"]} == {
        "ret_1d", "ret_20d", "vol_5_over_20", "dist_ma20", "us_slope_chg_5d"}
    generated = ROOT/"latex/generated"
    dictionary = (generated/"dictionary.tex").read_text()
    for f in features:
        assert r"\path{"+f["name"]+"}" in dictionary
    runs = {}
    checked = 0
    for name, key, quarters in [
        ("run_a_quarterly.csv", "A", range(9,17)),
        ("run_b_quarterly.csv", "B", range(9,17)),
        ("no_wrapper_quarterly.csv", "C", range(9,17)),
        ("reserved_quarterly.csv", "reserved", range(1,5))]:
        runs[key] = {r["model"]:r for r in rows(name)}
        for r in runs[key].values():
            qs = [float(r[("R" if key == "reserved" else "Q")+str(i)]) for i in quarters]
            assert abs(statistics.mean(qs)-float(r["mean_quarterly_auc"])) < 1e-12
            assert abs(statistics.stdev(qs)-float(r["quarterly_sd"])) < 1e-12
            checked += 2
    text = (generated/"reserved_table.tex").read_text()
    for model, r in runs["reserved"].items():
        if model == "majority": continue
        values = " & ".join(f'{float(r[c]):.3f}' for c in
                            ["R1","R2","R3","R4","mean_quarterly_auc","pooled_auc"])
        assert values in text, model
        checked += 6
    dev = (generated/"development_comparison.tex").read_text()
    for m, name in [("lightgbm","LightGBM"),("xgboost","XGBoost"),("catboost","CatBoost")]:
        a,b,c = runs["A"][m],runs["B"][m],runs["C"]["nw_"+m]
        wins = sum(float(c[f"Q{i}"])>float(b[f"Q{i}"]) for i in range(9,17))
        expected = name+" & "+" & ".join(f'{float(r["mean_quarterly_auc"]):.3f}' for r in [a,b,c])
        expected += f' & {float(c["mean_quarterly_auc"])-float(b["mean_quarterly_auc"]):+.3f} & {wins}/8'
        assert expected in dev; checked += 5
    classes = {(r["model"],r["quarter"]):r for r in rows("reserved_classification.csv")}
    main = (ROOT/"latex/main.tex").read_text()
    for m,name in [("xgboost","XGBoost"),("catboost","CatBoost"),("lightgbm","LightGBM")]:
        r = classes["nw_"+m,"pooled"]
        assert int(r["n"]) == 229 and int(r["n_positive"]) == 64
        row = f'{name} & {r["tp"]} & {r["fp"]} & {float(r["precision_1"]):.3f} & {float(r["recall_1"]):.3f} & {float(r["mean_predicted"]):.3f}'
        assert row in main, row; checked += 5
    for r in rows("posthoc_auc_summary.csv"):
        if r["cohort"] not in ["original229","companion258"]: continue
        assert int(r["n"]) == (229 if r["cohort"] == "original229" else 258)
        assert abs(statistics.mean(float(r[f"R{i}"]) for i in range(1,5))-float(r["mean_quarterly_auc"])) < 1e-12
    links = 0
    for p in ROOT.glob("*.md"):
        for url in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)",p.read_text()):
            u = urlsplit(url)
            if u.scheme or u.netloc: continue
            assert (p.parent/unquote(u.path)).exists(),(p.name,url)
            links += 1
    for name in ["usdjpy_research.pdf","usdjpy_technical_appendix.pdf"]:
        assert (ROOT/"reports"/name).read_bytes().startswith(b"%PDF-"),name
    bad = []
    for p in ROOT.rglob("*"):
        if not p.is_file(): continue
        if p.suffix in [".joblib",".pkl",".cbm",".parquet",".pyc"] or p.name in [".DS_Store",".env"]:
            bad.append(str(p.relative_to(ROOT)))
        if any(x in p.relative_to(ROOT).parts for x in ["raw","private","local_only","__pycache__"]):
            bad.append(str(p.relative_to(ROOT)))
    assert not bad,bad
    result = {"status":"passed","scope":"Public snapshot only; not a research rerun",
              "evidence_and_code_hashes":len(hashes),"python_syntax_files":syntax,
              "candidate_features":39,"frozen_features":34,"numeric_checks":checked,
              "readme_local_links":links,"pdf_files_present":2,
              "private_data_models_credentials":"Not bundled; this check rejects known artifact types"}
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
