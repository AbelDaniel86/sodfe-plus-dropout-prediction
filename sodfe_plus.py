#!/usr/bin/env python3
"""
SOD-FE+ : student dropout prediction (UCI dataset 697, Portugal)
=================================================================
End-to-end pipeline: clean -> EDA -> feature engineering -> models -> evaluation
-> fairness audit -> explainability -> written report.

Usage
-----
    python sodfe_plus.py --data data.csv --out outputs            # full run
    python sodfe_plus.py --data data.csv --out outputs --quick    # fast smoke test

Design rules that keep the numbers honest
-----------------------------------------
* Every data-dependent step (scaling, encoding, feature ranking, outlier
  filtering, SMOTE, SHAP feature selection, stacking) is fitted on the TRAINING
  part of each CV fold only. The test fold is never touched, never resampled.
* SMOTE / outlier removal change only the training data; metrics are always on
  real, untouched students.
* Dropout = positive class (1). Recall of this class is the headline metric for
  an early-warning system.
"""
import argparse, os, time, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, IsolationForest, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                             roc_auc_score, balanced_accuracy_score, confusion_matrix,
                             roc_curve, ConfusionMatrixDisplay)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
SEED = 0

COURSES = {33: "Biofuel Tech", 171: "Animation/Multimedia", 8014: "Social Service (eve)",
           9003: "Agronomy", 9070: "Communication Design", 9085: "Veterinary Nursing",
           9119: "Informatics Eng.", 9130: "Equinculture", 9147: "Management",
           9238: "Social Service", 9254: "Tourism", 9500: "Nursing", 9556: "Oral Hygiene",
           9670: "Advertising/Mktg", 9773: "Journalism", 9853: "Basic Education",
           9991: "Management (eve)"}


# =====================================================================
# 1. LOAD + CLEAN
# =====================================================================
def load_and_clean(path):
    """Load the semicolon-separated UCI file and fix its known quirks."""
    raw = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    log = {"raw_shape": raw.shape}

    # --- column hygiene: BOM, stray tab in 'Daytime/evening attendance\t', typo 'Nacionality'
    raw.columns = [c.replace("\t", "").replace("\ufeff", "").strip() for c in raw.columns]
    raw = raw.rename(columns={"Nacionality": "Nationality"})

    df = raw.copy()
    log["missing_cells"] = int(df.isna().sum().sum())          # this file has none
    log["duplicate_rows"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    # --- target: drop 'Enrolled' (still studying -> outcome unknown), Dropout = 1
    log["target_counts_raw"] = df["Target"].value_counts().to_dict()
    df = df[df["Target"] != "Enrolled"].reset_index(drop=True)
    df["Target"] = (df["Target"] == "Dropout").astype(int)

    # --- logical consistency checks (flagged and reported, not silently "fixed")
    checks = {}
    for s in ("1st", "2nd"):
        enr, app = f"Curricular units {s} sem (enrolled)", f"Curricular units {s} sem (approved)"
        ev, g = f"Curricular units {s} sem (evaluations)", f"Curricular units {s} sem (grade)"
        checks[f"{s}: approved > enrolled"] = int((df[app] > df[enr]).sum())
        checks[f"{s}: approved>0 but grade==0"] = int(((df[app] > 0) & (df[g] == 0)).sum())
        checks[f"{s}: enrolled==0 (no activity)"] = int((df[enr] == 0).sum())
    checks["age < 17"] = int((df["Age at enrollment"] < 17).sum())
    checks["admission grade outside 0-200"] = int((~df["Admission grade"].between(0, 200)).sum())
    log["consistency_checks"] = checks
    log["clean_shape"] = df.shape

    # --- types: binary / nominal / numeric
    log["course_names"] = True
    df["Course_name"] = df["Course"].map(COURSES).fillna(df["Course"].astype(str))
    return df, log


BINARY = ["Daytime/evening attendance", "Displaced", "Educational special needs", "Debtor",
          "Tuition fees up to date", "Gender", "Scholarship holder", "International"]
NOMINAL = ["Marital status", "Application mode", "Course", "Previous qualification",
           "Nationality", "Mother's qualification", "Father's qualification",
           "Mother's occupation", "Father's occupation"]
CONTINUOUS_RAW = ["Previous qualification (grade)", "Admission grade", "Age at enrollment",
                  "Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)",
                  "Unemployment rate", "Inflation rate", "GDP"]


# =====================================================================
# 2. EDA
# =====================================================================
def run_eda(df, fig_dir, log):
    d = df.copy()
    d["Outcome"] = d["Target"].map({0: "Graduate", 1: "Dropout"})
    pal = {"Graduate": "#2a9d8f", "Dropout": "#e76f51"}

    # 2.1 class balance + dropout rate by course
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 2]})
    cnt = d["Outcome"].value_counts()
    ax[0].bar(cnt.index, cnt.values, color=[pal[i] for i in cnt.index])
    for i, v in enumerate(cnt.values):
        ax[0].text(i, v, f"{v}\n({v/len(d):.1%})", ha="center", va="bottom")
    ax[0].set_title("Class balance (Enrolled removed)"); ax[0].set_ylim(0, cnt.max() * 1.18)
    cr = d.groupby("Course_name")["Target"].agg(["mean", "size"]).sort_values("mean")
    ax[1].barh(cr.index, cr["mean"], color="#e76f51")
    for y, (m, n) in enumerate(zip(cr["mean"], cr["size"])):
        ax[1].text(m + .005, y, f"{m:.0%} (n={n})", va="center", fontsize=8)
    ax[1].axvline(d["Target"].mean(), color="k", ls="--", lw=1)
    ax[1].set_title("Dropout rate by course (dashed = overall)"); ax[1].set_xlim(0, cr["mean"].max() * 1.2)
    plt.tight_layout(); plt.savefig(f"{fig_dir}/eda_01_balance_course.png", dpi=150); plt.close()

    # 2.2 numeric distributions by outcome
    cols = ["Admission grade", "Age at enrollment", "Curricular units 1st sem (approved)",
            "Curricular units 1st sem (grade)", "Curricular units 2nd sem (approved)",
            "Curricular units 2nd sem (grade)"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for a, c in zip(axes.ravel(), cols):
        sns.kdeplot(data=d, x=c, hue="Outcome", palette=pal, fill=True, common_norm=False, ax=a, warn_singular=False)
        a.set_title(c.replace("Curricular units ", "CU "), fontsize=10)
    plt.suptitle("Key numeric features by outcome", y=1.0); plt.tight_layout()
    plt.savefig(f"{fig_dir}/eda_02_numeric_dists.png", dpi=150); plt.close()

    # 2.3 dropout rate by binary / socio-economic flags
    flags = ["Gender", "Scholarship holder", "Debtor", "Tuition fees up to date", "Displaced",
             "International", "Daytime/evening attendance", "Educational special needs"]
    rows = []
    for f in flags:
        for v, g in d.groupby(f):
            rows.append((f, str(v), g["Target"].mean(), len(g)))
    fr = pd.DataFrame(rows, columns=["feature", "value", "dropout_rate", "n"])
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=fr, x="feature", y="dropout_rate", hue="value", ax=ax, palette="Set2")
    ax.axhline(d["Target"].mean(), color="k", ls="--", lw=1)
    ax.set_title("Dropout rate across binary indicators (0/1 values)"); plt.xticks(rotation=30, ha="right")
    plt.tight_layout(); plt.savefig(f"{fig_dir}/eda_03_binary_dropout.png", dpi=150); plt.close()
    fr.to_csv(f"{fig_dir}/../tables/eda_binary_dropout_rates.csv", index=False)

    # 2.4 correlation with target (point-biserial) -- top 15
    num = df.drop(columns=["Course_name"]).select_dtypes(include=[np.number])
    corr = num.corr()["Target"].drop("Target").sort_values()
    top = pd.concat([corr.head(8), corr.tail(8)])
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.index, top.values, color=np.where(top.values > 0, "#e76f51", "#2a9d8f"))
    ax.set_title("Strongest linear correlations with dropout"); plt.tight_layout()
    plt.savefig(f"{fig_dir}/eda_04_target_corr.png", dpi=150); plt.close()
    # NOTE: nominal codes (Course, Mother's occupation...) have no numeric meaning; correlation is
    # only indicative for numeric/binary columns.

    # 2.5 curricular-units correlation heatmap (multicollinearity)
    cu = [c for c in df.columns if c.startswith("Curricular")]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[cu].corr(), cmap="vlag", center=0, annot=False, ax=ax)
    ax.set_title("Correlation among curricular variables (strong multicollinearity)")
    ax.set_xticklabels([t.get_text().replace("Curricular units ", "") for t in ax.get_xticklabels()], rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels([t.get_text().replace("Curricular units ", "") for t in ax.get_yticklabels()], fontsize=8)
    plt.tight_layout(); plt.savefig(f"{fig_dir}/eda_05_curricular_corr.png", dpi=150); plt.close()

    # 2.6 univariate (IQR) vs multivariate (Isolation Forest) outlier view
    out_tbl = {}
    for c in CONTINUOUS_RAW:
        q1, q3 = df[c].quantile([.25, .75]); i = q3 - q1
        out_tbl[c] = int(((df[c] < q1 - 1.5 * i) | (df[c] > q3 + 1.5 * i)).sum())
    log["iqr_outliers_per_feature"] = out_tbl
    Z = StandardScaler().fit_transform(df[CONTINUOUS_RAW + [c for c in df.columns if c.startswith("Curricular")]])
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=SEED).fit(Z)
    flag_if = iso.predict(Z) == -1
    q = df[CONTINUOUS_RAW]
    iqr_any = ((q < q.quantile(.25) - 1.5 * (q.quantile(.75) - q.quantile(.25))) |
               (q > q.quantile(.75) + 1.5 * (q.quantile(.75) - q.quantile(.25)))).any(axis=1).values
    log["outlier_overlap"] = {"IQR any-feature": int(iqr_any.sum()), "IsolationForest 5%": int(flag_if.sum()),
                              "both": int((iqr_any & flag_if).sum()),
                              "IF only (multivariate)": int((~iqr_any & flag_if).sum())}
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["Curricular units 1st sem (grade)"], df["Curricular units 2nd sem (grade)"], s=8,
               c=np.where(flag_if, "#d62828", "#adb5bd"), alpha=.6)
    ax.set_xlabel("1st-sem grade"); ax.set_ylabel("2nd-sem grade")
    ax.set_title("Isolation-Forest outliers (red) live in joint space, not only in the tails")
    plt.tight_layout(); plt.savefig(f"{fig_dir}/eda_06_outliers.png", dpi=150); plt.close()


# =====================================================================
# 3. FEATURE ENGINEERING (row-wise, no fitting -> leakage-free)
# =====================================================================
def engineer(df):
    X = df.drop(columns=["Target", "Course_name"], errors="ignore").copy()
    sd = lambda a, b: np.where(b > 0, a / np.where(b == 0, 1, b), 0.0)
    for s, k in (("1st", "s1"), ("2nd", "s2")):
        enr = X[f"Curricular units {s} sem (enrolled)"]; app = X[f"Curricular units {s} sem (approved)"]
        ev = X[f"Curricular units {s} sem (evaluations)"]
        X[f"approval_rate_{k}"] = sd(app, enr)
        X[f"eval_pass_rate_{k}"] = sd(app, ev)
        X[f"fail_count_{k}"] = (enr - app).clip(lower=0)
        X[f"no_activity_{k}"] = (enr == 0).astype(int)
    X["total_approved"] = X["Curricular units 1st sem (approved)"] + X["Curricular units 2nd sem (approved)"]
    X["total_enrolled"] = X["Curricular units 1st sem (enrolled)"] + X["Curricular units 2nd sem (enrolled)"]
    X["total_credited"] = X["Curricular units 1st sem (credited)"] + X["Curricular units 2nd sem (credited)"]
    X["overall_approval_rate"] = sd(X["total_approved"], X["total_enrolled"])
    X["approval_rate_trend"] = X["approval_rate_s2"] - X["approval_rate_s1"]
    X["grade_trend"] = X["Curricular units 2nd sem (grade)"] - X["Curricular units 1st sem (grade)"]
    X["approved_trend"] = X["Curricular units 2nd sem (approved)"] - X["Curricular units 1st sem (approved)"]
    g = X[["Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)"]].replace(0, np.nan)
    X["mean_grade_nonzero"] = g.mean(axis=1).fillna(0)
    X["financial_risk"] = X["Debtor"] + (1 - X["Tuition fees up to date"])
    X["is_mature_student"] = (X["Age at enrollment"] >= 25).astype(int)
    X["admission_minus_prev_grade"] = X["Admission grade"] - X["Previous qualification (grade)"]
    X["application_order_high"] = (X["Application order"] >= 3).astype(int)
    return X


def make_preprocessor(X):
    binary = [c for c in X.columns if X[c].nunique() <= 2 and c not in NOMINAL]
    nominal = [c for c in NOMINAL if c in X.columns]
    numeric = [c for c in X.columns if c not in binary + nominal]
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("bin", "passthrough", binary),
        # rare levels (<2%) pooled -> controls cardinality of occupation/qualification/nationality
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.02, sparse_output=False), nominal),
    ], verbose_feature_names_out=False)
    pre.set_output(transform="pandas")
    return pre, numeric


# =====================================================================
# 4. SOD-FE+ BUILDING BLOCKS
# =====================================================================
def iqr_mask(Xdf, cols):
    """True = inlier under the univariate 1.5*IQR rule (any column outside => outlier)."""
    q1, q3 = Xdf[cols].quantile(.25), Xdf[cols].quantile(.75); i = q3 - q1
    return ~(((Xdf[cols] < q1 - 1.5 * i) | (Xdf[cols] > q3 + 1.5 * i)).any(axis=1)).values


def rank_top_beta(Xdf, y, beta=15):
    """SOD-FE style ranking: mean of rank(|Pearson|) and rank(MI) -> top-beta columns."""
    rho = np.array([abs(np.corrcoef(Xdf[c], y)[0, 1]) if Xdf[c].std() > 0 else 0 for c in Xdf.columns])
    mi = mutual_info_classif(Xdf, y, random_state=SEED)
    comb = (np.argsort(np.argsort(-rho)) + np.argsort(np.argsort(-mi))) / 2
    return list(Xdf.columns[np.argsort(comb)[:beta]])


def shap_rfe(Xdf, y, k=20, drop_frac=0.3, n_shap=600):
    """Recursive feature elimination driven by mean |SHAP| of an XGBoost model."""
    active = list(Xdf.columns)
    rng = np.random.RandomState(SEED)
    while len(active) > k:
        m = XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.1, subsample=.8,
                          colsample_bytree=.8, eval_metric="logloss", random_state=SEED, n_jobs=1)
        m.fit(Xdf[active], y)
        idx = rng.choice(len(Xdf), min(n_shap, len(Xdf)), replace=False)
        sv = shap.TreeExplainer(m).shap_values(Xdf[active].iloc[idx])
        imp = np.abs(sv).mean(0)
        n_keep = max(k, int(len(active) * (1 - drop_frac)))
        active = [active[i] for i in np.argsort(-imp)[:n_keep]]
    return active


def build_model(kind, n_jobs=1, fast=False):
    nt = 150 if fast else 300
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    if kind == "rf":
        return RandomForestClassifier(n_estimators=nt, min_samples_leaf=2, random_state=SEED, n_jobs=n_jobs)
    if kind == "xgb":
        return XGBClassifier(n_estimators=nt, max_depth=5, learning_rate=0.05, subsample=.8,
                             colsample_bytree=.8, gamma=0.5, eval_metric="logloss", random_state=SEED, n_jobs=n_jobs)
    if kind == "stack":
        return StackingClassifier(
            estimators=[("rf", build_model("rf", n_jobs, fast)), ("xgb", build_model("xgb", n_jobs, fast)),
                        ("mlp", make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-2,
                                                                              early_stopping=True, max_iter=300, random_state=SEED)))],
            final_estimator=LogisticRegression(max_iter=1000, C=1.0), cv=3, n_jobs=n_jobs, stack_method="predict_proba")
    raise ValueError(kind)


def run_pipeline(Xtr, ytr, Xte, cfg, fast=False):
    """Fit everything on (Xtr, ytr) and return P(dropout) for Xte + diagnostics.

    cfg keys
      model        : lr | rf | xgb | stack
      rank_beta    : None | int          (SOD-FE feature ranking, Pearson+MI)
      outlier      : none | iqr | iforest
      scope        : majority | all      (label-aware cleaning; majority = graduates only)
      order        : clean_first | smote_first
      add_score    : bool                (append Isolation-Forest anomaly score as a feature)
      balance      : none | smote
      shap_k       : None | int          (SHAP-RFE)
    """
    pre, numeric = make_preprocessor(Xtr)
    A = pre.fit_transform(Xtr); B = pre.transform(Xte)
    ytr = np.asarray(ytr); info = {}

    if cfg.get("rank_beta"):
        keep = rank_top_beta(A, ytr, cfg["rank_beta"]); A, B = A[keep], B[keep]

    cont_cols = [c for c in A.columns if c in numeric and A[c].nunique() > 10]
    iso = None
    if cfg["outlier"] == "iforest" or cfg.get("add_score"):
        iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=SEED, n_jobs=1).fit(A)

    def clean(Ad, yd):
        if cfg["outlier"] == "none":
            return Ad, yd
        inl = (iso.predict(Ad) == 1) if cfg["outlier"] == "iforest" else iqr_mask(Ad, [c for c in cont_cols if c in Ad.columns])
        keep = inl | (yd == 1) if cfg.get("scope", "majority") == "majority" else inl   # majority = graduates (0)
        info["removed"] = info.get("removed", 0) + int((~keep).sum())
        return Ad[keep].reset_index(drop=True), yd[keep]

    def smote(Ad, yd):
        return Ad, yd if cfg["balance"] == "none" else None

    def do_smote(Ad, yd):
        if cfg["balance"] != "smote":
            return Ad, yd
        Ar, yr = SMOTE(random_state=SEED).fit_resample(Ad, yd)
        return Ar, np.asarray(yr)

    if cfg["order"] == "clean_first":
        A, ytr = clean(A, ytr)
        if cfg.get("add_score"):
            A = A.assign(outlier_score=iso.decision_function(A)); B = B.assign(outlier_score=iso.decision_function(B))
        A, ytr = do_smote(A, ytr)
    else:   # smote_first (the SOD-FE baseline order: balance, then clean)
        A, ytr = do_smote(A, ytr)
        A, ytr = clean(A, ytr)
        if cfg.get("add_score"):
            A = A.assign(outlier_score=iso.decision_function(A)); B = B.assign(outlier_score=iso.decision_function(B))

    if cfg.get("shap_k"):
        sel = shap_rfe(A, ytr, k=cfg["shap_k"]); A, B = A[sel], B[sel]; info["selected"] = sel

    model = build_model(cfg["model"], fast=fast).fit(A, ytr)
    info["n_train"] = len(A)
    return model.predict_proba(B)[:, 1], info


# =====================================================================
# 5. EXPERIMENTS
# =====================================================================
EXPERIMENTS = {
    # name : cfg
    "M1 LR (raw)":                         dict(model="lr",    outlier="none",    balance="none",  order="clean_first"),
    "M2 RF (raw)":                         dict(model="rf",    outlier="none",    balance="none",  order="clean_first"),
    "M3 XGB (raw)":                        dict(model="xgb",   outlier="none",    balance="none",  order="clean_first"),
    "M4 RF + SMOTE":                       dict(model="rf",    outlier="none",    balance="smote", order="clean_first"),
    "M5 SOD-FE baseline (rank+SMOTE>IQR)": dict(model="rf",    outlier="iqr",     balance="smote", order="smote_first", rank_beta=15, scope="majority"),
    "M6 +IForest, clean>SMOTE":            dict(model="rf",    outlier="iforest", balance="smote", order="clean_first", scope="majority"),
    "M7 +outlier_score feature":           dict(model="rf",    outlier="iforest", balance="smote", order="clean_first", scope="majority", add_score=True),
    "M8 +SHAP-RFE (top-20)":               dict(model="rf",    outlier="iforest", balance="smote", order="clean_first", scope="majority", add_score=True, shap_k=20),
    "M9 SOD-FE+ (full, stacked)":          dict(model="stack", outlier="iforest", balance="smote", order="clean_first", scope="majority", add_score=True, shap_k=20),
    "M10 Stack only (no SOD+, no SHAP)":   dict(model="stack", outlier="none",    balance="smote", order="clean_first"),
}


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return dict(Accuracy=accuracy_score(y, pred), F1=f1_score(y, pred), Recall=recall_score(y, pred),
                Precision=precision_score(y, pred, zero_division=0), BalAcc=balanced_accuracy_score(y, pred),
                AUC=roc_auc_score(y, p))


def run_cv(X, y, n_splits, n_repeats, fast, out_dir):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    folds = list(cv.split(X, y))
    per_fold = []; oof = {m: np.zeros((n_repeats, len(y))) for m in EXPERIMENTS}; info_log = {}
    for name, cfg in EXPERIMENTS.items():
        t0 = time.time(); removed = []
        for fi, (tr, te) in enumerate(folds):
            p, info = run_pipeline(X.iloc[tr], y[tr], X.iloc[te], cfg, fast)
            oof[name][fi // n_splits, te] = p
            per_fold.append(dict(model=name, fold=fi, **metrics(y[te], p)))
            removed.append(info.get("removed", 0))
        info_log[name] = float(np.mean(removed))
        d = pd.DataFrame([r for r in per_fold if r["model"] == name])
        print(f"  {name:<40s} F1={d.F1.mean():.4f} Recall={d.Recall.mean():.4f} AUC={d.AUC.mean():.4f}  ({time.time()-t0:.0f}s)", flush=True)
    pf = pd.DataFrame(per_fold)
    pf.to_csv(f"{out_dir}/tables/cv_per_fold.csv", index=False)
    return pf, oof, info_log, folds


def corrected_ttest(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated k-fold."""
    d = np.asarray(a) - np.asarray(b); k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    return t, 2 * stats.t.sf(abs(t), k - 1)


# =====================================================================
# 6. FAIRNESS
# =====================================================================
def fairness_table(df_raw, y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    groups = {
        "Gender": df_raw["Gender"].map({0: "Female", 1: "Male"}),
        "Nationality": np.where(df_raw["Nationality"] == 1, "Portuguese", "Other"),
        "Age": np.where(df_raw["Age at enrollment"] >= 25, "25+", "<25"),
        "Scholarship": df_raw["Scholarship holder"].map({0: "No", 1: "Yes"}),
    }
    rows = []
    for gname, gv in groups.items():
        gv = pd.Series(np.asarray(gv))
        for lvl in sorted(gv.unique()):
            m = (gv == lvl).values
            tp = ((pred == 1) & (y == 1) & m).sum(); fn = ((pred == 0) & (y == 1) & m).sum()
            fp = ((pred == 1) & (y == 0) & m).sum(); tn = ((pred == 0) & (y == 0) & m).sum()
            rows.append(dict(attribute=gname, group=lvl, n=int(m.sum()), base_rate=y[m].mean(),
                             flag_rate=pred[m].mean(), TPR=tp / max(tp + fn, 1), FPR=fp / max(fp + tn, 1),
                             precision=tp / max(tp + fp, 1), accuracy=(pred[m] == y[m]).mean()))
    t = pd.DataFrame(rows)
    gaps = t.groupby("attribute").agg(
        demographic_parity_gap=("flag_rate", lambda s: s.max() - s.min()),
        equal_opportunity_gap=("TPR", lambda s: s.max() - s.min()),
        fpr_gap=("FPR", lambda s: s.max() - s.min()),
        disparate_impact_ratio=("flag_rate", lambda s: s.min() / s.max() if s.max() > 0 else np.nan)).reset_index()
    return t, gaps


# =====================================================================
# 7. MAIN
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.csv"); ap.add_argument("--out", default="outputs")
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--quick", action="store_true", help="1 repeat, fewer trees")
    a = ap.parse_args()
    if a.quick:
        a.repeats = 1
    fig_dir, tab_dir = f"{a.out}/figures", f"{a.out}/tables"
    os.makedirs(fig_dir, exist_ok=True); os.makedirs(tab_dir, exist_ok=True)
    T0 = time.time(); log = {}

    print("[1/7] Loading & cleaning ...")
    df, clog = load_and_clean(a.data); log.update(clog)
    print(f"      {clog['raw_shape']} -> {df.shape}; missing cells={clog['missing_cells']}, duplicates={clog['duplicate_rows']}")

    print("[2/7] EDA ...")
    run_eda(df, fig_dir, log)

    print("[3/7] Feature engineering ...")
    X = engineer(df); y = df["Target"].values
    log["n_raw_features"] = df.shape[1] - 2; log["n_engineered_features"] = X.shape[1]
    print(f"      {log['n_raw_features']} raw -> {X.shape[1]} features (before one-hot)")

    print(f"[4/7] Cross-validated experiments ({a.folds}-fold x {a.repeats} repeats) ...")
    pf, oof, removed, folds = run_cv(X, y, a.folds, a.repeats, a.quick, a.out)
    summ = pf.groupby("model", sort=False)[["Accuracy", "F1", "Recall", "Precision", "BalAcc", "AUC"]].agg(["mean", "std"])
    summ.columns = [f"{m}_{s}" for m, s in summ.columns]; summ["avg_train_rows_removed"] = pd.Series(removed)
    summ.to_csv(f"{tab_dir}/model_comparison.csv"); print(summ[["Accuracy_mean", "F1_mean", "Recall_mean", "AUC_mean"]].round(4).to_string())

    # hypothesis test: SOD-FE baseline (M5) vs SOD-FE+ (M9), paired over identical folds
    ntr = int(len(y) * (a.folds - 1) / a.folds); nte = len(y) - ntr
    base, plus = "M5 SOD-FE baseline (rank+SMOTE>IQR)", "M9 SOD-FE+ (full, stacked)"
    tests = []
    for met in ("F1", "Recall", "AUC", "Accuracy"):
        va = pf[pf.model == plus].sort_values("fold")[met].values; vb = pf[pf.model == base].sort_values("fold")[met].values
        t, p = corrected_ttest(va, vb, ntr, nte)
        try: w = stats.wilcoxon(va, vb).pvalue
        except ValueError: w = np.nan
        tests.append(dict(metric=met, SODFE_plus=va.mean(), baseline=vb.mean(), diff=va.mean() - vb.mean(),
                          t_corrected=t, p_corrected=p, p_wilcoxon=w, folds_plus_wins=int((va > vb).sum()), n_folds=len(va)))
    tests = pd.DataFrame(tests); tests.to_csv(f"{tab_dir}/hypothesis_test_M9_vs_M5.csv", index=False)
    print("\n      H0 test (SOD-FE+ vs SOD-FE baseline):\n", tests.round(4).to_string(index=False))

    print("[5/7] Figures: comparison, ROC, confusion, ablation ...")
    # 5.1 comparison bars
    plot = summ.reset_index().melt(id_vars="model", value_vars=[f"{m}_mean" for m in ("Accuracy", "F1", "Recall", "AUC")],
                                   var_name="metric", value_name="score")
    plot["metric"] = plot["metric"].str.replace("_mean", "")
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(data=plot, y="model", x="score", hue="metric", ax=ax, palette="viridis")
    lo = max(0.5, plot.score.min() - 0.05); ax.set_xlim(lo, 1.0); ax.set_title("Cross-validated performance by model (mean over folds)")
    ax.legend(loc="lower right"); plt.tight_layout(); plt.savefig(f"{fig_dir}/res_01_model_comparison.png", dpi=150); plt.close()

    # 5.2 ROC (repeat-0 out-of-fold probabilities)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for m in ["M1 LR (raw)", "M2 RF (raw)", base, "M8 +SHAP-RFE (top-20)", plus]:
        fpr, tpr, _ = roc_curve(y, oof[m][0]); ax.plot(fpr, tpr, lw=2 if m == plus else 1.3, label=f"{m} (AUC={roc_auc_score(y, oof[m][0]):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (out-of-fold predictions)"); ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout(); plt.savefig(f"{fig_dir}/res_02_roc.png", dpi=150); plt.close()

    # 5.3 confusion matrices baseline vs plus
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for a_, m in zip(axes, (base, plus)):
        ConfusionMatrixDisplay(confusion_matrix(y, oof[m][0] >= .5), display_labels=["Graduate", "Dropout"]).plot(ax=a_, cmap="Blues", colorbar=False)
        a_.set_title(m, fontsize=9)
    plt.tight_layout(); plt.savefig(f"{fig_dir}/res_03_confusion.png", dpi=150); plt.close()

    # 5.4 ablation ladder (F1 and recall, with std error bars)
    lad = ["M2 RF (raw)", "M4 RF + SMOTE", base, "M6 +IForest, clean>SMOTE", "M7 +outlier_score feature", "M8 +SHAP-RFE (top-20)", plus]
    fig, ax = plt.subplots(figsize=(11, 5))
    for met, c in (("F1", "#264653"), ("Recall", "#e76f51")):
        mm = summ.loc[lad, f"{met}_mean"]; ss = summ.loc[lad, f"{met}_std"]
        ax.errorbar(range(len(lad)), mm, yerr=ss, marker="o", capsize=3, label=met, color=c)
    ax.set_xticks(range(len(lad))); ax.set_xticklabels([l.split(" ", 1)[0] + "\n" + l.split(" ", 1)[1][:22] for l in lad], fontsize=8)
    ax.set_title("Ablation ladder: what does each SOD-FE+ component add? (error bars = fold std)"); ax.legend()
    plt.tight_layout(); plt.savefig(f"{fig_dir}/res_04_ablation.png", dpi=150); plt.close()

    # 5.5 threshold trade-off for final model
    ths = np.linspace(0.1, 0.9, 33)
    th = pd.DataFrame([dict(threshold=t, **metrics(y, oof[plus][0], t)) for t in ths])
    th.to_csv(f"{tab_dir}/threshold_tradeoff_M9.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in ("Precision", "Recall", "F1"):
        ax.plot(th.threshold, th[c], label=c)
    ax.axvline(.5, color="k", ls=":", lw=.8); ax.set_xlabel("Decision threshold"); ax.set_title("SOD-FE+ precision / recall trade-off")
    ax.legend(); plt.tight_layout(); plt.savefig(f"{fig_dir}/res_05_threshold.png", dpi=150); plt.close()

    print("[6/7] Fairness audit ...")
    ft, fg = fairness_table(df, y, oof[plus][0]); ft.to_csv(f"{tab_dir}/fairness_by_group.csv", index=False); fg.to_csv(f"{tab_dir}/fairness_gaps.csv", index=False)
    print(fg.round(3).to_string(index=False))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for a_, met in zip(axes, ("flag_rate", "TPR", "FPR")):
        sns.barplot(data=ft, x="attribute", y=met, hue="group", ax=a_, palette="Set2"); a_.set_title(met); a_.legend(fontsize=7)
    plt.suptitle("SOD-FE+ error profile by group (OOF predictions)"); plt.tight_layout(); plt.savefig(f"{fig_dir}/res_06_fairness.png", dpi=150); plt.close()

    print("[7/7] SHAP explainability (hold-out fit) ...")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    pre, numeric = make_preprocessor(Xtr); A = pre.fit_transform(Xtr); B = pre.transform(Xte)
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=SEED).fit(A)
    keep = (iso.predict(A) == 1) | (ytr == 1); A2, y2 = A[keep].reset_index(drop=True), ytr[keep]
    A2 = A2.assign(outlier_score=iso.decision_function(A2)); B2 = B.assign(outlier_score=iso.decision_function(B))
    A2, y2 = SMOTE(random_state=SEED).fit_resample(A2, y2)
    sel = shap_rfe(A2, y2, k=20); A2, B2 = A2[sel], B2[sel]
    final = build_model("stack", fast=a.quick).fit(A2, y2)
    pte = final.predict_proba(B2)[:, 1]; hold = metrics(yte, pte); pd.Series(hold).to_csv(f"{tab_dir}/holdout_metrics_M9.csv")
    print("      hold-out:", {k: round(v, 4) for k, v in hold.items()})
    xgb = XGBClassifier(n_estimators=250, max_depth=5, learning_rate=.05, subsample=.8, colsample_bytree=.8, random_state=SEED, eval_metric="logloss").fit(A2, y2)
    expl = shap.TreeExplainer(xgb); sv = expl.shap_values(B2)
    shap.summary_plot(sv, B2, show=False, max_display=20); plt.title("SHAP summary: drivers of predicted dropout risk"); plt.tight_layout()
    plt.savefig(f"{fig_dir}/res_07_shap_beeswarm.png", dpi=150, bbox_inches="tight"); plt.close()
    imp = pd.Series(np.abs(sv).mean(0), index=B2.columns).sort_values(ascending=False); imp.to_csv(f"{tab_dir}/shap_importance.csv", header=["mean_abs_shap"])
    # local explanations: two worst misclassifications (confident wrong) of the final stack
    wrong = np.where((pte >= .5) != yte)[0]; conf = np.abs(pte[wrong] - .5); pick = wrong[np.argsort(-conf)[:2]]
    ev = expl(B2)
    for j, i in enumerate(pick):
        plt.figure(); shap.plots.waterfall(ev[i], max_display=12, show=False)
        plt.title(f"Misclassified student #{i}: true={'Dropout' if yte[i] else 'Graduate'}, P(dropout)={pte[i]:.2f}", fontsize=9)
        plt.savefig(f"{fig_dir}/res_08_shap_local_{j+1}.png", dpi=150, bbox_inches="tight"); plt.close()

    log["runtime_sec"] = round(time.time() - T0); log["holdout"] = hold
    json.dump(log, open(f"{tab_dir}/run_log.json", "w"), indent=2, default=str)
    write_report(a.out, log, summ, tests, fg, ft, imp, hold, a)
    print(f"Done in {log['runtime_sec']}s -> {a.out}/")


def write_report(out, log, summ, tests, fg, ft, imp, hold, a):
    base, plus = "M5 SOD-FE baseline (rank+SMOTE>IQR)", "M9 SOD-FE+ (full, stacked)"
    f = lambda m, c: summ.loc[m, f"{c}_mean"]
    best_f1 = summ["F1_mean"].idxmax()
    L = []; w = L.append
    w("# SOD-FE+ — Student Dropout Prediction: Results\n")
    w(f"*Data: UCI 697 (Portugal), {log['clean_shape'][0]:,} students after removing 'Enrolled'. {a.folds}-fold stratified CV x {a.repeats} repeats; all preprocessing, outlier removal, SMOTE and feature selection fitted inside training folds only.*\n")
    w("## 1. Data cleaning\n")
    w(f"- Raw shape {log['raw_shape']}; missing cells: **{log['missing_cells']}**; duplicate rows: **{log['duplicate_rows']}**.")
    w("- Fixed file quirks: UTF-8 BOM, stray tab in `Daytime/evening attendance`, typo `Nacionality`, `;` separator.")
    w(f"- 'Enrolled' students ({log['target_counts_raw'].get('Enrolled', 0)}) removed: outcome unknown. Target: Dropout=1.")
    w(f"- Consistency checks (counts): {json.dumps(log['consistency_checks'])}")
    w(f"- Outliers: IQR flags {log['outlier_overlap']['IQR any-feature']} students on any continuous feature; Isolation Forest (5%) flags {log['outlier_overlap']['IsolationForest 5%']}, of which {log['outlier_overlap']['IF only (multivariate)']} are invisible to IQR (multivariate-only).\n")
    w("## 2. Main EDA findings\n")
    w("![](figures/eda_01_balance_course.png)\n\n![](figures/eda_02_numeric_dists.png)\n\n![](figures/eda_03_binary_dropout.png)\n\n![](figures/eda_04_target_corr.png)\n")
    w("## 3. Model comparison (CV mean ± std)\n")
    cols = ["Accuracy", "F1", "Recall", "Precision", "AUC"]
    w("| Model | " + " | ".join(cols) + " | avg rows removed |\n|---|" + "---|" * (len(cols) + 1))
    for m in summ.index:
        w(f"| {m} | " + " | ".join(f"{summ.loc[m, c+'_mean']:.3f} ± {summ.loc[m, c+'_std']:.3f}" for c in cols) + f" | {summ.loc[m, 'avg_train_rows_removed']:.0f} |")
    w("\n![](figures/res_01_model_comparison.png)\n\n![](figures/res_04_ablation.png)\n\n![](figures/res_02_roc.png)\n\n![](figures/res_03_confusion.png)\n")
    w("## 4. Hypothesis test — SOD-FE+ (M9) vs SOD-FE baseline (M5)\n")
    w("| Metric | SOD-FE+ | Baseline | Diff | corrected-t p | Wilcoxon p | folds won |\n|---|---|---|---|---|---|---|")
    for _, r in tests.iterrows():
        w(f"| {r.metric} | {r.SODFE_plus:.4f} | {r.baseline:.4f} | {r['diff']:+.4f} | {r.p_corrected:.4f} | {r.p_wilcoxon:.4f} | {r.folds_plus_wins}/{r.n_folds} |")
    sig = tests.set_index("metric").loc["F1", "p_corrected"] < 0.05
    w(f"\n**Verdict on H0 (no F1 difference):** {'rejected at α=0.05' if sig else 'NOT rejected at α=0.05 (corrected resampled t-test)'}; F1 difference = {tests.set_index('metric').loc['F1','diff']:+.4f}. Best mean F1 overall: **{best_f1}** ({summ.loc[best_f1,'F1_mean']:.4f}).\n")
    w(f"Hold-out (20%) result for SOD-FE+: " + ", ".join(f"{k}={v:.3f}" for k, v in hold.items()) + "\n")
    w("## 5. Fairness audit (SOD-FE+, out-of-fold)\n")
    w(fg.round(3).to_markdown(index=False)); w("\n![](figures/res_06_fairness.png)\n")
    w("## 6. Explainability\n")
    w("Top SHAP drivers: " + ", ".join(imp.index[:8]) + ".\n\n![](figures/res_07_shap_beeswarm.png)\n")
    w("## 7. Caveats\n")
    w("- Features include 2nd-semester results, so this is an end-of-first-year early-warning model, not an enrolment-time one.")
    w("- SMOTE on one-hot columns creates fractional dummy values; stacking's inner CV sees SMOTE-generated neighbours. Both are standard but slightly optimistic for CV of the stack.")
    w("- Single institution, 2008–2019; no causal claims. Demographic-parity gaps partly reflect real base-rate differences — compare `flag_rate` with `base_rate` in `tables/fairness_by_group.csv`.")
    open(f"{out}/REPORT.md", "w").write("\n".join(L))


if __name__ == "__main__":
    main()
