# SOD-FE+: Student Dropout Prediction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/sodfe-plus-dropout-prediction/blob/main/sodfe_plus_colab.ipynb)

Reproducible pipeline for predicting student dropout (UCI dataset 697, Polytechnic Institute of Portalegre, Portugal). It implements **SOD-FE+**, an enhanced version of the SOD-FE approach, and evaluates each component with an ablation study, a significance test, a fairness audit and SHAP explanations.

**SOD-FE+** = Isolation Forest outlier filter (label-aware, applied before SMOTE) + anomaly-score feature + SHAP-based recursive feature elimination + stacked ensemble (Random Forest, XGBoost, MLP → logistic regression).

![Methodology flowchart](flowchart/sodfe_plus_flowchart.png)

## Key results (5-fold CV × 3 repeats, 3,630 students, dropout = positive class)

| Model | F1 | Recall | AUC |
|---|---|---|---|
| SOD-FE baseline (reconstruction) | 0.861 | 0.871 | 0.929 |
| SOD-FE+ (full) | 0.880 | 0.872 | 0.951 |
| Stacked ensemble only | 0.889 | 0.865 | 0.958 |

SOD-FE+ improves F1 over the baseline by +0.019 (corrected resampled t-test p = 0.023), but its individual components add little over a plain stacked ensemble. See `results/REPORT.md` for the full tables, figures, fairness audit and caveats.

## Repository layout

```
sodfe_plus.py              complete pipeline (cleaning, EDA, models, evaluation, fairness, SHAP)
sodfe_plus_colab.ipynb     Colab notebook with the script embedded
requirements.txt           exact package versions used
data/data.csv              UCI dataset 697 (semicolon-separated)
results/                   figures/, tables/ and REPORT.md from the reported run
flowchart/                 methodology flowchart (PNG, SVG) and generating script
```

## Reproduce

```bash
pip install -r requirements.txt
python sodfe_plus.py --data data/data.csv --out outputs --repeats 3
```

- Full run: about 8 minutes on one CPU core (no GPU needed).
- `--quick` runs one repeat with fewer trees (about 2 minutes) to check your setup; its numbers differ slightly.
- In Google Colab: open the notebook with the badge above, run all cells and upload `data/data.csv` when asked.

Tested with Python 3.12.3. All random seeds are fixed (`random_state=0`). Every data-dependent step is fitted on the training folds only, so test folds are never cleaned, resampled or used for feature selection. Small numerical differences are possible across library versions or operating systems.

## Notes and limitations

- The SOD-FE baseline is a reconstruction from the published description, not the authors' code.
- Features include second-semester results, so this is an end-of-first-year early-warning model.
- Single institution, 2008–2019; results may not generalise.
- SMOTE on one-hot columns produces fractional dummy values.

## Data source and citation

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89 (licence: CC BY 4.0).
