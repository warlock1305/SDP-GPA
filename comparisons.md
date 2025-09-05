# Repository Type Comparisons: Methods and Formulas

This document explains how the comparison scripts identify significant differences between repository categories (e.g., CLI tools vs Data Science, Data Science vs Web Application), and which statistical models/formulas are used.

## Embeddings and Data Preparation
- **Embeddings**: Each repository is represented by a 768-dimensional vector from CodeBERT (`microsoft/codebert-base`).
- **Sources**: Embeddings are loaded from `scripts/extraction/CodeBERTEmbeddings` matching patterns like `cli_tool_*_embeddings.json`, `data_science_*_embeddings.json`, `web_application_*_embeddings.json`, `library_*_embeddings.json`.
- **Matrices**: For a pairwise comparison, embeddings are stacked into a feature matrix \(X \in \mathbb{R}^{N \times D}\) with labels \(y \in \{0,1\}\) for the two categories.

## What "Significant Dimensions" Means Here
A "dimension" is a single CodeBERT feature index \(d \in [0, D)\). A dimension is considered significant if its distribution differs between the two categories according to statistical criteria. The scripts use multiple, complementary criteria and then combine their results.

## Methods Used

### 1) Independent Two-Sample t-test (per dimension)
- Script: `find_significant_dimensions_ds_vs_web.py`
- For each dimension \(d\): compare values for group A vs group B using SciPy’s `stats.ttest_ind` (default `equal_var=True`).
- Test statistic (conceptually):
  \( t_d = \frac{\bar{x}_{A,d} - \bar{x}_{B,d}}{\sqrt{ s^2_{A,d}/n_A + s^2_{B,d}/n_B }} \)
- Decision: mark dimensions with `p < 0.05` as statistically significant; also rank by absolute \(|t_d|\) and report top dimensions.

### 2) ANOVA F-test feature scoring (SelectKBest)
- Script: `find_significant_dimensions_ds_vs_web.py`
- Standardize \(X\) using `StandardScaler`, then apply `SelectKBest(score_func=f_classif, k=20)`.
- `f_classif` computes an ANOVA F-statistic per dimension measuring between-class variance vs within-class variance.
- Output: F-scores and p-values per dimension; report top-\(k\) by F-score.

### 3) Mutual Information feature scoring (SelectKBest)
- Script: `find_significant_dimensions_ds_vs_web.py`
- Using the same standardized \(X\), apply `SelectKBest(score_func=mutual_info_classif, k=20)`.
- Mutual Information (non-parametric) measures dependency between each feature and the class label without assuming linearity or normality.
- Output: MI score per dimension; report top-\(k\) by MI.

### 4) Random Forest feature importance
- Script: `find_significant_dimensions_ds_vs_web.py`
- Standardize \(X\), train `RandomForestClassifier(n_estimators=100, random_state=42)`.
- Feature importances are computed from impurity reductions (Gini importance) aggregated across trees.
- Validation: `cross_val_score(rf, X_scaled, y, cv=5)` is printed to give an accuracy estimate.
- Output: importance score per dimension; report top dimensions by importance.

### 5) Cross-method aggregation of significant dimensions
- Script: `find_significant_dimensions_ds_vs_web.py`
- Take the top 50 dimensions from each method: |t|, F-score, MI, and RF importance.
- Compute:
  - The intersection across all four methods (most robust consensus).
  - Dimensions appearing in ≥3 methods, ranked by count.
- These aggregated sets are the recommended "significant dimensions" for that category pair.

## Additional Analyses in the Pairwise Comparison Scripts
The comparison scripts for specific pairs (e.g., `compare_cli_vs_datascience.py`, `compare_cli_vs_library_fixed.py`, `compare_datascience_vs_webapp.py`) provide context and validation via descriptive statistics and projections:

- **Summary statistics per repository** over the embedding vector: mean, standard deviation, min, max, L2 norm.
- **Two-sample t-tests on aggregated metrics**: t-tests compare distributions of repository-level means, stds, and norms between the two categories.
- **Cosine similarity analyses**: within-category and cross-category similarity matrices and their aggregate means/stds.
- **Correlation heatmaps**: correlation matrix over the first 50 embedding dimensions for a visual sense of redundancy/structure.
- **Low-dimensional projections**:
  - PCA (2 components) with explained variance ratios.
  - t-SNE (2D) for non-linear visualization of separability.
- **Dimension-wise mean comparison plots**: bar plots comparing the average value of the first 20 dimensions between categories.

These analyses corroborate whether the categories exhibit separable patterns in the embedding space and help interpret the significant-dimension findings.

## Notes and Assumptions
- CodeBERT embedding dimensionality is typically 768.
- No multiple-testing correction is applied to the per-dimension t-tests in the script; significance uses a nominal `p < 0.05` threshold.
- Standardization is applied before F-test/MI/RF to stabilize scales across dimensions.
- Random Forest importance reflects impurity-based importance; alternative models (e.g., linear SVM with absolute weights) could be added similarly.

## File-to-Method Mapping
- `compare_cli_vs_datascience.py`: descriptive stats, t-tests on aggregated metrics (mean/std/norm), cosine similarity, PCA, t-SNE, correlation heatmaps, dimension-wise mean plots.
- `compare_cli_vs_library_fixed.py`: same structure as above for CLI vs Library.
- `compare_datascience_vs_webapp.py`: same structure as above for Data Science vs Web App.
- `find_significant_dimensions_ds_vs_web.py`: per-dimension significance via t-test, ANOVA F-test, Mutual Information, Random Forest; plus cross-method aggregation into final recommended dimensions.
