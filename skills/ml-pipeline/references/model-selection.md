# Model selection playbook

Read at step 3 (when writing the prediction contract) and again before the Gate B report. Key on
`ml_pipeline/data_profile.json → traits`. Always start with a baseline (a dummy predictor and one
simple model), judged on validation with the contract's metric. "Avoid" means: not without a
written reason in the `Model rationale:` block.

## Tabular, small (`small_data`: fewer than 5,000 rows)
- Baseline: majority-class / mean dummy, then logistic or linear regression with L2.
- Strong default: gradient boosting with shallow trees (depth 2–4, strong regularization, early
  stopping) or the regularized linear model itself.
- Worth trying: random forest (few knobs); GAM / EBM when the model must be read by a person.
- Avoid: neural nets (overfit and hide it — the hook denies this); large hyperparameter searches
  (use `guard.cv_splits(train, n_repeats=3)` and report the spread); anything you cannot explain to the user.

## Tabular, medium (5,000 – 500,000 rows)
- Baseline as above. Strong default: LightGBM / XGBoost / CatBoost with early stopping on validation.
- Worth trying: a linear model with feature crosses; an MLP only if boosting plateaus and rows > 50,000.
- Avoid: k-NN and RBF-SVM above ~50,000 rows (quadratic time); unregularized trees.

## Tabular, large (`large_data`: more than 500,000 rows)
- Keep it on disk: `guard.profile(path)`, `guard.eda_figures(path)`, `guard.split_to_files(path)`,
  `guard.final_test(model.predict, test_path)` all stream. Convert CSV to Parquet once.
- Strong default: histogram gradient boosting (`HistGradientBoosting*`, LightGBM) with subsampling and
  early stopping; linear models with `SGD*` / `partial_fit` on hashed or sparse features when the table
  is very wide or does not fit in memory.
- Tune on a stated sample of train, then refit the chosen configuration on all of train.
- Avoid: anything O(n²); exact k-NN; kernel SVMs; loading the whole file with pandas "to look at it".

## Wide (`wide_data`: more than 200 columns)
- Read `data_profile_summary.md`; `02_feature_overview` and the variable pages cover every column. Rank
  features by target strength only on the training split (`guard.train_associations` at step 7). Drop
  constant and near-duplicate columns at step 7 using train statistics only.
- Strong default: regularized linear (L1 / elastic net) and gradient boosting with column subsampling.
- Avoid: feature selection on all rows (fit selectors on train only — the hook treats them as fits).

## Imbalanced target (`imbalanced`: minority class under 10%)
- Metric: average precision (PR-AUC) primary; recall at a fixed precision or F1 at a chosen
  threshold; never accuracy for model selection (the hook denies `scoring="accuracy"`).
- First try `class_weight="balanced"` / `scale_pos_weight`; if you resample, `fit_resample` on the
  training split only (the hook denies anything else).
- The majority-class dummy baseline shows exactly what accuracy hides.

## High-cardinality categoricals (`high_card_categoricals`: more than 50 levels)
- Prefer CatBoost, or target / ordinal encoding fit on train inside a pipeline object.
- Avoid one-hot encoding (width explosion, and leakage if fit on all rows).

## Temporal data (`has_datetime`)
- Dates stored as text (`"2024-03-01"`) or epoch numbers count (`traits.datetime_as_text`); convert
  them with `guard.parse_datetimes(df)`.
- Split: chronological — `guard.split(time_col=..., gap=...)`; validation is the period after train,
  test the latest period; `guard.cv_splits(time_col=..., gap=...)` or `TimeSeriesSplit(gap=...)` for
  CV. Use a gap at least as long as the prediction horizon. Random splits and k-fold are denied.
- Features: lags and rolling statistics computed only from the past; calendar features.
- Baseline: last value / seasonal naive for forecasting; logistic on lag features for temporal
  classification. Strong default: gradient boosting on lag features; ETS / ARIMA for univariate series.
- Avoid: tree models extrapolating a trend (detrend first); LSTMs before boosting has been tried.

## Repeated entities (`has_groups`: patients, users, devices, sites)
- `guard.profile` finds repeating entity columns itself (`traits.group_candidates`); `guard.split`
  refuses a row-level split on them unless the user agreed and you pass `not_grouped_reason=`.
- Split by group — `guard.split(group_col=...)`, `guard.cv_splits(group_col=...)` or `GroupKFold` for
  CV. Row-level splits are denied.
- Avoid: features that identify the entity; the model memorizes it instead of learning.

## Text
- Baseline: TF-IDF (1–2 grams) + logistic regression or linear SVM.
- Strong default: a fine-tuned small transformer only when the baseline is clearly insufficient and
  there are more than ~5,000 labelled examples. Avoid training embeddings from scratch on small corpora.

## Images
- Step 1: `guard.manifest(folder, kind="image", group_pattern=r"(patient\d+)")`; split the manifest
  with `group_col="group"` so one source — and every exact duplicate file — stays in one split.
- Baseline: features from a pretrained CNN + logistic regression. Strong default: fine-tune a
  pretrained backbone with augmentation applied after the split.
- Avoid: training from scratch below ~100,000 images; augmenting before splitting; random splits when
  images share a source (patient, device, session → group split).

## Interpretability required
- Logistic / linear with monotone constraints, GAM / EBM, shallow trees; gradient boosting with
  monotone constraints plus SHAP for explanation. Avoid black-box ensembles where a clinician or
  regulator must read the model.

## Wrong applications the hook denies (red flags) and their override names

| red flag | what fires it | do instead |
|---|---|---|
| `neural-net-small-data` | MLP / keras / torch training with fewer than 5,000 rows | regularized linear, gradient boosting |
| `random-split-temporal` | `train_test_split` / `KFold` with a datetime column | `guard.split(time_col=)`, `TimeSeriesSplit` |
| `group-split` | any non-Group split with `group_col` set | `guard.split(group_col=)`, `GroupKFold` |
| `accuracy-imbalanced` | `scoring="accuracy"` with minority under 10% | `average_precision`, `f1`, `balanced_accuracy` |
| `resample-before-split` | `fit_resample` on anything but the train split | split first, then resample train |
| `baseline-first` | a non-baseline model or training API before `- Step 9 done` | dummy + one simple model, logged as `baseline` |
| `tune-before-train` | a hyperparameter search before `- Step 10 done` | compare candidates on validation first |

Override one with `- Override: red flag <name> - user approved <what> YYYY-MM-DD - reason: <why>` in
`PIPELINE.md`, only after the user said "override red flag <name>" in chat (the hook records it). Advisory smells (warned, not denied): k-NN or
kernel SVM above 50,000 rows; one-hot on high-cardinality columns; `accuracy_score` reported on an
imbalanced target.
