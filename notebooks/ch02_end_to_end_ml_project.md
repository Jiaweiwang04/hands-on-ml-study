# Chapter 2 — End-to-End Machine Learning Project

## 1. The End-to-End Workflow

A practical machine learning project follows a repeatable sequence:

```text
Frame the problem → Get the data → Explore it → Split the data
→ Prepare features → Train baselines → Validate and tune
→ Evaluate once on the test set
```

The California housing task is **supervised regression**: the inputs are district-level
measurements and the target is a continuous median house value.

---

## 2. Evaluation Metrics

For $m$ predictions, **Root Mean Squared Error (RMSE)** is:

$RMSE = \sqrt{\frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i-y_i)^2}$

RMSE gives more weight to large errors. **Mean Absolute Error (MAE)** is:

$MAE = \frac{1}{m}\sum_{i=1}^{m}|\hat{y}_i-y_i|$

MAE is easier to interpret and is less sensitive to large errors. The right metric should match
the cost of mistakes; this chapter uses RMSE as the primary metric.

---

## 3. Splitting and Sampling

A **train/test split** reserves data that the model does not see during training or model
selection. Looking repeatedly at the test set creates **data snooping bias**: decisions become
adapted to that particular sample, so its score no longer estimates performance on new data.

**Stratified sampling** preserves the proportions of important subgroups in each split. For the
housing data, binned median income is a reasonable stratification feature because income is
strongly related to house value. Stratification is useful only when the chosen groups are
meaningful and each contains enough examples.

---

## 4. Explore Before Modeling

**Exploratory data analysis (EDA)** checks shapes, data types, summary statistics, missing
values, unusual distributions, and a few important plots. A **correlation** measures linear
association, not causation, and can miss nonlinear relationships.

**Feature engineering** combines or transforms raw columns to make useful patterns easier to
learn. Examples for housing data include rooms per household or bedrooms per room. New features
must be created without using information from the target or future data.

---

## 5. Preprocessing Without Leakage

Common preparation steps include:

- **missing-value handling**, such as replacing missing numeric values with the training median;
- **categorical encoding**, such as `OneHotEncoder` for unordered categories;
- **feature scaling**, such as `StandardScaler`, especially for models sensitive to feature
  magnitude.

A scikit-learn **Pipeline** chains preprocessing and a model. A **ColumnTransformer** applies
different transformations to different column groups, such as imputation and scaling for numeric
columns and one-hot encoding for categorical columns.

These tools prevent **data leakage** when they are fitted inside each training fold. Fitting an
imputer, encoder, or scaler on the full dataset lets validation or test information influence the
model.

---

## 6. Baseline Models

- **Linear Regression** learns a linear relationship. It is fast and interpretable, but may
  underfit nonlinear data.
- **Decision Tree** recursively partitions the feature space. An unrestricted tree can fit the
  training data extremely closely and overfit.
- **Random Forest** averages many randomized decision trees. This usually reduces variance and
  generalizes better than one tree, at the cost of computation and interpretability.

**Underfitting** means the model is too limited: both training and validation errors are high.
**Overfitting** means training error is much lower than validation error: the model has learned
training-specific noise or detail.

---

## 7. Cross-Validation and Tuning

In **k-fold cross-validation**, the training set is divided into $k$ folds. The model trains on
$k-1$ folds and validates on the remaining fold, repeating until every fold has served as the
validation set. The mean estimates generalization performance; the standard deviation shows how
sensitive the estimate is to the particular split.

**Parameters** are learned from data, such as regression coefficients or tree split thresholds.
**Hyperparameters** are chosen before fitting, such as tree depth or the number of trees.

**GridSearchCV** evaluates every combination in a specified grid. **RandomizedSearchCV** samples
a fixed number of combinations from distributions or lists, which is often more efficient for a
large search space. Both must operate only on the training set and should include preprocessing
inside the searched pipeline.

---

## 8. Final Test-Set Evaluation

After selecting the model and hyperparameters with cross-validation, fit the chosen pipeline on
the full training set and evaluate it **exactly once** on the untouched test set. This final score
is the best available estimate of performance on unseen data. If the score drives more tuning,
the test set has effectively become another validation set and a new test set is needed.

---

# Chapter Takeaway

Reliable machine learning is a process, not just an algorithm. Split early, learn every
preprocessing step from training data, compare simple baselines with cross-validation, tune a
small number of justified hyperparameters, and protect the test set until the final evaluation.
