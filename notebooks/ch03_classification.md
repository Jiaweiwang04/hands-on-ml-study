# Chapter 3 — Classification

## 1. Classification

Classification predicts **discrete classes** rather than continuous values.

Examples:

- spam / not spam
- cancer / healthy
- handwritten digits $0$–$9$

Main types:

- **Binary classification**: two classes
- **Multiclass classification**: more than two classes
- **Multilabel classification**: one sample can have multiple binary labels
- **Multioutput classification**: one sample has multiple outputs, and each output can have multiple possible classes

---

## 2. Binary Classification

A binary classifier predicts:

$$
y \in \{0,1\}
$$

Example:

$$
y =
\begin{cases}
1 & \text{digit is 5} \\
0 & \text{otherwise}
\end{cases}
$$

The class of interest is usually called the **positive class**, while the other is the **negative class**.

---

## 3. Accuracy and Class Imbalance

Accuracy is:

$$
\text{Accuracy}
=
\frac{\text{correct predictions}}{\text{total predictions}}
$$

Accuracy can be misleading for **imbalanced datasets**.

Example:

- 99% negative samples
- 1% positive samples

A classifier that always predicts negative still achieves:

$$
99\%
$$

accuracy.

Therefore, classification performance should usually also be evaluated using:

- confusion matrix
- precision
- recall
- F1 score
- PR curve
- ROC curve / AUC

---

## 4. Confusion Matrix

For binary classification:

| | Predicted Positive | Predicted Negative |
|---|---:|---:|
| Actual Positive | TP | FN |
| Actual Negative | FP | TN |

### TP — True Positive

Correctly predicts a positive sample.

### TN — True Negative

Correctly predicts a negative sample.

### FP — False Positive

Predicts positive when the true class is negative.

Also called a **false alarm**.

### FN — False Negative

Predicts negative when the true class is positive.

This corresponds to a **missed positive case**.

---

## 5. Precision

Precision measures how reliable positive predictions are.

$$
\text{Precision}
=
\frac{TP}{TP+FP}
$$

Interpretation:

> Among all samples predicted as positive, how many are actually positive?

High precision means fewer **false positives**.

---

## 6. Recall

Recall measures how many real positive samples are detected.

$$
\text{Recall}
=
\frac{TP}{TP+FN}
$$

Recall is also called:

- sensitivity
- True Positive Rate (TPR)

Interpretation:

> Among all actual positive samples, how many were detected?

High recall means fewer **false negatives**.

---

## 7. Precision–Recall Trade-off

Many classifiers internally produce a **score** rather than directly outputting a class.

A threshold determines the prediction:

$$
\text{score} \geq \text{threshold}
\Rightarrow
\text{positive}
$$

Increasing the threshold usually gives:

$$
\text{Precision} \uparrow
$$

$$
\text{Recall} \downarrow
$$

Lowering the threshold usually gives:

$$
\text{Recall} \uparrow
$$

$$
\text{Precision} \downarrow
$$

The correct threshold depends on the cost of different errors.

Examples:

- cancer screening → often prioritize recall
- spam filtering → may prioritize precision

---

## 8. F1 Score

The F1 score combines precision and recall using their harmonic mean:

$$
F_1
=
2
\frac{
\text{Precision}\times\text{Recall}
}{
\text{Precision}+\text{Recall}
}
$$

Equivalent form:

$$
F_1
=
\frac{2TP}{2TP+FP+FN}
$$

F1 is high only when both precision and recall are reasonably high.

It is useful when both FP and FN matter.

---

## 9. Cross-Validation for Classification

Evaluating predictions directly on the training data can give misleading results.

Use cross-validation:

```python
from sklearn.model_selection import cross_val_score

cross_val_score(
    clf,
    X_train,
    y_train,
    cv=3,
    scoring="accuracy"
)
```

To obtain predictions from models that did not train on the corresponding samples:

```python
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(
    clf,
    X_train,
    y_train,
    cv=3
)
```

These are **out-of-fold predictions**.

---

## 10. Decision Scores

Some classifiers provide a decision score:

```python
y_scores = clf.decision_function(X)
```

The score can be compared with different thresholds.

To obtain cross-validated decision scores:

```python
y_scores = cross_val_predict(
    clf,
    X_train,
    y_train,
    cv=3,
    method="decision_function"
)
```

This allows evaluation across many thresholds.

---

## 11. Precision–Recall Curve

A Precision–Recall curve shows the relationship between precision and recall for different thresholds.

```python
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(
    y_train,
    y_scores
)
```

The PR curve is especially useful when:

- the positive class is rare
- the dataset is highly imbalanced

---

## 12. ROC Curve

The ROC curve compares:

$$
TPR
$$

against:

$$
FPR
$$

where:

$$
TPR
=
\frac{TP}{TP+FN}
=
\text{Recall}
$$

and:

$$
FPR
=
\frac{FP}{FP+TN}
$$

An ideal classifier has:

$$
TPR \rightarrow 1
$$

and:

$$
FPR \rightarrow 0
$$

so the ideal point lies near the top-left corner.

---

## 13. ROC AUC

AUC is the **Area Under the ROC Curve**.

$$
0 \leq \text{AUC} \leq 1
$$

Typical interpretation:

- random classifier: $\text{AUC} \approx 0.5$
- perfect classifier: $\text{AUC} = 1$

A higher AUC means the classifier is generally better at separating positive and negative samples.

For highly imbalanced datasets, the **PR curve is often more informative than ROC**.

---

## 14. Multiclass Classification

Multiclass classification predicts one class from more than two possibilities.

Example:

$$
y \in \{0,1,\dots,9\}
$$

Some algorithms directly support multiclass classification, including:

- Logistic Regression
- Random Forest
- Naive Bayes

Binary classifiers can also be extended to multiclass tasks.

### One-vs-Rest (OvR)

Train one classifier for each class:

- 0 vs rest
- 1 vs rest
- ...
- 9 vs rest

For $N$ classes, train:

$$
N
$$

classifiers.

The class with the highest score is selected.

### One-vs-One (OvO)

Train one classifier for every pair of classes.

For $N$ classes:

$$
\frac{N(N-1)}{2}
$$

classifiers are required.

For 10 classes:

$$
\frac{10\times9}{2}=45
$$

OvO is commonly associated with **Support Vector Machines (SVMs)**.

---

## 15. Error Analysis

Overall accuracy alone does not explain how the model fails.

A confusion matrix can reveal which classes are commonly confused.

```python
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred
)
```

Error analysis should investigate:

- which classes are confused
- whether certain samples are systematically difficult
- whether preprocessing or features should be improved

The goal is not only to know **how often the model is wrong**, but also **how it is wrong**.

---

## 16. Multilabel Classification

In multilabel classification, one sample can have multiple labels simultaneously.

Example:

```text
person = 1
dog = 1
outdoor = 1
night = 0
```

Output:

$$
[1,1,1,0]
$$

Each label is usually a binary prediction.

A sample can belong to several labels at the same time.

---

## 17. Multioutput Classification

Multioutput classification predicts multiple outputs for each sample.

General form:

$$
y=(y_1,y_2,\dots,y_k)
$$

Unlike multilabel classification, each output can have more than two possible values.

Example: image denoising.

For a $28\times28$ grayscale image, the model predicts:

$$
784
$$

pixel values.

Each output can take a value from:

$$
\{0,1,\dots,255\}
$$

Therefore:

- **multilabel** → multiple binary outputs
- **multioutput** → multiple outputs, each possibly multiclass

Multilabel classification can be viewed as a special case of multioutput classification.

---

## 18. Main Algorithms Mentioned

Important classifiers introduced or used in this chapter include:

- SGDClassifier
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Naive Bayes
- K-Nearest Neighbors (KNN)

Important multiclass strategies:

- One-vs-Rest (OvR)
- One-vs-One (OvO)

---

## Key Takeaways

Classification metrics must be chosen according to the problem.

Important formulas:

$$
\text{Precision}
=
\frac{TP}{TP+FP}
$$

$$
\text{Recall}
=
\frac{TP}{TP+FN}
$$

$$
F_1
=
2
\frac{PR}{P+R}
$$

Key ideas:

- accuracy can be misleading on imbalanced datasets
- precision measures the reliability of positive predictions
- recall measures how many actual positives are detected
- threshold selection creates a precision–recall trade-off
- PR curves are especially useful for rare positive classes
- confusion matrices are essential for error analysis
- multiclass, multilabel and multioutput classification describe different output structures