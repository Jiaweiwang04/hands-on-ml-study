# Chapter 1 — The Machine Learning Landscape

## 1. What Is Machine Learning?

Machine Learning builds systems that learn patterns from data instead of relying only on hand-written rules.

A model can be viewed as:

$$
\hat y = f_\theta(x)
$$

where:

- $x$: input features
- $y$: target
- $\theta$: model parameters
- $f_\theta$: learned mapping

The main goal is not low training error, but **good generalization to unseen data**.

---

## 2. Main Types of Machine Learning

### Supervised Learning

Training data contains input-target pairs $(x_i, y_i)$.

Main tasks:

- **Classification**: predict discrete classes
- **Regression**: predict continuous values

Examples:

- spam detection
- image classification
- house price prediction
- patient outcome prediction

Common model names introduced in this chapter:

- **Linear Regression**
- **Decision Tree**
- **Random Forest**
- **SVM**
- **Neural Network**
- **K-Nearest Neighbors (KNN)**

---

### Unsupervised Learning

Training data has no labels.

Main tasks:

- **Clustering**
- **Dimensionality Reduction**
- **Anomaly Detection**
- **Visualization**

Important methods:

- **K-Means** — clustering
- **PCA** — dimensionality reduction
- **Manifold Learning** — nonlinear dimensionality reduction
- **Learned Embeddings** — learned low-dimensional representations

---

### Semi-Supervised Learning

Uses a small amount of labeled data together with a large amount of unlabeled data.

Useful when labels are expensive, especially in medicine and biology.

---

### Self-Supervised Learning

Creates labels automatically from the data itself.

This is widely used in:

- large language models
- vision models
- speech models
- multimodal models

---

## 3. Batch Learning vs Online Learning

### Batch Learning

Train once on an existing dataset:

```text
Dataset → Training → Model → Deployment
```

New data usually requires retraining.

---

### Online Learning

The model updates continuously or incrementally as new data arrives.

The **learning rate** controls how strongly new data changes the model.

---

## 4. Instance-Based vs Model-Based Learning

### Instance-Based Learning

Predictions are based on similarity to stored training examples.

Main example:

- **K-Nearest Neighbors (KNN)**

Core idea:

> Similar inputs should have similar outputs.

---

### Model-Based Learning

Assume a parameterized model and learn its parameters.

Example:

$$
\hat y = \theta_0 + \theta_1 x
$$

Training often means solving:

$$
\theta^* = \arg\min_\theta L(\theta)
$$

This optimization perspective is fundamental to machine learning.

---

## 5. Loss Functions

A loss function measures model error.

For regression, a common example is **Mean Squared Error (MSE)**:

$$
MSE = \frac{1}{n}\sum_{i=1}^{n}(\hat y_i-y_i)^2
$$

For classification, a common loss is:

- **Cross-Entropy**

Different models may use different objectives, but training usually means minimizing a loss.

---

## 6. Parameters vs Hyperparameters

### Parameters

Learned from data.

Examples:

- linear model weights
- neural network weights
- biases

### Hyperparameters

Chosen by the practitioner.

Examples:

- learning rate
- number of neighbors in KNN
- tree depth
- regularization strength
- number of trees
- neural network architecture

---

## 7. Generalization

The real goal of ML is performance on unseen data.

Typical split:

```text
Dataset
├── Training Set
├── Validation Set
└── Test Set
```

### Training Set

Used to learn parameters.

### Validation Set

Used for:

- model selection
- hyperparameter tuning
- feature selection
- threshold selection

### Test Set

Used only for final evaluation.

Repeatedly tuning against the test set makes the test score unreliable.

---

## 8. Data Leakage

Data leakage happens when information that should be unavailable during training enters the training process.

Wrong:

```text
Fit preprocessing on all data
→ Split train/test
```

Correct:

```text
Split first
→ Fit preprocessing on training data
→ Apply to validation/test data
```

Example:

```python
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Leakage often produces unrealistically good evaluation results.

---

## 9. Overfitting and Underfitting

### Overfitting

The model learns training-specific noise or accidental patterns.

Typical pattern:

```text
Training performance: very good
Validation/Test performance: poor
```

Common causes:

- model too complex
- too little data
- noisy labels
- too many irrelevant features

Common solutions:

- more data
- simpler model
- better feature selection
- **regularization**
- **cross-validation**

Important regularization methods to remember:

- **L1 Regularization**
- **L2 Regularization**

General form:

$$
Loss = Prediction\ Error + \lambda \times Complexity
$$

---

### Underfitting

The model is too simple to capture the real pattern.

Typical pattern:

```text
Training performance: poor
Validation/Test performance: poor
```

Possible solutions:

- more expressive model
- better features
- weaker regularization
- better optimization

---

## 10. Main Challenges in Machine Learning

Important real-world problems:

- insufficient training data
- nonrepresentative training data
- sampling bias
- poor-quality data
- missing values
- noisy or incorrect labels
- irrelevant features
- data leakage
- distribution shift
- overfitting
- underfitting

In practice, data problems are often more important than choosing a more complicated algorithm.

---

## 11. Feature Engineering

Feature engineering transforms raw data into more useful model inputs.

Example:

```text
Raw date
→ day of week
→ month
→ season
```

Traditional ML often depends heavily on feature engineering.

Deep Learning can learn many useful representations automatically, but input representation still matters.

---

## 12. Distribution Shift

ML often assumes training and deployment data are similar:

$$
P_{train}(x,y) \approx P_{deployment}(x,y)
$$

If this assumption fails, performance can drop.

Examples:

- different hospitals
- new patient populations
- changing user behavior
- different devices
- changing environments

---

## 13. No Free Lunch

There is no single algorithm that is best for every problem.

Different models make different assumptions:

- **Linear Regression** assumes approximately linear relationships
- **KNN** relies on meaningful similarity
- **Decision Trees** partition the feature space
- **Random Forests** combine many trees
- **SVMs** search for separating boundaries with large margins
- **Neural Networks** learn flexible nonlinear functions

Model choice should depend on the data, task, and empirical evaluation.

---

## 14. Typical Machine Learning Workflow

```text
1. Define the problem
2. Collect data
3. Understand the data
4. Create train/validation/test split
5. Clean and preprocess data
6. Engineer/select features
7. Train a baseline
8. Evaluate
9. Diagnose errors
10. Tune and compare models
11. Final test evaluation
12. Deploy
13. Monitor
```

---

## 15. Key Algorithms and Methods Mentioned

Keep these names familiar:

- **Linear Regression**
- **K-Nearest Neighbors (KNN)**
- **K-Means**
- **PCA**
- **Manifold Learning**
- **Learned Embeddings**
- **Decision Tree**
- **Random Forest**
- **SVM**
- **Neural Network**
- **Mean Squared Error (MSE)**
- **Cross-Entropy**
- **Cross-Validation**
- **L1 Regularization**
- **L2 Regularization**

---

## 16. Connection to Biomedical ML

Biomedical ML often has:

```text
Small sample size
+
High-dimensional features
+
Expensive labels
+
Missing data
+
Sampling bias
+
Leakage risk
```

Example:

```text
500 patients
20,000 genes
```

This creates a strong risk of overfitting.

Important ideas from this chapter for biomedical projects:

- careful train/test splitting
- leakage prevention
- representative datasets
- dimensionality reduction
- regularization
- reliable validation
- generalization to new patients or hospitals

---

# Chapter Takeaway

The most important idea in Chapter 1 is:

> **Machine Learning is not about fitting the training data as closely as possible. It is about learning patterns that generalize to unseen data.**

When approaching a new ML problem, ask:

1. Is it supervised or unsupervised?
2. Is it classification or regression?
3. Is learning batch or online?
4. Is the method instance-based or model-based?
5. What loss is being optimized?
6. How are train, validation, and test sets separated?
7. Is there any data leakage?
8. Is the model overfitting or underfitting?
9. Is the dataset representative?
10. Does the evaluation really measure generalization?
