# Chapter 4 — Training Models

Chapter 4 explains how a model learns its parameters from data.

The main training process is:

$$
\boxed{
\text{define a model}
\rightarrow
\text{define a loss}
\rightarrow
\text{minimize the loss}
\rightarrow
\text{obtain the parameters}
}
$$

In mathematical form:

$$
\boldsymbol{\theta}^{*}
=
\arg\min_{\boldsymbol{\theta}}
J(\boldsymbol{\theta})
$$

where:

- $\boldsymbol{\theta}$ contains the model parameters
- $J(\boldsymbol{\theta})$ is the objective or cost function
- $\boldsymbol{\theta}^{*}$ is the parameter vector that minimizes the objective

---

## 1. Linear Regression

Linear Regression predicts a continuous value using a weighted sum of the input features.

For $n$ features:

$$
\hat y
=
\theta_0
+
\theta_1x_1
+
\cdots
+
\theta_nx_n
$$

The intercept is $\theta_0$.

If we define $x_0=1$, the model can be written compactly as:

$$
\boxed{
\hat y
=
\boldsymbol{\theta}^{\mathsf T}\mathbf x
}
$$

where $\boldsymbol{\theta}$ contains the intercept and feature coefficients.

---

## 2. Mean Squared Error

Linear Regression is usually trained by minimizing Mean Squared Error.

For $m$ training samples:

$$
\operatorname{MSE}(\boldsymbol{\theta})
=
\frac{1}{m}
\sum_{i=1}^{m}
\left(
\boldsymbol{\theta}^{\mathsf T}\mathbf x^{(i)}
-
y^{(i)}
\right)^2
$$

MSE gives extra weight to large errors because each residual is squared.

The training objective is:

$$
\hat{\boldsymbol{\theta}}
=
\arg\min_{\boldsymbol{\theta}}
\operatorname{MSE}(\boldsymbol{\theta})
$$

The MSE objective for Linear Regression is convex. Therefore, it has no bad local minima: every local minimum is also a global minimum.

---

## 3. Training with scikit-learn

```python
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

predictions = linear_model.predict(X_test)

print(linear_model.intercept_)
print(linear_model.coef_)
```

The learned parameters are stored in:

- `intercept_` for $\theta_0$
- `coef_` for the remaining coefficients

---

## 4. The Normal Equation

Linear Regression has a direct mathematical solution.

Let $\mathbf X$ be the design matrix and $\mathbf y$ be the vector of target values.

Then the Normal Equation is:

$$
\boxed{
\hat{\boldsymbol{\theta}}
=
(\mathbf X^{\mathsf T}\mathbf X)^{-1}
\mathbf X^{\mathsf T}\mathbf y
}
$$

In NumPy:

```python
import numpy as np

X_b = np.c_[np.ones((len(X), 1)), X]

theta_normal = (
    np.linalg.inv(X_b.T @ X_b)
    @ X_b.T
    @ y
)
```

This method does not require iterative optimization.

However, it requires $\mathbf X^{\mathsf T}\mathbf X$ to be invertible and can be numerically unstable.

---

## 5. The Pseudoinverse

If features are redundant or highly correlated, then $\mathbf X^{\mathsf T}\mathbf X$ may be singular or poorly conditioned.

A more stable solution uses the Moore–Penrose pseudoinverse:

$$
\boxed{
\hat{\boldsymbol{\theta}}
=
\mathbf X^{+}\mathbf y
}
$$

In NumPy:

```python
theta_pinv = np.linalg.pinv(X_b) @ y
```

The pseudoinverse is computed using Singular Value Decomposition (SVD).

If:

$$
\mathbf X
=
\mathbf U\boldsymbol{\Sigma}\mathbf V^{\mathsf T}
$$

then:

$$
\mathbf X^{+}
=
\mathbf V
\boldsymbol{\Sigma}^{+}
\mathbf U^{\mathsf T}
$$

The pseudoinverse:

- works when the matrix is singular
- is more numerically stable than explicitly computing an inverse
- returns the minimum-norm least-squares solution when several solutions exist

`LinearRegression` uses a stable least-squares solver rather than explicitly evaluating the Normal Equation.

---

## 6. Direct Solutions vs Iterative Optimization

The direct solution works well when:

- the number of features is moderate
- the dataset fits comfortably in memory
- an exact least-squares solution is wanted

Gradient-based methods are often preferred when:

- the number of features is very large
- the number of samples is very large
- data arrives continuously
- the model has no closed-form solution

---

## 7. Gradient Descent

Gradient Descent starts with an initial parameter vector and repeatedly moves toward lower loss.

The gradient:

$$
\nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta})
$$

points in the direction of the fastest increase.

Therefore, Gradient Descent moves in the opposite direction:

$$
\boxed{
\boldsymbol{\theta}
\leftarrow
\boldsymbol{\theta}
-
\eta
\nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta})
}
$$

where $\eta$ is the **learning rate**.

---

## 8. Learning Rate

The learning rate controls the size of each update.

### Learning rate too small

- training is stable
- progress is slow
- many iterations are required

### Learning rate too large

- the algorithm can jump across the minimum
- the loss may oscillate
- the optimization may diverge

The learning rate is therefore one of the most important optimization hyperparameters.

---

## 9. Feature Scaling and Gradient Descent

When features have very different scales, the MSE surface becomes elongated.

Gradient Descent may then zigzag toward the minimum and converge slowly.

Standardization transforms a feature using:

$$
x_{\text{scaled}}
=
\frac{x-\mu}{\sigma}
$$

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
```

Scaling is especially important for Gradient Descent and regularized models.

---

## 10. Batch Gradient Descent

Batch Gradient Descent uses the complete training set for every update.

For Linear Regression:

$$
\nabla_{\boldsymbol{\theta}}
\operatorname{MSE}(\boldsymbol{\theta})
=
\frac{2}{m}
\mathbf X^{\mathsf T}
(\mathbf X\boldsymbol{\theta}-\mathbf y)
$$

Example:

```python
theta = np.zeros(X_b.shape[1])
learning_rate = 0.1

for epoch in range(1000):
    residuals = X_b @ theta - y
    gradient = 2 / len(X_b) * X_b.T @ residuals
    theta -= learning_rate * gradient
```

Advantages:

- stable updates
- smooth convergence
- accurate gradient

Disadvantages:

- every update uses the entire dataset
- slow for very large datasets
- cannot naturally learn from an endless stream

---

## 11. Stochastic Gradient Descent

Stochastic Gradient Descent uses one randomly selected training sample for each update.

Instead of computing the exact gradient, it computes a noisy estimate.

Advantages:

- very fast updates
- suitable for large datasets
- supports online learning
- noise can help escape shallow local minima in non-convex problems

Disadvantages:

- the loss fluctuates
- the final parameters keep moving around the minimum
- the result depends on the learning schedule and random order

```python
from sklearn.linear_model import SGDRegressor

sgd_regressor = SGDRegressor(
    max_iter=1000,
    tol=1e-5,
    penalty=None,
    eta0=0.1,
    random_state=42,
)

sgd_regressor.fit(X_train_scaled, y_train)
```

---

## 12. Learning Schedules

SGD should usually begin with relatively large steps and gradually reduce them.

A common schedule is:

$$
\eta(t)
=
\frac{t_0}{t+t_1}
$$

where $t$ is the iteration number.

If the learning rate decreases too quickly, training may stop before reaching a good solution. If it decreases too slowly, the parameters may continue moving around the minimum.

---

## 13. Mini-batch Gradient Descent

Mini-batch Gradient Descent computes each update from a small batch of samples.

Typical batch sizes include:

$$
16,
32,
64,
128,
256
$$

Advantages:

- more stable than SGD
- faster updates than Batch GD
- efficient matrix operations
- works well with GPU acceleration

Disadvantages:

- introduces the batch-size hyperparameter
- still produces noisy updates

Mini-batch Gradient Descent is the standard optimization approach for modern neural networks.

---

## 14. Comparing Gradient Descent Methods

| Method | Samples per update | Main advantage | Main disadvantage |
|---|---:|---|---|
| Batch GD | all samples | stable and accurate gradient | expensive updates |
| SGD | one sample | fast and supports online learning | highly noisy path |
| Mini-batch GD | a small batch | good speed–stability balance | requires batch-size tuning |

All three methods can approach the same optimum for a convex objective, but they take different paths.

---

## 15. Polynomial Regression

Linear models can fit nonlinear relationships by transforming the input features.

For one original feature $x$, a degree-two transformation creates:

$$
x
\longrightarrow
(x,x^2)
$$

The model becomes:

$$
\hat y
=
\theta_0
+
\theta_1x
+
\theta_2x^2
$$

This model is nonlinear in $x$, but it is still linear in its parameters.

```python
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

polynomial_model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression(),
)

polynomial_model.fit(X_train, y_train)
```

For two original features $a$ and $b$, degree-two polynomial features include:

$$
a,
b,
a^2,
ab,
b^2
$$

The number of generated features can grow rapidly as the degree and original feature count increase.

---

## 16. Model Complexity

Polynomial degree controls model complexity.

### Degree too low

- model is too simple
- training error is high
- validation error is high
- this is underfitting or high bias

### Degree too high

- model can follow noise in the training data
- training error becomes very low
- validation error remains much higher
- this is overfitting or high variance

Model selection should use validation data or cross-validation, not training error alone.

---

## 17. Learning Curves

A learning curve plots model error as the training-set size increases.

Usually, it shows training error and validation error.

### Underfitting pattern

- training and validation errors are both high
- the two curves are close together
- adding more data is unlikely to solve the main problem

Possible solutions:

- use a more expressive model
- add better features
- reduce regularization

### Overfitting pattern

- training error is low
- validation error is clearly higher
- the gap between the curves is large

Possible solutions:

- collect more training data
- simplify the model
- reduce the polynomial degree
- increase regularization

---

## 18. Regularized Linear Models

Regularization reduces overfitting by restricting the model parameters.

The general objective becomes:

$$
\boxed{
J(\boldsymbol{\theta})
=
\text{data loss}
+
\text{regularization penalty}
}
$$

The intercept is normally not regularized.

Because penalties depend on coefficient size, features should usually be scaled before regularized regression.

---

## 19. Ridge Regression

Ridge Regression adds an $\ell_2$ penalty:

$$
J(\boldsymbol{\theta})
=
\operatorname{MSE}(\boldsymbol{\theta})
+
\alpha
\sum_{j=1}^{n}
\theta_j^2
$$

The hyperparameter $\alpha$ controls regularization strength.

### If $\alpha=0$

Ridge becomes ordinary Linear Regression.

### If $\alpha$ is large

- coefficients are strongly reduced
- the model becomes smoother
- variance decreases
- bias increases

Ridge usually shrinks coefficients toward zero without making them exactly zero.

```python
from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
```

---

## 20. Lasso Regression

Lasso Regression adds an $\ell_1$ penalty:

$$
J(\boldsymbol{\theta})
=
\operatorname{MSE}(\boldsymbol{\theta})
+
\alpha
\sum_{j=1}^{n}
|\theta_j|
$$

Lasso tends to make some coefficients exactly zero.

Therefore, it performs a form of automatic feature selection and produces a sparse model.

```python
from sklearn.linear_model import Lasso

lasso_model = Lasso(
    alpha=0.05,
    max_iter=20000,
)

lasso_model.fit(X_train_scaled, y_train)
```

Lasso can behave unstably when several features are strongly correlated because it may select one feature and suppress the others.

---

## 21. Elastic Net

Elastic Net combines Ridge and Lasso penalties.

$$
J(\boldsymbol{\theta})
=
\operatorname{MSE}(\boldsymbol{\theta})
+
r\alpha
\sum_{j=1}^{n}|\theta_j|
+
\frac{1-r}{2}\alpha
\sum_{j=1}^{n}\theta_j^2
$$

In scikit-learn, $r=\texttt{l1\_ratio}$.

- $r=1$ gives Lasso
- $r=0$ gives an $\ell_2$-only penalty
- $0<r<1$ combines both penalties

```python
from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(
    alpha=0.05,
    l1_ratio=0.5,
    max_iter=20000,
)

elastic_net.fit(X_train_scaled, y_train)
```

Elastic Net is often safer than pure Lasso when features are correlated.

---

## 22. Ridge vs Lasso vs Elastic Net

| Model | Penalty | Main behavior |
|---|---|---|
| Ridge | $\ell_2$ | smoothly shrinks all coefficients |
| Lasso | $\ell_1$ | can set coefficients exactly to zero |
| Elastic Net | $\ell_1+\ell_2$ | combines sparsity with greater stability |

A practical default is:

- prefer Ridge when most features are expected to contribute
- prefer Lasso when only a small subset of features should matter
- prefer Elastic Net when sparsity is useful but features may be correlated

---

## 23. Early Stopping

Early stopping is a regularization method for iterative learning algorithms.

The procedure is:

1. train for one epoch
2. measure validation error
3. save the model if validation error improves
4. stop after validation error stops improving
5. restore the best saved model

At first, both training error and validation error decrease.

After overfitting begins, training error continues to decrease while validation error increases.

The best model is the checkpoint with the lowest validation error.

The validation set chooses the stopping point. The test set must remain untouched until final evaluation.

---

## 24. Logistic Regression

Despite its name, Logistic Regression is a classification model.

It first computes a linear score:

$$
z
=
\boldsymbol{\theta}^{\mathsf T}\mathbf x
$$

It then transforms the score into a probability using the sigmoid function:

$$
\boxed{
\sigma(z)
=
\frac{1}{1+e^{-z}}
}
$$

Therefore:

$$
\hat p
=
P(y=1\mid\mathbf x)
=
\sigma(\boldsymbol{\theta}^{\mathsf T}\mathbf x)
$$

The standard decision rule is:

$$
\hat y
=
\begin{cases}
1 & \text{if } \hat p\geq0.5 \\
0 & \text{if } \hat p<0.5
\end{cases}
$$

The probability threshold can be changed when false positives and false negatives have different costs.

---

## 25. Logistic Loss

Logistic Regression is trained using binary cross-entropy, also called log loss.

$$
J(\boldsymbol{\theta})
=
-\frac{1}{m}
\sum_{i=1}^{m}
\left[
y^{(i)}\log\hat p^{(i)}
+
(1-y^{(i)})\log(1-\hat p^{(i)})
\right]
$$

Confident incorrect predictions receive a very large penalty.

The objective is convex, so optimization can find the global minimum.

---

## 26. Logistic Regression in scikit-learn

```python
from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression(
    C=1.0,
    random_state=42,
)

logistic_model.fit(X_train, y_train)

classes = logistic_model.predict(X_test)
probabilities = logistic_model.predict_proba(X_test)
```

The parameter `C` is the inverse of regularization strength.

- small `C` means stronger regularization
- large `C` means weaker regularization

---

## 27. Decision Boundaries

Logistic Regression creates a linear decision boundary in the input feature space.

The boundary for threshold $0.5$ occurs where:

$$
\boldsymbol{\theta}^{\mathsf T}\mathbf x=0
$$

Polynomial features can create nonlinear boundaries, while the classifier remains linear in the transformed feature space.

---

## 28. Softmax Regression

Softmax Regression extends Logistic Regression to multiple mutually exclusive classes.

For each class $k$, the model calculates a score:

$$
s_k(\mathbf x)
=
\boldsymbol{\theta}_k^{\mathsf T}\mathbf x
$$

The softmax function converts all class scores into probabilities:

$$
\boxed{
\hat p_k
=
\frac{
e^{s_k(\mathbf x)}
}{
\sum_{j=1}^{K}e^{s_j(\mathbf x)}
}
}
$$

The probabilities satisfy:

$$
0\leq\hat p_k\leq1
$$

and:

$$
\sum_{k=1}^{K}\hat p_k=1
$$

The predicted class is:

$$
\hat y
=
\arg\max_k
\hat p_k
$$

---

## 29. Cross-Entropy Loss

Softmax Regression uses multiclass cross-entropy:

$$
J(\boldsymbol{\Theta})
=
-\frac{1}{m}
\sum_{i=1}^{m}
\sum_{k=1}^{K}
y_k^{(i)}
\log\left(\hat p_k^{(i)}\right)
$$

The loss is small when the true class receives high probability.

---

## 30. Softmax Regression in scikit-learn

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

softmax_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        C=10,
        random_state=42,
    ),
)

softmax_model.fit(X_train, y_train)

predictions = softmax_model.predict(X_test)
probabilities = softmax_model.predict_proba(X_test)
```

Softmax Regression assumes each sample belongs to exactly one class.

It is not appropriate for multilabel classification, where several labels may be true simultaneously.

---

## 31. Model Connections

| Model | Output | Typical loss | Training method |
|---|---|---|---|
| Linear Regression | continuous value | MSE | direct solution or Gradient Descent |
| Ridge / Lasso / Elastic Net | continuous value | MSE + penalty | direct or iterative optimization |
| Logistic Regression | binary probability | binary cross-entropy | iterative optimization |
| Softmax Regression | multiclass probabilities | multiclass cross-entropy | iterative optimization |

The common pattern is:

$$
\text{parameters}
\rightarrow
\text{predictions}
\rightarrow
\text{loss}
\rightarrow
\text{optimization}
$$

---

## 32. Practical Workflow

A practical training workflow is:

1. separate training, validation and test data
2. choose a model and loss function
3. put preprocessing inside a pipeline
4. scale features when optimization or regularization requires it
5. train using a direct solver or Gradient Descent
6. inspect training and validation errors
7. use learning curves to diagnose bias and variance
8. tune model complexity and regularization
9. use early stopping for iterative models when appropriate
10. evaluate once on the untouched test set

The test set should not choose:

- polynomial degree
- regularization strength
- learning rate
- number of epochs
- probability threshold

These choices belong to the training and validation process.

---

## Key Takeaways

The main optimization objective is:

$$
\boldsymbol{\theta}^{*}
=
\arg\min_{\boldsymbol{\theta}}
\left(
\text{data loss}
+
\text{regularization penalty}
\right)
$$

Important ideas:

- Linear Regression minimizes prediction error for continuous targets.
- The Normal Equation gives a direct solution, but the pseudoinverse is more stable.
- Batch GD, SGD and Mini-batch GD trade gradient accuracy for computational speed.
- Feature scaling makes Gradient Descent and regularization behave more predictably.
- Polynomial features allow linear models to represent nonlinear relationships.
- Learning curves distinguish high bias from high variance.
- Ridge shrinks coefficients, Lasso creates sparse models and Elastic Net combines both behaviors.
- Early stopping limits overfitting by keeping the best validation checkpoint.
- Logistic Regression models binary probabilities using the sigmoid function.
- Softmax Regression models mutually exclusive multiclass probabilities.
- Validation data guides model choices; the test set is reserved for final evaluation.
