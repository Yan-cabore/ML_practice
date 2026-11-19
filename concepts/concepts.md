# Concepts Explored with Scikit-learn

This document contains concepts explored during my initial experiments with Scikit-learn, using datasets such as Iris.

The purpose is to document the concepts, methods, and tools encountered throughout the learning process, along with practical examples and explanations of how they work.

## 1. Fitting (`fit`)

*Fitting* is the process of adjusting a model to the training data, allowing it to learn the patterns and relationships present in that data.

In Scikit-learn, we use the `fit()` method to perform this process, providing the features (`X`) and the expected outputs (`y`).

```python
model.fit(X, y)
```

During this process, the algorithm adjusts its parameters based on the provided data.

## 2. Predicting (`predict`)

*Predicting* is the process of making predictions using a model that has already gone through the fitting process.

At this stage, we provide new data to the model so it can use the patterns learned during training to generate a prediction.

In Scikit-learn, we use the `predict()` method:

```python
prediction = model.predict(X_new)
```

## 3. Random State (`random_state`)

`random_state` is a parameter used to control randomness in certain processes during machine learning.

Algorithms such as `RandomForestClassifier` use random operations during execution. Therefore, different executions may produce different results.

By defining a `random_state`, we fix the seed of the pseudorandom number generator, allowing us to reproduce the same results across different executions, provided the other conditions remain the same.

```python
model = RandomForestClassifier(random_state=42)
```

The value `42` is just a popular convention. Any integer can be used.

**Important:** `random_state` does not directly improve model accuracy. Its main purpose is to ensure the reproducibility of experiments.

## 4. Transformers and Pre-processors

Transformers are components responsible for modifying the representation of input data, making it more suitable for subsequent machine learning operations.

Unlike predictive estimators, transformers do not use the `predict()` method. Instead, they use `transform()` to return a modified version of the input data.

A common example is `StandardScaler`, which standardizes numerical features.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_transformed = scaler.fit_transform(X)
```

### 4.1. Scalers

Scalers are a type of transformer used to modify the scale of numerical features.

Different features may have different ranges of values. Scaling helps put them on comparable scales, which can be useful for algorithms that are sensitive to feature magnitudes.

Some examples include:

- `StandardScaler`: standardizes features using their mean and standard deviation.
- `MinMaxScaler`: scales features to a specified range, usually between 0 and 1.
- `Normalizer`: normalizes individual samples using a specified norm.

### 4.2. Fitting and Transforming (`fit_transform`)

The `fit_transform()` method combines two operations into a single call.

First, `fit()` learns the parameters required for the transformation. Then, `transform()` applies the learned parameters to the input data.

For example, when using `StandardScaler`:

```python
scaler.fit(X)
X_transformed = scaler.transform(X)
```

This is equivalent to:

```python
X_transformed = scaler.fit_transform(X)
```

In the case of `StandardScaler`, the fitting process calculates the mean and standard deviation of each feature.

The transformation then applies the following formula:

\[
z = \frac{x-\mu}{\sigma}
\]

Where:

- `x`: original value.
- `μ`: feature mean.
- `σ`: feature standard deviation.
- `z`: standardized value.

**Important:** When working with training and test sets, `fit_transform()` should be applied to the training data, while `transform()` should be applied to the test data using the parameters learned from the training set.

## 5. Pipelines

A Pipeline is an object that combines multiple processing steps into a single workflow.

It allows transformers and predictive estimators to be chained together, making it easier to organize and execute machine learning operations.

For example, we can combine a `StandardScaler` with a `LogisticRegression` model:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)
```

When calling `fit()`, the pipeline executes the steps in sequence:

1. The `StandardScaler` learns the scaling parameters and transforms the training data.
2. The `LogisticRegression` model receives the transformed data and is fitted.

We can then use the pipeline to make predictions:

```python
pipeline.fit(X_train, y_train)

prediction = pipeline.predict(X_test)
```

**Important:** Pipelines help prevent data leakage by ensuring that preprocessing steps are fitted using only the training data during model training and evaluation.

## 6. Model Evaluation

Model evaluation is the process of assessing how well a trained model performs when making predictions on data.

Fitting a model does not necessarily mean that it will perform well on new, unseen data. Therefore, its performance needs to be evaluated.

### 6.1. Train and Test Sets

A common approach is to divide the dataset into two subsets:

- **Training set:** used to fit the model.
- **Test set:** used to evaluate the model using data that was not included during training.

In Scikit-learn, we can use `train_test_split()` to perform this division.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

In this example, `test_size=0.2` reserves 20% of the dataset for testing, while the remaining 80% is used for training.

### 6.2. Accuracy

Accuracy is a classification metric that measures the proportion of correct predictions made by a model.

It is calculated by dividing the number of correct predictions by the total number of predictions.

\[
Accuracy = \frac{\text{Correct Predictions}}{\text{Total Predictions}}
\]

In Scikit-learn, we can calculate accuracy using `accuracy_score()`:

```python
from sklearn.metrics import accuracy_score

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print(accuracy)
```

An accuracy of `0.95`, for example, means that the model correctly classified 95% of the evaluated samples.

### 6.3. Cross-Validation

Cross-validation is an evaluation technique that divides the dataset into multiple subsets, called folds.

In a typical 5-fold cross-validation:

1. The dataset is divided into five subsets.
2. The model is trained using four subsets.
3. The remaining subset is used for evaluation.
4. This process is repeated five times, changing the subset used for evaluation.

The final result provides multiple evaluation scores, helping us understand how consistently the model performs across different data splits.

In Scikit-learn, we can use `cross_validate()`:

```python
from sklearn.model_selection import cross_validate

results = cross_validate(model, X, y, cv=5)

print(results['test_score'])
```

Here, `cv=5` specifies that five folds will be used.

## 7. Automatic Parameter Searches

Machine learning estimators have parameters that influence how they learn from data. Some of these parameters are called *hyperparameters* because they are configured before the model fitting process.

For example, `RandomForestClassifier` has hyperparameters such as:

- `n_estimators`: determines the number of decision trees in the forest.
- `max_depth`: determines the maximum depth of each decision tree.

Choosing appropriate hyperparameter values can influence the model's performance.

Scikit-learn provides tools to automatically search for suitable hyperparameter combinations.

### 7.1. RandomizedSearchCV

`RandomizedSearchCV` is a tool that evaluates randomly selected hyperparameter combinations using cross-validation.

Instead of testing every possible combination, it samples a specified number of configurations from the parameter search space.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

model = RandomForestClassifier(random_state=42)

param_distributions = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None, 5, 10, 15]
}

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=5,
    cv=5,
    random_state=42
)

search.fit(X_train, y_train)
```

After the search is completed, we can access the best hyperparameter combination:

```python
print(search.best_params_)
```

We can also access the best estimator found during the search:

```python
best_model = search.best_estimator_
```

**Important:** Hyperparameter searches should be performed using training data. The test set should remain separate for the final evaluation of the selected model.