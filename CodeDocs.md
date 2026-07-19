# Code Doc for Midterm 

## Table of Contents
- [Code Doc for Midterm](#code-doc-for-midterm)
  - [Table of Contents](#table-of-contents)
  - [Use](#use)
  - [Data Preprocessing](#data-preprocessing)
  - [Stratify](#stratify)
  - [Pipeline](#pipeline)
  - [GridSearchCV](#gridsearchcv)
  - [Helper Functions](#helper-functions)
    - [Evaluate](#evaluate)
    - [Scorer](#scorer)
    - [GridSearch Summarizer](#gridsearch-summarizer)
  - [Questions](#questions)
    - [1. Which model performed best overall? Define what you mean by “best” and justify your choice using more than one metric.](#1-which-model-performed-best-overall-define-what-you-mean-by-best-and-justify-your-choice-using-more-than-one-metric)
    - [2. Which model achieved the highest recall for malignant cases? Why is this metric especially important for this task?](#2-which-model-achieved-the-highest-recall-for-malignant-cases-why-is-this-metric-especially-important-for-this-task)
    - [3. How did scaling affect logistic regression, SVM, and KNN/Decision Tree? Explain the observed changes using the way each algorithm works.](#3-how-did-scaling-affect-logistic-regression-svm-and-knndecision-tree-explain-the-observed-changes-using-the-way-each-algorithm-works)
    - [4. For the Decision Tree, how did maxDepth affect training and validation/cross-validation performance? What methods can be used to control overfitting?](#4-for-the-decision-tree-how-did-maxdepth-affect-training-and-validationcross-validation-performance-what-methods-can-be-used-to-control-overfitting)
    - [5. How does a Decision Tree select a feature and split point at each node? Explain the role of an impurity criterion such as Gini impurity or entropy.](#5-how-does-a-decision-tree-select-a-feature-and-split-point-at-each-node-explain-the-role-of-an-impurity-criterion-such-as-gini-impurity-or-entropy)
    - [6. For KNN, how did you select k? Explain why a very small k can overfit and why a very large k can underfit.](#6-for-knn-how-did-you-select-k-explain-why-a-very-small-k-can-overfit-and-why-a-very-large-k-can-underfit)
    - [7. For each of the five classifiers, explain the main assumptions or inductive biases, the types of decision boundaries it can represent, important sensitivities (for example, scaling, noise, outliers, correlated features, dimensionality, or hyperparameters), and situations in which performance may degrade.](#7-for-each-of-the-five-classifiers-explain-the-main-assumptions-or-inductive-biases-the-types-of-decision-boundaries-it-can-represent-important-sensitivities-for-example-scaling-noise-outliers-correlated-features-dimensionality-or-hyperparameters-and-situations-in-which-performance-may-degrade)


## Use
This is the docs the midterm project. This will be use to explain functions, steps used, and other topics we maybe questioned on as part of our project. Please add in any notes you see fit.

## Data Preprocessing
In this step we loaded the breast cancer dataset and and split it 
using train_test_split. Stratify was step to yes(explained in the next section)

## Stratify
Since some classification tasks can exhibit rare classes such has with the cancer dataset(cancer being rare), Splitting can result in training or validation sets without any occurrence of a particular class.
This can led to errors in code during training. To mitigate such problems, splitters such as `StratifiedKFold` and `StratifiedShuffleSplit` implement stratified sampling to ensure that relative class frequencies are approximately preserved in each fold.
`StratifiedKFold` returns sets that contain approximately the same percentage of samples of each target class as the complete set.

## Pipeline
A pipeline chains a sequence of preprocessing steps and a final model into one object that behaves like a single estimator. Every step except the last must be a transformer (has fit and transform: scalers, PCA, feature selectors). The last step is the estimator (the classifier).
```
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", KNeighborsClassifier()),
])
```

Once built it has normal estimator interface:`pipe.fit(X, y)`, `pipe.predict(X)`, `pipe.score(X, y)`.

This is used because preprocessing statistics are always computed from training data only — including inside every cross-validation fold, so there is no leakage between training and validation data. 

## GridSearchCV

GridSearchCV performs an exhaustive search over a grid of hyperparameter values, evaluating each combination with cross-validation on the __training set only__, then refits the best combination on the full training set.
```
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

knn_grid = GridSearchCV(
    estimator=Pipeline([("scaler", StandardScaler()),
                        ("clf", KNeighborsClassifier())]),
    param_grid={
        "clf__n_neighbors": [3, 5, 7, 9, 11, 13],
        "clf__weights": ["uniform", "distance"],
    },
    cv=skf,
    scoring=make_scorer(recall_score, pos_label=0),  # malignant recall
    n_jobs=-1,          # use all CPU cores
)
```
With 6 values of k × 2 weight options × 5 folds, this trains 60 models, plus one final refit.

__For every point on the grid, GridSearchCV:__

    1. Splits the training data into cv folds.
    2. For each fold: fits the estimator on the other folds, scores it on the held-out fold.
    3. Averages the fold scores → that combination's mean_test_score.

__Attributes__
    
    1. best_params_ : dict of the winning hyperparameters
    2. best_score_ : mean CV score of the winner
    3. best_estimator_ : the refitted pipeline — pass this to evaluate()
    4. cv_results_ : full results table for every combination
## Helper Functions
### Evaluate 
This helper predicts y to make a confusion matrix. It displays the confusion matrix and classification report. It also returns model metrics; ensure you save them for the comparison chart at the end.

Input parameters: model see example below, name : String
```
Knn_grid = GridSearchCV(...)
evaluate(knn_grid.best_estimator_, "K-Nearest Neighbors")

```
### Scorer

### GridSearch Summarizer

## Questions

### 1. Which model performed best overall? Define what you mean by “best” and justify your choice using more than one metric.

We decided to use a "weighted recall" by taking a portion of the malignant and benign recalls. We took a larger portion of the malignant recall since we decided to place an emphasis on preventing false negatives.

###  2. Which model achieved the highest recall for malignant cases? Why is this metric especially important for this task?

We had a tie between Logistic Regression, SVM, and Decision tree with a value of (0.9762). This  metric was important because it measures false negatives. In a medical context, a false negative may lead to someone not receiving treatment. This is very serious with a disease such as cancer.

### 3. How did scaling affect logistic regression, SVM, and KNN/Decision Tree? Explain the observed changes using the way each algorithm works.

###  4. For the Decision Tree, how did maxDepth affect training and validation/cross-validation performance? What methods can be used to control overfitting?

###  5. How does a Decision Tree select a feature and split point at each node? Explain the role of an impurity criterion such as Gini impurity or entropy.

###  6. For KNN, how did you select k? Explain why a very small k can overfit and why a very large k can underfit.

### 7. For each of the five classifiers, explain the main assumptions or inductive biases, the types of decision boundaries it can represent, important sensitivities (for example, scaling, noise, outliers, correlated features, dimensionality, or hyperparameters), and situations in which performance may degrade.

