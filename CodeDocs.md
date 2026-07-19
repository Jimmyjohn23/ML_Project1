# Code Doc for Midterm 

## Table of Contents
- [Code Doc for Midterm](#code-doc-for-midterm)
  - [Table of Contents](#table-of-contents)
  - [Use](#use)
  - [Data Preprocessing](#data-preprocessing)
  - [Stratify](#stratify)
  - [Pipeline](#pipeline)
  - [GridSearchCV](#gridsearchcv)
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

## Questions

### 1. Which model performed best overall? Define what you mean by “best” and justify your choice using more than one metric.

We decided to use a "weighted recall" by taking a portion of the malignant and benign recalls. We took a larger portion of the malignant recall since we decided to place an emphasis on preventing false negatives.

###  2. Which model achieved the highest recall for malignant cases? Why is this metric especially important for this task?

We had a tie between Logistic Regression, SVM, and Decision tree with a value of (0.9762). This  metric was important because it measures false negatives. In a medical context, a false negative may lead to someone not receiving treatment. This is very serious with a disease such as cancer.

### 3. How did scaling affect logistic regression, SVM, and KNN/Decision Tree? Explain the observed changes using the way each algorithm works.

Scaling had a large effect on Logistic Regression, SVM, and KNN because these models use the size of the feature values during training or prediction. In this dataset, some features have much larger number ranges than others. For example, area values can be much larger than smoothness values. If scaling is not used, the model may treat the larger-numbered features as more important even if they are not actually more useful.

For Logistic Regression, StandardScaler gave the best result. Logistic Regression learns coefficients for each feature, and scaling helps the solver converge better. It also makes the regularization penalty treat each feature more fairly because all features are on a similar scale.

For SVM, scaling was also important. The best SVM used MinMaxScaler with the rbf kernel. SVM tries to separate points using distances and margins, so features with large ranges can dominate the boundary if they are not scaled. The rbf kernel is especially sensitive to scale because gamma controls how far the influence of each point reaches.

For KNN, MinMaxScaler gave the best result. KNN classifies a point by looking at nearby points, so distance is the main idea behind the algorithm. If one feature has a very large range, it can dominate the distance calculation. Scaling keeps each feature from unfairly controlling the neighbor search.

For Decision Tree, scaling did not really affect the outcome. A Decision Tree does not use distance or gradient-based weight updates. Instead, it chooses feature thresholds such as "feature <= value". If a feature is scaled, the threshold value changes, but the ordering of the samples usually stays the same. Because of this, Decision Trees are mostly scale-invariant.

###  4. For the Decision Tree, how did maxDepth affect training and validation/cross-validation performance? What methods can be used to control overfitting?

MaxDepth controls how many levels the tree is allowed to grow. If maxDepth is very small, the tree is simple and may underfit. This means it may not capture enough patterns in the data. If maxDepth is very large or set to none, the tree can keep splitting until it captures many details in the training set. This can improve training performance, but it can also overfit and perform worse on validation or cross-validation data.

In our grid search, we tested maxDepth values of none, 3, 5, and 10. The best Decision Tree selected maxDepth as none, but it also selected min_samples_leaf = 20. This means the tree was technically allowed to grow deep, but it could not create leaves with very few samples. That helped control overfitting because the tree could not make very specific branches for only one or two training examples.

There are multiple ways to control overfitting in a Decision Tree. One way is to limit maxDepth so the tree cannot become too complex. Another way is to increase min_samples_leaf or min_samples_split so each split needs enough data to be useful. We also tested ccp_alpha, which is used for cost-complexity pruning. Pruning removes branches that do not improve the model enough. Class weights can also be tested when the dataset is imbalanced, but in our best model the default class weights were selected.

###  5. How does a Decision Tree select a feature and split point at each node? Explain the role of an impurity criterion such as Gini impurity or entropy.

A Decision Tree selects a feature and split point by testing possible splits and choosing the one that makes the child groups more pure. A pure group means most or all of the samples in that group belong to the same class. In this project, that means a split is better if it separates malignant and benign samples more clearly.

At each node, the tree looks through the features and possible threshold values. For example, it may test a rule such as "worst radius <= 16.8". That rule divides the data into a left branch and a right branch. The tree then measures how mixed the classes are after the split. The split that reduces impurity the most is selected.

Gini impurity and entropy are two ways to measure how mixed a node is. Gini impurity is low when most samples in a node belong to one class and high when the classes are mixed. Entropy works similarly, but it comes from information theory. A split with high information gain, or large impurity reduction, is preferred. In our grid search, the best Decision Tree selected the gini criterion, so it used Gini impurity to decide which splits were best.

###  6. For KNN, how did you select k? Explain why a very small k can overfit and why a very large k can underfit.

When choosing K, we had grid searched odd values ranging from 3 - 13. We decided odd numbers so that our model has a majority vote on the class output, where as an even number of K's could leave the modle unsure which lable to assign from a voting tie. Using 5 fold cross validation on the training set with our weighted recall scorer, we found k = 7 was the best value for our model.

    K is important because it controls the bias-varriance trade off. A small K like k = 1  will copy, and essentially, memorize the training data set, causing overfitting. this comes from high vvarriance  and low bias, which results in a complex and jagged decision boundary. Contrairily, too large of a decision boundary underfits due to high bias and low varriance. In this case, the modle will predict the majority class of the entire data set for every single input, casuing a smoothed decsison boundary. In extreme cases, every prediction is just going to produce "bening" for everyone due to high bias. K=7 deemed to be the sweet spot from our cross validaation, where it is big enough to vote away noise, while small enough to stay local.


### 7. For each of the five classifiers, explain the main assumptions or inductive biases, the types of decision boundaries it can represent, important sensitivities (for example, scaling, noise, outliers, correlated features, dimensionality, or hyperparameters), and situations in which performance may degrade.


Three of our models share hyperplane decision boundaries, but differ on how they are found. Logisic regression computes a weighted sum of the features and passes it through the sigmoid fuction to produce the prob that a sample is "bening". 

Logistic regression, percetron, and SVM with a linear kernel all draw a hyperplane between the maglignant and bennign classes, but are found differnetly. Logistic regression computes a weighted sum of features and passes them through a sigmoid function to find the probability of a sample being "benign". Its main sensitives are feature scalling and regularization strength C (1/lambda. higher C, more complex model). We found this out when an over-regularized model collapsed into predicting every sample as malignant. The perceptron assumes data is linearly seperable initially. Since our classes overlap slightly, the update rule never fully settled on a clean boundary, which we beileve generated six false positives. It also provides no probabilyt estimates, just hard a label. 

The SVM instead looks for the boundary with the maximum margin, and with the RBF kernel it can curve that boundary. It was the most scaling-sensitive model we tested: the unscaled mean CV score was around 0.79 compared to roughly 0.95 when scaled, since the kernel's distance calculations get dominated by large-range features like area. Naive Bayes assumes every feature is independent given the class, which is not really true for this dataset (radius, perimeter, and area essentially measure the same thing), so it ends up counting the same evidence multiple times. We believe this is why it had the lowest malignant recall (0.9048). On the plus side, it is unaffected by scaling and trains almost instantly.

The decision tree and KNN make no assumption about the overall shape of the boundary. The tree greedily picks whichever feature and threshold reduce impurity (Gini or entropy) the most at each node, so its boundary is built from axis-aligned rectangles. Scaling has no effect on it, but it overfits easily; without our minimum leaf size of 20 it would essentially memorize the training set. Even tuned, it had the lowest accuracy (0.9035) and the largest variation between CV folds, and it struggles to represent diagonal boundaries since it can only approximate them with a staircase of splits. KNN assumes that points near each other share a label. It does not really train at all; it stores the data and measures distances at prediction time, which makes scaling and the choice of k the two settings it depends on most. It also degrades in high dimensions, where every point ends up about the same distance from every other point, and irrelevant features corrupt the distances. Overall, the ranking we observed makes sense given the data: the classes are close to linearly separable, many features are correlated, and there are only 455 training samples, which favors the simpler linear models and works against the high-variance tree and the independence-assuming Naive Bayes.

