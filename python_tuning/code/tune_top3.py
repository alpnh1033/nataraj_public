"""
Tunes the top 3 classifiers picked from the Orange screen (by confusion-
matrix behavior, not just accuracy -- see the paper's Methods 2.4):
Logistic Regression, SVM, Neural Network.

Methodology, identical for all three: grid search with 5-fold cross-
validation on training.npz only; the same-source (testing.npz) and
cross-source (cross_testing.npz) test sets are each touched exactly once,
after tuning -- never used to pick hyperparameters.
"""
import warnings
from pathlib import Path
import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=ConvergenceWarning)

FEATURES_DIR = Path("PATH/TO/final-dataset/features")  # <-- point this at your own copy
MODELS_DIR = Path("../model")

# Each entry pairs a model with the parameter settings GridSearchCV will try.
GRIDS = {
    # C = regularization strength. Smaller C penalizes complexity more (a
    # simpler, smoother decision boundary); larger C penalizes it less (fits
    # the training data more closely, risking overfitting).
    "logistic_regression": (LogisticRegression, [
        {"C": [0.1], "max_iter": [2000], "random_state": [42]},
        {"C": [1.0], "max_iter": [2000], "random_state": [42]},
        {"C": [10.0], "max_iter": [2000], "random_state": [42]},
        {"C": [100.0], "max_iter": [2000], "random_state": [42]},
    ]),
    # C=0.1 included deliberately -- a wider grid than the first pass used.
    # Blind, same-source-CV-based selection still has to pick it on its own
    # merits; whether it actually does (vs. a larger C) is the honest test
    # behind the Results 3.4 tuning-gap finding, not assumed in advance.
    # gamma = how far one training point's influence reaches (RBF kernel).
    # Small gamma: smoother, more global boundary. Large gamma: more local,
    # can bend tightly around individual points.
    "svm": (SVC, [
        {"kernel": ["rbf"], "C": [0.1], "gamma": ["scale"], "random_state": [42]},
        {"kernel": ["rbf"], "C": [0.1], "gamma": [0.01], "random_state": [42]},
        {"kernel": ["rbf"], "C": [1], "gamma": ["scale"], "random_state": [42]},
        {"kernel": ["rbf"], "C": [1], "gamma": [0.01], "random_state": [42]},
        {"kernel": ["rbf"], "C": [10], "gamma": ["scale"], "random_state": [42]},
        {"kernel": ["rbf"], "C": [10], "gamma": [0.01], "random_state": [42]},
    ]),
    # Grid explores around Orange's own Neural Network widget defaults
    # (100 relu units, adam, alpha=0.0001), plus a couple of alternatives.
    # hidden_layer_sizes = the network's shape: (100,) is one layer of 100
    # neurons, (100, 50) is two layers (100 neurons, then 50). alpha =
    # regularization strength, same idea as Logistic Regression's C but
    # inverted -- here a BIGGER alpha means MORE regularization.
    "neural_network": (MLPClassifier, [
        {"hidden_layer_sizes": [(100,)], "alpha": [0.0001], "activation": ["relu"], "solver": ["adam"], "max_iter": [1000], "random_state": [42]},
        {"hidden_layer_sizes": [(100,)], "alpha": [0.01], "activation": ["relu"], "solver": ["adam"], "max_iter": [1000], "random_state": [42]},
        {"hidden_layer_sizes": [(50,)], "alpha": [0.001], "activation": ["relu"], "solver": ["adam"], "max_iter": [1000], "random_state": [42]},
        {"hidden_layer_sizes": [(100, 50)], "alpha": [0.001], "activation": ["relu"], "solver": ["adam"], "max_iter": [1000], "random_state": [42]},
    ]),
}


def tune_and_evaluate(name, model_class, grid, X_train, y_train, same_test, cross_test, classes):
    # Choose settings using training data only; refit the best model on all training data.
    print(f"\n=== {name} tuning (5-fold CV on training.npz) ===")
    search = GridSearchCV(model_class(), grid, cv=5, n_jobs=-1)
    search.fit(X_train, y_train)
    final = search.best_estimator_
    print(f"Best {name} params: {search.best_params_} (cv {search.best_score_*100:.2f}%)")

    same_acc = accuracy_score(same_test["y"], final.predict(same_test["X"]))
    cross_pred = final.predict(cross_test["X"])
    cross_acc = accuracy_score(cross_test["y"], cross_pred)
    print(f"{name}: same-source test {same_acc*100:.2f}%, cross-source test {cross_acc*100:.2f}%")

    recalls = recall_score(cross_test["y"], cross_pred, labels=classes, average=None, zero_division=0)
    below = [(c, r * 100) for c, r in zip(classes, recalls) if r < 0.80]
    print(f"Mudras below 80% cross-source recall: {len(below)}")
    for c, r in sorted(zip(classes, recalls * 100), key=lambda cr: cr[1]):
        flag = " *" if r < 80 else ""
        print(f"  {c:<16}{r:>7.2f}%{flag}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(final, MODELS_DIR / f"{name}.joblib")
    print(f"Saved to {MODELS_DIR / f'{name}.joblib'}")


# Load the three saved datasets and keep their files closed after evaluation.
with np.load(FEATURES_DIR / "training.npz", allow_pickle=True) as train, \
     np.load(FEATURES_DIR / "testing.npz", allow_pickle=True) as same_test, \
     np.load(FEATURES_DIR / "cross_testing.npz", allow_pickle=True) as cross_test:
    X_train, y_train = train["X"], train["y"]
    classes = sorted(set(cross_test["y"]))

    for model_name, (model_class, grid) in GRIDS.items():
        tune_and_evaluate(
            model_name, model_class, grid, X_train, y_train, same_test, cross_test, classes
        )
