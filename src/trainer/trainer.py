import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import joblib


def load_data(filepath):
    return pd.read_csv(filepath)


def preprocess_data(data, target):
    features = [col for col in data.columns if col != target]
    X = data[features]
    y = data[target]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def tune_hyperparameters(
    model, X_train, y_train, param_space, cv=5, n_iter=10, scoring="f1"
):
    search = RandomizedSearchCV(
        model, param_space, cv=cv, n_iter=n_iter, scoring=scoring
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


def select_features(model, features, X_train, X_test, k=10):
    importances = model.feature_importances_
    features_scores = list(zip(features, importances))
    features_scores = sorted(features_scores, key=lambda x: x[1], reverse=True)
    selected_features = [f[0] for f in features_scores[:k]]
    return X_train[selected_features], X_test[selected_features], selected_features


def evaluate_model(
    model,
    selected_features,
    best_params,
    X_train,
    y_train,
    X_test,
    y_test,
    n_bootstraps=100,
):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    precision_train = precision_score(y_train, y_pred_train)
    precision_test = precision_score(y_test, y_pred_test)

    recall_train = recall_score(y_train, y_pred_train)
    recall_test = recall_score(y_test, y_pred_test)

    f1_train = f1_score(y_train, y_pred_train)
    f1_test = f1_score(y_test, y_pred_test)

    f1_scores = []
    for i in range(n_bootstraps):
        X_boot, y_boot = resample(X_test, y_test, random_state=i)
        y_pred_boot = model.predict(X_boot)
        f1_boot = f1_score(y_boot, y_pred_boot)
        f1_scores.append(f1_boot)

    f1_mean = np.mean(f1_scores)
    f1_std = np.std(f1_scores)

    metrics = {
        "Selected Features": selected_features,
        "Best Hyperparameters": best_params,
        "Training Accuracy": accuracy_train,
        "Training Precision": precision_train,
        "Training Recall": recall_train,
        "Training F1-score": f1_train,
        "Test Accuracy": accuracy_test,
        "Test Precision": precision_test,
        "Test Recall": recall_test,
        "Test F1-score Mean": f1_mean,
        "Test F1-score Std": f1_std,
    }

    return metrics


def save_model(model, filepath):
    joblib.dump(model, filepath)


def main():
    # Load and preprocess data
    data = load_data("../../data/processed_data/cards_prepared.csv")
    X, y = preprocess_data(data, "popular")

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Tune hyperparameters
    model = RandomForestClassifier(random_state=42)
    param_space = {
        "max_depth": [10, 20, 30],
        "min_samples_leaf": [1, 2, 4],
        "min_samples_split": [2, 5, 10],
        "min_weight_fraction_leaf": [0.01, 0.05, 0.1],
    }
    best_model = tune_hyperparameters(model, X_train, y_train, param_space)

    # Select features
    X_train_selected, X_test_selected, selected_features = select_features(
        best_model, X.columns, X_train, X_test
    )

    with open("../../data/train_misc/selected_features.txt", "w") as f:
        for feature in selected_features:
            f.write(f"{feature}\n")

    # Train a new model on the selected features
    model_selected = RandomForestClassifier(**best_model.get_params())
    model_selected.fit(X_train_selected, y_train)

    # Evaluate the performance of the new model
    metrics = evaluate_model(
        model_selected,
        selected_features,
        best_model.get_params(),
        X_train_selected,
        y_train,
        X_test_selected,
        y_test,
    )
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # Save the model
    save_model(model_selected, "../../models/hive_mind_queen_model.pkl")


if __name__ == "__main__":
    main()
