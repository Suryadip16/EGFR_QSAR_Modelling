import pandas as pd
import numpy as np
import seaborn as sns
from tabulate import tabulate
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize
from xgboost import XGBClassifier, XGBRFClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from itertools import cycle


# Function to plot ROC Curve for multi-class
def plot_roc_auc(y_test, y_score, n_classes, model_name):
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    colors = cycle(["blue", "red", "green"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i], tpr[i], color=color, lw=2,
            label=f"Class {i} (AUC = {roc_auc[i]:.2f})"
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.show()


# Function to plot PR Curve for multi-class
def plot_pr_auc(y_test, y_score, n_classes, model_name):
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    precision, recall, pr_auc = {}, {}, {}

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        pr_auc[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

    plt.figure()
    colors = cycle(["purple", "orange", "cyan"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            recall[i], precision[i], color=color, lw=2,
            label=f"Class {i} (PR-AUC = {pr_auc[i]:.2f})"
        )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.legend(loc="lower left")
    plt.show()


def run_model(model, x_data, y_data, model_name):
    kf = KFold(n_splits=10, random_state=40, shuffle=True)
    cv_scores = []
    y_true, y_proba = [], []

    for train_index, test_index in kf.split(x_data):
        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_pred_proba = model.predict_proba(x_test)

        score = accuracy_score(y_test, y_pred)
        cv_scores.append(score)

        y_true.extend(y_test)
        y_proba.extend(y_pred_proba)

    mean_score = np.mean(cv_scores)
    sd = np.std(cv_scores)
    print(f"{model_name} - Mean Accuracy: {mean_score:.2f}, SD: {sd:.2f}")

    # Plot ROC-AUC and PR-AUC
    plot_roc_auc(np.array(y_true), np.array(y_proba), n_classes=3, model_name=model_name)
    plot_pr_auc(np.array(y_true), np.array(y_proba), n_classes=3, model_name=model_name)

    return mean_score, sd


def main():
    # Load the dataset
    data_df = pd.read_csv("final_model_data_bioactivity_class.csv")
    data_x = data_df.drop(["Name", "pIC50", "Molecule_ChEMBL_ID", "bioactivity_class"], axis=1)
    data_label = data_df["bioactivity_class"]

    plt.figure(figsize=(5.5, 5.5))

    sns.countplot(x='bioactivity_class', data=data_df, edgecolor='black', hue='bioactivity_class')
    plt.xlabel("Bioactivity Class")
    plt.ylabel("Frequency")
    plt.show()

    # Encode labels
    encoder = LabelEncoder()
    data_y = encoder.fit_transform(data_label)

    # Initialize models
    models = [
        ("XGBoost", XGBClassifier(learning_rate=0.2, max_depth=7, n_estimators=200, objective="multi:softmax", num_class=3)),
        ("XGBRF", XGBRFClassifier(learning_rate=0.2, max_depth=10, n_estimators=200, subsample=1.0,
                                  reg_lambda=1, colsample_bynode=0.7, random_state=42, objective='multi:softmax',
                                  num_class=3, use_label_encoder=False)),
        ("Gradient Boosting", GradientBoostingClassifier(learning_rate=0.2, max_depth=7, n_estimators=100)),
        ("CatBoost", CatBoostClassifier(learning_rate=0.2, l2_leaf_reg=3, random_strength=5,
                                        iterations=1000, depth=8, border_count=128, bagging_temperature=1.0,
                                        verbose=0, loss_function='MultiClass'))
    ]

    # Store results for each model
    results = []
    for model_name, model in models:
        score, sd = run_model(model, data_x.values, data_y, model_name)
        results.append([model_name, score, sd])

    # Display results in a tabular format
    headers = ["Model", "Mean Accuracy Score", "Standard Deviation"]
    print(tabulate(results, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()
