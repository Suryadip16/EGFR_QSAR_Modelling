import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from tabulate import tabulate
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb


def logistic_reg(x, y):
    kf = KFold(n_splits=10, random_state=40, shuffle=True)
    params = {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs', 'saga']}
    classifier = LogisticRegression(max_iter=1000, multi_class='auto')

    grid = GridSearchCV(classifier, params, cv=3)
    cv_scores = []

    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        grid.fit(x_train, y_train)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(x_test)
        score = accuracy_score(y_test, y_pred)
        cv_scores.append(score)

    mean_score = np.mean(cv_scores)
    sd = np.std(cv_scores)
    return mean_score, sd


def random_forest(x, y):
    kf = KFold(n_splits=10, random_state=40, shuffle=True)
    params = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'criterion': ['gini', 'entropy']}
    classifier = RandomForestClassifier()

    grid = GridSearchCV(classifier, params, cv=3)
    cv_scores = []

    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        grid.fit(x_train, y_train)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(x_test)
        score = accuracy_score(y_test, y_pred)
        cv_scores.append(score)

    mean_score = np.mean(cv_scores)
    sd = np.std(cv_scores)
    return mean_score, sd


def decision_tree(x, y):
    kf = KFold(n_splits=10, random_state=40, shuffle=True)
    params = {'max_depth': range(1, 20), 'criterion': ['gini', 'entropy']}

    fold_acc_scores_list = []

    for train_index, test_index in kf.split(x):
        x_train1, x_test = x[train_index], x[test_index]
        y_train1, y_test = y[train_index], y[test_index]

        x_train2, x_val, y_train2, y_val = train_test_split(
            x_train1, y_train1, shuffle=True, random_state=40, test_size=0.3
        )

        grid = GridSearchCV(DecisionTreeClassifier(random_state=40), params, cv=3)
        grid.fit(x_train2, y_train2)
        best_model = grid.best_estimator_

        test_pred = best_model.predict(x_test)
        fold_acc_score = accuracy_score(y_test, test_pred)
        fold_acc_scores_list.append(fold_acc_score)

    mean_acc_score = np.mean(fold_acc_scores_list)
    sd_acc_score = np.std(fold_acc_scores_list)
    return mean_acc_score, sd_acc_score


def xg_boost(x, y):
    params = {'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2]}

    classifier = xgb.XGBClassifier(objective="multi:softmax", num_class=len(np.unique(y)))
    grid = GridSearchCV(classifier, params, cv=10, scoring='accuracy')  # 10-fold CV inside GridSearch

    grid.fit(x, y)  # Fit the model using the entire dataset with CV inside GridSearch
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    mean_score = grid.best_score_

    print(f"Best parameters (XGBoost): {best_params}")
    print(f"Mean CV Accuracy (XGBoost): {mean_score}")

    return mean_score, best_params


def gradient_boost(x, y):
    params = {'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2]}

    classifier = GradientBoostingClassifier()
    grid = GridSearchCV(classifier, params, cv=10, scoring='accuracy')  # 10-fold CV inside GridSearch

    grid.fit(x, y)  # Fit the model using the entire dataset with CV inside GridSearch
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    mean_score = grid.best_score_

    print(f"Best parameters (GradientBoost): {best_params}")
    print(f"Mean CV Accuracy (GradientBoost): {mean_score}")

    return mean_score, best_params


def main():
    data_df = pd.read_csv("final_model_data_bioactivity_class.csv")
    print(data_df.columns)
    data_x = data_df.drop(["Name", "pIC50", "Molecule_ChEMBL_ID", "bioactivity_class"], axis=1)
    # print(data_x.shape)
    encoder = LabelEncoder()
    data_label = data_df['bioactivity_class']
    data_y = encoder.fit_transform(data_label)
    print(np.unique(data_y))

    # Initialize a list to store results
    results = []

    # Linear Regression
    # lin_reg_score, lin_reg_sd = logistic_reg(data_x.values, data_y)
    # results.append(["Logistic Regression", lin_reg_score, lin_reg_sd])
    # print("Done Logistic Regression")

    # Decision Tree
    # dt_score, dt_sd = decision_tree(data_x.values, data_y)
    # results.append(["Decision Tree", dt_score, dt_sd])
    # print("Done DT")

    # XGBoost
    xgb_score, xgb_sd = xg_boost(data_x.values, data_y)
    results.append(["XGBoost", xgb_score, xgb_sd])
    print("Done XGB")

    # Gradient Boosting
    gb_score, gb_sd = gradient_boost(data_x.values, data_y)
    results.append(["Gradient Boosting", gb_score, gb_sd])
    print("Done GB")

    # Random Forest
    # rf_score, rf_sd = random_forest(data_x.values, data_y)
    # results.append(["Random Forest", rf_score, rf_sd])
    # print("Done RF")

    # CatBoost
    # cb_score, cb_sd = cat_boost(data_x.values, data_y.values)
    # results.append(["CatBoost", cb_score, cb_sd])

    # Print the results in a tabular format
    headers = ["Model", "Mean R2 Score", "Standard Deviation"]
    print(tabulate(results, headers=headers, tablefmt="grid"))


if __name__ == '__main__':
    main()
