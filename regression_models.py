import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor, Pool
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split, RandomizedSearchCV
from tabulate import tabulate


# Function for RandomForest with Hyperparameter Tuning
def random_forest(x, y):
    regressor = RandomForestRegressor(random_state=40)
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    search = RandomizedSearchCV(regressor, param_grid, n_iter=10, cv=5, random_state=40, scoring='r2')
    search.fit(x, y)

    best_model = search.best_estimator_
    print(f"RandomForest Best Params: {search.best_params_}")
    return evaluate_model(best_model, x, y)


# Function for Decision Tree with Hyperparameter Tuning
def decision_tree(x, y):
    regressor = DecisionTreeRegressor(random_state=40)
    param_grid = {
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    search = RandomizedSearchCV(regressor, param_grid, n_iter=10, cv=5, random_state=40, scoring='r2')
    search.fit(x, y)

    best_model = search.best_estimator_
    print(f"Decision Tree Best Params: {search.best_params_}")
    return evaluate_model(best_model, x, y)


# Function for XGBoost with Hyperparameter Tuning
def xg_boost(x, y):
    regressor = xgb.XGBRegressor(objective="reg:squarederror", random_state=40)
    param_grid = {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 10]
    }
    search = RandomizedSearchCV(regressor, param_grid, n_iter=10, cv=5, random_state=40, scoring='r2')
    search.fit(x, y)

    best_model = search.best_estimator_
    print(f"XGBoost Best Params: {search.best_params_}")
    return evaluate_model(best_model, x, y)


# Function for GradientBoosting with Hyperparameter Tuning
def gradient_boost(x, y):
    regressor = GradientBoostingRegressor(random_state=40)
    param_grid = {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 10]
    }
    search = RandomizedSearchCV(regressor, param_grid, n_iter=10, cv=5, random_state=40, scoring='r2')
    search.fit(x, y)

    best_model = search.best_estimator_
    print(f"Gradient Boosting Best Params: {search.best_params_}")
    return evaluate_model(best_model, x, y)


# Function for CatBoost with Hyperparameter Tuning
def cat_boost(x, y):
    train_pool = Pool(data=x, label=y)
    param_grid = {
        'iterations': [100, 300, 500],
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    search = CatBoostRegressor(random_seed=40)
    search.set_params(**{'loss_function': 'RMSE'})
    search.fit(x, y, verbose=0)

    print(f"CatBoost Best Params: {search.get_params()}")
    return evaluate_model(search, x, y)


# Generic function to evaluate the model using K-Fold Cross-Validation
def evaluate_model(model, x, y):
    kf = KFold(n_splits=10, random_state=40, shuffle=True)
    cv_scores = []
    all_y_test = []  # Collect all actual values from all folds
    all_y_pred = []  # Collect all predicted values from all folds

    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        score = r2_score(y_test, y_pred)
        cv_scores.append(score)

        # Collect predictions and actuals for plots
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

    mean_score = np.mean(cv_scores)
    sd = np.std(cv_scores)

    # Convert lists to numpy arrays for plotting
    all_y_test = np.array(all_y_test)
    all_y_pred = np.array(all_y_pred)

    # Plot Predicted vs Actual Values
    plt.figure(figsize=(8, 6))
    plt.scatter(all_y_test, all_y_pred, alpha=0.7, color='blue')
    plt.plot([all_y_test.min(), all_y_test.max()], [all_y_test.min(), all_y_test.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    plt.show()

    # Plot Residual Distribution
    residuals = all_y_test - all_y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color='green')
    plt.xlabel('Residuals')
    plt.title('Distribution of Residuals')
    plt.show()

    return mean_score, sd


# Main function to load data and execute all models
def main():
    data = pd.read_csv("final_model_data.csv")
    data.drop(["Molecule_ChEMBL_ID", "Name"], inplace=True, axis=1)

    data_x = data.drop(["pIC50"], axis=1).values
    data_y = data["pIC50"].values

    results = []

    # Random Forest
    rf_score, rf_sd = random_forest(data_x, data_y)
    results.append(["Random Forest", rf_score, rf_sd])

    # Decision Tree
    dt_score, dt_sd = decision_tree(data_x, data_y)
    results.append(["Decision Tree", dt_score, dt_sd])

    # XGBoost
    xgb_score, xgb_sd = xg_boost(data_x, data_y)
    results.append(["XGBoost", xgb_score, xgb_sd])

    # Gradient Boosting
    gb_score, gb_sd = gradient_boost(data_x, data_y)
    results.append(["Gradient Boosting", gb_score, gb_sd])

    # CatBoost
    cb_score, cb_sd = cat_boost(data_x, data_y)
    results.append(["CatBoost", cb_score, cb_sd])

    # Print results in tabular format
    headers = ["Model", "Mean R2 Score", "Standard Deviation"]
    print(tabulate(results, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()
