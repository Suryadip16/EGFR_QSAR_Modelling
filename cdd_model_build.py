import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor, cv, Pool
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from tabulate import tabulate


# Models:

def lin_reg(x, y):
    kf = KFold(n_splits=10, random_state=40, shuffle=True)
    regressor = LinearRegression()
    cv_scores = []
    for fold, [train_index, test_index] in enumerate(kf.split(x)):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        regressor.fit(x_train, y_train)
        y_pred = regressor.predict(x_test)
        score = r2_score(y_test, y_pred)
        cv_scores.append(score)
    mean_score = np.mean(cv_scores)
    sd = np.std(cv_scores)
    return mean_score, sd


def random_forest(x, y):
    kf = KFold(n_splits=10, shuffle=True, random_state=40)
    regressor = RandomForestRegressor(n_estimators=200)
    cv_scores = []
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        regressor.fit(x_train, y_train)
        y_pred = regressor.predict(x_test)
        score = r2_score(y_test, y_pred)
        cv_scores.append(score)
    mean_score = np.mean(cv_scores)
    sd = np.std(cv_scores)
    return mean_score, sd


def Decision_tree(x, y):
    kf = KFold(n_splits=10, random_state=40, shuffle=True)
    fold_acc_scores_list = []
    for fold, [train_index, test_index] in enumerate(kf.split(x)):
        x_train1, x_test = x[train_index], x[test_index]
        y_train1, y_test = y[train_index], y[test_index]
        x_train2, x_val, y_train2, y_val = train_test_split(x_train1, y_train1, shuffle=True, random_state=40,
                                                            test_size=0.3)

        depth_score = 0
        for depth in range(1, 20):
            regressor = DecisionTreeRegressor(max_depth=depth, random_state=40)
            regressor.fit(x_train2, y_train2)
            val_pred = regressor.predict(x_val)
            val_acc = r2_score(y_val, val_pred)
            if val_acc > depth_score:
                depth_score = val_acc
                depth_value = depth
        regressor = DecisionTreeRegressor(max_depth=depth_value, random_state=40)
        regressor.fit(x_train1, y_train1)
        test_pred = regressor.predict(x_test)
        fold_acc_score = r2_score(y_test, test_pred)
        fold_acc_scores_list.append(fold_acc_score)
    mean_acc_score = np.mean(fold_acc_scores_list)
    sd_acc_score = np.std(fold_acc_scores_list)
    return mean_acc_score, sd_acc_score


def xg_boost(x, y):
    kf = KFold(n_splits=10, random_state=40, shuffle=True)
    regressor = xgb.XGBRegressor(objective="reg:linear")
    cv_scores = []
    for fold, [train_index, test_index] in enumerate(kf.split(x)):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        regressor.fit(x_train, y_train)
        y_pred = regressor.predict(x_test)
        score = r2_score(y_test, y_pred)
        cv_scores.append(score)
    mean_score = sum(cv_scores) / len(cv_scores)
    sd = np.std(cv_scores)
    return mean_score, sd


def gradient_boost(x, y):
    kf = KFold(n_splits=10, random_state=40, shuffle=True)
    regressor = GradientBoostingRegressor()
    cv_scores = []
    for fold, [train_index, test_index] in enumerate(kf.split(x)):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        regressor.fit(x_train, y_train)
        y_pred = regressor.predict(x_test)
        score = r2_score(y_test, y_pred)
        cv_scores.append(score)
    mean_score = sum(cv_scores) / len(cv_scores)
    sd = np.std(cv_scores)
    return mean_score, sd


# def cat_boost(x, y):
#     kf = KFold(n_splits=10, random_state=40, shuffle=True)
#     fold_acc_scores_list = []
#     for fold, [train_index, test_index] in enumerate(kf.split(x)):
#         x_train1, x_test = x[train_index], x[test_index]
#         y_train1, y_test = y[train_index], y[test_index]
#         x_val, x_test1, y_val, y_test1 = train_test_split(x_test, y_test, shuffle=True, random_state=40,
#                                                           test_size=0.5)
#         regressor = CatBoostRegressor(iterations=1000,
#                                       learning_rate=0.1,
#                                       depth=6,
#                                       loss_function='RMSE',
#                                       use_best_model=True,  # Enable best model selection
#                                       verbose=100)
#
#         # Train the model with eval_set
#         regressor.fit(x_train1, y_train1,
#                       eval_set=(x_val, y_val),  # Validation set for performance monitoring
#                       early_stopping_rounds=100)
#
#         # Make predictions on the test set
#         y_pred = regressor.predict(x_test1)
#         fold_acc_score = r2_score(y_test1, y_pred)
#         fold_acc_scores_list.append(fold_acc_score)
#     mean_acc_score = np.mean(fold_acc_scores_list)
#     sd_acc_score = np.std(fold_acc_scores_list)
#     return mean_acc_score, sd_acc_score

def cat_boost(x, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Create a Pool object for the training data
    train_pool = Pool(data=X_train, label=y_train)

    # Define the initial parameters for CatBoostRegressor
    params = {
        'iterations': 1000,
        'depth': 6,
        'learning_rate': 0.1,
        'loss_function': 'RMSE',
        'verbose': 100
    }

    # Step 1: Perform Cross-Validation to find the best number of iterations
    cv_data = cv(
        pool=train_pool,  # Pool object with training data and labels
        params=params,  # Parameters for the model
        fold_count=10,  # Number of folds
        shuffle=True,  # Shuffle the dataset before splitting into folds
        partition_random_seed=42,  # Random seed for reproducibility
        early_stopping_rounds=100  # Early stopping after 50 rounds of no improvement
    )

    # Get the optimal number of iterations
    optimal_iterations = len(cv_data)

    # Update the parameters with the optimal number of iterations
    params['iterations'] = optimal_iterations

    # Step 2: Fit the final model on the entire training dataset with optimal iterations
    final_model = CatBoostRegressor(**params)
    final_model.fit(X_train, y_train)

    # Step 3: Make predictions on the test set
    y_pred = final_model.predict(X_test)

    # Evaluate the model
    r2 = r2_score(y_test, y_pred)
    print(f"R2 Score: {r2}")


def main():
    data = pd.read_csv("final_model_data.csv")
    # print(data.columns)
    data.drop(["Molecule_ChEMBL_ID"], inplace=True, axis=1)
    # print(data.columns)

    # Make and X and Y Matrix

    x = data.drop(["Name", "pIC50"], axis=1)
    # print(data_x.columns)

    selection = VarianceThreshold(threshold=(.9 * (1 - .9)))
    data_x = selection.fit_transform(x)
    print(data_x.shape)

    data_y = data["pIC50"]

    # Initialize a list to store results
    results = []

    # Linear Regression
    lin_reg_score, lin_reg_sd = lin_reg(data_x, data_y)
    results.append(["Linear Regression", lin_reg_score, lin_reg_sd])

    # Random Forest
    rf_score, rf_sd = random_forest(data_x, data_y)
    results.append(["Random Forest", rf_score, rf_sd])

    # Decision Tree
    dt_score, dt_sd = Decision_tree(data_x, data_y)
    results.append(["Decision Tree", dt_score, dt_sd])

    # XGBoost
    xgb_score, xgb_sd = xg_boost(data_x, data_y)
    results.append(["XGBoost", xgb_score, xgb_sd])

    # Gradient Boosting
    gb_score, gb_sd = gradient_boost(data_x, data_y)
    results.append(["Gradient Boosting", gb_score, gb_sd])

    # CatBoost
    # cb_score, cb_sd = cat_boost(data_x.values, data_y.values)
    # results.append(["CatBoost", cb_score, cb_sd])

    # Print the results in a tabular format
    headers = ["Model", "Mean R2 Score", "Standard Deviation"]
    print(tabulate(results, headers=headers, tablefmt="grid"))
    # cat_boost(data_x, data_y)


if __name__ == "__main__":
    main()
