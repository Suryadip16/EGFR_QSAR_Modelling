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


def main():
    data_df = pd.read_csv("EGFR_data_processed.csv")

    unique_count = data_df['Molecule_ChEMBL_ID'].nunique()
    print(f"Number of unique Molecule_ChEMBL_ID entries: {unique_count}")

    # Sort by 'pIC50' column and then remove duplicates, keeping the row with the highest pIC50 value
    data_df_sorted = data_df.sort_values(by='pIC50', ascending=False)
    data_df_unique = data_df_sorted.drop_duplicates(subset='Molecule_ChEMBL_ID', keep='first')
    print(data_df_unique)
    fp_df = pd.read_csv("final_model_data.csv")
    print(data_df_unique.columns)
    print(fp_df.columns)

    # Assuming data_df_unique and model_data_df are already loaded as pandas DataFrames.

    # Select the columns to be added from data_df_unique
    columns_to_add = ['Molecule_ChEMBL_ID', 'MW', 'LogP', 'NumHDonors', 'NumHAcceptors']

    # Merge the selected columns from data_df_unique into model_data_df on 'Molecule_ChEMBL_ID'
    merged_df = pd.merge(fp_df, data_df_unique[columns_to_add], on='Molecule_ChEMBL_ID', how='left')

    # The merged_df now contains the additional columns
    print(merged_df.columns)

    # Make and X and Y Matrix

    data_x = merged_df.drop(["Name", "pIC50", "Molecule_ChEMBL_ID"], axis=1)
    # print(data_x.columns)

    # selection = VarianceThreshold(threshold=(.9 * (1 - .9)))
    # data_x = selection.fit_transform(x)
    # print(data_x.shape)

    data_y = merged_df["pIC50"]

    # Initialize a list to store results
    results = []

    # Linear Regression
    lin_reg_score, lin_reg_sd = lin_reg(data_x.values, data_y.values)
    results.append(["Linear Regression", lin_reg_score, lin_reg_sd])

    # Random Forest
    rf_score, rf_sd = random_forest(data_x.values, data_y.values)
    results.append(["Random Forest", rf_score, rf_sd])

    # Decision Tree
    dt_score, dt_sd = Decision_tree(data_x.values, data_y.values)
    results.append(["Decision Tree", dt_score, dt_sd])

    # XGBoost
    xgb_score, xgb_sd = xg_boost(data_x.values, data_y.values)
    results.append(["XGBoost", xgb_score, xgb_sd])

    # Gradient Boosting
    gb_score, gb_sd = gradient_boost(data_x.values, data_y.values)
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
