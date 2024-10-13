import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import r2_score
from sklearn.feature_selection import VarianceThreshold


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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)

# Define the parameter grid for RandomizedSearchCV
param_dist = {
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'iterations': [500, 1000, 2000],
    'l2_leaf_reg': [1, 3, 5, 7, 10],
    'bagging_temperature': [0.5, 1.0, 2.0],
    'random_strength': [1, 2, 5, 10],
    'border_count': [32, 64, 128]
}

# Initialize the CatBoostRegressor model
catboost_model = CatBoostRegressor(
    loss_function='RMSE',
    verbose=100
)

# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=catboost_model,
    param_distributions=param_dist,
    n_iter=10,  # Number of random combinations to try
    cv=5,  # 5-fold cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1  # Use all available cores
)

# Fit the RandomizedSearchCV model
random_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best Parameters: ", random_search.best_params_)
print("Best Score: ", random_search.best_score_)

# Use the best estimator to make predictions
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
print(f"R2 Score: {r2}")
