import pandas as pd

# Load the dataset
activity_df = pd.read_csv("EFGR_data.csv")
print("Columns in the dataset:", activity_df.columns)

# Check and remove missing values
activity_df2 = activity_df.dropna(subset=['Standard_Value', 'AlogP', 'MW', 'Smiles'])
print("Data after removing missing values:", activity_df2)

# Filter the dataset to include only entries with Standard_Units as 'nM'
activity_df16 = activity_df2[activity_df2['Standard_Units'] == "nM"]
print("Data with Standard_Units as 'nM':", activity_df16)

# Classify bioactivity based on Standard_Value
bioactivity_class = []
for i in activity_df16.Standard_Value:
    if float(i) <= 1000:
        bioactivity_class.append("Active")
    elif 10000 > float(i) > 1000:
        bioactivity_class.append("Intermediate")
    else:
        bioactivity_class.append("Inactive")

# Select the relevant columns and append bioactivity_class
selection = ["Molecule_ChEMBL_ID", "Smiles", "Standard_Value", "Standard_Units", "AlogP"]
data_df = activity_df16[selection]
data_df['bioactivity_class'] = bioactivity_class

# Display the final dataframe
print("Final dataframe columns:", data_df.columns)
print(data_df)
data_df.to_csv("Bioactivity_Data_preprocessed_EFGR_no_BE.csv", index=False)
