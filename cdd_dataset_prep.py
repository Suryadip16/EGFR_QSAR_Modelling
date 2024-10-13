import pandas as pd

data_df = pd.read_csv("EGFR_data_processed.csv")
# Print the number of unique Molecule_ChEMBL_ID entries
unique_count = data_df['Molecule_ChEMBL_ID'].nunique()
print(f"Number of unique Molecule_ChEMBL_ID entries: {unique_count}")

# Sort by 'pIC50' column and then remove duplicates, keeping the row with the highest pIC50 value
data_df_sorted = data_df.sort_values(by='pIC50', ascending=False)
data_df_unique = data_df_sorted.drop_duplicates(subset='Molecule_ChEMBL_ID', keep='first')
print(data_df_unique)

selection = ['Smiles', 'Molecule_ChEMBL_ID']
data_df_selection = data_df_unique[selection]
print(data_df_selection)
# data_df_selection.to_csv("egfr_molecules.smi", sep='\t', index=False, header=False)
fp_df = pd.read_csv("descriptors_output.csv")
print(fp_df)
merged_df = pd.merge(fp_df, data_df_unique[['Molecule_ChEMBL_ID', 'pIC50']], left_on='Name', right_on='Molecule_ChEMBL_ID', how='left')
print(merged_df)
merged_df.to_csv("final_model_data.csv", index=False)



