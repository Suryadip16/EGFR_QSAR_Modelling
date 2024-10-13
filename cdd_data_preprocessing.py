import pandas as pd

activity_df = pd.read_csv("EFGR_data.csv")
print(activity_df.columns)

# check and remove missing values
activity_df2 = activity_df[activity_df.Standard_Value.notna()]
print(activity_df2)
activity_df3 = activity_df2[activity_df2.AlogP.notna()]
print(activity_df3)
activity_df4 = activity_df3[activity_df3.MW.notna()]
print(activity_df4)
activity_df5 = activity_df4[activity_df4.Smiles.notna()]
print(activity_df5)
activity_df6 = activity_df5[activity_df5.Standard_Value.notna()]
print(activity_df6)
activity_df16 = activity_df6[activity_df6.Standard_Units == "nM"]
print(activity_df16)
# activity_df7 = activity_df6[activity_df6.Ligand_Efficiency_BEI.notna()]
# print(activity_df7)
# activity_df8 = activity_df7[activity_df7.Ligand_Efficiency_LE.notna()]
# print(activity_df8)
# activity_df9 = activity_df8[activity_df8.Ligand_Efficiency_LLE.notna()]
# print(activity_df9)
# activity_df10 = activity_df9[activity_df9.Ligand_Efficiency_SEI.notna()]
# print(activity_df10)

# activity_df3 = activity_df[activity_df2.]
# no missing values were found

# classifying compounds as "Active"(IC50 <= 1000nM); "Intermediate"(10000>IC50>1000); "Inactive"(IC50>10000)
bioactivity_class = []
for i in activity_df16.Standard_Value:
    if float(i) <= 1000:
        bioactivity_class.append("Active")
    elif 10000 > float(i) > 1000:
        bioactivity_class.append("Intermediate")
    else:
        bioactivity_class.append("Inactive")

selection = ["Molecule_ChEMBL_ID", "Smiles", "Standard_Value", "Standard_Units", "AlogP"]
data_df = activity_df16[selection]
data_df = pd.concat([data_df, pd.Series(bioactivity_class)], axis=1)
data_df.rename(columns={0: 'bioactivity_class'}, inplace=True)
print(data_df.columns)
print(data_df)

data_df.to_csv("Bioactivity_Data_preprocessed_EFGR_no_BE.csv", index=False)



