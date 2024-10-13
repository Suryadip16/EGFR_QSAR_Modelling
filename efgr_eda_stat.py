import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from numpy.random import seed
from scipy.stats import normaltest, f_oneway, kruskal, levene

sns.set(style="ticks")

data_df = pd.read_csv("Bioactivity_Data_preprocessed_EFGR_no_BE.csv")

print(data_df.columns)


# Calculate Lipinski Descriptors: Christopher Lipinski, a scientist at Pfizer, came up with a set of rule of thumb
# metrics for evaluating the "druglikeness" of compounds. Such Druglikeness is based on the Absorption, Distribution,
# Metabolism and Excretion (ADME), that is also known as the pharmacokinetic profile. Lipinski analyzed all orally active
# FDA-approved drugs in the formulation of what is to be known as the Rule-of-Five or Lipinski's Rule:

# Molecular Weight: The molecular weight of the compound should be less than 500 Daltons.
# Lipophilicity (LogP): The partition coefficient (LogP), which measures the compound's solubility in fats versus water,
# should be less than 5. This indicates that the compound is neither too hydrophilic (water-loving) nor too lipophilic (fat-loving).
# Hydrogen Bond Donors: The compound should have 5 or fewer hydrogen bond donors, which are typically NH or OH groups.
# Hydrogen Bond Acceptors: The compound should have 10 or fewer hydrogen bond acceptors, typically nitrogen or oxygen atoms.

# Exceptions and Considerations
# Violations: A compound that violates more than one of these rules is less likely to be orally bioavailable. However,
# there are exceptions where compounds violating these rules have still been found to be effective drugs.
# Application: These rules are most relevant during the early stages of drug design to help filter out compounds that
# are unlikely to be successful as oral drugs.

# CALCULATE DESCRIPTORS:

def lipinski(smiles):
    moldata = []
    for element in smiles:
        mol = Chem.MolFromSmiles(element)
        moldata.append(mol)

    base_data = np.arange(1, 1)
    i = 0
    for mol in moldata:
        molwt = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        numHdonors = Lipinski.NumHDonors(mol)
        numHacceptors = Lipinski.NumHAcceptors(mol)

        row = np.array([molwt, logp, numHdonors, numHacceptors])

        if i == 0:
            base_data = row
        else:
            base_data = np.vstack([base_data, row])
        i = i + 1

    col_names = ["MW", "LogP", "NumHDonors", "NumHAcceptors"]
    descriptors = pd.DataFrame(data=base_data, columns=col_names)
    return descriptors


df_lipinski = lipinski(data_df.Smiles)
# print(df_lipinski)
# print(df_lipinski.describe())

combined_df = pd.concat([data_df, df_lipinski], axis=1)
print(combined_df.describe())


# plt.scatter(combined_df.standard_value, list(range(len(combined_df.standard_value))))
# plt.show()

# To allow IC50 data values to be more evenly distributed, we convert it to pIC50, which is essentially -log10(IC50)

def pIC50(df):
    pIC50 = []

    for i in df["Standard_Value"]:
        molar = i * (10 ** -9)  # Converts nM to M
        pIC50.append(-np.log10(molar))
    df["pIC50"] = pIC50
    x = df.drop('Standard_Value', axis=1)
    return x


# print(combined_df.standard_value.describe())
final_df = pIC50(combined_df)
print(final_df.columns)
print(final_df.describe())
# final_df.to_csv("EGFR_data_processed.csv", index=False)
plt.scatter(final_df.pIC50, list(range(len(combined_df.pIC50))))
plt.show()

# final_df_2class = final_df[final_df.bioactivity_class != 'Intermediate']
#
# print(final_df_2class)

# EDA (Chemical Space Analysis)

# Frequency Plot of the 2 Bioactivity Classes:

plt.figure(figsize=(5.5, 5.5))

sns.countplot(x='bioactivity_class', data=final_df, edgecolor='black', hue='bioactivity_class')
plt.xlabel("Bioactivity Class")
plt.ylabel("Frequency")
plt.show()
# plt.savefig("plot_bioactivity_class.pdf")
#
# # Scatter Plot of MW and LogP
#
plt.figure(figsize=(5.5, 5.5))

sns.scatterplot(x='MW', y='LogP', data=final_df, hue="bioactivity_class", size='pIC50', edgecolor='black', alpha=0.7)
plt.xlabel("MW")
plt.ylabel("LogP")
plt.legend(bbox_to_anchor=(0.97, 1), loc=2, borderaxespad=0)
plt.show()
# # plt.savefig("plot_MW_vs_LogP.png")
#
# # Boxplot
#
# For pIC50

sns.boxplot(x='bioactivity_class', y='pIC50', data=final_df, hue='bioactivity_class')
plt.xlabel("Bioactivity Class")
plt.ylabel("pIC50")
plt.show()
print(
    "****************************************************************************************************************")
# Statistical Tests:
alpha = 0.05
print("STATISTICAL TESTS: ")
print("pIC50: ")

# Normality and Variance Test:


def normality_and_variance_tests(descriptor, verbose=False):
    seed(42)

    # Select the relevant columns
    selection = [descriptor, 'bioactivity_class']
    df = final_df[selection]

    # Separate the data into active, inactive, and intermediate classes
    active = df[df.bioactivity_class == 'Active'][descriptor]
    inactive = df[df.bioactivity_class == 'Inactive'][descriptor]
    intermediate = df[df.bioactivity_class == 'Intermediate'][descriptor]

    # Perform D'Agostino's K-squared test for normality
    dagostino_active_stat, dagostino_active_p = normaltest(active)
    dagostino_inactive_stat, dagostino_inactive_p = normaltest(inactive)
    dagostino_intermediate_stat, dagostino_intermediate_p = normaltest(intermediate)

    # Perform Levene's test for homogeneity of variances
    levene_stat, levene_p = levene(active, inactive, intermediate)

    # Create a results DataFrame
    results = pd.DataFrame({
        'Class': ['Active', 'Inactive', 'Intermediate'],
        'D_Agostino_K_squared_p_value': [dagostino_active_p, dagostino_inactive_p, dagostino_intermediate_p]
    })

    # Use pd.concat instead of append to add the 'Overall' row
    overall_row = pd.DataFrame({
        'Class': ['Overall'],
        'D_Agostino_K_squared_p_value': [None],
        'Levenes_p_value': [levene_p]
    })

    results = pd.concat([results, overall_row], ignore_index=True)
    print(results)

    return results


# Parametric Test: One-Way ANOVA:


def anova_test(descriptor, verbose=False):
    seed(42)

    # Select the relevant columns
    selection = [descriptor, 'bioactivity_class']
    df = final_df[selection]

    # Separate the data into active, inactive, and intermediate classes
    active = df[df.bioactivity_class == 'Active'][descriptor]
    inactive = df[df.bioactivity_class == 'Inactive'][descriptor]
    intermediate = df[df.bioactivity_class == 'Intermediate'][descriptor]

    # Perform ANOVA
    stat, p = f_oneway(active, inactive, intermediate)

    # Interpretation
    alpha = 0.05
    if p > alpha:
        interpretation = "Same Distribution (Failed to Reject H0)"
    else:
        interpretation = "Different Distribution (Reject H0)"

    # Create a results DataFrame
    results = pd.DataFrame({
        'Descriptor': [descriptor],
        'Statistics': [stat],
        'p_value': [p],
        'alpha': [alpha],
        'Interpretation': [interpretation]
    })

    return results


# Non-Parametric Test: Kruskal-Wallis Test


def kruskal_wallis_test(descriptor, verbose=False):
    seed(42)

    # Select the relevant columns
    selection = [descriptor, 'bioactivity_class']
    df = final_df[selection]

    # Separate the data into active, inactive, and intermediate classes
    active = df[df.bioactivity_class == 'Active'][descriptor]
    inactive = df[df.bioactivity_class == 'Inactive'][descriptor]
    intermediate = df[df.bioactivity_class == 'Intermediate'][descriptor]

    # Perform Kruskal-Wallis test
    stat, p = kruskal(active, inactive, intermediate)

    # Interpretation
    alpha = 0.05
    if p > alpha:
        interpretation = "Same Distribution (Failed to Reject H0)"
    else:
        interpretation = "Different Distribution (Reject H0)"

    # Create a results DataFrame
    results = pd.DataFrame({
        'Descriptor': [descriptor],
        'Statistics': [stat],
        'p_value': [p],
        'alpha': [alpha],
        'Interpretation': [interpretation]
    })

    return results


def choose_test_and_run(descriptor):
    seed(42)

    # Perform normality and variance tests
    results = normality_and_variance_tests(descriptor)

    # Extract p-values from results
    dagostino_active_p = results.loc[results['Class'] == 'Active', 'D_Agostino_K_squared_p_value'].values[0]
    dagostino_inactive_p = results.loc[results['Class'] == 'Inactive', 'D_Agostino_K_squared_p_value'].values[0]
    dagostino_intermediate_p = results.loc[results['Class'] == 'Intermediate', 'D_Agostino_K_squared_p_value'].values[0]
    levene_p = results.loc[results['Class'] == 'Overall', 'Levenes_p_value'].values[0]

    # Set significance level
    alpha = 0.05

    # Check for normality
    normality_pass = all(p > alpha for p in [dagostino_active_p, dagostino_inactive_p, dagostino_intermediate_p])

    # Check for homogeneity of variances
    variance_pass = levene_p > alpha

    if normality_pass and variance_pass:
        # Perform ANOVA if both tests pass
        print("Performing ANOVA:")
        test_results = anova_test(descriptor)
    else:
        # Perform Kruskal-Wallis test if either test fails
        print("Performing Kruskal-Wallis:")
        test_results = kruskal_wallis_test(descriptor)

    return test_results


# Testing for normality among the two groups

inactive_list = final_df[final_df['bioactivity_class'] == 'Inactive']['pIC50'].tolist()
active_list = final_df[final_df['bioactivity_class'] == 'Active']['pIC50'].tolist()
intermediate_list = final_df[final_df['bioactivity_class'] == 'Intermediate']['pIC50'].tolist()

# print(inactive_list)
# print(active_list)
# print(len(inactive_list))
# print(len(active_list))

# Histplot of the 3 classes:

sns.histplot(inactive_list, kde=True)
plt.title("Histogram of Drugs labelled Inactive")
plt.show()

sns.histplot(active_list, kde=True)
plt.title("Histogram of Drugs labelled Active")
plt.show()

sns.histplot(intermediate_list, kde=True)
plt.title("Histogram of Drugs labelled Intermediate")
plt.show()

# Test and Run :
pIC50_res = choose_test_and_run("pIC50")
print(pIC50_res)

# For MW
print("**************************************************************************************************************")
print("MW")
# Boxplot

plt.figure(figsize=(5.5, 5.5))

sns.boxplot(x='bioactivity_class', y='MW', data=final_df, hue="bioactivity_class")

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('MW', fontsize=14, fontweight='bold')
plt.show()

inactive_list_mw = final_df[final_df['bioactivity_class'] == 'Inactive']['MW'].tolist()
active_list_mw = final_df[final_df['bioactivity_class'] == 'Active']['MW'].tolist()
intermediate_list_mw = final_df[final_df['bioactivity_class'] == 'Intermediate']['MW'].tolist()
# Histplot of the 3 classes:

sns.histplot(inactive_list_mw, kde=True)
plt.title("Histogram of Drugs labelled Inactive(MW)")
plt.show()

sns.histplot(active_list_mw, kde=True)
plt.title("Histogram of Drugs labelled Active(MW)")
plt.show()

sns.histplot(intermediate_list_mw, kde=True)
plt.title("Histogram of Drugs labelled Intermediate(MW)")
plt.show()

# Test and Run :
mw_res = choose_test_and_run("MW")
print(mw_res)

# For LogP
print("**************************************************************************************************************")
print("LogP")
# Boxplot

plt.figure(figsize=(5.5, 5.5))

sns.boxplot(x='bioactivity_class', y='LogP', data=final_df, hue="bioactivity_class")

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('LogP', fontsize=14, fontweight='bold')
plt.show()

# Shapiro-Wilk:

inactive_list_logp = final_df[final_df['bioactivity_class'] == 'Inactive']['LogP'].tolist()
active_list_logp = final_df[final_df['bioactivity_class'] == 'Active']['LogP'].tolist()
intermediate_list_logp = final_df[final_df['bioactivity_class'] == 'Intermediate']['LogP'].tolist()
# Histplot of the 3 classes:

sns.histplot(inactive_list_logp, kde=True)
plt.title("Histogram of Drugs labelled Inactive(LogP)")
plt.show()

sns.histplot(active_list_logp, kde=True)
plt.title("Histogram of Drugs labelled Active(LogP)")
plt.show()

sns.histplot(intermediate_list_logp, kde=True)
plt.title("Histogram of Drugs labelled Intermediate(LogP)")
plt.show()

# Test and Run :
logp_res = choose_test_and_run("LogP")
print(logp_res)

# For NumHDonors
print("**************************************************************************************************************")
print("NumHDonors")
# Boxplot

plt.figure(figsize=(5.5, 5.5))

sns.boxplot(x='bioactivity_class', y='NumHDonors', data=final_df, hue="bioactivity_class")

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('Number of H Donors', fontsize=14, fontweight='bold')
plt.show()

# Shapiro-Wilk:

inactive_list_nhd = final_df[final_df['bioactivity_class'] == 'Inactive']['NumHDonors'].tolist()
active_list_nhd = final_df[final_df['bioactivity_class'] == 'Active']['NumHDonors'].tolist()
intermediate_list_nhd = final_df[final_df['bioactivity_class'] == 'Intermediate']['NumHDonors'].tolist()

# Histplot of the 2 classes:

sns.histplot(inactive_list_nhd, kde=True)
plt.title("Histogram of Drugs labelled Inactive(NumHDonors)")
plt.show()

sns.histplot(active_list_nhd, kde=True)
plt.title("Histogram of Drugs labelled Active(NumHDonors)")
plt.show()

sns.histplot(intermediate_list_nhd, kde=True)
plt.title("Histogram of Drugs labelled Intermediate(NumHDonors)")
plt.show()

# Test and Run :
nhd_res = choose_test_and_run("NumHDonors")
print(nhd_res)

# For NumHAcceptors
print("**************************************************************************************************************")
print("NumHAcceptors")
# Boxplot

plt.figure(figsize=(5.5, 5.5))

sns.boxplot(x='bioactivity_class', y='NumHAcceptors', data=final_df, hue="bioactivity_class")

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('Number of H Acceptors', fontsize=14, fontweight='bold')
plt.show()

# Shapiro-Wilk:

inactive_list_nha = final_df[final_df['bioactivity_class'] == 'Inactive']['NumHAcceptors'].tolist()
active_list_nha = final_df[final_df['bioactivity_class'] == 'Active']['NumHAcceptors'].tolist()
intermediate_list_nha = final_df[final_df['bioactivity_class'] == 'Intermediate']['NumHAcceptors'].tolist()

# Histplot of the 2 classes:

sns.histplot(inactive_list_nha, kde=True)
plt.title("Histogram of Drugs labelled Inactive(NumHAcceptors)")
plt.show()

sns.histplot(active_list_nha, kde=True)
plt.title("Histogram of Drugs labelled Active(NumHAcceptors)")
plt.show()

sns.histplot(intermediate_list_nha, kde=True)
plt.title("Histogram of Drugs labelled Intermediate(NumHAcceptors)")
plt.show()

# Test and Run :
nhd_res = choose_test_and_run("NumHAcceptors")
print(nhd_res)
