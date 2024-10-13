import pandas as pd
from chembl_webresource_client.new_client import new_client

target = new_client.target
target_query = target.search('Epidermal growth factor receptor erbB1')
targets_df = pd.DataFrame.from_dict(target_query)
print(targets_df)

selected_target = targets_df.target_chembl_id[3]
print(selected_target)

activity = new_client.activity
res = activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")

activity_df = pd.DataFrame.from_dict(res)

print(activity_df)
# activity_df.to_csv("Bioactivity_data.csv", index=False)
#
# # check for missing standard values
# activity_df2 = activity_df[activity_df.standard_value.notna()]
