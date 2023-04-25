import os
import pandas as pd
import numpy as np


data_ini = pd.read_csv("./ml-100k/u.data",sep='\t', header=None)
data_ini = data_ini.drop(3, axis=1)

# the idea is to create a 200*200 matrix with roughly 50% sparsity, based on the assumption that 50% of the survey entry is noisy.
# we take 200 mostly rated movies and 200 most frequent rating user, thus the sparsity is roughly 50%

top_users = data_ini[0].value_counts()[:200]
top_movies = data_ini[1].value_counts()[:200]

filtered_data = data_ini[(data_ini[0].isin(top_users.index.tolist())) & (data_ini[1].isin(top_movies.index.tolist()))]
# print(filtered_data.count) 
# the sparsity is roughly 50%

df = filtered_data.pivot_table(index=0, columns=1, values=2, aggfunc='first')
print("the original 200*200 data matrix without synthetic imputation:", df.head(50))

# create df_fill3 to replace all the NaN by 3, based on the assumption that classmate will rate unfamiliar restaurant 3
df_fill3 = df.fillna(3)
df_fill3.to_csv(os.path.join("noisy_truth_matrix", f"df_fill3.csv"))

# create df_c to replace all the NaN by user's mean rating minus c, with c values 0.1, 0.2, 0.3, 0.5, 0.75, 1
# based on the assumption that classmate will rate unfamiliar restaurant c points below his average

c_values = [0.1, 0.2, 0.3, 0.5, 0.75, 1]
output_dir = "noisy_truth_matrix"
os.makedirs(output_dir, exist_ok=True)
for c in c_values:
    row_means = df.mean(axis=1, skipna=True)

    df_c = df.apply(lambda x: x.fillna(row_means[x.index] - c))

    print(f"DataFrame with NaN values replaced by row mean - {c}:")
    print(df_c)
    print("\n")
    output_file = os.path.join(output_dir, f"df_c_{c}.csv")
    df_c.to_csv(output_file)
                
