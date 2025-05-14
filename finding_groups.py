import random
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, roc_curve
from copy import deepcopy
import numpy as np
import math
from collections import defaultdict, Counter

# Change this threshold to generate a different grouping
threshold = 0.75
file_path = 'grouping/grouping_750.pkl'


loci_df = pd.read_csv("Data/esm2_embeddings_loci.csv")
rbp_df = pd.read_csv("Data/esm2_embeddings_rbp.csv")


def dataframe_to_padded_dict(df, key_column, exclude_columns, min_size):
    result_dict = defaultdict(list)
    
    # Identify columns to include (all except key and excluded ones)
    value_columns = [col for col in df.columns if col not in exclude_columns and col != key_column]
    
    for _, row in df.iterrows():
        key = row[key_column]
        values = row[value_columns].tolist()
        result_dict[key].extend(values)  # Append directly to a single list
    
    # Apply zero-padding if needed
    for key in result_dict:
        while len(result_dict[key]) < min_size:
            result_dict[key].append(0)

        if len(result_dict[key]) > min_size:
            result_dict[key] = result_dict[key][:min_size]

    return dict(result_dict)

# Convert dataframe to dictionary using 'ID' as key, excluding 'Category' and 'Name', and padding to length 6

def dict_to_dataframe(data_dict):
    # Find the max number of columns needed
    max_length = max(len(v) for v in data_dict.values())
    
    # Convert dictionary to list of rows
    rows = []
    for key, values in data_dict.items():
        # Ensure all lists are of max length (already padded)
        row = [key] + values
        rows.append(row)
    
    # Create column names
    columns = ["Phage_ID"] + [str(i+1) for i in range(max_length)]
    
    # Convert to DataFrame
    return pd.DataFrame(rows, columns=columns)

# Convert back to DataFrame
# rbp_df_reconstructed = dict_to_dataframe(result)



mode = "train"

RBP_embeddings = deepcopy(rbp_df)
loci_embeddings = deepcopy(loci_df)

interactions = pd.read_csv('Data/phage_host_interactions.csv', index_col=0)

# Comment out this block to work with extended embeddings
################################################################################

# construct multi-RBP representations
multi_embeddings = []
names = []
for phage_id in list(set(RBP_embeddings['phage_ID'])):
    rbp_embeddings = RBP_embeddings.iloc[:,2:][RBP_embeddings['phage_ID'] == phage_id]
    multi_embedding = np.mean(np.asarray(rbp_embeddings), axis=0)
    names.append(phage_id)
    multi_embeddings.append(multi_embedding)

multiRBP_embeddings = pd.concat([pd.DataFrame({'phage_ID': names}), pd.DataFrame(multi_embeddings)], axis=1)
# 105 rows × 1281 columns
# column "phage_ID", and 1280 columns of averages of proteins

################################################################################

# COMMENT IN THIS LINE TO WORK WITH EXTENDED EMBEDDINGS, also shift to "Phage_ID" in the lines below
# multiRBP_embeddings = deepcopy(rbp_df_reconstructed)

# construct dataframe for training
features_lan = []
labels = []
groups_loci = []
groups_phage = []

keys_features_lan = []

for i, accession in enumerate(loci_embeddings['accession']):
    for j, phage_id in enumerate(multiRBP_embeddings['phage_ID']): # use "phage_ID" if running the code with the original features
        if mode == 'train':
            interaction = interactions.loc[accession][phage_id]
            if math.isnan(interaction) == False: # if the interaction is known
                # language embeddings
                features_lan.append(pd.concat([loci_embeddings.iloc[i, 1:], multiRBP_embeddings.iloc[j, 1:]]))

                # append labels and groups
                labels.append(int(interaction))
                groups_loci.append(i)
                groups_phage.append(j)
                keys_features_lan.append((accession, phage_id)) 

        elif mode == 'test':
            # language embeddings
            features_lan.append(pd.concat([loci_embeddings.iloc[i, 1:], multiRBP_embeddings.iloc[j, 1:]]))
            
            # append groups
            groups_loci.append(i)
            groups_phage.append(j)

            
features_lan = np.asarray(features_lan)
# 10'006 interactions are known (0 or 1), number of rows
# 2'560 concatenated features, number of columns

print("Dimensions match?", features_lan.shape[1] == (loci_embeddings.shape[1]+multiRBP_embeddings.shape[1]-2))

#np.save(general_path+'/esm2_features'+data_suffix+'.txt', features_lan)
# if mode == 'train':
# features_lan, labels, groups_loci, groups_phage


labels = np.asarray(labels)

# ESM-2 FEATURES + XGBoost model
features_esm2 = features_lan

imbalance = sum([1 for i in labels if i==1]) / sum([1 for i in labels if i==0])
xgb = XGBClassifier(scale_pos_weight=1/imbalance, learning_rate=0.3, n_estimators=250, max_depth=9,
                    eval_metric='logloss', use_label_encoder=False, 
                    tree_method = "hist", device = "cuda")  # Uses GPU for inference (optional))
xgb.fit(features_esm2, labels)
# xgb.save_model('phagehostlearn_vbeta.json')

# if we want to set a threshold for grouping
matrix = np.loadtxt('all_loci_score_matrix.txt', delimiter='\t')
# matrix = features_lan
group_i = 0
new_groups = [np.nan] * len(groups_loci)
for i in range(matrix.shape[0]):
    cluster = np.where(matrix[i,:] >= threshold)[0]
    oldgroups_i = [k for k, x in enumerate(groups_loci) if x in cluster]
    if i in groups_loci and np.isnan(new_groups[groups_loci.index(i)]):
        for ogi in oldgroups_i:
            new_groups[ogi] = group_i
        group_i += 1
groups_loci = new_groups
print('Number of unique groups: ', len(set(groups_loci)))
print(len(groups_loci))



def build_dict_and_check_conflicts(keys_list, values_list, key_selector=None):
    """
    Builds a dictionary from keys_list and values_list and checks for conflicts.
    
    Args:
        keys_list (list of tuples): The list containing tuple keys.
        values_list (list of int): The list containing integer values.
        key_selector (function, optional): A function to extract the desired key from the tuple.
                                           If None, the entire tuple is used as the key.

    Returns:
        dict: A dictionary with selected keys and their associated value.
        int: The number of keys associated with multiple different values.
        dict: Detailed conflicts with the counts of occurrences.
    """
    key_value_map = defaultdict(list)

    # Build the dictionary (storing all values per key to check for conflicts)
    for k, v in zip(keys_list, values_list):
        key = k if key_selector is None else key_selector(k)
        key_value_map[key].append(v)

    # Detect conflicts (if a key has multiple distinct values)
    conflict_count = 0
    conflicts_detail = {}
    final_dict = {}

    for key, vals in key_value_map.items():
        unique_vals = set(vals)
        if len(unique_vals) > 1:
            conflict_count += 1
            conflicts_detail[key] = Counter(vals)
        # Store the most common value or the first if no preference is defined
        final_dict[key] = max(unique_vals, key=vals.count)

    return final_dict, conflict_count, conflicts_detail



dict_first, conflicts_first, details_first = build_dict_and_check_conflicts(
    keys_features_lan, groups_loci, key_selector=lambda x: x[0])
print("### First Element as Key ###")
print(f"Number of conflicting keys: {conflicts_first}")
if conflicts_first:
    print("Conflicts detail:", details_first)
print("Resulting dictionary:\n", dict_first, "\n")


# --- Save (dump) the tuple into a pickle file ---
with open(file_path, 'wb') as f:
    pickle.dump(dict_first, f)