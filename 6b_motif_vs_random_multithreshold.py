# === CONFIGURABLE MODEL TO COMPARE MOTIF VS. RANDOM PROTEIN ===

import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBClassifier
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, auc, roc_curve
import matplotlib.pyplot as plt
import pickle
import os.path
import random

predsss = []

# === LOAD MAIN DATASET ===
if not os.path.isfile('combined_embeddings_per_protein.csv'):
    embeddings_loci_protein = pd.read_csv("esm2_embeddings_loci_per_protein.csv")
    embeddings_rbp = pd.read_csv("esm2_embeddings_rbp.csv")
    phage_host_interactions = pd.read_csv('phage_host_interactions.csv')

    interactions_melted = phage_host_interactions.melt(
        id_vars=['Unnamed: 0'], var_name='phage_ID', value_name='label'
    ).rename(columns={'Unnamed: 0': 'accession'})
    interactions_melted = interactions_melted.dropna(subset=['label'])

    merged = interactions_melted.merge(embeddings_loci_protein, on='accession', how='inner')
    merged = merged.merge(embeddings_rbp, on='phage_ID', how='inner')

    host_embedding_cols = [col for col in merged.columns if col.endswith('_x')]
    virus_embedding_cols = [col for col in merged.columns if col.endswith('_y')]

    host_embeddings = merged[host_embedding_cols].astype(np.float32)
    virus_embeddings = merged[virus_embedding_cols].astype(np.float32)

    host_embeddings.columns = [f"host_{i}" for i in range(host_embeddings.shape[1])]
    virus_embeddings.columns = [f"virus_{i}" for i in range(virus_embeddings.shape[1])]

    final_df = pd.concat([
        merged[['accession', 'phage_ID', 'protein_index', 'protein_ID']],
        host_embeddings,
        virus_embeddings,
        merged[['label']]
    ], axis=1)

    final_df.to_csv('combined_embeddings_per_protein.csv', index=False)
else:
    final_df = pd.read_csv('combined_embeddings_per_protein.csv', dtype={'accession': str})

# === LOAD ADDITIONAL DATA ===
df_sero = pd.read_csv("kaptive_results.tsv", sep="\t")
df_sero = df_sero[["Assembly", "Best match type", "Match confidence"]]

df_motifs = pd.read_csv("SingleHostProteins/full_onehost_found.csv")

with open("grouping/grouping_1.pkl", 'rb') as f:
    groups_dictionary = pickle.load(f)
final_df['group_loci'] = final_df['accession'].map(groups_dictionary)

# === PREPROCESS SEROTYPE ENCODING ===
df_sero_filtered = df_sero[
    (df_sero["Match confidence"] == "Typeable") &
    (df_sero["Best match type"] != "Capsule null")
].copy()

one_hot_sero = pd.get_dummies(df_sero_filtered["Best match type"], prefix="sero_")
df_sero_encoded = pd.concat([df_sero_filtered[["Assembly"]], one_hot_sero], axis=1)



thresholds = [1.0, 0.995, 0.99, 0.95, 0.9, 0.85, 0.8, 0.75]
tstr = ['100', '99.5', '99', '95', '90', '85', '80', '75']
grouping_files = [
    "grouping/grouping_1.pkl",
    "grouping/grouping_995.pkl",
    "grouping/grouping_990.pkl",
    "grouping/grouping_950.pkl",
    "grouping/grouping_900.pkl",
    "grouping/grouping_850.pkl",
    "grouping/grouping_800.pkl",
    "grouping/grouping_750.pkl"
]

for i, threshold in enumerate(thresholds):
    print(f"===== Running LOGO CV for threshold {tstr[i]}% =====")
    with open(grouping_files[i], 'rb') as f:
        groups_dictionary = pickle.load(f)
    final_df['group_loci'] = final_df['accession'].map(groups_dictionary)

    # === LOGO CROSS VALIDATION ===
    logo = LeaveOneGroupOut()
    scores_max, label_max = [], []
    all_predictions = []
    pbar = tqdm(total=len(set(final_df['group_loci'])), desc=f"LOGO CV @ 100%")
    
    for train_index, test_index in logo.split(final_df, final_df['label'], final_df['group_loci']):
        train_df = final_df.iloc[train_index].copy()
        test_df = final_df.iloc[test_index].copy()
    
        host_cols = [col for col in train_df.columns if col.startswith("host_")]
        train_df.drop(columns=host_cols, inplace=True)
        test_df.drop(columns=host_cols, inplace=True)
    
        train_df = train_df.merge(df_sero_encoded, how="left", left_on="accession", right_on="Assembly").drop(columns=["Assembly"])
        test_df = test_df.merge(df_sero_encoded, how="left", left_on="accession", right_on="Assembly").drop(columns=["Assembly"])
        train_df.fillna(0, inplace=True)
        test_df.fillna(0, inplace=True)
    
        # Identify motif positives and build random control
        train_df = train_df.merge(df_motifs.assign(motif_positive=1), on=["accession", "phage_ID", "protein_ID"], how="left")
        train_df["motif_positive"] = train_df["motif_positive"].fillna(0)
    
        pos_df = train_df[train_df["motif_positive"] == 1].copy()
    
        # Select a different protein from same phage-host pair
        def get_random_other(row):
            candidates = train_df[(train_df['accession'] == row['accession']) &
                                  (train_df['phage_ID'] == row['phage_ID']) &
                                  (train_df['protein_ID'] != row['protein_ID'])]
            return candidates.sample(1) if len(candidates) else row.to_frame().T
    
        neg_samples = pd.concat([get_random_other(row) for _, row in pos_df.iterrows()], ignore_index=True)
        neg_samples = neg_samples.drop(columns=['motif_positive'])
    
        # Label and combine
        pos_df['label'] = 1
        neg_samples['label'] = 0
        train_df = pd.concat([pos_df, neg_samples], ignore_index=True)
    
        # Train model
        feature_cols = [col for col in train_df.columns if col.startswith(('sero_', 'virus_'))]
        X_train = train_df[feature_cols].values
        y_train = train_df['label'].astype(int).values
    
        if len(set(y_train)) < 2:
            continue
    
        imbalance = sum(y_train == 1) / sum(y_train == 0) if sum(y_train == 0) else 1
    
        xgb = XGBClassifier(
            scale_pos_weight=1 / imbalance,
            learning_rate=0.3,
            n_estimators=250,
            max_depth=7,
            eval_metric='logloss',
            use_label_encoder=False,
            tree_method="gpu_hist",
            predictor="gpu_predictor",
            device="cuda"
        )
        xgb.fit(X_train, y_train)
    
        # Predict on test
        feature_cols_test = [col for col in test_df.columns if col.startswith(('sero_', 'virus_'))]
        X_test = test_df[feature_cols_test].values
        y_test = test_df['label'].astype(int).values if 'label' in test_df else np.zeros(len(test_df))
    
        score_xgb = xgb.predict_proba(X_test)[:, 1]
        test_df['score'] = score_xgb
        test_df['true_label'] = y_test
    
        all_predictions.append(test_df[['accession', 'phage_ID', 'protein_ID', 'score', 'true_label']])
    
        max_scores = test_df.groupby(['accession', 'phage_ID']).agg({
            'score': 'max',
            'true_label': 'first'
        }).reset_index()
    
        scores_max.append(max_scores['score'].values)
        label_max.append(max_scores['true_label'].values)
        pbar.update(1)
    
    pbar.close()


    if len(scores_max) == 0:
        print(f"No valid folds for threshold {tstr[i]}%. Skipping evaluation.")
        continue

    scores_max = np.concatenate(scores_max)
    label_max = np.concatenate(label_max)

    if len(set(label_max)) > 1:
        fpr, tpr, _ = roc_curve(label_max, scores_max)
        rauclr = round(auc(fpr, tpr), 3)
        print(f"Final AUC with motif vs. random protein @ {tstr[i]}%: {rauclr}")
        predsss.append((label_max, scores_max, rauclr))
    else:
        print(f"Final evaluation failed at {tstr[i]}% threshold due to single-class predictions.")

# === SAVE ALL RESULTS ===
with open('Results/6b_AUCs_motif_vs_random_multithreshold.pkl', 'wb') as f:
    pickle.dump(predsss, f)
