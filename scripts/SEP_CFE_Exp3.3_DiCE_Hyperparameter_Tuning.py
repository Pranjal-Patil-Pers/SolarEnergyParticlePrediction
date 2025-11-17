import joblib
import numpy as np
import pandas as pd
import os
from dice_ml import Dice
import wandb
from scripts.SEP_CFE_functions import *
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

# === Logging the Results === #
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
wandb.init(
    entity="gsu-dmlab",
    project="SEPCFE",  # You choose the project name
    name="Exp3.3_Dice_Hyperparameter_Tuning",  # Give each run a name
)


# === Load Models ===
best_rf = joblib.load('../models/RandomForestClassifier_model_sep_cfe.pkl')
exp_random = joblib.load("../models/sep_cfe_random_explainer.pkl")
exp_genetic = joblib.load("../models/sep_cfe_genetic_explainer.pkl")

# === Load a representative test instance ===
# Replace this with how you actually load your test instance
# Should be a pandas DataFrame with one row
df_combined_labels=pd.read_csv('../data/processed/df_combined_labels.csv',sep=',',index_col=0)
query_ts_filename = '1986-02-07_01-00.csv'
query_instance1 = get_query_instance(query_ts_filename,df_combined_labels,best_rf)
# instance_df = pd.read_csv("../data/sample_test_instance.csv")  # shape: (1, d)
instance_df=query_instance1

# === Feature Weights Dictionary === #
feature_names = df_combined_labels.columns
k=50
sorted_normalized_weights_df  = get_feature_importance(best_rf,feature_names)
top_k_features = sorted_normalized_weights_df['Feature'].head(k).to_list()
top_k_normalized_weights_dict = sorted_normalized_weights_df.head(k).set_index('Feature')['Normalized Importance'].to_dict()
normalized_weights = {f: top_k_normalized_weights_dict.get(f, 0.01) for f in feature_names}




# === Helper Functions ===
def compute_proximity(cfs, instance):
    return np.mean([np.linalg.norm(cf - instance) for cf in cfs])

def compute_diversity(cfs):
    pairwise_dists = [np.linalg.norm(cfs[i] - cfs[j])
                      for i in range(len(cfs)) for j in range(i+1, len(cfs))]
    return np.mean(pairwise_dists) if pairwise_dists else 0.0

def compute_sparsity(cfs, instance):
    diff_matrix = (cfs != instance).astype(int)
    return np.mean(np.sum(diff_matrix, axis=1))



# === Experiment Config ===
proximity_weights = [0.1, 0.5, 1, 5]
diversity_weights = [0.1, 0.5, 1, 5]
feature_weights_options = {
    "uniform": None,
    "exponential": {f: 2 ** i for i, f in enumerate(feature_names)},
    "rf_importance": normalized_weights
}
total_CFs = 4
desired_class = "opposite"

results = []

# === Run experiments ===
for explainer_name, explainer in [("genetic", exp_genetic)]:  #("random", exp_random),
    for proximity_weight in proximity_weights:
        for diversity_weight in diversity_weights:
            for fw_name, fw_dict in feature_weights_options.items():
                try:
                    gen_cf = explainer.generate_counterfactuals(
                        instance_df,
                        total_CFs=total_CFs,
                        desired_class=desired_class,
                        proximity_weight=proximity_weight,
                        diversity_weight=diversity_weight,
                        feature_weights=fw_dict
                    )
                    cf_df = gen_cf.cf_examples_list[0].final_cfs_df
                    cfs = cf_df.values
                    instance_vals = instance_df.values[0]

                    proximity = compute_proximity(cfs, instance_vals)
                    diversity = compute_diversity(cfs)
                    sparsity = compute_sparsity(cfs, instance_vals)
                    metrics = {
                        "explainer": explainer_name,
                        "proximity_weight": proximity_weight,
                        "diversity_weight": diversity_weight,
                        "feature_weights": fw_name,
                        "avg_proximity": proximity,
                        "avg_diversity": diversity,
                        "avg_sparsity": sparsity,
                        "num_counterfactuals": len(cfs)
                    }

                    # Log to wandb and local results list
                    wandb.log(metrics)
                    results.append(metrics)

                except Exception as e:
                    print(f"[Error] {explainer_name} | prox={proximity_weight}, div={diversity_weight}, fw={fw_name}")
                    print(str(e))

# === Save Results ===

results_df = pd.DataFrame(results)
os.makedirs("results", exist_ok=True)
results_df.to_csv(f"../GSEP/evaluation_results/experiment3.3_hpt_{run_id}.csv", index=False)

print(f"Experiment completed. Results saved to '../GSEP/evaluation_results/experiment3.3_hpt_{run_id}.csv'.")
wandb.finish()
