"""
Native_guide_DBA_lib.py

Library for multivariate time series data loading, preprocessing, neighbor retrieval,
counterfactual generation, and visualization for classification tasks.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tslearn.neighbors import KNeighborsTimeSeries
from tslearn.barycenters import dtw_barycenter_averaging
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy
from datetime import timedelta
import matplotlib.pyplot as plt
import joblib

# ==================
# Data Loading Utils
# ==================

def load_labels(label_file):
    """
    Load class labels from a CSV file.
    """
    df_labels = pd.read_csv(label_file)
    label_map = dict(zip(df_labels['File'], df_labels['Label']))
    return label_map

def load_mvts_and_labels(data_dir, label_map,
                         start_offset_min=300, end_offset_min=660,
                         cols_to_keep=None, exclude_files=None):
    """
    Load multivariate time series, labels, and metadata from directory.
    """
    if exclude_files is None:
        exclude_files = []
    X_list, y, meta_list = [], [], []
    for fname in os.listdir(data_dir):
        if fname.endswith('.csv') and fname in label_map and fname not in exclude_files:
            file_path = os.path.join(data_dir, fname)
            ts_df = pd.read_csv(file_path, sep=",")
            ts_df = ts_df.rename(columns={'time_tag': 'time_stamp'})
            ts_df['time_stamp'] = pd.to_datetime(ts_df['time_stamp'], format='%Y-%m-%d %H:%M:%S')
            event_start = ts_df['time_stamp'].iloc[0] + timedelta(minutes=start_offset_min)
            event_end = ts_df['time_stamp'].iloc[0] + timedelta(minutes=end_offset_min)
            df_obs = ts_df[(ts_df['time_stamp'] >= event_start) & (ts_df['time_stamp'] < event_end)].copy()
            row_dict = {col: pd.Series(df_obs[col].values) for col in cols_to_keep}
            X_list.append(row_dict)
            y.append(label_map[fname])
            meta_list.append({
                "filename": fname,
                "timestamps": df_obs["time_stamp"].tolist()
            })
    X = pd.DataFrame(X_list)
    y = np.array(y)
    meta = pd.DataFrame(meta_list)
    return X, y, meta

def train_val_test_split(X, y, meta, test_size=0.15, val_size=0.17, random_state=42):
    """
    Stratified splitting into train, val, test sets (with metadata).
    """
    X_temp, X_test, y_temp, y_test, meta_temp, meta_test = train_test_split(
        X, y, meta, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
        X_temp, y_temp, meta_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )
    return (X_train, X_val, X_test, y_train, y_val, y_test, meta_train, meta_val, meta_test)

def convert_to_3d_numpy(df, cols_to_keep):
    """
    Convert DataFrame to 3D numpy array (samples, channels, timepoints).
    """
    return from_nested_to_3d_numpy(df[cols_to_keep])

# =========================
# Time Series Model & NN
# =========================

def native_guide_retrieval_mvts(query, predicted_label, y_train, X_train_3D, distance="dtw", n_neighbors=3):
    """
    Retrieve nearest unlike neighbors for a multivariate time series query.
    """
    unlike_indices = np.where(y_train != predicted_label)[0]
    X_unlike = X_train_3D[unlike_indices, :, :]
    knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric=distance)
    knn.fit(X_unlike)
    query_reshaped = query.reshape(1, query.shape[0], query.shape[1])
    dist, ind = knn.kneighbors(query_reshaped, return_distance=True)
    neighbor_indices = unlike_indices[ind[0]]
    return dist[0], neighbor_indices

def target_mvts(instance_3d, model):
    """
    Find the second-most likely class for a multivariate time series.
    """
    instance_batch = instance_3d[np.newaxis, :, :]
    probs = model.predict_proba(instance_batch)[0]
    target_class = np.argsort(probs)[-2]
    return target_class

def get_generated_cf(query, insample_cf, model, pred_threshold=0.5, beta_step=0.01):
    """
    Generate a counterfactual time series using DTW barycenter averaging.
    """
    beta = 0.0
    target_class = target_mvts(query, model)  # second most likely class for the query timeseries
    generated_cf = dtw_barycenter_averaging([query, insample_cf], weights=np.array([1-beta, beta]))
    prob_of_generated_cf_for_target_class = model.predict_proba(generated_cf[np.newaxis, :, :])[0][target_class] # checking the prediction probability for the barycenter being classified as a opposite class
    while prob_of_generated_cf_for_target_class < pred_threshold and beta <= 1.0:
        beta += beta_step
        generated_cf = dtw_barycenter_averaging([query, insample_cf], weights=np.array([1-beta, beta]))
        prob_of_generated_cf_for_target_class = model.predict_proba(generated_cf[np.newaxis, :, :])[0][target_class]
    # print("Beta used for CF:", beta)
    # print("Probability for target class:", prob_of_generated_cf_for_target_class)
    return generated_cf

# ==================
# Visualization
# ==================

def plot_query(query, query_label, cols_to_keep):
    """Plot a multivariate time series query."""
    n_channels = query.shape[0]
    plt.figure(figsize=(12, 3 * n_channels))
    for i, ch in enumerate(cols_to_keep):
        plt.subplot(n_channels, 1, i+1)
        plt.plot(query[i], label=f"Query (Class Label:{query_label})", color='red', linewidth=2)
        plt.title(f"{ch}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
    plt.tight_layout()
    plt.show()

def plot_query_and_neighbors(query, query_label, X_train_3D, y_train, neighbor_indices, cols_to_keep, title_suffix=""):
    """Plot query and nearest unlike neighbors."""
    n_channels = query.shape[0]
    plt.figure(figsize=(12, 3 * n_channels))
    for i, ch in enumerate(cols_to_keep):
        plt.subplot(n_channels, 1, i+1)
        plt.plot(query[i], label=f"Query (Class Label:{query_label})", color='red', linewidth=2)
        for j, idx in enumerate(neighbor_indices):
            plt.plot(X_train_3D[idx, i, :], label=f"Unlike Neighbor {j+1}(Class:{y_train[idx]})", linestyle='--')
        plt.title(f"{ch} {title_suffix}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
    plt.tight_layout()
    plt.show()

def plot_mvts_cf(query, query_label, generated_cf, generated_cf_label, cols_to_keep, title="Query vs Counterfactual"):
    """Plot original and counterfactual time series."""
    n_channels = query.shape[0]
    timepoints = query.shape[1]
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 2 * n_channels), sharex=True)
    if n_channels == 1:
        axes = [axes]
    for i, ch in enumerate(cols_to_keep):
        axes[i].plot(range(timepoints), query[i], label=f"Query (class_label:{query_label})", alpha=0.7)
        axes[i].plot(range(timepoints), generated_cf[i], label=f"Counterfactual (class label:{generated_cf_label})", alpha=0.7, linestyle="--")
        axes[i].set_title(f"{ch}")
        axes[i].legend()
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# ==================
# Example Usage
# ==================
#
# label_file = '../../data/raw/class_labels.csv'
# data_dir = '../../data/raw/data/'
# cols_to_keep = ['p3_flux_ic', 'p5_flux_ic', 'p7_flux_ic', 'long']
#
# label_map = load_labels(label_file)
# X, y, meta = load_mvts_and_labels(data_dir, label_map, cols_to_keep=cols_to_keep, exclude_files=['2005-09-07_14-25.csv'])
#
# X_train, X_val, X_test, y_train, y_val, y_test, meta_train, meta_val, meta_test = train_val_test_split(X, y, meta)
# X_train_3D = convert_to_3d_numpy(X_train, cols_to_keep)
# X_val_3D   = convert_to_3d_numpy(X_val, cols_to_keep)
# X_test_3D  = convert_to_3d_numpy(X_test, cols_to_keep)
#
# model = joblib.load('../../models/KNN_1_TS__classifier_v1.3.pkl')
#
# query = X_test_3D[0]
# query_label = y_test[0]
# dist, neighbor_indices = native_guide_retrieval_mvts(query, query_label, y_train, X_train_3D)
# plot_query_and_neighbors(query, query_label, X_train_3D, y_train, neighbor_indices, cols_to_keep)
#
# insample_cf = X_train_3D[neighbor_indices[0]]
# generated_cf = get_generated_cf(query, insample_cf, model)
# cf_label = target_mvts(query, model)
# plot_mvts_cf(query, query_label, generated_cf, cf_label, cols_to_keep)
#
# ==================

