"""
python_mvtslib.py

Class-based library for multivariate time series data loading, preprocessing,
nearest-neighbor retrieval, counterfactual generation, and visualization.
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

class DataLoader:
    """
    Handles reading class labels and time series data from disk.
    """
    def __init__(self, label_file, data_dir, cols_to_keep, start_offset_min=300, end_offset_min=660, exclude_files=None):
        self.label_file = label_file
        self.data_dir = data_dir
        self.cols_to_keep = cols_to_keep
        self.start_offset_min = start_offset_min
        self.end_offset_min = end_offset_min
        self.exclude_files = exclude_files if exclude_files is not None else []

    def load_labels(self):
        df = pd.read_csv(self.label_file)
        self.label_map = dict(zip(df['File'], df['Label']))
        return self.label_map

    def load_data(self):
        X_list, y, meta_list = [], [], []
        label_map = self.load_labels()
        for fname in os.listdir(self.data_dir):
            if fname.endswith('.csv') and fname in label_map and fname not in self.exclude_files:
                file_path = os.path.join(self.data_dir, fname)
                ts_df = pd.read_csv(file_path)
                ts_df = ts_df.rename(columns={'time_tag': 'time_stamp'})
                ts_df['time_stamp'] = pd.to_datetime(ts_df['time_stamp'], format='%Y-%m-%d %H:%M:%S')
                event_start = ts_df['time_stamp'].iloc[0] + timedelta(minutes=self.start_offset_min)
                event_end = ts_df['time_stamp'].iloc[0] + timedelta(minutes=self.end_offset_min)
                df_obs = ts_df[(ts_df['time_stamp'] >= event_start) & (ts_df['time_stamp'] < event_end)].copy()
                row_dict = {col: pd.Series(df_obs[col].values) for col in self.cols_to_keep}
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

class Preprocessor:
    """
    Handles dataset splitting and conversion to 3D numpy format.
    """
    def __init__(self, cols_to_keep):
        self.cols_to_keep = cols_to_keep

    def split(self, X, y, meta, test_size=0.15, val_size=0.17, random_state=42):
        X_temp, X_test, y_temp, y_test, meta_temp, meta_test = train_test_split(
            X, y, meta, test_size=test_size, random_state=random_state, stratify=y
        )
        X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
            X_temp, y_temp, meta_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )
        return (X_train, X_val, X_test, y_train, y_val, y_test, meta_train, meta_val, meta_test)

    def to_3d(self, df):
        return from_nested_to_3d_numpy(df[self.cols_to_keep])

class MVTSModel:
    """
    Utilities for prediction and counterfactual computation.
    """
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict_proba(self, instance_3d):
        return self.model.predict_proba(instance_3d[np.newaxis, :, :])[0]

    def target_class(self, instance_3d):
        probs = self.predict_proba(instance_3d)
        return np.argsort(probs)[-2]

    def generate_cf(self, query, insample_cf, pred_threshold=0.5, beta_step=0.01):
        beta = 0.0
        target_class = self.target_class(query)
        generated_cf = dtw_barycenter_averaging([query, insample_cf], weights=np.array([1-beta, beta]))
        prob_cf = self.model.predict_proba(generated_cf[np.newaxis, :, :])[0][target_class]
        while prob_cf < pred_threshold and beta <= 1.0:
            beta += beta_step
            generated_cf = dtw_barycenter_averaging([query, insample_cf], weights=np.array([1-beta, beta]))
            prob_cf = self.model.predict_proba(generated_cf[np.newaxis, :, :])[0][target_class]
        print("Beta used for CF:", beta)
        print("Probability for target class:", prob_cf)
        return generated_cf, target_class

class MVTSNeighbors:
    """
    Finds nearest unlike neighbors for a time series query.
    """
    def __init__(self, X_train_3D, y_train, distance="dtw", n_neighbors=3):
        self.X_train_3D = X_train_3D
        self.y_train = y_train
        self.distance = distance
        self.n_neighbors = n_neighbors

    def find_unlike(self, query, predicted_label):
        unlike_indices = np.where(self.y_train != predicted_label)[0]
        X_unlike = self.X_train_3D[unlike_indices, :, :]
        knn = KNeighborsTimeSeries(n_neighbors=self.n_neighbors, metric=self.distance)
        knn.fit(X_unlike)
        query_reshaped = query.reshape(1, query.shape[0], query.shape[1])
        dist, ind = knn.kneighbors(query_reshaped, return_distance=True)
        neighbor_indices = unlike_indices[ind[0]]
        return dist[0], neighbor_indices

class MVTSPlotter:
    """
    Plots single queries, neighbors, and counterfactuals.
    """
    def __init__(self, cols_to_keep):
        self.cols_to_keep = cols_to_keep

    def plot_query(self, query, query_label):
        n_channels = query.shape[0]
        plt.figure(figsize=(12, 3 * n_channels))
        for i, ch in enumerate(self.cols_to_keep):
            plt.subplot(n_channels, 1, i+1)
            plt.plot(query[i], label=f"Query (Class Label:{query_label})", color='red', linewidth=2)
            plt.title(f"{ch}")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_query_and_neighbors(self, query, query_label, X_train_3D, y_train, neighbor_indices, title_suffix=""):
        n_channels = query.shape[0]
        plt.figure(figsize=(12, 3 * n_channels))
        for i, ch in enumerate(self.cols_to_keep):
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

    def plot_mvts_cf(self, query, query_label, generated_cf, generated_cf_label, title="Query vs Counterfactual"):
        n_channels = query.shape[0]
        timepoints = query.shape[1]
        fig, axes = plt.subplots(n_channels, 1, figsize=(12, 2 * n_channels), sharex=True)
        if n_channels == 1:
            axes = [axes]
        for i, ch in enumerate(self.cols_to_keep):
            axes[i].plot(range(timepoints), query[i], label=f"Query (Class:{query_label})", alpha=0.7)
            axes[i].plot(range(timepoints), generated_cf[i], label=f"Counterfactual (Class:{generated_cf_label})", alpha=0.7, linestyle="--")
            axes[i].set_title(f"{ch}")
            axes[i].legend()
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

# ===========================
# Example Usage (not executed)
# ===========================
#
# # Initialize loader
# loader = DataLoader(
#     label_file='../../data/raw/class_labels.csv',
#     data_dir='../../data/raw/data/',
#     cols_to_keep=['p3_flux_ic', 'p5_flux_ic', 'p7_flux_ic', 'long'],
#     exclude_files=['2005-09-07_14-25.csv']
# )
#
# X, y, meta = loader.load_data()
#
# # Preprocess
# processor = Preprocessor(cols_to_keep=loader.cols_to_keep)
# X_train, X_val, X_test, y_train, y_val, y_test, meta_train, meta_val, meta_test = processor.split(X, y, meta)
# X_train_3D = processor.to_3d(X_train)
# X_val_3D   = processor.to_3d(X_val)
# X_test_3D  = processor.to_3d(X_test)
#
# # Model and neighbors
# model = MVTSModel(model_path='../../models/KNN_1_TS__classifier_v1.3.pkl')
# neighbors_util = MVTSNeighbors(X_train_3D, y_train)
#
# # Pick a test query
# query = X_test_3D[0]
# query_label = y_test[0]
#
# dist, neighbor_indices = neighbors_util.find_unlike(query, query_label)
#
# plotter = MVTSPlotter(cols_to_keep=loader.cols_to_keep)
# plotter.plot_query_and_neighbors(query, query_label, X_train_3D, y_train, neighbor_indices)
#
# insample_cf = X_train_3D[neighbor_indices[0]]
# generated_cf, cf_label = model.generate_cf(query, insample_cf)
# plotter.plot_mvts_cf(query, query_label, generated_cf, cf_label)

