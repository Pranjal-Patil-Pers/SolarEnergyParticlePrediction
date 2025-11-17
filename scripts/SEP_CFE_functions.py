import numpy as np
import pandas as pd
import re
from typing import Tuple, List
from datetime import timedelta
from collections import Counter

from sklearn.metrics import confusion_matrix


# Get feature importances
def get_feature_importance(model, feature_names):
    feature_importances = model.feature_importances_
    weight_vector = (feature_importances / np.sum(feature_importances))
    # Create dictionary
    normalized_weights_dict = dict(zip(feature_names, weight_vector))
    sorted_normalized_weights_dict = dict(
        sorted(normalized_weights_dict.items(), key=lambda item: item[1], reverse=True))
    sorted_normalized_weights_df = pd.DataFrame(list(sorted_normalized_weights_dict.items()),
                                                    columns=['Feature', 'Normalized Importance'])
    return sorted_normalized_weights_df

def get_query_instance(query_ts_filename, df_combined_labels, model):
    """
    Retrieve the query instance and its target label for a given filename.

    Args:
           query_ts_filename (str): The filename identifying the query instance.
           df_combined_labels (pd.DataFrame): DataFrame containing labeled data.
           model: A trained classifier with a `.predict()` method.

    Returns:
        Tuple[pd.DataFrame, Any]: The query instance (features only) and its true label.
     """
    query_instance_raw = df_combined_labels[df_combined_labels['File'] == query_ts_filename]
    query_instance = query_instance_raw.drop(["Label", "Event_Y_N", "Multi_Label", "File"], axis=1)

    true_label = query_instance_raw["Event_Y_N"].values[0]
    predicted_label = model.predict(query_instance)

    print("Target value for the query instance:\n", true_label)
    print("Predicted value:\n", predicted_label)

    return query_instance, true_label, predicted_label

def extract_slices_from_headers(header_list, pattern, target_metric):
    """
    Given a list of column names and a regex pattern, extract slices for a specific metric.

    Args:
        header_list (List[str]): List of column names.
        pattern (str): Regex pattern to match metric slices.
        target_metric (str): The metric to extract slice (start, end) tuples for.
    Returns:
        List[Tuple[int, int]]: Sorted list of (start, end) tuples for the target metric.
    """
    # Step 1: Match headers to extract relevant entries
    result = []
    for col in header_list:
        m = re.match(pattern, col)
        if m:
            metric = m.group(1)
            slice_str = m.group(2)
            start, end = map(int, slice_str.split(':'))
            result.append((metric, start, end))

    # Step 2: Filter for target metric and sort
    filtered = [(start, end) for metric, start, end in result if metric == target_metric]
    filtered.sort(key=lambda x: (x[0], x[1]))

    return filtered

def extract_feature_ranges(csv_path, delim, slices, top_k_features, flux_types):
    """
    Extract min-max ranges for selected time slices from a CSV file containing time series data.

    Args:
        csv_path (str): Full path to the CSV file (including filename).
        slices (List[Tuple[int, int]]): List of (start_minute, end_minute) tuples.
        top_k_features (List[str]): List of feature keys to include in the final output.

    Returns:
        Dict[str, List[float]]: Dictionary mapping feature slice keys to their [min, max] values.
    """
    # Load the data
    df = pd.read_csv(csv_path, delimiter=delim)
    df = df.rename(columns={'time_tag': 'time_stamp'})
    df['time_stamp'] = pd.to_datetime(df['time_stamp'], format='%Y-%m-%d %H:%M:%S')

    # Define observation window based on event offset
    event_start = df['time_stamp'].iloc[0] + timedelta(minutes=300)
    event_end = df['time_stamp'].iloc[0] + timedelta(minutes=660)

    # Filter the relevant time window
    df_obs = df[(df['time_stamp'] >= event_start) & (df['time_stamp'] < event_end)].copy()
    df_obs['minutes'] = (df_obs['time_stamp'] - event_start).dt.total_seconds() / 60

    # Compute min-max ranges for each slice
    range_dict = {}
    for flux in flux_types:
        for start_min, end_min in slices:
            slice_data = df_obs[(df_obs['minutes'] >= start_min) & (df_obs['minutes'] < end_min)]
            if slice_data.empty:
                continue
            key = f"{flux}_mean@[{start_min}:{end_min}]"
            ranges = [slice_data[flux].min(), slice_data[flux].max()]
            range_dict[key] = ranges

    # Filter using top_k_features
    filtered_dict = {k: v for k, v in range_dict.items() if k in top_k_features}
    return filtered_dict

def get_pertubed_series(csv_path, sample_cfe, flux_type, slices, start_offset_min=300, end_offset_min=660):
    df = pd.read_csv(csv_path, delimiter=',')
    df = df.rename(columns={'time_tag': 'time_stamp'})
    df['time_stamp'] = pd.to_datetime(df['time_stamp'], format='%Y-%m-%d %H:%M:%S')
    event_start = df['time_stamp'].iloc[0] + timedelta(minutes=start_offset_min)
    event_end = df['time_stamp'].iloc[0] + timedelta(minutes=end_offset_min)
    df_obs = df[(df['time_stamp'] >= event_start) & (df['time_stamp'] < event_end)].copy()
    df_obs['minutes'] = (df_obs['time_stamp'] - event_start).dt.total_seconds() / 60

    offset_accum = np.zeros_like(df_obs['minutes'], dtype=float)
    offset_count = np.zeros_like(df_obs['minutes'], dtype=int)

    for start_min, end_min in slices:
        slice_data = df_obs[(df_obs['minutes'] >= start_min) & (df_obs['minutes'] < end_min)]
        if slice_data.empty:
            continue
        mask = (df_obs['minutes'] >= start_min) & (df_obs['minutes'] < end_min)
        if np.sum(mask) == 0:
            continue
        flux_data = slice_data[flux_type].values
        pattern = f'^{flux_type}_mean@\\[{start_min}:{end_min}\\]$'
        cfe_value = sample_cfe.filter(regex=pattern).iloc[0]  # should be a single value
        global_adjustment = cfe_value - flux_data.mean()
        delta = flux_data + global_adjustment
        offset_accum[mask] += delta
        offset_count[mask] += 1

    final_offset = np.zeros_like(df_obs['minutes'], dtype=float)
    nonzero = offset_count > 0
    final_offset[nonzero] = offset_accum[nonzero] / offset_count[nonzero]

    original = df_obs[flux_type].values
    final_series = original + final_offset
    perturbed_series = pd.Series(final_series, index=df_obs.index)
    original_series = pd.Series(original, index=df_obs.index)
    min_y = min(final_series.min(), final_offset.min(), original.min())
    max_y = max(final_series.max(), final_offset.max(), original.max())
    return df_obs, perturbed_series, original_series, min_y, max_y

def TSS(y_true,y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    return  (TP / (TP + FN) - FP / (FP + TN))

def HSS(y_true,y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    return 2 * (TP * TN - FP * FN) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))


def analyze_counterfactuals(
        df: pd.DataFrame,
        min_max_dict: dict,
        query_instance,
        window_pattern=r'\[(.*?)\]'
):
    """
    Analyzes a DataFrame of counterfactuals for:
    1. Min/max violations.
    2. Features changed from a query instance (ordered by frequency).
    3. Windows changed (if feature names contain window pattern).

    Parameters:
        df (pd.DataFrame): DataFrame of instances to check.
        min_max_dict (dict): {feature: [min, max], ...}
        query_instance (pd.Series or pd.DataFrame): The original instance.
        window_pattern (str): Regex pattern to extract window from feature name.

    Returns:
        dict: {
            'violations': list of dicts,
            'changed_features_ordered': pd.Series,
            'changed_windows_ordered': list of (window, count)
        }
    """
    # --- 1. Min/max violations ---
    violations = []
    for col, (min_val, max_val) in min_max_dict.items():
        if col in df.columns:
            out_of_range = (df[col] < min_val) | (df[col] > max_val)
            if out_of_range.any():
                violating_rows = df.index[out_of_range].tolist()
                violations.append({'column': col, 'rows': violating_rows, 'min': min_val, 'max': max_val})

    # --- 2. Features changed from query instance ---
    # Ensure query_instance is a Series and columns are aligned
    if isinstance(query_instance, pd.DataFrame):
        query_instance = query_instance.iloc[0]
    query_instance = query_instance.reindex(df.columns)
    changed = df.ne(query_instance, axis=1)
    feature_change_counts = changed.sum(axis=0)
    changed_features = feature_change_counts[feature_change_counts > 0]
    changed_features_sorted = changed_features.sort_values(ascending=False)
    features_sorted_by_change = changed_features_sorted.index.tolist()

    # --- 3. Windows changed ---
    # Remove 'Event_Y_N' if present (or any other non-feature columns as needed)
    features_for_window = [f for f in features_sorted_by_change if isinstance(f, str) and f != 'Event_Y_N']

    windows = []
    for feat in features_for_window:
        match = re.search(window_pattern, feat)
        if match:
            windows.append(match.group(1))
    window_counts = Counter(windows)
    sorted_windows = window_counts.most_common()

    # --- Return results ---
    return {
        'violations': violations,
        'changed_features_ordered': changed_features_sorted,
        'changed_windows_ordered': sorted_windows
    }




# #===================================================
# import numpy as np
# import pandas as pd
# from datetime import timedelta
# import re
#
# def get_perturbed_series(csv_path, sample_cfe, flux_type, slices,
#                          start_offset_min=300, end_offset_min=660):
#
#     # ------------------------------------------------------
#     # 1. Load data and extract observation window
#     # ------------------------------------------------------
#     df = pd.read_csv(csv_path, delimiter=',')
#     df = df.rename(columns={'time_tag': 'time_stamp'})
#     df['time_stamp'] = pd.to_datetime(df['time_stamp'])
#
#     event_start = df['time_stamp'].iloc[0] + timedelta(minutes=start_offset_min)
#     event_end   = df['time_stamp'].iloc[0] + timedelta(minutes=end_offset_min)
#
#     df_obs = df[(df['time_stamp'] >= event_start) &
#                 (df['time_stamp'] < event_end)].copy()
#     df_obs['minutes'] = (df_obs['time_stamp'] - event_start).dt.total_seconds() / 60
#
#     # Original time series
#     x = df_obs[flux_type].values
#     n = len(x)
#
#     # ------------------------------------------------------
#     # 2. Build interval list in index coordinates
#     # ------------------------------------------------------
#     # slices are in minute units → convert to row indices
#     index_intervals = []
#     for start_min, end_min in slices:
#         mask = (df_obs['minutes'] >= start_min) & (df_obs['minutes'] < end_min)
#         idx = np.where(mask)[0]
#         if len(idx) > 0:
#             index_intervals.append((idx[0], idx[-1] + 1))  # end-exclusive
#         else:
#             index_intervals.append((None, None))  # keep alignment
#
#     # Filter out empty slices but keep mapping
#     valid_intervals = [(s, e) for (s, e) in index_intervals if s is not None]
#
#     # ------------------------------------------------------
#     # 3. Build matrix A for overlapping interval means
#     # ------------------------------------------------------
#     k = len(valid_intervals)
#     A = np.zeros((k, n), dtype=float)
#
#     for i, (start, end) in enumerate(valid_intervals):
#         L = end - start
#         A[i, start:end] = 1.0 / L
#
#     # ------------------------------------------------------
#     # 4. Compute new means from sample_cfe
#     # ------------------------------------------------------
#     new_means = []
#
#     valid_slice_counter = 0
#     for (start_min, end_min), (s, e) in zip(slices, index_intervals):
#         if s is None:
#             continue  # skip empty
#
#         pattern = rf'^{flux_type}_mean@\[{start_min}:{end_min}\]$'
#         matches = sample_cfe.filter(regex=pattern)
#
#         if matches.shape[1] != 1:
#             raise ValueError(f"Could not find unique mean column for slice {start_min}:{end_min}")
#
#         new_mean = matches.iloc[0, 0]
#         new_means.append(new_mean)
#
#     new_means = np.array(new_means)
#
#     # ------------------------------------------------------
#     # 5. Compute the adjusted series y:
#     #    y = x - A^T (A A^T)^(-1) (A x - new_means)
#     # ------------------------------------------------------
#     Ax = A @ x
#     AA_T_inv = np.linalg.inv(A @ A.T)
#
#     correction = A.T @ AA_T_inv @ (Ax - new_means)
#     y = x - correction
#
#     # ------------------------------------------------------
#     # 6. Convert to Pandas Series and compute bounds
#     # ------------------------------------------------------
#     perturbed_series = pd.Series(y, index=df_obs.index)
#     original_series  = pd.Series(x, index=df_obs.index)
#
#     min_y = min(y.min(), x.min())
#     max_y = max(y.max(), x.max())
#
#     return df_obs, perturbed_series, original_series, min_y, max_y



