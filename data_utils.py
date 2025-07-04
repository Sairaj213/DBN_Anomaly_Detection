import numpy as np
import pandas as pd


def generate_normal_series(seq_length=100, num_series=1000):
    x = np.linspace(0, 4 * np.pi, seq_length)
    data = []
    for _ in range(num_series):
        noise = np.random.normal(0, 0.1, seq_length)
        series = np.sin(x) + noise
        data.append(series)
    return np.array(data)


def inject_anomalies(data, anomaly_fraction=0.1):
    num_series, seq_length = data.shape
    num_anomaly_points = int(seq_length * anomaly_fraction)
    data_with_anomalies = data.copy()
    for series in data_with_anomalies:
        anomaly_indices = np.random.choice(seq_length, num_anomaly_points, replace=False)
        series[anomaly_indices] += np.random.choice([3, -3], size=num_anomaly_points)
    return data_with_anomalies


def create_sequences(series, seq_length=100):
    """
    Converts a 1D time series into a 2D array of overlapping windows.
    Each row is a sequence of length `seq_length`.
    """
    sequences = []
    for i in range(len(series) - seq_length + 1):
        sequences.append(series[i:i + seq_length])
    return np.array(sequences)


def load_time_series(input_type="csv", manual_data=None, filename=None, column_index=1, normalize=True):
    """
    Loads time-series data from CSV or manual input or synthetic.
    
    Args:
        input_type (str): 'csv', 'manual', or 'synthetic'
        manual_data (list or np.ndarray): 1D data if input_type='manual'
        filename (str): path to CSV if input_type='csv'
        column_index (int): index of column to use from CSV (default: 1)
        normalize (bool): whether to normalize data to 0–1 range

    Returns:
        np.ndarray: 1D time-series array
    """
    if input_type == "csv":
        if not filename:
            raise ValueError("Filename must be provided for CSV input.")
        df = pd.read_csv(filename)
        if column_index >= df.shape[1]:
            raise IndexError(f"CSV only has {df.shape[1]} columns. Invalid column_index: {column_index}")
        time_series_data = df.iloc[:, column_index].to_numpy()

    elif input_type == "manual":
        if manual_data is None:
            raise ValueError("Manual data must be provided if input_type is 'manual'.")
        time_series_data = np.array(manual_data)

    elif input_type == "synthetic":
        time_series_data = generate_normal_series(seq_length=100, num_series=1)[0]

    else:
        raise ValueError("Invalid input type. Use 'csv', 'manual', or 'synthetic'.")

    if normalize:
        time_series_data = (time_series_data - np.min(time_series_data)) / (np.max(time_series_data) - np.min(time_series_data))

    return time_series_data
