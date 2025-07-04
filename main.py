# main.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from data_utils import generate_normal_series, inject_anomalies, create_sequences
from dbn import DBN
from anomaly_utils import compute_reconstruction_errors, flag_anomalies, reconstruct

# Uncomment if using your own CSV
# from data_utils import load_time_series
# import pandas as pd


def main():
    # For reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # --------------------------------------------------------------------------
    # 🔧 SECTION 1: CHOOSE DATA SOURCE
    # --------------------------------------------------------------------------

    # ✅ DEFAULT: Use synthetic data with injected anomalies
    normal_data = generate_normal_series(seq_length=100, num_series=1000)
    anomalous_data = inject_anomalies(normal_data, anomaly_fraction=0.1)

    # 🚧 ALTERNATIVE: Use your own CSV file
    # 📌 To use your own time-series data:
    # 1. Make sure your CSV has a column with raw values (e.g., "value")
    # 2. Uncomment the block below
    # 3. Comment out the `generate_normal_series` and `inject_anomalies` lines above

    """
    df = pd.read_csv("your_data.csv")
    series = df['value'].values

    # Normalize the series
    series = (series - np.min(series)) / (np.max(series) - np.min(series))

    # Create overlapping windows (sequence length = 100)
    norm_data = create_sequences(series, seq_length=100)
    """

    # --------------------------------------------------------------------------
    # 🔄 Normalize synthetic data (skip if using own CSV with normalization)
    data_min = anomalous_data.min()
    data_max = anomalous_data.max()
    norm_data = (anomalous_data - data_min) / (data_max - data_min)

    # --------------------------------------------------------------------------
    # 📈 Visualize a sample time-series
    plt.plot(norm_data[0], label="Anomalous Sample")
    plt.title("Sample Time-Series")
    plt.legend()
    plt.show()  # ❗ CLOSE this window to proceed

    # --------------------------------------------------------------------------
    # 🧠 Train the DBN
    dbn = DBN(n_visible=norm_data.shape[1], hidden_layers=[64, 32], k=1)
    dbn.train_dbn(norm_data, epochs=10, lr=0.01, batch_size=128)

    # --------------------------------------------------------------------------
    # 📏 Compute reconstruction error using only the first RBM
    first_rbm = dbn.rbm_layers[0]
    recon_errors = compute_reconstruction_errors(norm_data, first_rbm, batch_size=128)

    threshold = np.percentile(recon_errors, 90)
    print(f"\nAnomaly Threshold (90th percentile): {threshold:.4f}")

    anomalies = flag_anomalies(recon_errors, threshold)
    print(f"Number of anomalies detected: {np.sum(anomalies)}")

    # --------------------------------------------------------------------------
    # 📊 Plot reconstruction error distribution
    plt.hist(recon_errors, bins=50)
    plt.xlabel("Reconstruction Error")
    plt.title("Reconstruction Error Distribution")
    plt.show()  # ❗ CLOSE this window to continue

    # --------------------------------------------------------------------------
    # 🔍 Visualize one detected anomaly
    anomalous_indices = np.where(anomalies)[0]
    if len(anomalous_indices):
        idx = anomalous_indices[0]
        sample_tensor = torch.FloatTensor(norm_data[idx:idx+1])
        sample_recon = reconstruct(first_rbm, sample_tensor)

        plt.figure(figsize=(12, 4))
        plt.plot(sample_tensor.squeeze().detach().numpy(), label="Original")
        plt.plot(sample_recon.squeeze().detach().numpy(), label="Reconstructed")
        plt.title(f"Anomaly Sample Index: {idx}")
        plt.legend()
        plt.show()  # ❗ CLOSE this window to end
    else:
        print("No anomalies detected based on the threshold.")


if __name__ == "__main__":
    main()
