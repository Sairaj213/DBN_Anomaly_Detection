import torch
import numpy as np
import matplotlib.pyplot as plt


def reconstruct(rbm, v):
    p_h, _ = rbm.v_to_h(v)
    p_v, _ = rbm.h_to_v(p_h)
    return p_v


def compute_reconstruction_errors(data, rbm, batch_size=128):
    from torch.utils.data import DataLoader, TensorDataset
    tensor_data = torch.FloatTensor(data)
    dataloader = DataLoader(TensorDataset(tensor_data), batch_size=batch_size, shuffle=False)
    all_errors = []
    for batch in dataloader:
        v = batch[0]
        v_recon = reconstruct(rbm, v)
        errors = torch.mean((v - v_recon) ** 2, dim=1)
        all_errors.extend(errors.detach().numpy())
    return np.array(all_errors)


def flag_anomalies(recon_errors, threshold):
    return recon_errors > threshold


def apply_model(time_series_data, dbn_model):
    time_series_tensor = torch.FloatTensor(time_series_data).unsqueeze(0)
    first_rbm = dbn_model.rbm_layers[0]
    reconstructed = reconstruct(first_rbm, time_series_tensor).squeeze().detach().numpy()
    error = np.mean((time_series_data - reconstructed) ** 2)
    print("\nTime-Series Reconstruction Error:", error)
    plt.figure(figsize=(12, 4))
    plt.plot(time_series_data, label="Original Time-Series")
    plt.plot(reconstructed, label="Reconstructed")
    plt.legend()
    plt.title("Model Output: Original vs Reconstructed")
    plt.show()
    return error