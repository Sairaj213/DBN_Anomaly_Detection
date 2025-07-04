from torch.utils.data import DataLoader, TensorDataset
from rbm import RBM
import torch


class DBN:
    def __init__(self, n_visible, hidden_layers, k=1):
        self.n_layers = len(hidden_layers)
        self.rbm_layers = []
        for n_hidden in hidden_layers:
            rbm = RBM(n_visible, n_hidden, k)
            self.rbm_layers.append(rbm)
            n_visible = n_hidden

    def train_dbn(self, data, epochs=10, lr=0.01, batch_size=64):
        input_data = torch.FloatTensor(data)
        for layer_idx, rbm in enumerate(self.rbm_layers):
            print(f"\nTraining RBM layer {layer_idx+1}")
            dataset = TensorDataset(input_data)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for epoch in range(epochs):
                epoch_loss = 0.0
                for batch in dataloader:
                    v = batch[0]
                    p_h, _ = rbm.v_to_h(v)
                    for _ in range(rbm.k):
                        p_v, v_sample = rbm.h_to_v(p_h)
                        p_h, _ = rbm.v_to_h(v_sample)
                    loss = torch.mean((v - v_sample) ** 2)
                    epoch_loss += loss.item()
                    rbm.contrastive_divergence(v, lr)
                print(f"Epoch {epoch+1}: Loss {epoch_loss/len(dataloader):.4f}")
            p_h, _ = rbm.v_to_h(input_data)
            input_data = p_h

    def get_representation(self, x):
        for rbm in self.rbm_layers:
            p_h, _ = rbm.v_to_h(x)
            x = p_h
        return x