import torch
import torch.nn as nn
import torch.nn.functional as F


class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden, k=1):
        super(RBM, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k = k
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.1)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))

    def sample_from_prob(self, p):
        return torch.bernoulli(p)

    def v_to_h(self, v):
        p_h = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return p_h, self.sample_from_prob(p_h)

    def h_to_v(self, h):
        p_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return p_v, self.sample_from_prob(p_v)

    def contrastive_divergence(self, v, lr=0.01):
        p_h, h_sample = self.v_to_h(v)
        positive_grad = torch.matmul(p_h.t(), v)
        h = h_sample
        for _ in range(self.k):
            p_v, v_sample = self.h_to_v(h)
            p_h, h = self.v_to_h(v_sample)
        negative_grad = torch.matmul(p_h.t(), v_sample)
        self.W.data += lr * (positive_grad - negative_grad) / v.size(0)
        self.v_bias.data += lr * torch.mean(v - v_sample, dim=0)
        self.h_bias.data += lr * torch.mean(p_h - h, dim=0)