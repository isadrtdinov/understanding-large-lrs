import torch
import numpy as np
from torch import nn


def get_weights(model):
    weights = []
    for param in model.parameters():
        if param.requires_grad:
            weights.append(param.data.clone().flatten())
    weights = torch.cat(weights, dim=0)
    return weights


def set_weights(model, weights):
    offset = 0
    for param in model.parameters():
        if param.requires_grad:
            numel = param.numel()
            param.data.copy_(weights[offset:offset + numel].reshape(*param.shape))
            offset += numel


def get_grads(model):
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.clone().flatten())
    grads = torch.cat(grads, dim=0)
    return grads


def num_params(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def low_rank_init(model, dirichlet_alpha=1):
    for param in model.parameters():
        if param.requires_grad:
            U, S, V = torch.svd(param.data)
            S_new = torch.from_numpy(
                np.random.dirichlet(alpha=[dirichlet_alpha] * S.shape[0])
            ).to(dtype=torch.float, device=param.device)
            S_new = S_new / S_new.norm() * S.norm()
            param.data.copy_(U @ torch.diag(S_new) @ V.T)


def init_model(
    num_layers=3, num_hidden=32, num_features=32,
    last_layer_norm=None, activation='relu'
):
    if activation == 'relu':
        act_class = nn.ReLU
    elif activation == 'tanh':
        act_class = nn.Tanh
    else:
        raise ValueError('Unknown activation type')

    assert num_layers >= 2
    layers = [
        nn.Linear(num_features, num_hidden, bias=False),
        nn.LayerNorm(num_hidden, elementwise_affine=False, eps=1e-7),
        act_class()
    ]

    for _ in range(num_layers - 2):
        layers += [
            nn.Linear(num_hidden, num_hidden, bias=False),
            nn.LayerNorm(num_hidden, elementwise_affine=False, eps=1e-7),
            act_class()
        ]

    layers += [nn.Linear(num_hidden, 1, bias=False)]
    layers[-1].weight.requires_grad = False
    if last_layer_norm is not None:
        weight = layers[-1].weight.data
        layers[-1].weight.data = (last_layer_norm / torch.norm(weight.flatten())) * weight

    return nn.Sequential(*layers)
