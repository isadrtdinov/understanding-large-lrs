import os
import sys
from copy import deepcopy

import numpy as np
import torch
from collections import OrderedDict
import torch.nn.functional as F
# from matplotlib import pyplot as plt


def cross_entropy(model, input, target, reduction="mean"):
    "standard cross-entropy loss function"
    output = model(input)
    loss = F.cross_entropy(output, target, reduction=reduction)
    return loss, output


def interpolate_state_dicts(model_A_state_dict, model_B_state_dict, t=0.5):
    """
    LINEAR INTERPOLATION! no casting on the sphere.
    t = 0 -> A
    t = 1 -> B
    """
    state_dict = deepcopy(model_A_state_dict)
    for key in state_dict.keys():
        state_dict[key] = (1.0 - t) * model_A_state_dict[key] + t * model_B_state_dict[key]
    return state_dict


def load_state_dict_from_swa_to_usual(path: str):
    pkl = torch.load(path)
    if 'state_dict' in pkl.keys():
        sd = pkl['state_dict']
    else:
        sd = pkl
        
    layers = []
    for key, value in sd.items():
        if key == 'n_averaged':
            continue
        elif key.startswith('module.'):
            layers.append((key[7:], value))
        
    new_sd = OrderedDict(layers)
        
    return new_sd


def interpolate_on_the_sphere(point_a: np.ndarray, point_b: np.ndarray, t: float) -> np.ndarray:
    point_a_proj = point_a * np.dot(point_a, point_b) / np.linalg.norm(point_b)
    point_b_orth = point_b - point_a_proj
    out = np.cos(t) * point_a + np.sin(t) * point_b_orth
    return out


def make_flatten_vec(state_dict, param_list):
    values = []
    for key, value in state_dict.items():
        if key in param_list:
            values.append(torch.flatten(value))
    vec = torch.cat(values, 0).to(torch.float64)
    return vec


def make_state_dict_orthogonal(project_where_state_dict: np.ndarray, project_what_state_dict: np.ndarray, param_list):
    """
    """
    
    orth_dict = deepcopy(project_what_state_dict)
    with torch.no_grad():
        
        vec_A = make_flatten_vec(project_where_state_dict, param_list).detach().cpu().numpy()
        vec_B = make_flatten_vec(project_what_state_dict,  param_list).detach().cpu().numpy()

        dot_A_B = np.dot(vec_A, vec_B)
        pnorm = np.linalg.norm(vec_A)

        orth_norm = 0.0
        for key in param_list:
            orth_dict[key] = project_what_state_dict[key] - project_where_state_dict[key] * dot_A_B / pnorm ** 2
            orth_norm += (orth_dict[key] ** 2).sum().item()
        
        orth_norm = orth_norm ** 0.5
        
        for key in param_list:
            orth_dict[key] = orth_dict[key] * pnorm / orth_norm
        
    return orth_dict


def sphere_interpolate_orth_state_dicts(model_A_state_dict, model_B_state_dict_orthogonal, t: float, 
                                   param_list):
    """
    t \in [-pi, pi]
    t = 0 -> A
    t = pi/2 -> B
    """
    state_dict = deepcopy(model_A_state_dict)
    with torch.no_grad():   
        for key in param_list:
            state_dict[key] = torch.cos(t) * model_A_state_dict[key] + torch.sin(t) * model_B_state_dict_orthogonal[key]
        
    return state_dict


def sphere_get_t_coord_from_orth_basis(point_A_orth, point_B_orth, point_X, param_list):
    v_A = make_flatten_vec(point_A_orth, param_list).detach().cpu().numpy()
    v_B = make_flatten_vec(point_B_orth, param_list).detach().cpu().numpy()
    v_X = make_flatten_vec(point_X,      param_list).detach().cpu().numpy()
    
    n_A = np.linalg.norm(v_A)
    n_B = np.linalg.norm(v_B)
    n_X = np.linalg.norm(v_X)
    
    
    cos_t = np.dot(v_X, v_A) / (n_A * n_X)
    sin_t = np.dot(v_X, v_B) / (n_B * n_X)
    
    t = np.arctan2(sin_t, cos_t)
    return t


def get_gradients_for_single_batch(model, input_batch, label_batch, criterion, pnames):
    params = []
    param_names = []
    for n, p in model.named_parameters():
        if n in pnames and p.requires_grad == True:
            params.append(p)
            param_names.append(n)

    loss, outputs = criterion(model, input_batch, label_batch)

    grads = torch.autograd.grad(loss, params)
    
    named_grads = {k: v for k, v in zip(param_names, grads)}

    return named_grads, loss, outputs


def get_tangent_state_dict(sd_base, sd_what, param_list):
    """
    """
    
    sd_tan = deepcopy(sd_what)
    with torch.no_grad():
        
        vec_A = make_flatten_vec(sd_base, param_list).detach()  #.cpu().numpy()
        vec_B = make_flatten_vec(sd_what, param_list).detach()  #.cpu().numpy()

        dot_A_B = torch.dot(vec_A, vec_B)
        pnorm_A = torch.linalg.norm(vec_A)

        tan_norm = 0.0
        for key in param_list:
            sd_tan[key] = sd_what[key] - sd_base[key] * dot_A_B / pnorm_A ** 2
            tan_norm += (sd_tan[key] ** 2).sum().item()
        
        tan_norm = tan_norm ** 0.5
        
        for key in param_list:
            sd_tan[key] = sd_tan[key] * pnorm_A / tan_norm
        
    return tan_norm, pnorm_A, sd_tan


def renorm_state_dict(sd_what, param_list, pnorm_W):
    with torch.no_grad():
        tan_norm = 0.0
        for key in param_list:
            tan_norm += (sd_what[key] ** 2).sum().item()
        
        tan_norm = tan_norm ** 0.5
        
        for key in param_list:
            sd_what[key] = sd_what[key] * pnorm_W / tan_norm
        
    return tan_norm, pnorm_W, sd_what