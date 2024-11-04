import os
import json
import shutil
import torch
import numpy as np
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from dataset import generate_data, get_dataloaders
from model import (
    init_model, get_weights, set_weights, 
    num_params, low_rank_init
)
from train import train_model


def pretrain(lr, init_point, init_last_layer, config, data, device):
    model = init_model(
        config.num_layers, config.num_hidden, config.num_features,
        config.last_layer_norm, config.activation
    )
    if init_last_layer is not None:
        model[-1].weight.data.copy_(init_last_layer)
    if init_point is None:
        init_point = torch.randn(num_params(model))
        init_point = init_point / torch.norm(init_point)

        if getattr(config, 'init_dirichlet', None) is not None:
            set_weights(model, init_point)
            low_rank_init(model, config.init_dirichlet)
            init_point = get_weights(model)

    set_weights(model, init_point)

    X_train, X_test, y_train, y_test = data
    train_loader, test_loader = \
        get_dataloaders(X_train, X_test, y_train, y_test, config.batch_size, config.pt_seed)

    batch_losses, trace = train_model(
        model.to(device), train_loader, test_loader,
        lr=lr, wd=0, num_iters=config.pt_iters,
        ckpt_iters=config.ckpt_iters, log_iters=config.log_iters,
        riemann_opt=getattr(config, 'riemann_opt', False)
    )

    torch.save({
        'model': model.state_dict(),
        'batch_losses': batch_losses,
        'trace': trace
    }, f'{config.savedir}/pt_lr={lr:.3e}.pt')


def launch_pretraining(config, device):
    if os.path.isdir(config.savedir):
        shutil.rmtree(config.savedir)
    os.makedirs(config.savedir)

    with open(f'{config.savedir}/config.json', 'w') as f:
        json.dump(config.__dict__, f)

    if isinstance(config.data_protocol, str):
        # load data from provided path
        if not os.path.isfile(config.data_protocol):
            raise ValueError('Dataset file not found')

        ckpt = torch.load(config.data_protocol)
        X_train, X_test, y_train, y_test = \
            ckpt['X_train'], ckpt['X_test'], ckpt['y_train'], ckpt['y_test']

    else:
        # generate data by specified protocol
        X_train, X_test, y_train, y_test, views = \
            generate_data(
                config.data_protocol, config.multiview_probs,
                config.data_seed, config.num_features,
                config.train_samples, config.test_samples, device
            )
        torch.save((X_train, X_test, y_train, y_test, views),
                   f'{config.savedir}/data.pt')

    if config.init_point_seed is not None:
        model = init_model(
            config.num_layers, config.num_hidden,
            config.num_features, config.last_layer_norm
        )

        torch.manual_seed(config.init_point_seed)
        init_last_layer = model[-1].weight.data.clone()
        init_point = torch.randn(num_params(model))
        init_point = init_point / torch.norm(init_point)

        if getattr(config, 'init_dirichlet', None) is not None:
            np.random.seed(config.init_point_seed)
            set_weights(model, init_point)
            low_rank_init(model, config.init_dirichlet)
            init_point = get_weights(model)

        del model
    else:
        init_last_layer, init_point = None, None

    if config.pt_seed is not None:
        torch.manual_seed(config.pt_seed)

    result = Parallel(n_jobs=8)(
        delayed(pretrain)(lr, init_point, init_last_layer, config,
                          (X_train, X_test, y_train, y_test), device)
        for lr in tqdm(config.lrs)
    )


def finetune(pt_lr, ft_lr, config, data, device):
    model = init_model(
        config.num_layers, config.num_hidden, config.num_features,
        config.last_layer_norm, config.activation
    )
    pt_ckpt = torch.load(f'{config.savedir}/pt_lr={pt_lr:.3e}.pt')
    model.load_state_dict(pt_ckpt['model'])

    X_train, X_test, y_train, y_test = data
    train_loader, test_loader = \
        get_dataloaders(X_train, X_test, y_train, y_test, config.batch_size, config.pt_seed)

    batch_losses, trace = train_model(
        model.to(device), train_loader, test_loader,
        lr=ft_lr, wd=0, num_iters=config.ft_iters,
        ckpt_iters=config.ckpt_iters, log_iters=config.log_iters,
        riemann_opt=getattr(config, 'riemann_opt', False)
    )

    torch.save({
        'model': model.state_dict(),
        'batch_losses': batch_losses,
        'trace': trace
    }, f'{config.ft_savedir}/pt_lr={pt_lr:.3e}-ft_lr={ft_lr:.3e}.pt')


def launch_finetuning(config, device, num_ft_lr=None):
    if isinstance(config.data_protocol, str):
        # load data from provided path
        if not os.path.isfile(config.data_protocol):
            raise ValueError('Dataset file not found')

        ckpt = torch.load(config.data_protocol)
        X_train, X_test, y_train, y_test = \
            ckpt['X_train'], ckpt['X_test'], ckpt['y_train'], ckpt['y_test']
    else:
        X_train, X_test, y_train, y_test = torch.load(f'{config.savedir}/data.pt')[:4]

    if os.path.isdir(config.ft_savedir):
        shutil.rmtree(config.ft_savedir)

    os.makedirs(config.ft_savedir)

    if num_ft_lr is None:
        num_ft_lr = len(config.lrs)

    if config.pt_seed is not None:
        torch.manual_seed(config.ft_seed)

    for i, ft_lr in enumerate(tqdm(config.lrs[:num_ft_lr])):
        result = Parallel(n_jobs=8)(
            delayed(finetune)(pt_lr, ft_lr, config,
                              (X_train, X_test, y_train, y_test), device)
            for pt_lr in config.lrs
        )
