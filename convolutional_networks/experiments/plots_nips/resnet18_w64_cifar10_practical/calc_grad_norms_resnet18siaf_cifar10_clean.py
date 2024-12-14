#!/usr/bin/env python
# coding: utf-8

# CALCULATE GRAD NORMS AFTER DROP TRAINING


import sys
sys.path.append('../..')
sys.path.append('../../..')


import os
import time
import math
from glob import glob
from pathlib import Path

import numpy as np
import tabulate
import torch
import torch.nn.functional as F

import data
import training_utils
import nets as models
from datetime import datetime
from parser_train import parser

from torch.optim.swa_utils import AveragedModel


import argparse
parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--gpu', type=int, help='gpu_to_use')
parser.add_argument('--directory_with_checkpoints', type=str, help='init checkpoint')
parser.add_argument('--loader', type=str, help='train or test', default='train')
parser.add_argument('--aug', type=int, help='if true, adds augmentation', default=0)
parser.add_argument('--train_mode', type=int, help='if true, uses train mode for calc gradients', default=1)


forced_args = parser.parse_args()

gpunumber = forced_args.gpu

args = {
    'model': 'ResNet18SIAf',
    'seed': 1,
    'no_aug': not (bool(forced_args.aug)),
    'dataset': 'CIFAR10',
    'data_path': '~/datasets/',
    'batch_size': 128,
    'num_workers': 4,
    'use_test': True,
    'use_data_size': None,
    'split_classes': None,
    'corrupt_train': None,
    'noninvlr': 0.0,
    
    'num_channels': 64,
    'init_scale': 10,
      
    'fix_si_pnorm': True,
    'fix_si_pnorm_value': -1,
    
    'num_classes': 10,
    'eval_freq': 1,
    'save_freq': 1,
    'save_freq_int': 0,
    'fbgd': False,
}

# setup cuda -----------------------------------------------------

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=str(gpunumber)

device = 'cuda'

print('Using random seed {}'.format(args['seed']))
torch.backends.cudnn.benchmark = True
torch.manual_seed(args['seed'])
torch.cuda.manual_seed(args['seed'])

# ----------------------------------------------------------------

print('='*80)
print('start calculating grad norms...')
print('='*80)

def check_si_name(n, model_name):
    if model_name == 'ResNet18':
        return "conv1" in n or "1.bn1" in n or "1.0.bn1" in n or (("conv2" in n or "short" in n) and "4" not in n)
    elif model_name == 'ResNet18SI':
        return 'linear' not in n
    elif model_name == 'ResNet18SIAf':
        return ('linear' not in n and 'bn' not in n and 'shortcut.0' not in n)
    elif 'ConvNet' in model_name:
        return 'conv_layers.0.' in n or 'conv_layers.3.' in n or 'conv_layers.7.' in n or 'conv_layers.11.' in n
    return False


def cross_entropy(model, input, target, reduction="mean"):
    "standard cross-entropy loss function"
    output = model(input)
    loss = F.cross_entropy(output, target, reduction=reduction)
    return loss, output


if args['model'] == 'ResNet18SIAf':
    model_cfg = models.ResNet18SIAf
else:
    raise NotImplementedError('unknown model: {}'.format(args['model']))


print("Preparing model")
print(*model_cfg.args)


# add extra args for varying names
if 'ResNet18SI' in args['model']:
    extra_args = {
        'init_channels': args['num_channels'],
        'linear_norm': args['init_scale']
    }
elif 'ConvNet' in args['model']:
    extra_args = {
        'init_channels': args['num_channels'], 
        'max_depth': args['depth'], 
        'init_scale': args['init_scale']
    }
elif args['model'] == 'LeNet':
    extra_args = {
        'scale':args.scale
    }
else:
    extra_args = {}


model = model_cfg.base(*model_cfg.args, num_classes=args['num_classes'], **model_cfg.kwargs,
                       **extra_args)
_ = model.to(device)

# dataset -----------------------------------------

print("Loading dataset %s from %s" % (args['dataset'], args['data_path']))
transform_train = model_cfg.transform_test if args['no_aug'] else model_cfg.transform_train
loaders, num_classes = data.loaders(
    args['dataset'],
    args['data_path'],
    args['batch_size'],
    args['num_workers'],
    transform_train,
    model_cfg.transform_test,
    use_validation=not args['use_test'],
    use_data_size=args['use_data_size'],
    split_classes=args['split_classes'],
    corrupt_train=args['corrupt_train'],
    shuffle_train=False
)


def calc_grads_and_metrics(model, dataloader, criterion, train_mode=False, return_numpy=False, pnames=None):
    if train_mode:
        model.train()
    else:
        model.eval()

    if pnames is None:
        params = list([param for param in model.parameters() if param.requires_grad == True])
    else:
        params = [p for n, p in model.named_parameters() if (n in pnames and p.requires_grad == True)]
        
#     grads_list = []
    
    loss_sum = 0.0
    correct = 0
    for images, labels in dataloader:
        images = images.cuda()
        labels = labels.cuda()

#         outputs = model(images)
        loss, outputs = criterion(model, images, labels)
    
        loss_sum += loss.item() * images.size(0)
        predicted = outputs.argmax(-1)
        correct += (predicted == labels).sum().item()
        
#         grads = torch.autograd.grad(loss, params)
#         if return_numpy:
#             grads = np.concatenate([g.cpu().numpy().ravel() for g in grads])
#         grads_list.append(grads)

    final_loss = loss_sum / len(dataloader.dataset)
    final_acc  = correct / len(dataloader.dataset)
    return  final_loss, final_acc


# main procedure -----------------------------------------------------------------

if args['model'] in ['ResNet18SI', 'ResNet18', 'ConvNetSI', 'ConvNet', 'ResNet18SIAf']:
    pnames = [n for n, _ in model.named_parameters() if check_si_name(n, args['model'])]
else:
    raise ValueError("Using pre-BN parameters currently is not allowed for this model!")

    
glob_list = list(glob(os.path.join(forced_args.directory_with_checkpoints, 'checkpoint*.pt'))) \
          + list(glob(os.path.join(forced_args.directory_with_checkpoints, 'interp_result_*.pt'))) \
          + list(glob(os.path.join(forced_args.directory_with_checkpoints, 'to_checkpoint*.pt')))

            
for ind, ckpt_fname in enumerate(sorted(glob_list)):
    print(ind, ckpt_fname)
    
    if 'batch' in os.path.basename(ckpt_fname):
        print('skip batch')
        continue
    
    if ('interp_result_' not in ckpt_fname) and ('-seed-' not in ckpt_fname):
        bname = os.path.basename(ckpt_fname)[:-3].replace('checkpoint-', '')
        ep = int(bname)
        
#     if not (399 < ep < 401):
#         print('skip (out of interest)')
#         continue
#         if ep not in [2425, 2430, 2435]:
#             print('skip (out of interest)')
#             continue
    
#         if not (300 < ep < 303):
#             continue
    #     if ep > 500:
    #         print('skip (epoch is too large')
    #         continue

    #     if ep not in [51, 54, 59, 99, 101, 104, 109, 149, 151, 154, 159, 199, 201, 204, 209, 249, 251, 254, 259, 299, 301, 304, 309, 349, 399, 449, 499]:
    #         print('Skip (epoch is out of interest)')
    #         continue

    #     if ep < 5190:
    #         print('skip (epoch is out of interest)')
    #         continue
    else:
        pass
    
    checkpoint = torch.load(ckpt_fname)
    
    if 'n_averaged' in checkpoint['state_dict']:
        
        swa_model = AveragedModel(model)#.to(args.device)
        swa_model.load_state_dict(checkpoint["state_dict"])
        print(type(swa_model.module))
        model = swa_model.module.cuda()
    else:
        model.load_state_dict(checkpoint["state_dict"])
    
    if ("gnorm_trainmode_m_" + forced_args.loader in checkpoint and forced_args.train_mode):
        if checkpoint["loss_trainmode_" + forced_args.loader] is not None:
            print('skip {} {}'.format(ind, ckpt_fname))
            continue
    if ("gnorm_evalmode_m_"  + forced_args.loader in checkpoint and not forced_args.train_mode):
        if checkpoint["loss_evalmode_"  + forced_args.loader] is not None:
            print('skip {} {}'.format(ind, ckpt_fname))
            continue
    
    final_loss, final_acc = calc_grads_and_metrics(model, loaders[forced_args.loader], cross_entropy, train_mode=bool(forced_args.train_mode), pnames=pnames)
    with torch.no_grad():
#         gnorms = []
#         for p_grads in grads_list:  
#             gnorm = 0
#             for t in p_grads:
#                 gnorm += (t**2).sum().item()
#             gnorms.append(np.sqrt(gnorm))
        if bool(forced_args.train_mode):
#             checkpoint["gnorm_trainmode_m_" + forced_args.loader] = np.array(gnorms).mean()
            checkpoint["loss_trainmode_" + forced_args.loader] = final_loss
            checkpoint["acc_trainmode_" + forced_args.loader] = final_acc
#             print('GRAD NORM', checkpoint["gnorm_trainmode_m_" + forced_args.loader])
        else:
#             checkpoint["gnorm_evalmode_m_" + forced_args.loader] = np.array(gnorms).mean()
            checkpoint["loss_evalmode_" + forced_args.loader] = final_loss
            checkpoint["acc_evalmode_" + forced_args.loader] = final_acc
#             print('GRAD NORM', checkpoint["gnorm_evalmode_m_" + forced_args.loader])
    torch.cuda.empty_cache()
    torch.save(checkpoint, ckpt_fname)