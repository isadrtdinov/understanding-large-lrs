#!/usr/bin/env python
# coding: utf-8

# # Custom reimplementations of drop training
# 
# The idea is to check if there is some problems with training or it is some interesting effect:
# 
# "While learning rate drop, higher drop values lead to smaller angle distance"

# # imports

# In[8]:


import sys
sys.path.append('../..')

import os
import time
import math
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


# config

import argparse
parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--gpu', type=int, help='gpu_to_use')
parser.add_argument('--init_elr', type=float, help='init elr')
parser.add_argument('--drop_elr', type=float, help='drop elr')
parser.add_argument('--drop_epoch', type=int, help='epoch where make drop')
parser.add_argument('--seed', type=int, help='random_seed')
parser.add_argument('--wd', type=float, help='weight decay', default=0.0)
parser.add_argument('--aug', type=int, help='if true, adds augmentation', default=0)
parser.add_argument('--epochs', type=int, help='amount of epochs to conduct, default = 1000', default=1000)


forced_args = parser.parse_args()

gpunumber = forced_args.gpu

print("Using Augmentations is turned " + "on" if bool(forced_args.aug) else "off")

args = {
    'model': 'ResNet18SI',
    'seed': forced_args.seed,
    'no_aug': not (bool(forced_args.aug)),
    'dataset': 'CIFAR10',
    'data_path': '~/datasets/',
    'batch_size': 128,
    'num_workers': 4,
    'use_test': True,
    'use_data_size': None,    # !
    'split_classes': None,
    'corrupt_train': None,
    'noninvlr': 0.0,
    
    'num_channels': 32,
    'init_scale': 10,
    
    'elr_init': forced_args.init_elr, 
    'elr_drop': forced_args.drop_elr,
    'drop_epoch': forced_args.drop_epoch,
    'momentum': 0.0, 
    'wd': forced_args.wd,
    
    'epochs': forced_args.epochs,
    'fix_si_pnorm': True,
    'fix_si_pnorm_value': 28.0,
    
    'num_classes': 10,
    'eval_freq': 1,
    'save_freq': 1,
    'save_freq_int': 0,
    'fbgd': False,
}

output_dir = f"./Experiments/{args['model']}_{args['dataset']}_elri_{args['elr_init']}_" +\
    f"elrd_{args['elr_drop']}_dropepoch_{args['drop_epoch']}_wd_{args['wd']}_seed_{args['seed']}_noaug_{args['no_aug']}"

from pathlib import Path
Path(output_dir).mkdir(parents=True, exist_ok=True)


# setup cuda

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=str(gpunumber)

device = 'cuda'

print('Using random seed {}'.format(args['seed']))
torch.backends.cudnn.benchmark = True
torch.manual_seed(args['seed'])
torch.cuda.manual_seed(args['seed'])


print('='*80)
print('start process...')
print('='*80)

print('Weight decay = {}'.format(args['wd']))


def cross_entropy(model, input, target, reduction="mean"):
    "standard cross-entropy loss function"
    output = model(input)
    loss = F.cross_entropy(output, target, reduction=reduction)
    return loss, output


def check_si_name(n, model_name='$$$'):
    if model_name == 'ResNet18':
        return "conv1" in n or "1.bn1" in n or "1.0.bn1" in n or (("conv2" in n or "short" in n) and "4" not in n)
    elif model_name == 'ResNet18SI':
        return 'linear' not in n
    elif model_name == 'ResNet18SIAf':
        return ('linear' not in n and 'bn' not in n and 'shortcut.0' not in n)
    elif 'ConvNet' in model_name:
        return 'conv_layers.0.' in n or 'conv_layers.3.' in n or 'conv_layers.7.' in n or 'conv_layers.11.' in n
    raise NotImplementedError('Unknown model: {}'.format(model_name))
    return False


def train_epoch_custom(
    loader,
    model,
    model_name: str,
    criterion,
    optimizer,
    cuda=True,
    regression=False,
    verbose=False,
    subset=None,
    fbgd=False,
    save_freq_int = 0,
    epoch=None,
    output_dir = None,
    si_pnorm_0 = None
):
    loss_sum = 0.0
    correct = 0.0
    verb_stage = 0
    save_ind = 0

    num_objects_current = 0
    num_batches = len(loader)

    model.train()

    if subset is not None:
        num_batches = int(num_batches * subset)
        loader = itertools.islice(loader, num_batches)

    if verbose:
        loader = tqdm.tqdm(loader, total=num_batches)

    reduction = "sum" if fbgd else "mean"
    optimizer.zero_grad()

    for i, (input, target) in enumerate(loader):
        if cuda:
            input = input.cuda(non_blocking=True)
            target = target.type(torch.LongTensor)
            target = target.cuda(non_blocking=True)
            
        loss, output = criterion(model, input, target, reduction)

        if fbgd:
            loss_sum += loss.item()
            loss /= len(loader.dataset)
            loss.backward()
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if si_pnorm_0 is not None:
                fix_si_pnorm(model, si_pnorm_0, model_name)
                
            loss_sum += loss.data.item() * input.size(0)

        if not regression:
            pred = output.data.argmax(1, keepdim=True)
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        num_objects_current += input.size(0)

        if verbose and 10 * (i + 1) / num_batches >= verb_stage + 1:
            print(
                "Stage %d/10. Loss: %12.4f. Acc: %6.2f"
                % (
                    verb_stage + 1,
                    loss_sum / num_objects_current,
                    correct / num_objects_current * 100.0,
                )
            )
            verb_stage += 1
            
        if (save_freq_int > 0) and (save_freq_int*(i+1)/ num_batches >= save_ind + 1) and (save_ind + 1 < save_freq_int):
            save_checkpoint_int(
                output_dir,
                epoch,
                save_ind + 1,
                state_dict=model.state_dict(),
                optimizer=optimizer.state_dict()
            )
            save_ind += 1
            

    if fbgd:
        optimizer.step()
        optimizer.zero_grad()
        
        if si_pnorm_0 is not None:
            fix_si_pnorm(model, si_pnorm_0, model_name)

    return {
        "loss": loss_sum / num_objects_current,
        "accuracy": None if regression else correct / num_objects_current * 100.0,
    }


# In[16]:

columns = ["ep", "lr", "tr_loss", "tr_acc", "te_loss", "te_acc", "time"]\

def train_epoch_with_drop(model, model_name,  loaders, criterion, optimizer, epoch, drop_epoch, end_epoch,
                          eval_freq=1, save_freq=10, save_freq_int=0, output_dir='./', 
                          elr_init=0.01, elr_drop=0.0001,
                          noninvlr = -1, si_pnorm_0=None, fbgd=False):

    time_ep = time.time()

    if si_pnorm_0 is None:
        raise ValueError('SI_PNORM_0 must be provided!')
    
    if epoch >= drop_epoch:
        lr = elr_drop * si_pnorm_0 ** 2
    else:
        lr = elr_init * si_pnorm_0 ** 2
        
    if noninvlr >= 0:
        training_utils.adjust_learning_rate_only_conv(optimizer, lr)
    else:
        training_utils.adjust_learning_rate(optimizer, lr)
   
    if epoch > 0:
        train_res = train_epoch_custom(
            loader=loaders["train"], 
            model=model, 
            model_name=model_name, 
            criterion=criterion, 
            optimizer=optimizer, 
            fbgd=fbgd,             
            si_pnorm_0=si_pnorm_0,
            save_freq_int=save_freq_int,
            epoch=epoch,
            output_dir=output_dir
        )
    else:
        train_res = {"loss": None, "accuracy": None}
    if (
        epoch == 0
        or epoch % eval_freq == eval_freq - 1
        or epoch == end_epoch - 1
    ):
        test_res = training_utils.eval(loaders["test"], model, criterion)
    else:
        test_res = {"loss": None, "accuracy": None}
       
    def save_epoch(epoch):
        training_utils.save_checkpoint(
            output_dir,
            epoch,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(),
            train_res=train_res,
            test_res=test_res
        )
    
    if save_freq is None:
        if training_utils.do_report(epoch):
            save_epoch(epoch)
    elif epoch % save_freq == 0:
        save_epoch(epoch)
        
    time_ep = time.time() - time_ep
    values = [
        epoch,
        lr,
        train_res["loss"],
        train_res["accuracy"],
        test_res["loss"],
        test_res["accuracy"],
        time_ep,
    ]
    table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
    if epoch % 40 == 0:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)



# # creating model and dataset

if args['model'] == 'ResNet18SI':
    model_cfg = models.ResNet18SI
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
        'scale': args.scale
    }
else:
    extra_args = {}


model = model_cfg.base(*model_cfg.args, num_classes=args['num_classes'], **model_cfg.kwargs,
                       **extra_args)
_ = model.to(device)


# param groups

param_groups = model.parameters()

if args['noninvlr'] >= 0:
    print('Separate LR for last layer!')
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if check_si_name(n, args['model'])]},  # SI params are convolutions
        {'params': [p for n, p in model.named_parameters() if not check_si_name(n, args['model'])], 'lr': args['noninvlr']},  # other params
    ]
    for n, p in model.named_parameters():
        if check_si_name(n, args['model']):
            print(n, ' ELR =', args['elr_init'])
        else:
            print(n, '  LR =', args['noninvlr'])


# dataset

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
    use_data_size = args['use_data_size'],
    split_classes=args['split_classes'],
    corrupt_train=args['corrupt_train']
)

epoch_from = 0  # from init
epoch_to = args['epochs']
print(f"Training from {epoch_from} to {epoch_to} epochs")

def fix_si_pnorm(model, si_pnorm_0, model_name):
    "Fix SI-pnorm to si_pnorm_0 value"
    si_pnorm = np.sqrt(sum((p ** 2).sum().item() for n, p in model.named_parameters() if check_si_name(n, model_name)))
    p_coef = si_pnorm_0 / si_pnorm
    for n, p in model.named_parameters():
        if check_si_name(n, model_name):
            p.data *= p_coef


si_pnorm_0 = None
if args['fix_si_pnorm']:
    if args['fix_si_pnorm_value'] > 0:
        si_pnorm_0 = args['fix_si_pnorm_value']
        fix_si_pnorm(model, si_pnorm_0, args['model'])
    else:
        si_pnorm_0 = np.sqrt(sum((p ** 2).sum().item() for n, p in model.named_parameters() if check_si_name(n, args['model'])))

    print(f"Fixing SI-pnorm to value {si_pnorm_0:.4f}")

    
optimizer = torch.optim.SGD(
    param_groups, 
    lr=args['elr_init'] * si_pnorm_0 ** 2, 
    momentum=args['momentum'], 
    weight_decay=args['wd'],
)
 
    
for epoch in range(epoch_from, epoch_to + 1):  # i.e. [0, 500], where 0 = init checkpoint
    print('{} Epoch {:04d}'.format(datetime.now(), epoch))
    train_epoch_with_drop(
        model, args['model'],
        loaders, cross_entropy, optimizer, 
        epoch=epoch, 
        drop_epoch=args['drop_epoch'],
        end_epoch=epoch_to,
        
        eval_freq=args['eval_freq'], 
        save_freq=args['save_freq'],
        save_freq_int=args['save_freq_int'],
        output_dir=output_dir,
        elr_init=args['elr_init'],
        elr_drop=args['elr_drop'],
        noninvlr=args['noninvlr'],
        si_pnorm_0=si_pnorm_0,
        fbgd=args['fbgd'],
    )

print("="*80)
print("model done")