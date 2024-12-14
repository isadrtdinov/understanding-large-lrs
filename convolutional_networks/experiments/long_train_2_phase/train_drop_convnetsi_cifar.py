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


# # config

# In[44]:

import argparse
parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--gpu', type=int, help='gpu_to_use')
parser.add_argument('--init_lr', type=float, help='init lr')
parser.add_argument('--drop_lr', type=float, help='drop lr')
parser.add_argument('--drop_epoch', type=int, help='epoch where make drop')
parser.add_argument('--seed', type=int, help='random_seed')
parser.add_argument('--wd', type=float, help='weight decay', default=0.0)
parser.add_argument('--aug', type=int, help='if true, adds augmentation', default=0)
parser.add_argument('--epochs', type=int, help='epochs to train', default=1000)


forced_args = parser.parse_args()

gpunumber = forced_args.gpu

print("Using Augmentations is turned " + "on" if bool(forced_args.aug) else "off")

args = {
    'model': 'ConvNetSI',
    'seed': forced_args.seed,
    'no_aug': not (bool(forced_args.aug)),
    'dataset': 'CIFAR10',
    'data_path': '~/datasets/',
    'batch_size': 128,
    'num_workers': 4,
    'use_test': False,
    'use_data_size': 50000,
    'split_classes': None,
    'corrupt_train': None,
    'noninvlr': 0.0,
    
    'num_channels': 32,
    'init_scale': 10,
    
    'lr_init': forced_args.init_lr, 
    'lr_drop': forced_args.drop_lr,
    'drop_epoch': forced_args.drop_epoch,
    'momentum': 0.0, 
    'wd': forced_args.wd,
    
    'epochs': forced_args.epochs,
    'fix_si_pnorm': True,
    'fix_si_pnorm_value': -1,
    
    'num_classes': 10,
    'eval_freq': 5,
    'save_freq': 5,
    'save_freq_int': 0,
    'fbgd': False,
    'depth': 3,
}

output_dir = f"./Experiments/{args['model']}_{args['dataset']}_lri_{args['lr_init']}_" +\
    f"lrd_{args['lr_drop']}_dropepoch_{args['drop_epoch']}_wd_{args['wd']}_seed_{args['seed']}_noaug_{args['no_aug']}"

from pathlib import Path
Path(output_dir).mkdir(parents=True, exist_ok=True)

# In[12]:


# setup cuda

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=str(gpunumber)

device = 'cuda'

print('Using random seed {}'.format(args['seed']))
torch.backends.cudnn.benchmark = True
torch.manual_seed(args['seed'])
torch.cuda.manual_seed(args['seed'])


# In[ ]:


# import train

# import argparse


# In[ ]:


# parser = argparse.ArgumentParser(prog='PROG')
# parser.add_argument('--gpu', type=int, help='gpu_to_use')
# parser.add_argument('--lr', type=float, help='init lr')
# parser.add_argument('--lr_drop_epoch', type=int, help='epoch at which we are doing simple drop learning rate')
# parser.add_argument('--lr_drop_value', type=float, help='new lr value')

# forced_args = parser.parse_args()


# LR_START = float(forced_args.lr)
# LR_DROP_EPOCH = int(forced_args.lr_drop_epoch)
# LR_DROP_VALUE = float(forced_args.lr_drop_value)


# args = type('', (), {})()

# args.gpu = str(forced_args.gpu)

# args.dataset = "CIFAR10"
# args.data_path = "~/datasets/"  # path to datasets location (default: ~/datasets/)
# args.use_test = False      # use test dataset instead of validation (default: False)
# args.corrupt_train = 0.0  # train data corruption fraction (default: None --- no corruption)",
# args.split_classes = None  # split classes for CIFAR-10 (default: None --- no split)",
# args.fbgd = False  # train with full-batch GD (default: False)",
# args.batch_size = 128
# args.num_workers = 4
# args.model = "ResNet18SI"
# args.trial = 0          # trial number (default: 0)",
# args.resume_epoch = -1  # checkpoint epoch to resume training from (default: -1 --- start from scratch)",
# args.epochs = 1001      # number of epochs to train (default: 1001)",
# args.use_data_size = 50000  # how many examples to use for training",
# args.save_freq = 1   # save frequency (default: None --- custom saving)",
# args.save_freq_int = 0  # internal save frequency - how many times to save at each epoch (default: None --- custom saving)",
# args.eval_freq = 1     # "evaluation frequency (default: 10)",
# args.lr_init = LR_START      # initial learning rate (default: 0.01)",
# args.noninvlr = 0.0      # learning rate for not scale-inv parameters
# args.momentum = 0.0     # SGD momentum (default: 0.9)
# args.wd = 1e-3          # weight decay (default: 1e-4)
# args.loss = "CE"        # loss to use for training model (default: Cross-entropy)
# args.seed = 1           # random seed
# args.num_channels = 32  # number of channels for resnet
# args.depth = 3          # depth of convnet
# args.scale = 25         # scale of lenet
# args.no_schedule = True  # disable lr schedule
# args.c_schedule = None  # continuous schedule - decrease lr linearly after 1/4 epochs so that at the end it is x times lower "
# args.d_schedule = None  # discrete schedule - decrease lr x times after each 1/4 epochs "
# args.init_scale = 10    # init norm of the last layer weights "
# args.no_aug = True      # disable augmentation
# args.fix_si_pnorm = True  # set SI-pnorm to init after each iteration"
# args.fix_si_pnorm_value = -1  # custom fixed SI-pnorm value (must go with --fix_si_pnorm flag; default: -1 --- use init SI-pnorm value)",
# args.cosan_schedule = False  # cosine anealing schedule"


# # creating model

# In[13]:


print('='*80)
print('start process...')
print('='*80)

print('Weight decay = {}'.format(args['wd']))

def cross_entropy(model, input, target, reduction="mean"):
    "standard cross-entropy loss function"
    output = model(input)
    loss = F.cross_entropy(output, target, reduction=reduction)
    return loss, output


# In[14]:


def check_si_name(n, model_name='$$$'):
    if model_name == 'ResNet18':
        return "conv1" in n or "1.bn1" in n or "1.0.bn1" in n or (("conv2" in n or "short" in n) and "4" not in n)
    elif model_name == 'ResNet18SI':
        return 'linear' not in n
    elif model_name == 'ResNet18SIAf':
        return ('linear' not in n and 'bn' not in n and 'shortcut.0' not in n)
    elif 'ConvNet' in model_name:
        return 'conv_layers.0.' in n or 'conv_layers.3.' in n or 'conv_layers.7.' in n or 'conv_layers.11.' in n
    raise NotImplementedError("You should not get there")
    return False


# In[15]:





# In[ ]:


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
            
#         print('>>>', save_freq_int, save_ind)
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
                eval_freq=1, save_freq=10, save_freq_int=0, output_dir='./', lr_init=0.01,
                lr_drop=0.0001,
                noninvlr = -1, si_pnorm_0=None,fbgd=False):

    time_ep = time.time()

    if epoch >= drop_epoch:
        lr = lr_drop
    else:
        lr = lr_init
        
    if noninvlr >= 0:
        training_utils.adjust_learning_rate_only_conv(optimizer, lr)
    else:
        training_utils.adjust_learning_rate(optimizer, lr)
   
    train_res = train_epoch_custom(
        loader=loaders["train"], 
        model=model, 
        model_name=model_name, 
        criterion=criterion, 
        optimizer=optimizer, 
        fbgd=fbgd,             #  ??
        si_pnorm_0=si_pnorm_0,
        save_freq_int=save_freq_int,
        epoch=epoch,
        output_dir=output_dir
    )
    if (
        epoch == 0
        or epoch % eval_freq == 0
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

# In[25]:


if args['model'] == 'ConvNetSI':
    model_cfg = models.ConvNetSI
else:
    raise NotImplementedError('unknown model: {}'.format(args['model']))


# In[26]:


print("Preparing model")
print(*model_cfg.args)


# In[30]:


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


# #### setting up optimizers

# In[38]:


param_groups = model.parameters()

if args['noninvlr'] >= 0:
    print('Separate LR for last layer!')
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if check_si_name(n, args['model'])]},  # SI params are convolutions
        {'params': [p for n, p in model.named_parameters() if not check_si_name(n, args['model'])], 'lr': args['noninvlr']},  # other params
    ]
    for n, p in model.named_parameters():
        if check_si_name(n, args['model']):
            print(n, ' LR =', args['lr_init'])
        else:
            print(n, ' LR =', args['noninvlr'])

optimizer = torch.optim.SGD(
    param_groups, 
    lr=args['lr_init'], 
    momentum=args['momentum'], 
    weight_decay=args['wd'],
)

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


# In[45]:


epoch_from = 0
epoch_to = args['epochs']
print(f"Training from {epoch_from} to {epoch_to} epochs")

# if epoch_from > 0:
#     # Warning: due to specific lr schedule, resuming is generally not recommended!
#     print(f"Loading checkpoint from the {args.resume_epoch} epoch")
#     state = training_utils.load_checkpoint(resume_dir, args.resume_epoch)
#     model.load_state_dict(state['state_dict'])
#     optimizer.load_state_dict(state['optimizer'])
#     if args.noninvlr >= 0:
#         optimizer.param_groups[1]["lr"] = args.noninvlr

si_pnorm_0 = None
if args['fix_si_pnorm']:
    if args['fix_si_pnorm_value'] > 0:
        si_pnorm_0 = args['fix_si_pnorm_value']
    else:
        si_pnorm_0 = np.sqrt(sum((p ** 2).sum().item() for n, p in model.named_parameters() if check_si_name(n, args['model'])))

    print(f"Fixing SI-pnorm to value {si_pnorm_0:.4f}")


# # training

# In[46]:


def fix_si_pnorm(model, si_pnorm_0, model_name):
    "Fix SI-pnorm to si_pnorm_0 value"
    si_pnorm = np.sqrt(sum((p ** 2).sum().item() for n, p in model.named_parameters() if check_si_name(n, model_name)))
    p_coef = si_pnorm_0 / si_pnorm
    for n, p in model.named_parameters():
        if check_si_name(n, model_name):
            p.data *= p_coef

# In[ ]:

for epoch in range(epoch_from, epoch_to + 1):
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
        lr_init=args['lr_init'],
        lr_drop=args['lr_drop'],
        noninvlr=args['noninvlr'],
        si_pnorm_0=si_pnorm_0,
        fbgd=args['fbgd'],
    )

print("="*80)
print("model done")
