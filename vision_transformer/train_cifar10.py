# -*- coding: utf-8 -*-
"""
Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47
adapted by @isadrtdinov
"""

from __future__ import print_function

import random
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import csv
import time

from models import *
from torch import nn
from utils import progress_bar
from randomaug import RandAugment
from models.vit import ViT
from models.convmixer import ConvMixer

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
parser.add_argument('--plr', default=1e-4, type=float, help='pre-training learning rate')
parser.add_argument('--flr', default=None, type=float, help='fine-tuning learning rate')
parser.add_argument('--seed', default=None, type=int, help='training random seed')
parser.add_argument('--opt', default="adam", help='optimizer')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay value')
parser.add_argument('--warmup_epochs', default=0, type=int, help='learning rate warmup epochs')
parser.add_argument('--noaug', action='store_false', help='disable use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit', help='network architecture')
parser.add_argument('--dp', action='store_true', help='use data parallel')
parser.add_argument('--bs', default='512', help='batch size')
parser.add_argument('--size', default="32", help='image size')
parser.add_argument('--n_epochs', type=int, default='200', help='number of training epochs')
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int, help='network dimensionality')
parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")
parser.add_argument('--dataset', default='cifar10', type=str, help='training dataset (cifar10 or cifar100)')
parser.add_argument('--wandb_group', default=None, type=str, help='wandb experiment group')
parser.add_argument('--ckpt_dir', default='checkpoint', type=str, help='directory to store checkpoints')
parser.add_argument('--ckpt_epochs', default=None, type=int, nargs='+', help='epochs to save checkpoints')
args = parser.parse_args()

# take in args
usewandb = not args.nowandb
finetune = args.flr is not None
if not finetune:
    print(f'Launching pre-training with plr={args.plr}')
    watermark = f'PT_{args.dataset}_{args.net}_plr={args.plr}_wd={args.wd}_warmup={args.warmup_epochs}'
else:
    print(f'Launching fine-tuning from plr={args.plr} with flr={args.flr}')
    watermark = f'FT_{args.dataset}_{args.net}_plr={args.plr}_flr={args.flr}_wd={args.wd}_warmup={args.warmup_epochs}'
if args.wandb_group is None:
    args.wandb_group = f'{"FT" if finetune else "PT"}-{args.dataset}-'\
                       f'{args.net}-{args.opt}-{"aug" if args.noaug else "noaug"}'
if usewandb:
    import wandb
    wandb.init(project="vit-cifar10", name=watermark, group=args.wandb_group)
    wandb.config.update(args)

bs = int(args.bs)
imsize = int(args.size)
if args.ckpt_epochs is None:
    args.ckpt_epochs = [args.n_epochs]

use_amp = not args.noamp
aug = args.noaug
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Random seed
if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

# Data
print('==> Preparing data..')
if args.net == "vit_timm":
    size = 384
else:
    size = imsize

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Add RandAugment with N, M (hyperparameter)
if aug:
    N, M = 2, 14
    transform_train.transforms.insert(0, RandAugment(N, M))

# Prepare dataset
if args.dataset == 'cifar10':
    num_classes = 10
    train_set = torchvision.datasets.CIFAR10(
        root='~/datasets/cifar10', train=True, download=True,
        transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR10(
        root='~/datasets/cifar10', train=False, download=True,
        transform=transform_test
    )
elif args.dataset == 'cifar100':
    num_classes = 100
    train_set = torchvision.datasets.CIFAR100(
        root='~/datasets/cifar100', train=True, download=True,
        transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR100(
        root='~/datasets/cifar100', train=False, download=True,
        transform=transform_test
    )
else:
    raise ValueError(f'Unexpected dataset {args.dataset}')

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=bs, shuffle=True, num_workers=8,
)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=bs, shuffle=False, num_workers=8,
)

# Model factory..
print('==> Building model..')
# net = VGG('VGG19')
if args.net == 'res18':
    net = ResNet18()
elif args.net == 'vgg':
    net = VGG('VGG19')
elif args.net == 'res34':
    net = ResNet34()
elif args.net == 'res50':
    net = ResNet50()
elif args.net == 'res101':
    net = ResNet101()
elif args.net == "convmixer":
    # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
    net = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=num_classes)
elif args.net == "mlpmixer":
    from models.mlpmixer import MLPMixer

    net = MLPMixer(
        image_size=32,
        channels=3,
        patch_size=args.patch,
        dim=512,
        depth=6,
        num_classes=num_classes
    )
elif args.net == "vit_small":
    from models.vit_small import ViT

    net = ViT(
        image_size=size,
        patch_size=args.patch,
        num_classes=num_classes,
        dim=int(args.dimhead),
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1
    )
elif args.net == "vit_small_si":
    from models.vit_small_si import ViT

    net = ViT(
        image_size=size,
        patch_size=args.patch,
        num_classes=num_classes,
        dim=int(args.dimhead),
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1
    )

elif args.net == "vit_tiny":
    from models.vit_small import ViT

    net = ViT(
        image_size=size,
        patch_size=args.patch,
        num_classes=num_classes,
        dim=int(args.dimhead),
        depth=4,
        heads=6,
        mlp_dim=256,
        dropout=0.1,
        emb_dropout=0.1
    )
elif args.net == "simplevit":
    from models.simplevit import SimpleViT

    net = SimpleViT(
        image_size=size,
        patch_size=args.patch,
        num_classes=num_classes,
        dim=int(args.dimhead),
        depth=6,
        heads=8,
        mlp_dim=512
    )
elif args.net == "vit":
    # ViT for cifar10
    net = ViT(
        image_size=size,
        patch_size=args.patch,
        num_classes=num_classes,
        dim=int(args.dimhead),
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1
    )
elif args.net == "vit_timm":
    import timm

    net = timm.create_model("vit_base_patch16_384", pretrained=True)
    net.head = nn.Linear(net.head.in_features, num_classes)
elif args.net == "cait":
    from models.cait import CaiTnum_classes

    net = CaiT(
        image_size=size,
        patch_size=args.patch,
        num_classes=num_classes,
        dim=int(args.dimhead),
        depth=6,  # depth of transformer for patch to patch attention only
        cls_depth=2,  # depth of cross attention of CLS tokens to patch
        heads=8,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1,
        layer_dropout=0.05
    )
elif args.net == "cait_small":
    from models.cait import CaiT

    net = CaiT(
        image_size=size,
        patch_size=args.patch,
        num_classes=num_classes,
        dim=int(args.dimhead),
        depth=6,  # depth of transformer for patch to patch attention only
        cls_depth=2,  # depth of cross attention of CLS tokens to patch
        heads=6,
        mlp_dim=256,
        dropout=0.1,
        emb_dropout=0.1,
        layer_dropout=0.05
    )
elif args.net == "swin":
    from models.swin import swin_t

    net = swin_t(
        window_size=args.patch,
        num_classes=num_classes,
        downscaling_factors=(2, 2, 2, 1)
    )

# For Multi-GPU
if 'cuda' in device:
    print(device)
    if args.dp:
        print("using data parallel")
        net = torch.nn.DataParallel(net)  # make parallel
        cudnn.benchmark = True

if finetune:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(f'{args.ckpt_dir}/PT'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'{args.ckpt_dir}/PT/{args.dataset}-{args.net}-plr={args.plr}-ep={args.n_epochs}.pt')
    net.load_state_dict(checkpoint['model'])

# Loss is CE
criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.flr if finetune else args.plr, weight_decay=args.wd)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.flr if finetune else args.plr,
                          momentum=0.9, weight_decay=args.wd)
else:
    raise ValueError('Unknown optimizer')

# use constant scheduler with a possible warmup
if finetune:
    args.warmup_epochs = 0
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda epoch: min(1, (epoch + 1) / (args.warmup_epochs + 1))
)

# Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return train_loss / (batch_idx + 1), 100. * correct / total


# Validation
def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    acc = 100. * correct / total
    return test_loss / (batch_idx + 1), acc


def save_checkpoint(epoch):
    if epoch in args.ckpt_epochs:
        print(f'Saving epoch {epoch}')
        state = {
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict()
        }
        if not finetune:
            os.makedirs(f'{args.ckpt_dir}/PT', exist_ok=True)
            path = f'{args.ckpt_dir}/PT/{args.dataset}-{args.net}-plr={args.plr}-ep={epoch}.pt'
        else:
            os.makedirs(f'{args.ckpt_dir}/FT', exist_ok=True)
            path = f'{args.ckpt_dir}/FT/{args.dataset}-{args.net}-plr={args.plr}-flr={args.flr}-ep={epoch}.pt'
        torch.save(state, path)
        print(f'Checkpoint saved in {path}')


list_loss = []
list_acc = []

if usewandb:
    wandb.watch(net)

net.to(device)
print(f'{args.net} has {sum(param.numel() for param in net.parameters())} parameters')
os.makedirs(f"log-{args.wandb_group}", exist_ok=True)

log_txt = f'log-{args.wandb_group}/{watermark}.txt'
log_csv = f'log-{args.wandb_group}/{watermark}.csv'
for path in [log_txt, log_csv]:
    if os.path.isfile(path):
        os.remove(path)

val_loss, val_acc = 0, 0
for epoch in range(1, args.n_epochs + 1):
    start = time.time()
    train_loss, train_acc = train(epoch)
    if epoch % 10 == 0:
        val_loss, val_acc = test(epoch)

    scheduler.step()

    list_loss.append(val_loss)
    list_acc.append(val_acc)

    params = list(net.parameters())
    with torch.no_grad():
        backbone_norm = sum(param.square().sum().item() for param in params[:-2])
        fc_norm = sum(param.square().sum().item() for param in params[-2:])

    log_dict = {
        'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
        'val_loss': val_loss, "val_acc": val_acc,
        'backbone_norm': backbone_norm, 'fc_norm': fc_norm,
        "lr": optimizer.param_groups[0]["lr"],
        "epoch_time": time.time() - start
    }
    # Log training to wandb
    if usewandb:
        wandb.log(log_dict)

    # Save checkpoint
    save_checkpoint(epoch)

    # Log training metrics
    content = (
            time.ctime() + ' ' + f'Epoch {epoch:03d}, lr: {optimizer.param_groups[0]["lr"]:.2e}; '
                                 f'train loss: {train_loss:.3f}, train acc: {train_acc}%; '
                                 f'val loss: {val_loss:.3f}, val acc: {val_acc:.2f}%'
    )
    print(content)

    with open(log_txt, 'a') as appender:
        appender.write(content + "\n")

    if not os.path.isfile(log_csv):
        with open(log_csv, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(sorted(log_dict.keys()))

    with open(log_csv, 'a') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(
            list(map(lambda x: x[1], sorted(log_dict.items())))
        )
