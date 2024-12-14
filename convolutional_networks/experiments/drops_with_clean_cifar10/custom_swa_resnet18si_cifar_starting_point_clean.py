import sys
sys.path.append('../..')

import math
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import time
import tabulate
import data
import training_utils
import nets as models
from datetime import datetime
from parser_train import parser
from pathlib import Path

from torch.optim.swa_utils import AveragedModel, SWALR

import train

from training_utils import fix_si_pnorm
from train import check_si_name

import argparse


parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--gpu', type=int, help='gpu_to_use')
parser.add_argument('--elr', type=float, help='init elr')
parser.add_argument('--k_epoch', type=int, help='how many epoch to perform SWA')
parser.add_argument('--stride', type=int, default=1, help='step, i.e. step 5 gives epochs 0,5,10,15 etc.')
parser.add_argument('--seed', type=int, help='random seed')
parser.add_argument('--start_swa_epoch', type=str, help='swa epochs listed and separated with comma (like "50,100,150")')
parser.add_argument('--aug', type=int, help='if true, enable aug')


forced_args = parser.parse_args()

ELR = float(forced_args.elr)


args = type('', (), {})()

args.gpu = str(forced_args.gpu)

args.dataset = "CIFAR10"
args.data_path = "~/datasets/"  # path to datasets location (default: ~/datasets/)
args.use_test = True      # use test dataset instead of validation (default: False)
args.corrupt_train = 0.0  # train data corruption fraction (default: None --- no corruption)",
args.split_classes = None  # split classes for CIFAR-10 (default: None --- no split)",
args.fbgd = False  # train with full-batch GD (default: False)",
args.batch_size = 128
args.num_workers = 4
args.model = "ResNet18SI"
args.trial = 0          # trial number (default: 0)",
args.resume_epoch = -1  # checkpoint epoch to resume training from (default: -1 --- start from scratch)",
args.epochs = 1001      # number of epochs to train (default: 1001)",
args.use_data_size = None  # how many examples to use for training",
args.save_freq = 1   # save frequency (default: None --- custom saving)",
args.save_freq_int = 0  # internal save frequency - how many times to save at each epoch (default: None --- custom saving)",
args.eval_freq = 1     # "evaluation frequency (default: 10)",
args.elr_init = forced_args.elr     # initial learning rate (default: 0.01)",
args.noninvlr = 0.0      # learning rate for not scale-inv parameters
args.momentum = 0.0     # SGD momentum (default: 0.9)
args.wd = 0.0           # weight decay (default: 1e-4)
args.loss = "CE"        # loss to use for training model (default: Cross-entropy)
args.seed = forced_args.seed  # random seed
args.num_channels = 32  # number of channels for resnet
args.depth = 3          # depth of convnet
args.scale = 25         # scale of lenet
args.no_schedule = True  # disable lr schedule
args.c_schedule = None  # continuous schedule - decrease lr linearly after 1/4 epochs so that at the end it is x times lower "
args.d_schedule = None  # discrete schedule - decrease lr x times after each 1/4 epochs "
args.init_scale = 10    # init norm of the last layer weights "
args.no_aug = not(bool(forced_args.aug))      # disable augmentation
args.fix_si_pnorm = True  # set SI-pnorm to init after each iteration"
args.fix_si_pnorm_value = 28.0  # custom fixed SI-pnorm value (must go with --fix_si_pnorm flag; default: -1 --- use init SI-pnorm value)",
args.cosan_schedule = False  # cosine anealing schedule"

if args.no_aug:
    args.dir = "./Experiments/SWA_K_{}_stride_{}_ResNet18SI_CIFAR10_elri_{}_elrd_{}_dropepoch_1001_wd_0.0".format(forced_args.k_epoch, forced_args.stride, ELR, ELR) # training directory (default: None)",
else:
    args.dir = "./Experiments/SWA_K_{}_stride_{}_ResNet18SI_CIFAR10_elri_{}_elrd_{}_dropepoch_1001_wd_0.0_noaug_{}".format(forced_args.k_epoch, forced_args.stride, ELR, ELR, 'False')


columns = ["ep", "lr", "tr_loss", "tr_acc", "te_loss", "te_acc", "time"]


def cross_entropy(model, input, target, reduction="mean"):
    "standard cross-entropy loss function"
    output = model(input)
    loss = F.cross_entropy(output, target, reduction=reduction)
    return loss, output


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if torch.cuda.is_available():
    args.device = torch.device("cuda")
    args.cuda = True
else:
    args.device = torch.device("cpu")
    args.cuda = False

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


# ---


print("Using model %s" % args.model)
model_cfg = getattr(models, args.model)

print("Loading dataset %s from %s" % (args.dataset, args.data_path))
transform_train = model_cfg.transform_test if args.no_aug else model_cfg.transform_train
loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    transform_train,
    model_cfg.transform_test,
    use_validation=not args.use_test,
    use_data_size = args.use_data_size,
    split_classes=args.split_classes,
    corrupt_train=args.corrupt_train
)

print("Preparing model")
print(*model_cfg.args)

# add extra args for varying names
if 'ResNet18' in args.model:
    extra_args = {'init_channels':args.num_channels}
    if "SI" in args.model:
        extra_args.update({'linear_norm':args.init_scale})
elif 'ConvNet' in args.model:
    extra_args = {'init_channels':args.num_channels, 'max_depth':args.depth,'init_scale':args.init_scale}
elif args.model == 'LeNet':
    extra_args = {'scale':args.scale}
else:
    extra_args = {}


model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs,
                       **extra_args)
model.to(args.device)

# --
K_EPOCH = forced_args.k_epoch

start_swa_epochs = [int(x) for x in forced_args.start_swa_epoch.split(',')]

for start_swa_epoch in start_swa_epochs:
    print('='*80)
    print('START SWA EPOCH:', start_swa_epoch)
    print('='*80)
    
    output_dir = args.dir + "/swa_start_{:03d}_k_{:03d}".format(start_swa_epoch, forced_args.stride)
    os.makedirs(output_dir, exist_ok=True)
    criterion = cross_entropy

    swa_model = AveragedModel(model).to(args.device)

    if args.no_aug:
        base = './Experiments/ResNet18SI_CIFAR10_elri_{}_elrd_{}_dropepoch_1000_wd_0.0_seed_{}_noaug_True/checkpoint-{}.pt'
    else:
        base = './Experiments/ResNet18SI_CIFAR10_elri_{}_elrd_{}_dropepoch_1000_wd_0.0_seed_{}_noaug_False/checkpoint-{}.pt'

    si_pnorm_0 = None
    for epoch in range(start_swa_epoch, start_swa_epoch + K_EPOCH, forced_args.stride):

        time_ep = time.time()

        # ------------------------------

        checkpoint_fname = base.format(ELR, ELR, forced_args.seed, epoch)

        model.load_state_dict(torch.load(checkpoint_fname, map_location=args.device)['state_dict'])

        if si_pnorm_0 is None:
            si_pnorm_0 = np.sqrt(sum((p ** 2).sum().item() for n, p in model.named_parameters() if check_si_name(n, args.model)))

        swa_model.update_parameters(model)

#         fix_si_pnorm(swa_model.module, si_pnorm_0, model_name=args.model)

        torch.optim.swa_utils.update_bn(loaders["train"], swa_model, device=args.device)

        # ------------------------------


        train_res = training_utils.eval(loaders["train"], swa_model, criterion)
        test_res = training_utils.eval(loaders["test"], swa_model, criterion)
        
        def save_epoch(epoch):
            training_utils.save_checkpoint(
                output_dir,
                epoch,
                state_dict=swa_model.state_dict(),
                train_res=train_res,
                test_res=test_res
            )

        save_epoch(epoch)

        time_ep = time.time() - time_ep
        values = [
            epoch,
            0,
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
