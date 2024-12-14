import sys
sys.path.append('../..')
sys.path.append('../../..')


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
from custom_training_utils import *
# from get_info_funcs import calc_grads, calc_grads_norms_small_memory


import argparse
parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--gpu', type=int, help='gpu_to_use')
parser.add_argument('--elr', type=float, help='init elr')
parser.add_argument('--n_interp', type=int, help='', default=2000)
parser.add_argument('--recalc_bn', type=int, help='whether to recalc bn on every epoch or not', default=1)
parser.add_argument('--point_a', type=str)
parser.add_argument('--point_b', type=str)
parser.add_argument('--save', type=str)

# args ------------------------------------------------

forced_args = parser.parse_args()

ELR = float(forced_args.elr)

args = type('', (), {})()

args.gpu = str(forced_args.gpu)
args.dir = forced_args.save #"./../../Experiments/ResNetSI_CIFAR10_noninvlr_0.0/checkpoints_interp/c32_d3_ds50000_lr{}_wd0.001_mom0.0_corr0.0_epoch1001_noschinitscale10.0_noaug".format(LR) # training directory (default: None)",
args.dataset = "CIFAR10"
args.data_path = "~/datasets/"  # path to datasets location (default: ~/datasets/)
args.use_test = True      # use test dataset instead of validation (default: False)
args.corrupt_train = None  # train data corruption fraction (default: None --- no corruption)",
args.split_classes = None  # split classes for CIFAR-10 (default: None --- no split)",
args.fbgd = False  # train with full-batch GD (default: False)",
args.batch_size = 128
args.num_workers = 4
args.model = "ResNet18SIAf"
args.trial = 0          # trial number (default: 0)",
args.resume_epoch = -1  # checkpoint epoch to resume training from (default: -1 --- start from scratch)",
args.epochs = 1001      # number of epochs to train (default: 1001)",
args.use_data_size = None  # how many examples to use for training",
args.save_freq = 1   # save frequency (default: None --- custom saving)",
args.save_freq_int = 0  # internal save frequency - how many times to save at each epoch (default: None --- custom saving)",
args.eval_freq = 1     # "evaluation frequency (default: 10)",
args.elr_init = ELR     # initial learning rate (default: 0.01)",
args.noninvlr = 0.0      # learning rate for not scale-inv parameters
args.momentum = 0.0     # SGD momentum (default: 0.9)
args.wd = 0.0          # weight decay (default: 1e-4)
args.loss = "CE"        # loss to use for training model (default: Cross-entropy)
args.seed = 1           # random seed
args.num_channels = 64  # number of channels for resnet
args.depth = 3          # depth of convnet
args.scale = 25         # scale of lenet
args.no_schedule = True  # disable lr schedule
args.c_schedule = None  # continuous schedule - decrease lr linearly after 1/4 epochs so that at the end it is x times lower "
args.d_schedule = None  # discrete schedule - decrease lr x times after each 1/4 epochs "
args.init_scale = 10    # init norm of the last layer weights "
args.no_aug = True      # disable augmentation
args.fix_si_pnorm = True  # set SI-pnorm to init after each iteration"
args.fix_si_pnorm_value = None  # custom fixed SI-pnorm value (must go with --fix_si_pnorm flag; default: -1 --- use init SI-pnorm value)",
args.cosan_schedule = False  # cosine anealing schedule"

columns = ["ep", "alpha", "tr_loss", "tr_acc", "te_loss", "te_acc", "time"]

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu


# ------------------------------------------------


if torch.cuda.is_available():
    args.device = torch.device("cuda")
    args.cuda = True
else:
    args.device = torch.device("cpu")
    args.cuda = False
    

print('='*80)
print('start process...')
print('='*80)

print('Weight decay = {}'.format(args.wd))


torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

output_dir = args.dir 
os.makedirs(output_dir, exist_ok=True)


# ------------------------------------------------


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


# ------------------------------------------------

criterion = cross_entropy

checkpoint_fname_A = forced_args.point_a
checkpoint_fname_B = forced_args.point_b

def load_sd(fname, map_location):
    ckpt = torch.load(fname, map_location=map_location)
    if 'n_averaged' in ckpt['state_dict']:
        swa_model = AveragedModel(model)#.to(args.device)
        swa_model.load_state_dict(ckpt["state_dict"])
        print(type(swa_model.module))
        sd = swa_model.module.state_dict()
    else:
        sd = ckpt['state_dict']
    return sd

# sd_A = torch.load(checkpoint_fname_A, map_location=args.device)['state_dict']
# sd_B = torch.load(checkpoint_fname_B, map_location=args.device)['state_dict']
sd_A = load_sd(checkpoint_fname_A, map_location=args.device)
sd_B = load_sd(checkpoint_fname_B, map_location=args.device)

model.load_state_dict(sd_A)

si_pnorm_0 = None
if si_pnorm_0 is None:
    si_pnorm_0 = np.sqrt(sum((p ** 2).sum().item() for n, p in model.named_parameters() if check_si_name(n, args.model)))

print('SI_NORM =', si_pnorm_0)

    
for alpha_it, alpha in enumerate(np.linspace(0.0, 1.0, forced_args.n_interp + 1)):
    
    time_ep = time.time()
    
    # ------------------------------
    
    sd = interpolate_state_dicts(sd_A, sd_B, alpha)
    
    model.load_state_dict(sd)
       
#     fix_si_pnorm(model, si_pnorm_0, model_name=args.model)      
    
    if forced_args.recalc_bn > 0:
        torch.optim.swa_utils.update_bn(loaders["train"], model, device=args.device)

    model.eval()
        
    # ------------------------------
    
    train_res = training_utils.eval(loaders["train"], model, criterion)
    test_res = training_utils.eval(loaders["test"], model, criterion)
    
#     grad_norm = calc_grads_norms_small_memory(model, loaders['train'])
    
    def save_epoch(alpha):
        training_utils.save_checkpoint(
            output_dir,
            alpha,
            state_dict=model.state_dict(),
            name='interp_result_{:5.4f}'.format(alpha),
            train_res=train_res,
            test_res=test_res
        )

    save_epoch(alpha)
        
    time_ep = time.time() - time_ep
    values = [
        alpha,
        0,
        train_res["loss"],
        train_res["accuracy"],
        test_res["loss"],
        test_res["accuracy"],
        time_ep,
    ]
    table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
    if alpha_it % 40 == 0:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)
