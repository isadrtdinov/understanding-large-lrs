import math
import torch
import torch.nn.functional as F
import numpy as np
import os, sys
import time
import tabulate
import data
import training_utils
from training_utils import check_si_name
import nets as models
from datetime import datetime
from parser_train import parser


columns = ["ep", "lr", "tr_loss", "tr_acc", "te_loss", "te_acc", "time"]

def cross_entropy(model, input, target, reduction="mean"):
    "standard cross-entropy loss function"
    output = model(input)
    loss = F.cross_entropy(output, target, reduction=reduction)
    return loss, output

    
def main():
    print('ENTER MAIN')
    args = parser()
    args.device = None
    
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

    if torch.cuda.is_available():
        args.device = torch.device("cuda")
        args.cuda = True
    else:
        args.device = torch.device("cpu")
        args.cuda = False
        
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # n_trials = 1
    
    print("Preparing base directory %s" % args.dir)
    os.makedirs(args.dir, exist_ok=True)

    # for trial in range(n_trials):
    trial = args.trial
    output_dir = args.dir + f"/trial_{trial}"
    
    ### resuming is modified!!!
    if args.resume_epoch > -1:
        resume_dir = output_dir
        output_dir = output_dir + f"/from_{args.resume_epoch}_for_{args.epochs}"
        if args.save_freq_int > 0:
            output_dir = output_dir + f"_save_int_{args.save_freq_int}"
        if args.noninvlr >= 0:
            output_dir = output_dir + f"_noninvlr_{args.noninvlr}"
    ### resuming is modified!!!
    print("Preparing directory %s" % output_dir)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "command.sh"), "w") as f:
        f.write(" ".join(sys.argv))
        f.write("\n")

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

    
    param_groups = model.parameters()

    if args.noninvlr >= 0:
        param_groups = [
            {'params': [p for n, p in model.named_parameters() if check_si_name(n, args.model)]},  # SI params are convolutions
            {'params': [p for n, p in model.named_parameters() if not check_si_name(n, args.model)],'lr':args.noninvlr},  # other params
        ]

    optimizer = torch.optim.SGD(param_groups, 
                                lr=args.lr_init, 
                                momentum=args.momentum, 
                                weight_decay=args.wd)
    
    if args.cosan_schedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    epoch_from = args.resume_epoch + 1
    epoch_to = epoch_from + args.epochs
    print(f"Training from {epoch_from} to {epoch_to - 1} epochs")

    if epoch_from > 0:
        # Warning: due to specific lr schedule, resuming is generally not recommended!
        print(f"Loading checkpoint from the {args.resume_epoch} epoch")
        state = training_utils.load_checkpoint(resume_dir, args.resume_epoch)
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        if args.noninvlr >= 0:
            optimizer.param_groups[1]["lr"] = args.noninvlr
        
    si_pnorm_0 = None
    if args.fix_si_pnorm:
        if args.fix_si_pnorm_value > 0:
            # No lr schedule, plz...
            si_pnorm_0 = args.fix_si_pnorm_value
        else:
            si_pnorm_0 = np.sqrt(sum((p ** 2).sum().item() for n, p in model.named_parameters() if check_si_name(n, args.model)))
            
        print(f"Fixing SI-pnorm to value {si_pnorm_0:.4f}")


    for epoch in range(epoch_from, epoch_to):
#         print('{} Epoch {:04d}'.format(datetime.now(), epoch))
        train_epoch(model, loaders, cross_entropy, optimizer, 
                    epoch=epoch, 
                    end_epoch=epoch_to, 
                    eval_freq=args.eval_freq, 
                    save_freq=args.save_freq,
                    save_freq_int=args.save_freq_int,
                    output_dir=output_dir,
                    lr_init=args.lr_init,
                    lr_schedule=not args.no_schedule,
                    noninvlr=args.noninvlr,
                    c_schedule=args.c_schedule,
                    d_schedule=args.d_schedule,
                    si_pnorm_0=si_pnorm_0,
                    fbgd=args.fbgd,
                    cosan_schedule = args.cosan_schedule)  
        if args.cosan_schedule:
            scheduler.step()
    

    print("model ", trial, " done")


def train_epoch(model, loaders, criterion, optimizer, epoch, end_epoch,
                eval_freq=1, save_freq=10, save_freq_int=0, output_dir='./', lr_init=0.01,
                lr_schedule=True, noninvlr = -1, c_schedule=None, d_schedule=None, si_pnorm_0=None,fbgd=False, 
               cosan_schedule = False):

    time_ep = time.time()

    if not cosan_schedule:
        if not lr_schedule:
            lr = lr_init
        elif c_schedule > 0:
            lr = training_utils.c_schedule(epoch, lr_init, end_epoch, c_schedule)
        elif d_schedule > 0:
            lr = training_utils.d_schedule(epoch, lr_init, end_epoch, d_schedule)
        else:
            lr = training_utils.schedule(epoch, lr_init, end_epoch, swa=False)
        if noninvlr >= 0:
            training_utils.adjust_learning_rate_only_conv(optimizer, lr)
        else:
            training_utils.adjust_learning_rate(optimizer, lr)
    else:
        for param_group in optimizer.param_groups:
            lr = param_group["lr"]
            break

    train_res = training_utils.train_epoch(loaders["train"], model, criterion, optimizer, fbgd=fbgd,si_pnorm_0=si_pnorm_0,
                                           save_freq_int=save_freq_int,epoch = epoch,output_dir = output_dir)
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

if __name__ == '__main__':
    main()
