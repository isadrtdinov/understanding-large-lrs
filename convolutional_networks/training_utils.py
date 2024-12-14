import itertools
import torch
import os
import copy
from datetime import datetime
import math
import numpy as np
import tqdm

import torch.nn.functional as F


def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        # n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:, i : i + n].view(tensor.shape))
        i += n
    return outList


def LogSumExp(x, dim=0):
    m, _ = torch.max(x, dim=dim, keepdim=True)
    return m + torch.log((x - m).exp().sum(dim=dim, keepdim=True))


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

def adjust_learning_rate_only_conv(optimizer, lr):
    optimizer.param_groups[0]["lr"] = lr
    return lr

def get_resnet_prebn_groups(g):
    groups = None

    if g == 1:  # single pre-BN params
        groups = [
            ["layer1.0.conv1.weight"], 
            ["layer1.1.conv1.weight"],
            ["layer2.0.conv1.weight"], 
            ["layer2.1.conv1.weight"],
            ["layer3.0.conv1.weight"], 
            ["layer3.1.conv1.weight"],
            ["layer4.0.conv1.weight"], 
            ["layer4.1.conv1.weight"],
        ]

    elif g == 2:  # pairs of pre-BN params
        groups = [
            ["layer1.0.bn1.weight", "layer1.0.bn1.bias"],
            ["layer1.1.bn1.weight", "layer1.1.bn1.bias"],
            ["layer2.1.bn1.weight", "layer2.1.bn1.bias"],
            ["layer3.1.bn1.weight", "layer3.1.bn1.bias"],
            ["layer4.1.bn1.weight", "layer4.1.bn1.bias"],
        ]

    elif g == 3:  # triples of pre-BN params
        groups = [
            ["conv1.weight", "layer1.0.conv2.weight", "layer1.1.conv2.weight"],
            ["layer2.0.conv2.weight", "layer2.0.shortcut.0.weight", "layer2.1.conv2.weight"],
            ["layer3.0.conv2.weight", "layer3.0.shortcut.0.weight", "layer3.1.conv2.weight"],
        ]

    return groups


def do_report(epoch):
    # Only log activity for some epochs.  Mainly this is to make things run faster.
    if epoch < 20:       # Log for all first 20 epochs
        return True
    elif epoch < 100:    # Then for every 5th epoch
        return (epoch % 5 == 0)
    elif epoch < 200:    # Then every 10th
        return (epoch % 10 == 0)
    elif epoch < 1000:    # Then every 50th
        return (epoch % 50 == 0)
    elif epoch < 2000:    # Then every 100th
        return (epoch % 100 == 0)
    elif epoch < 10000:    # Then every 500th
        return (epoch % 500 == 0)
    # Then every 1000th
    return (epoch % 1000 == 0)


def save_checkpoint(dir, epoch, name="checkpoint", **kwargs):
    state = {"epoch": epoch}
    state.update(kwargs)
    filepath = os.path.join(dir, "%s-%d.pt" % (name, epoch))
    torch.save(state, filepath)
    
def save_checkpoint_int(dir, epoch, index, name="checkpoint", **kwargs):
    state = {"epoch": epoch,"index":index}
    state.update(kwargs)
    filepath = os.path.join(dir, "%s-%d-%d.pt" % (name, epoch,index))
    torch.save(state, filepath)
    
def load_checkpoint(dir, epoch, name="checkpoint"):
    filepath = os.path.join(dir, "%s-%d.pt" % (name, epoch))
    state = torch.load(filepath)
    return state


def check_si_name(n, model_name):
    if model_name == 'ResNet18':
        return "conv1" in n or "1.bn1" in n or "1.0.bn1" in n or (("conv2" in n or "short" in n) and "4" not in n)
    elif model_name == 'ResNet18SI':
        return 'linear' not in n
    elif model_name == 'WideResNetSI':
        return ('linear' not in n and 'bn' not in n)
    elif model_name == 'ResNet18SIAf':
        return ('linear' not in n and 'bn' not in n and 'shortcut.0' not in n)
    elif 'ConvNet' in model_name:
        return 'conv_layers.0.' in n or 'conv_layers.3.' in n or 'conv_layers.7.' in n or 'conv_layers.11.' in n
    return False


def fix_si_pnorm(model, si_pnorm_0, model_name):
    "Fix SI-pnorm to si_pnorm_0 value"
    si_pnorm = np.sqrt(sum((p ** 2).sum().item() for n, p in model.named_parameters() if check_si_name(n, model_name)))
    p_coef = si_pnorm_0 / si_pnorm
    for n, p in model.named_parameters():
        if check_si_name(n, model_name):
            p.data *= p_coef
            

def train_epoch(
    loader,
    model,
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
                fix_si_pnorm(model, si_pnorm_0)
                
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


def eval(loader, model, criterion, cuda=True, regression=False, verbose=False):
    loss_sum = 0.0
    correct = 0.0
    num_objects_total = len(loader.dataset)

    model.eval()

    with torch.no_grad():
        if verbose:
            loader = tqdm.tqdm(loader)
        for i, (input, target) in enumerate(loader):
            if cuda:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            loss, output = criterion(model, input, target)

            loss_sum += loss.item() * input.size(0)

            if not regression:
                pred = output.data.argmax(1, keepdim=True)
                correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        "loss": loss_sum / num_objects_total,
        "accuracy": None if regression else correct / num_objects_total * 100.0,
    }


def predict(loader, model, verbose=False):
    predictions = list()
    targets = list()

    model.eval()

    if verbose:
        loader = tqdm.tqdm(loader)

    offset = 0
    with torch.no_grad():
        for input, target in loader:
            input = input.cuda(non_blocking=True)
            output = model(input)

            batch_size = input.size(0)
            predictions.append(F.softmax(output, dim=1).cpu().numpy())
            targets.append(target.numpy())
            offset += batch_size

    return {"predictions": np.vstack(predictions), "targets": np.concatenate(targets)}


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= 1.0 - alpha
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model, verbose=False, subset=None, **kwargs):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    num_batches = len(loader)

    with torch.no_grad():
        if subset is not None:
            num_batches = int(num_batches * subset)
            loader = itertools.islice(loader, num_batches)
        if verbose:

            loader = tqdm.tqdm(loader, total=num_batches)
        for input, _ in loader:
            input = input.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            b = input_var.data.size(0)

            momentum = b / (n + b)
            for module in momenta.keys():
                module.momentum = momentum

            model(input_var, **kwargs)
            n += b

    model.apply(lambda module: _set_momenta(module, momenta))


def inv_softmax(x, eps=1e-10):
    return torch.log(x / (1.0 - x + eps))


def predictions(test_loader, model, seed=None, cuda=True, regression=False, **kwargs):
    # will assume that model is already in eval mode
    # model.eval()
    preds = []
    targets = []
    for input, target in test_loader:
        if seed is not None:
            torch.manual_seed(seed)
        if cuda:
            input = input.cuda(non_blocking=True)
        output = model(input, **kwargs)
        if regression:
            preds.append(output.cpu().data.numpy())
        else:
            probs = F.softmax(output, dim=1)
            preds.append(probs.cpu().data.numpy())
        targets.append(target.numpy())
    return np.vstack(preds), np.concatenate(targets)


def schedule(epoch, lr_init, epochs, swa, swa_start=None, swa_lr=None):
    t = (epoch) / (swa_start if swa else epochs)
    lr_ratio = swa_lr / lr_init if swa else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return lr_init * factor

def d_schedule(epoch, lr_init, epochs, p):
    # discrete schedule - decrease lr x times after each 1/4 epochs
    t = epoch / epochs
    if t <= 0.25:
        factor = 1.0
    elif t <= 0.5:
        factor = 1.0/p
    elif t <= 0.75:
        factor = 1.0/p/p
    else:
        factor = 1.0/p/p/p
    return lr_init * factor

def c_schedule(epoch, lr_init, epochs, p):
    # continuous schedule - decrease lr linearly after 1/4 epochs so that at the end it is x times lower 
    t = epoch / epochs
    if t <= 0.25:
        factor = 1.0
    else:
        factor = 1.0/p+(1-1.0/p)*(1-t)/0.75
    return lr_init * factor
