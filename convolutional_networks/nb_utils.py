import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import nets as models


def get_model_str(lr, wd, ep, add_str=None):
    wd_str = f"{wd:.0e}".replace('0', '') if wd > 0.0 else str(wd)
    add_str = ('-' + add_str) if add_str is not None else ""
    return f"lr_{lr}-wd_{wd_str}-ep_{ep}" + add_str

def get_model_extra(model_name, **kwargs):
    channels = kwargs.get("channels", 32)
    depth = kwargs.get("depth", 3) 
    # scale = kwargs.get("scale", 25)

    extra_args, mid_str = {}, ""
    if model_name == 'ResNet18':
        extra_args = {'init_channels': channels}
        mid_str = f"resnet_{channels}"
    elif model_name == 'ConvNet' or model_name == 'ConvNetSI':
        extra_args = {'init_channels': channels, 'max_depth': depth}
        mid_str = f"channels_{channels}-depth_{depth}"
    return extra_args, mid_str

def load_checkpoint(models_dir, model_name='ResNet18', cuda=False, **kwargs):
    num_classes = kwargs.get("num_classes", 100)
    lrwdep = kwargs.get("lrwdep", (0.01, 1e-4, 1001)) 
    trial = kwargs.get("trial", 0)
    chkpt = kwargs.get("chkpt", 0)

    device = torch.device('cuda' if cuda else 'cpu')
    model_cfg = getattr(models, model_name)
    extra_args, mid_str = get_model_extra(model_name, **kwargs)

    model_path = os.path.join(models_dir, mid_str, get_model_str(*lrwdep), f"trial_{trial}", f"checkpoint-{chkpt}.pt")
    checkpoint = torch.load(model_path, map_location=device)
    checkpoint["num_classes"] = num_classes
    checkpoint["lrwdep"] = lrwdep
    checkpoint["trial"] = trial
    checkpoint["chkpt"] = chkpt
    checkpoint["device"] = device
    checkpoint["model_cfg"] = model_cfg
    checkpoint["extra_args"] = extra_args
    checkpoint["mid_str"] = mid_str
    return checkpoint

def load_model(checkpoint):
    model_cfg = checkpoint["model_cfg"]
    num_classes = checkpoint["num_classes"]
    extra_args = checkpoint["extra_args"]
    device = checkpoint["device"]
    state_dict = checkpoint["state_dict"]

    model = model_cfg.base(*model_cfg.args, 
                           num_classes=num_classes, 
                           **model_cfg.kwargs,
                           **extra_args)
    model = model.to(device)
    model.load_state_dict(state_dict)
    return model

def check_si_name(n, model_name='ResNet18'):
    if model_name == 'ResNet18':
        return "conv1" in n or "1.bn1" in n or "1.0.bn1" in n or (("conv2" in n or "short" in n) and "4" not in n)
    elif model_name == 'ResNet18SI':
        return 'linear' not in n
    elif model_name == 'ResNet18SIAf':
        return ('linear' not in n and 'bn' not in n and 'shortcut.0' not in n)
    elif 'ConvNet' in model_name:
        return 'conv_layers.0.' in n or 'conv_layers.3.' in n or 'conv_layers.7.' in n or 'conv_layers.11.' in n
    return False

def load_info(npz_path, model_name='ResNet18', **kwargs):
    results_dict = dict(np.load(npz_path))
    lr = kwargs.get("lr")
    params_dict = kwargs.get("params_dict")
    
    prebn_pnorm_sqr = 0.0
    prebn_numel = 0
    
    for key in list(results_dict.keys()):
        if "acc" in key:
            results_dict[key.replace("acc", "error")] = 1.0 - results_dict[key]  
        elif lr is not None and "_norm" in key and key != "params_norm":
            p_name = key[:-5]
            if check_si_name(p_name, model_name):
                p_norm_sqr = results_dict[key] ** 2
                prebn_pnorm_sqr += p_norm_sqr
                elr = lr / p_norm_sqr
                results_dict[key.replace("norm", "elr")] = elr

                if params_dict is not None:
                    p_numel = params_dict[p_name].numel()
                    prebn_numel += p_numel
                    results_dict[key.replace("norm", "elr_unit")] = elr * p_numel
    
    if not isinstance(prebn_pnorm_sqr, float):
        results_dict["prebn_pnorm"] = np.sqrt(prebn_pnorm_sqr)
        elr = lr / prebn_pnorm_sqr
        results_dict["elr"] = elr
        results_dict["elr_unit"] = elr * prebn_numel
        F_tr = results_dict.get("fisher_trace")
        if F_tr is not None:
            # reduced F_tr is almost the same as effective F_tr when the fraction of SI-params is high
            results_dict["fisher_trace_r"] = F_tr * prebn_pnorm_sqr  
    
    return results_dict

def load_check_lrwdep_info(exp_spec, check_lrwdep, base_keys, model_name='ResNet18', **kwargs):
    info_dir=f'./info/{exp_spec}/'
    models_dir = f'./models/{exp_spec}/'

    _, mid_str = get_model_extra(model_name, **kwargs)
    npz_prepath = os.path.join(info_dir, mid_str)

    checkpoint = load_checkpoint(models_dir, model_name, lrwdep=check_lrwdep[0], **kwargs)
    model = load_model(checkpoint)
    params_dict = dict(model.named_parameters())

    all_numel = sum(p.numel() for p in params_dict.values())
    prebn_numel = sum(p.numel() for n, p in params_dict.items() if check_si_name(n, model_name))
    others_numel = all_numel - prebn_numel
    w_all, w_prebn = all_numel / others_numel, prebn_numel / others_numel

    results_dict = dict()

    for lrwdep in check_lrwdep:
        results_dict[lrwdep] = dict()

        for key in base_keys:
            npz_path = os.path.join(npz_prepath, get_model_str(*lrwdep), f"{key}.npz")
            results_dict[lrwdep][key] = load_info(npz_path, model_name, lr=lrwdep[0], params_dict=params_dict)
            
        if "train-tm-prebn" in base_keys:
            params_norm = results_dict[lrwdep]["train-tm"]["params_norm"]
            prebn_pnorm = results_dict[lrwdep]["train-tm"]["prebn_pnorm"]
            results_dict[lrwdep]["train-tm-prebn"]["params_norm"] = prebn_pnorm
            results_dict[lrwdep]["train-tm"]["params_norm_others"] = np.sqrt(params_norm ** 2 - prebn_pnorm ** 2)

            for c in ['m', 's']:
                g_norm = results_dict[lrwdep]["train-tm"][f"g{c}_norm"]
                g_norm_prebn = results_dict[lrwdep]["train-tm-prebn"][f"g{c}_norm"]
                results_dict[lrwdep]["train-tm"][f"g{c}_norm_others"] = np.sqrt(g_norm ** 2 - g_norm_prebn ** 2)

            F_tr = results_dict[lrwdep]["train-tm"]["fisher_trace"]
            F_tr_prebn = results_dict[lrwdep]["train-tm-prebn"]["fisher_trace"]
            
            results_dict[lrwdep]["train-tm"]["fisher_trace_others"] = w_all * F_tr - w_prebn * F_tr_prebn
            results_dict[lrwdep]["train-tm-prebn"]["fisher_trace_e"] = F_tr_prebn * prebn_pnorm ** 2

        if "fixed_elr" in exp_spec:
            # This currently properly works for ConvNetSI only
            epoch = 0
            epoch_lr = epoch + 1  # taking the next epoch due to saving format
            checkpoint = load_checkpoint(models_dir, model_name, lrwdep=lrwdep, chkpt=epoch_lr, **kwargs)
            
            lr = checkpoint["optimizer"]["param_groups"][0]['lr']
            p_norm_sqr = results_dict[lrwdep]["train-tm"]["prebn_pnorm"][0] ** 2
            elr = lr / p_norm_sqr
            results_dict[lrwdep]["train-tm"]["elr"] = elr * np.ones_like(results_dict[lrwdep]["train-tm"]["elr"])

    return results_dict

def plot_stuff(results_dict, title, 
               left_title, right_title, 
               left_base_key, right_base_key, 
               left_key, right_key, 
               yscale='linear'):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=300)
    fig.suptitle(title)

    axes[0].set(title=left_title, xscale='log', yscale=yscale, xlabel="Epoch")
    axes[1].set(title=right_title, xscale='log', yscale=yscale, xlabel="Epoch")

    for lrwdep in results_dict.keys():
        report_epochs = results_dict[lrwdep][left_base_key]["report_epochs"]
        label = get_model_str(*lrwdep)
        axes[0].plot(report_epochs, results_dict[lrwdep][left_base_key][left_key], '.-', label=label)
        axes[1].plot(report_epochs, results_dict[lrwdep][right_base_key][right_key], '.-', label=label)

    for ax in axes:
        ax.legend()
        ax.grid()

    plt.show()

def plot_some_pnorms(results_dict, lrwdep, pnames):
    plt.figure(figsize=(6,3), dpi=150)
    plt.title("Some params norms for " + get_model_str(*lrwdep))

    plt.xlabel("Epoch")
    plt.loglog()
    
    res_dict = results_dict[lrwdep]["train-tm"]
    report_epochs = res_dict["report_epochs"]
    for pname in pnames:
        plt.plot(report_epochs, res_dict[pname + "_norm"], '.-', label=pname)
                
    plt.legend()
    plt.grid()
    plt.show()

def plot_params_norms(results_dict, lrwdep, params_substrs_left=[".conv1"], params_substrs_right=["linear", "4.1.conv"]):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=300)
    fig.suptitle("Params norms for " + get_model_str(*lrwdep))

    axes[0].set(title="Pre-BN", xscale='log', yscale='log', xlabel="Epoch")
    axes[1].set(title="Other", xscale='log', yscale='log', xlabel="Epoch")

    report_epochs = results_dict[lrwdep]["train-tm"]["report_epochs"]
    for key, value in results_dict[lrwdep]["train-tm"].items():
        if "_norm" in key and key != "params_norm":
            p_name = key[:-5]
            for p_susbstr in params_substrs_left:
                if p_susbstr in p_name:
                    axes[0].plot(report_epochs, value, '.-', label=p_name)
            for p_susbstr in params_substrs_right:
                if p_susbstr in p_name:
                    axes[1].plot(report_epochs, value, '.-', label=p_name)

    for ax in axes:
        ax.legend()
        ax.grid()

    plt.show()

def plot_elrs(results_dict, lrwdep, elr_params_substr=".conv1"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=300)
    fig.suptitle("Effective LRs for " + get_model_str(*lrwdep))

    axes[0].set(title="ELR", xscale='log', yscale='log', xlabel="Epoch")
    axes[1].set(title="Unit ELR", xscale='log', yscale='log', xlabel="Epoch")

    report_epochs = results_dict[lrwdep]["train-tm"]["report_epochs"]
    for key in results_dict[lrwdep]["train-tm"].keys():
        if "_norm" in key and key != "params_norm":
            p_name = key[:-5]
            if elr_params_substr in p_name:
                axes[0].plot(report_epochs, results_dict[lrwdep]["train-tm"][p_name + "_elr"], '.-', label=p_name)
                axes[1].plot(report_epochs, results_dict[lrwdep]["train-tm"][p_name + "_elr_unit"], '.-', label=p_name)

    for ax in axes:
        ax.legend()
        ax.grid()

    plt.show()

def plot_prebn_vs_others(results_dict, lrwdep, key_title, key_name):
    key = results_dict[lrwdep]["train-tm"][key_name]
    key_prebn = results_dict[lrwdep]["train-tm-prebn"][key_name]
    key_others = results_dict[lrwdep]["train-tm"][key_name + "_others"]
    
    plt.figure(figsize=(6,3), dpi=150)
    plt.title(key_title + " for " + get_model_str(*lrwdep))
    plt.loglog()
    plt.xlabel("Epoch")
    
    report_epochs = results_dict[lrwdep]["train-tm"]["report_epochs"]
    plt.plot(report_epochs, key, '.-', label="All parameters")
    plt.plot(report_epochs, key_prebn, '.-', label="Pre-BN parameters")
    plt.plot(report_epochs, key_others, '.-', label="Other parameters")
    
    plt.legend()
    plt.grid()    
    plt.show()

def plot_alot(results_dict, all_title, titles, keys, 
              main_stuff=None, save_path=None, 
              xscale='log', yscale='log', title_y=0.98):
    rows, cols = len(titles) // 2, 2
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows), dpi=300)
    fig.suptitle(all_title, fontsize=16, y=title_y)

    for ax, title in zip(axes.flat, titles):
        ax.set(title=title, xscale=xscale, yscale=yscale)

    for ax in (axes[-1] if rows > 1 else axes):
        ax.set_xlabel('Epoch')

    for lrwdep in results_dict.keys():
        label = get_model_str(*lrwdep)
        for ax, (base_key, key) in zip(axes.flat, keys):
            report_epochs = results_dict[lrwdep][base_key]["report_epochs"]
            ax.plot(report_epochs, results_dict[lrwdep][base_key][key], '.-', label=label)

    if main_stuff is not None:
        main_model_key, main_pts = main_stuff
        for ax, (base_key, key) in zip(axes.flat, keys):
            main_report_epochs = results_dict[main_model_key][base_key]["report_epochs"]
            ax.plot(main_report_epochs[main_pts], results_dict[main_model_key][base_key][key][main_pts], 'ro')

    for ax in axes.flat:
        ax.legend()
        ax.grid()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', format='png', dpi=300)
    plt.show()

def pltnsave(results_dict, title, base_key, key, 
             main_stuff=None, save_path=None, 
             xscale='log', yscale='log'):
    plt.figure(figsize=(12,6), dpi=150)
    plt.title(title)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.xlabel("Epoch")
    
    for lrwdep in results_dict.keys():
        label = get_model_str(*lrwdep)
        report_epochs = results_dict[lrwdep][base_key]["report_epochs"]
        plt.plot(report_epochs, results_dict[lrwdep][base_key][key], '.-', label=label)
    
    if main_stuff is not None:
        main_model_key, main_pts = main_stuff
        main_report_epochs = results_dict[main_model_key][base_key]["report_epochs"]
        plt.plot(main_report_epochs[main_pts], results_dict[main_model_key][base_key][key][main_pts], 'ro')

    plt.legend()
    plt.grid()
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', format='png', dpi=300)
    plt.show()
