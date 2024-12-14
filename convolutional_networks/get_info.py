"""
    script to get info from trained models
"""
import torch
import os, sys
import re
import numpy as np
from collections import defaultdict

import data
import nets as models
from get_info_funcs import eval_model, eval_trace, eval_eigs, calc_grads,calc_grads_norms_small_memory
from parser_get_info import parser
from training_utils import get_resnet_prebn_groups

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

def main():
    args = parser()

    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print("Using model %s" % args.model)
    print("Train mode:", args.train_mode)
    model_cfg = getattr(models, args.model)

    print("Loading dataset %s from %s" % (args.dataset, args.data_path))
    print("Augmentation:", args.use_aug)
    transform = model_cfg.transform_train if args.use_aug else model_cfg.transform_test
    data_part = "test" if args.use_test else "train"
    print("Data part:", data_part)
    return_train_subsets = args.eval_on_train_subsets and args.corrupt_train is not None and data_part == "train"
    print("Evaluate model on train subsets:", return_train_subsets)

    loaders, num_classes = data.loaders(
        args.dataset,
        args.data_path,
        args.batch_size,
        args.num_workers,
        transform,
        transform,
        use_validation=False,
        use_data_size = args.use_data_size,
        corrupt_train=args.corrupt_train,
        shuffle_train=False,
        return_train_subsets=return_train_subsets
    )
    loader = loaders[data_part]

    if args.eval_on_random_subset:
        # Currently is not compatible with eval_on_train_subsets flag
        dataset = loader.dataset
        bs = args.batch_size
        rs_size = min(10 * bs, len(dataset))
        inds = np.random.choice(len(dataset), rs_size, False)
        subset = torch.utils.data.Subset(dataset, inds)
        loader = torch.utils.data.DataLoader(
            subset,
            batch_size=bs,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

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

    model = model_cfg.base(*model_cfg.args, num_classes=num_classes, 
                        **model_cfg.kwargs, **extra_args)
    model.cuda()

    pnames = None
    if args.prebn_only:
        print("Using pre-BN parameters only!")

        if args.model == 'ResNet18':
            pnames = [n for g in range(1, 4) for group in get_resnet_prebn_groups(g) for n in group]
        if args.model == 'ResNet18SI' or args.model == 'ResNet18SIAf':
            pnames = [n for n, _ in model.named_parameters() if check_si_name(n, args.model)]  # SI params are all but linear
        elif 'ConvNetSI' in args.model:
            pnames = [n for n, _ in model.named_parameters() if check_si_name(n, args.model)]  # SI params are convolutions
        else:
            raise ValueError("Using pre-BN parameters currently is not allowed for this model!")

    results = defaultdict(list)
    chkpts = [f for f in os.listdir(args.models_dir) if 'checkpoint-' in f]
    results['report_epochs'] = sorted([int(re.findall(r'\d+', s)[0]) for s in chkpts])
    if args.save_freq_int > 0:
        results['report_epochs'] = sorted([int(re.findall(r'\d+', s)[0])+(int(re.findall(r'\d+', s)[1])/args.save_freq_int if len(re.findall(r'\d+', s))>1 else 1) for s in chkpts])
        epoch_indexes = {int(re.findall(r'\d+', s)[0])+(int(re.findall(r'\d+', s)[1])/args.save_freq_int if len(re.findall(r'\d+', s))>1 else 1):'-'.join(re.findall(r'\d+', s)) for s in chkpts}
    print()

    for epoch in results['report_epochs']:
        if args.save_freq_int > 0:
            model_path = os.path.join(args.models_dir, f"checkpoint-{epoch_indexes[epoch]}.pt")
        else:
            model_path = os.path.join(args.models_dir, f"checkpoint-{epoch}.pt")
        print("Loading model: ", model_path)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["state_dict"])

        #with torch.no_grad():
        #    results["params_norm"].append(np.sqrt(sum((p ** 2).sum().item() for p in model.parameters())))
        #    if args.all_pnorm:
        #        for n, p in model.named_parameters():
        #            results[n + '_norm'].append(p.norm().item())
                    
        with torch.no_grad():
            allowed_pnames = pnames if pnames is not None else [n for n, _ in model.named_parameters()]
            results["params_norm"].append(np.sqrt(sum((p ** 2).sum().item() for n, p in model.named_parameters() if n in allowed_pnames)))
            results["params_numel"].append(sum(p.numel() for n, p in model.named_parameters() if n in allowed_pnames))
            if args.all_pnorm:
                for n, p in model.named_parameters():
                    if n in allowed_pnames:
                        results[n + '_norm'].append(p.norm().item())
                        results[n + '_numel'].append(p.numel())

        if args.eval_model:
            print("Evaluating model...")   

            if return_train_subsets:         
                loss_corrupt, acc_corrupt = eval_model(model, loaders["train_corrupt"], args.train_mode, False)
                corrupt_size = len(loaders["train_corrupt"].dataset)
                results['loss_corrupt'].append(loss_corrupt / corrupt_size)
                results['acc_corrupt'].append(acc_corrupt / corrupt_size)

                loss_normal, acc_normal = eval_model(model, loaders["train_normal"], args.train_mode, False)
                normal_size = len(loaders["train_normal"].dataset)
                results['loss_normal'].append(loss_normal / normal_size)
                results['acc_normal'].append(acc_normal / normal_size)

                loss = (loss_corrupt + loss_normal) / len(loader.dataset)
                acc = (acc_corrupt + acc_normal) / len(loader.dataset)
            else:         
                loss, acc = eval_model(model, loader, args.train_mode)

            results['loss'].append(loss)
            results['acc'].append(acc)

        if args.calc_grads:
            print("Calculating gradients...")
            grads_list = calc_grads(model, loader, args.train_mode, pnames=pnames)
            with torch.no_grad():
                gm, gs = [], []
                
                for p_grads in zip(*grads_list):  # taking all the gradients w.r.t. a particular parameter
                    p_grads = torch.stack(p_grads)
                    gm.append(p_grads.mean(0))
                    gs.append(p_grads.std(0))
                
                results["gm_norm"].append(np.sqrt(sum((g ** 2).sum().item() for g in gm)))
                results["gs_norm"].append(np.sqrt(sum((g ** 2).sum().item() for g in gs)))
            torch.cuda.empty_cache()
            
        if args.calc_grad_norms:
            print("Calculating gradients norms...")
            gnorm = calc_grads_norms_small_memory(model, loader, args.train_mode, pnames=pnames)
            results["gnorm_m"].append(np.array(gnorm).mean())
            #print(results["gnorm_m"][-1])
        torch.cuda.empty_cache() 

        def custom_eval_trace(fisher):
            prefix = 'F' if fisher else 'H'
            print(f"Calculating tr({prefix}) / N...")
            prefix = 'fisher_' if fisher else 'hess_'
            
            if return_train_subsets:   
                trace_corrupt = eval_trace(model, loaders["train_corrupt"], fisher, args.train_mode, pnames=pnames)
                results[prefix + 'trace_corrupt'].append(trace_corrupt)
                w_corrupt = len(loaders["train_corrupt"].dataset) / len(loader.dataset)

                trace_normal = eval_trace(model, loaders["train_normal"], fisher, args.train_mode, pnames=pnames)
                results[prefix + 'trace_normal'].append(trace_normal)
                w_normal = len(loaders["train_normal"].dataset) / len(loader.dataset)

                trace = w_corrupt * trace_corrupt + w_normal * trace_normal
            else:
                trace = eval_trace(model, loader, fisher, args.train_mode, pnames=pnames)

            results[prefix + 'trace'].append(trace)

        if args.fisher_trace:
            custom_eval_trace(True)
        elif args.hess_trace:
            custom_eval_trace(False)

        def custom_eval_eigs(fisher):
            prefix = 'F' if fisher else 'H'
            print(f"Calculating eig({prefix})...")
            prefix = 'fisher_' if fisher else 'hess_'

            eigvals, eigvecs = eval_eigs(model, loader, fisher, args.train_mode, pnames=pnames)
            eigvals = eigvals.cpu().numpy()
            results[prefix + 'evals'].append(eigvals)

        if args.fisher_evals:
            custom_eval_eigs(True)
        elif args.hess_evals:
            custom_eval_eigs(False)

        print()

    for k, v in results.items():
        results[k] = np.array(v)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    if args.update and os.path.isfile(args.save_path):
        print("Updating the old file")
        results_old = dict(np.load(args.save_path))
        results_old.update(results)
        results = results_old

    print("Saving all results to ", args.save_path)
    np.savez(args.save_path, **results)

    print()
    print(100 * '=')
    print()


if __name__ == '__main__':
    main()
