import os
import sys
import joblib
import torch
import numpy as np
from tqdm.auto import tqdm


def load_stats(configs, pt=True, num_ft_lrs=8):
    cache_file = os.path.join(
        os.path.split(configs[0].savedir)[0],
        f'cached_{"pt" if pt else "ft"}_stats.pickle'
    )
    read_seeds = 0
    if os.path.isfile(cache_file):
        cached_stats = joblib.load(cache_file)

        read_seeds = cached_stats[0].shape[0]
        if read_seeds == len(configs):
            return cached_stats

    lrs = configs[0].lrs
    n_seeds = len(configs)
    if pt:
        iters = configs[0].pt_iters // configs[0].log_iters
        s1 = (n_seeds, len(lrs), iters + 1)
    else:
        iters = configs[0].ft_iters // configs[0].log_iters
        s1 = (n_seeds, len(lrs), num_ft_lrs, iters + 1)

    all_arrays = tuple([np.zeros(s1) for _ in range(5)])
    keys = ['train_loss', 'test_loss', 'train_acc', 'test_acc', 'stoch_grad_norm']

    if read_seeds > 0:
        for array, cached_array in zip(all_arrays, cached_stats):
            array[:read_seeds] = cached_array
        del cached_stats

    desc = f'Loading {"pre-training" if pt else "fine-tuning"} stats'
    for s, config in enumerate(tqdm(configs[read_seeds:], desc=desc), read_seeds):
        if pt:
            for j, lr in enumerate(lrs):
                ckpt = torch.load(f'{config.savedir}/pt_lr={lr:.3e}.pt')
                for array, key in zip(all_arrays, keys):
                    if key not in ckpt['trace']:
                        continue
                    array[s, j] = np.array(ckpt['trace'][key])

        else:
            for i, pt_lr in enumerate(lrs):
                for j, ft_lr in enumerate(lrs[:num_ft_lrs]):
                    ckpt = torch.load(f'{config.ft_savedir}/pt_lr={pt_lr:.3e}-ft_lr={ft_lr:.3e}.pt')
                    for array, key in zip(all_arrays, keys):
                        if key not in ckpt['trace']:
                            continue
                        array[s, i, j] = np.array(ckpt['trace'][key])

    stats_size = sum(sys.getsizeof(array) for array in all_arrays) / 10 ** 6
    print(f'{"Pre-training" if pt else "Fine-tuning"} stats size: {stats_size:.2f} Mb')
    joblib.dump(all_arrays, cache_file)

    return all_arrays
