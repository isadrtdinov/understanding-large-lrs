import os

plrs = [
    1e-5, 2e-5, 5e-5, 1e-4, 2e-4,
    5e-4, 1e-3, 2e-3, 5e-3, 1e-2
]
flrs = [1e-5, 2e-5, 5e-5, 1e-4]
dataset = 'cifar10'

for i, plr in enumerate(plrs, 1):
    for j, flr in enumerate(flrs, 1):
        python_command = (
            f'python train_cifar10.py '
            f'--net vit_small_si --plr {plr} --flr {flr} --seed {100 * j + i} '
            f'--patch 4 --opt adam --wd 1e-4 --noamp --n_epochs 500 --warmup_epochs 0 '
            f'--dataset {dataset} --ckpt_epochs 500 --ckpt_dir ckpts/{dataset}_ckpt_adam_wd=1e-4'
        )
        os.system(python_command)
