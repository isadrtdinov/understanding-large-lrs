import glob
import os.path

import torch
import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from tqdm.auto import tqdm
from fourier_slicer import FourierSlicer
from models.vit_small_si import ViT


# set these variable to process pre-training, fine-tuning or SWA
flr = 1e-5  # FLR to process fine-tuning, None for pre-training/SWA
num_swa = 1  # set number of models > 1 to process SWA (flr must be None)
dataset, model_name = 'cifar100', 'vit_small_si'
ckpt_dir = f'ckpts/{dataset}_ckpt_adam_wd=1e-4'


swa_epochs = list(range(500 - num_swa * 2 + 2, 501, 2))  # if num_swa == 1, swa_epochs == [500]
if flr is None:
    lrs = [float(file[file.find('plr=') + 4:file.find('-ep')]) for file in glob.glob(f'{ckpt_dir}/PT/*.pt')]
else:
    lrs = [float(file[file.find('plr=') + 4:file.find('-flr')]) for file in glob.glob(f'{ckpt_dir}/FT/*.pt')]
lrs = sorted(set(lrs))
assert lrs is not None, 'No PLRs found'
print('Found PLRs:', lrs)


ckpts_by_lr = {
    lr:
        [f'{ckpt_dir}/PT/{dataset}-{model_name}-plr={lr}-ep={ep}.pt' for ep in swa_epochs] if flr is None else
        [f'{ckpt_dir}/FT/{dataset}-{model_name}-plr={lr}-flr={flr}-ep={ep}.pt' for ep in swa_epochs]
    for lr in lrs
}
if flr is None:
    if len(swa_epochs) > 1:
        save_path = f'data-npy/{model_name}-{dataset}-swa={num_swa}.npy'
    else:
        save_path = f'data-npy/{model_name}-{dataset}-plr.npy'
else:
    save_path = f'data-npy/{model_name}-{dataset}-flr={flr}-prefix.npy'
os.makedirs('data-npy', exist_ok=True)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')


model = ViT(
    image_size=32,
    patch_size=4,
    num_classes=10 if dataset == 'cifar10' else 100,
    dim=512,
    depth=6,
    heads=8,
    mlp_dim=512,
    dropout=0.1,
    emb_dropout=0.1
).to(device)


def load_model(path):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt["model"])


def load_swa_model(paths):
    ckpts = [torch.load(path) for path in paths]
    keys = list(ckpts[0]['model'].keys())
    for key in keys:
        ckpts[0]['model'][key] = torch.stack([
            ckpt['model'][key] for ckpt in ckpts
        ], dim=0).mean(dim=0)
    model.load_state_dict(ckpts[0]["model"])


if dataset == 'cifar10':
    test_set = CIFAR10(
        '~/datasets/cifar10/', train=False,
        transform=T.Compose([
            T.Resize(32),
            T.ToTensor(),
        ]),
        download=True
    )
else:
    test_set = CIFAR100(
        '~/datasets/cifar100/', train=False,
        transform=T.Compose([
            T.Resize(32),
            T.ToTensor(),
        ]),
        download=True
    )


normalize = T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
test_loader = DataLoader(test_set, batch_size=2048, num_workers=8, shuffle=False, pin_memory=True)
slicer = FourierSlicer(32, blocks=[(0, 1), (1, 9), (9, 25), (25, 33)], pres_low_freq=0, mask_mode=False)

test_accs = np.zeros((len(lrs), 1, len(slicer.blocks) + 1))
for i, lr in enumerate(tqdm(lrs)):
    ckpts = ckpts_by_lr[lr]
    load_swa_model(ckpts)
    model.eval()
    k = 0

    all_preds = []
    for images, _ in test_loader:
        images = images.to(device)

        with torch.no_grad():
            preds = model(normalize(images)).argmax(dim=-1).cpu()
        cur_preds = [preds]

        for j, rec_images in enumerate(slicer(images)):
            with torch.no_grad():
                if (j == 0 and not slicer.mask_mode) or (j > 0 and slicer.mask_mode):
                    rec_images = normalize(rec_images)
                preds = model(rec_images).argmax(dim=-1).cpu()
                cur_preds.append(preds)

        all_preds.append(torch.stack(cur_preds, dim=0))

    all_preds = torch.cat(all_preds, dim=1)
    test_accs[i, k] = (
            all_preds == torch.tensor(test_set.targets).reshape(1, -1)
    ).to(torch.float).mean(dim=1).numpy()

np.save(save_path, test_accs)
