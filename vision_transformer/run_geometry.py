import os
import glob
import torch
import numpy as np
import torchvision.transforms as T
from copy import deepcopy
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from tqdm.auto import tqdm
from models.vit_small_si import ViT


dataset, model_name = 'cifar10', 'vit_small_si'
ckpt_dir = f'ckpts/{dataset}_ckpt_adam_wd=1e-4'
flr1, flr2 = 1e-5, 1e-4  # set FLRs for calculating geometry metrics
num_swa = 5  # set number of SWA models
alphas = np.linspace(0, 1, 7)  # interpolation coefficients to consider
num_epochs = 500  # number of training epochs
swa_epochs = list(range(500 - num_swa * 2 + 2, 501, 2))


lrs = [float(file[file.find('plr=') + 4:file.find('-flr')]) for file in glob.glob(f'{ckpt_dir}/FT/*.pt')]
lrs = sorted(set(lrs))
assert lrs is not None, 'No PLRs found'
print('Found PLRs:', lrs)


swa_ckpts = {
    lr: [f'{ckpt_dir}/PT/{dataset}-{model_name}-plr={lr}-ep={ep}.pt' for ep in swa_epochs] for lr in lrs
}
flr1_ckpts = {
    lr: f'{ckpt_dir}/FT/{dataset}-{model_name}-plr={lr}-flr={flr1}-ep={num_epochs}.pt' for lr in lrs
}
flr2_ckpts = {
    lr: f'{ckpt_dir}/FT/{dataset}-{model_name}-plr={lr}-flr={flr2}-ep={num_epochs}.pt' for lr in lrs
}
for lr in lrs:
    for path in swa_ckpts[lr]:
        assert os.path.isfile(path), f'File does not exist!: {path}'
    assert os.path.isfile(flr1_ckpts[lr]), f'File does not exist!: {flr1_ckpts[lr]}'
    assert os.path.isfile(flr2_ckpts[lr]), f'File does not exist!: {flr2_ckpts[lr]}'


save_path = f'data-npy/{model_name}-{dataset}-geometry.pt'
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


normalize = T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
transform = T.Compose([T.Resize(32), T.ToTensor(), normalize])
if dataset == 'cifar10':
    test_set = CIFAR10('~/datasets/cifar10/', train=False, transform=transform, download=True)
    train_set = CIFAR10('~/datasets/cifar10/', train=True, transform=transform, download=True)
else:
    test_set = CIFAR100('~/datasets/cifar100/', train=False, transform=transform, download=True)
    train_set = CIFAR100('~/datasets/cifar100/', train=True, transform=transform, download=True)

test_loader = DataLoader(test_set, batch_size=2048, num_workers=4, shuffle=False, pin_memory=True)
train_loader = DataLoader(train_set, batch_size=2048, num_workers=4, shuffle=False, pin_memory=True)


@torch.no_grad()
def load_swa_state(paths):
    ckpts = [torch.load(path, map_location=torch.device('cpu')) for path in paths]
    keys = list(ckpts[0]['model'].keys())
    for key in keys:
        ckpts[0]['model'][key] = torch.stack([
            ckpt['model'][key] for ckpt in ckpts
        ], dim=0).mean(dim=0)

    return ckpts[0]['model']


@torch.inference_mode()
def interpolate_states(state1, state2, alpha):
    state = deepcopy(state1)
    for key in state1.keys():
        state[key] = (1 - alpha) * state1[key] + alpha * state2[key]
    return state


@torch.inference_mode()
def get_error(model, loader):
    errors = 0
    model.eval()
    for images, labels in loader:
        preds = model(images.to(device)).argmax(dim=-1).cpu()
        errors += (preds != labels).sum().item()
    return errors / len(loader.dataset)


@torch.inference_mode()
def get_angle(state1, state2):
    p1 = torch.cat([p.reshape(-1) for p in state1.values()])
    p2 = torch.cat([p.reshape(-1) for p in state2.values()])
    cosine = (p1 @ p2) / (p1.norm() * p2.norm())
    cosine = torch.clip(cosine, -1, 1).item()
    return np.arccos(cosine)


@torch.inference_mode()
def get_barriers(state1, state2):
    train_errors = np.zeros_like(alphas)
    test_errors = np.zeros_like(alphas)

    for i, alpha in enumerate(alphas):
        state = interpolate_states(state1, state2, alpha)
        model.load_state_dict(state)
        model.to(device)
        train_errors[i] = get_error(model, train_loader)
        test_errors[i] = get_error(model, test_loader)

    train_barrier = np.max(train_errors - (1 - alphas) * train_errors[0] - alphas * train_errors[-1])
    train_error = np.max(test_errors - (1 - alphas) * test_errors[0] - alphas * test_errors[-1])
    return train_errors, test_errors, train_barrier, train_error


angles = torch.zeros((3, len(lrs)))
train_errors = torch.zeros((3, len(lrs), len(alphas)))
test_errors = torch.zeros((3, len(lrs), len(alphas)))
train_barriers = torch.zeros((3, len(lrs)))
test_barriers = torch.zeros((3, len(lrs)))

for i, lr in enumerate(tqdm(lrs)):
    swa_state = load_swa_state(swa_ckpts[lr])
    flr1_state = torch.load(flr1_ckpts[lr], map_location=torch.device('cpu'))['model']
    flr2_state = torch.load(flr2_ckpts[lr], map_location=torch.device('cpu'))['model']

    for j, (state1, state2) in enumerate([(swa_state, flr1_state), (swa_state, flr2_state), (flr1_state, flr2_state)]):
        angles[j, i] = get_angle(state1, state2)
        out = get_barriers(state1, state2)
        train_errors[j, i] = torch.from_numpy(out[0])
        test_errors[j, i] = torch.from_numpy(out[1])
        train_barriers[j, i] = out[2]
        test_barriers[j, i] = out[3]

torch.save((angles, train_errors, test_errors, train_barriers, test_barriers), save_path)
