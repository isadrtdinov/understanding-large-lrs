# Training ViT on CIFAR-10/100

This code is based on the [repo](https://github.com/kentaroy47/vision-transformers-cifar10/tree/main) by Kentaro Yoshioka

## Requirements

The following list of libraries is required:

`torch`, `torchvision`, `wandb`, `einops`

## Run training

To run a single training run:
```bash
python train_cifar10.py
```

Use `python train_cifar10.py --help` to see the training configuration.
The parameters are also listed at the top of `train_cifar10.py` file.

## Run experiments

To run pre-training for all PLRs:
```bash
python launch_pt.py
```

To run pre-training for all combinations of PLRs and FLRs:
```bash
python launch_ft.py
```

## Aggregating results

To calculate accuracy of different frequency components, run:
```bash
python run_fourier_accs.py
```

To calculate geometrical metrics (angular distance and train/test error barriers), run:
```bash
python run_geometry.py
```

**Note:** to calculate metrics for all models sets (i.e., pre-training, fine-tuning and SWA),
you need to set the parameters in the scripts manually. See `run_fourier_accs.py` and `run_geometry.py`
for more details.

## Drawing plots

To draw plots from the paper, use the `plots.ipynb` notebook.
