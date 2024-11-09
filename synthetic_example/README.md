# Synthetic feature learning example

## Requirements

The following list of libraries is required:

`torch`, `numpy`, `joblib`, `tqdm`

## Experiments

You can find the description of the **training config**
in the `launch-final.ipynb` notebook.

To run the synthetic example:

1. Run all cells of the `launch-final.ipynb` notebook, this will launch synthetic
example pre-training and fine-tuning for all data generation and training random seeds.
2. Once training is finished, use the notebook `features-barriers.ipynb` to calculate
feature importance, error barriers and swa accuracy.
3. Use the notebook `plots-final.ipynb` to draw figures from the paper.

**Note:** current outputs in the `plots-final.ipynb` are just sample
examples and differ from the ones presented in the paper.