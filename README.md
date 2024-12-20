# Where Do Large Learning Rates Lead Us?

This repo is the official source code of NeurIPS-2024 paper:

**Where Do Large Learning Rates Lead Us?** \
By [Ildus Sadrtdinov](https://isadrtdinov.github.io/)\*,
[Maxim Kodryan](https://scholar.google.com/citations?user=BGVWciMAAAAJ&hl=en)\*,
[Eduard Pokonechny](https://scholar.google.com/citations?user=lvAKVn4AAAAJ&hl=en)\*,
[Ekaterina Lobacheva](https://tipt0p.github.io/)†,
[Dmitry Vetrov](https://scholar.google.com/citations?user=7HU0UoUAAAAJ&hl=en)†

\* &mdash; equal contribution; † &mdash; shared senior authorship

[arXiv](https://arxiv.org/abs/2410.22113) / [OpenReview](https://openreview.net/forum?id=G5lMFOtFHa) / [Poster & video](https://neurips.cc/virtual/2024/poster/95929) /
[X thread](https://x.com/KateLobacheva/status/1866371463487631859)

## Abstract

<div align="justify">
<img align="right" width=42% src="logo.jpg" />
It is generally accepted that starting neural networks training with large learning rates (LRs) improves generalization.
Following a line of research devoted to understanding this effect, we conduct an empirical study in a controlled setting
focusing on two questions: 1) how large an initial LR is required for obtaining optimal quality, and
2) what are the key differences between models trained with different LRs?
We discover that only a narrow range of initial LRs slightly above the convergence threshold
lead to optimal results after fine-tuning with a small LR or weight averaging.
By studying the local geometry of reached minima, we observe that using LRs from this optimal range
allows for the optimization to locate a basin that only contains high-quality minima.
Additionally, we show that these initial LRs result in a sparse set of learned features,
with a clear focus on those most relevant for the task. In contrast, starting training with too small LRs
leads to unstable minima and attempts to learn all features simultaneously, resulting in poor generalization.
Conversely, using initial LRs that are too large fails to detect a basin with good solutions and extract meaningful patterns from the data.
</div>

## Code

- [Synthetic example](https://github.com/isadrtdinov/understanding-large-lrs/tree/main/synthetic_example)
- [Convolutional networks (ResNet/ConvNet)](https://github.com/isadrtdinov/understanding-large-lrs/tree/main/convolutional_networks)
- [ViT for CIFAR-10/100](https://github.com/isadrtdinov/understanding-large-lrs/tree/main/vision_transformer)

## Citation

If you found this code useful, please cite our paper:

```
@inproceedings{sadrtdinov2024where,
    title={Where Do Large Learning Rates Lead Us?},
    author={Ildus Sadrtdinov and Maxim Kodryan and Eduard Pokonechny and Ekaterina Lobacheva and Dmitry Vetrov},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS)},
    year={2024},
}
```
