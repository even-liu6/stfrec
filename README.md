# stfrec

stfrec is a PyTorch recommendation model built upon LightCCF. It combines the base model's user–item scores with scores from an item–item structural diffusion graph.

The model retains the base implementation's BPR, embedding regularization, and neighborhood aggregation losses. Structural diffusion is applied during recommendation scoring and is enabled by default for all three datasets below.

## Method

The structural diffusion component uses training interactions only:

1. Construct item–item affinities as `C = R^T D_u^-1 R`, where `R` is the user–item training interaction matrix and `D_u` contains user degrees.
2. Remove self-connections and penalize each target item by `(degree + 1)^(-beta)`.
3. Retain the top `k` neighbors per item and normalize each row to obtain the transition matrix `S`.
4. Multiply each user's normalized interaction history by `S`, then apply per-user min–max normalization.
5. Fuse scores as `base_score + alpha * diffusion_score`, followed by a sigmoid.

Observed training items are excluded during evaluation. The default encoder is matrix factorization (`MF`).

## Environment

The environment versions recorded in the original project are:

| Dependency | Version |
| --- | --- |
| Python | 3.8.18 |
| PyTorch | 2.1.0 |
| CUDA for the GPU setup | 12.1 |
| SciPy | 1.10.1 |
| NumPy | 1.24.3 |

Install PyTorch for your platform and the dependencies above in your Python environment. GPU execution is enabled by default when CUDA is available. Pass `--cuda false` to use the CPU.

## Repository structure

```text
stfrec/
├── main.py                 # Training entry point and logging
├── stfrec.py               # STFRec model, diffusion graph, and score fusion
├── utility/
│   ├── parser.py           # Arguments and dataset-specific defaults
│   ├── data_loader.py      # Interaction loading and adjacency construction
│   ├── losses.py           # Base losses and experimental loss helpers
│   ├── trainer.py          # Training loop and result output
│   ├── tester.py           # Ranking evaluation
│   └── tools.py            # Shared utilities
└── dataset/
    ├── douban-book/
    ├── tmall/
    └── amazon-book/
```

Each dataset directory contains `train.txt` and `test.txt`. Each line contains a user ID followed by one or more interacted item IDs, separated by single spaces:

```text
user_id item_id_1 item_id_2 ...
```

IDs must be nonnegative integers. The data loader loads `pre_Adj.npz` if available and constructs it otherwise. When replacing interaction data, remove the old adjacency cache so it can be rebuilt.

## Dataset-specific parameters

The parser automatically selects the following defaults based on `--dataset`:

| Dataset | `--dataset` | α (`--rw_alpha`) | β (`--rw_beta`) | k (`--rw_topk`) |
| --- | --- | --- | --- | --- |
| Douban-book | `douban-book` | 0.70 | 0.30 | 16 |
| Tmall | `tmall` | 0.40 | 0.30 | 16 |
| Amazon-book | `amazon-book` | 0.40 | 0.70 | 16 |

Here, α controls the diffusion score weight, β controls the target-item popularity penalty, and k is the number of neighbors retained per item in the diffusion graph. It is separate from the evaluation cutoffs in `--top_K`.

Explicit command-line values override the corresponding dataset defaults. Custom dataset names fall back to the Douban-book preset unless overridden. Diffusion weight decay is inactive with the default settings.

## Run

Run commands from the repository root. Selecting a dataset is sufficient to apply its diffusion parameters:

```bash
python main.py --dataset douban-book
python main.py --dataset tmall
python main.py --dataset amazon-book
```

The equivalent commands with explicit diffusion parameters are:

```bash
# Douban-book
python main.py --dataset douban-book --rw_alpha 0.70 --rw_beta 0.30 --rw_topk 16

# Tmall
python main.py --dataset tmall --rw_alpha 0.40 --rw_beta 0.30 --rw_topk 16

# Amazon-book
python main.py --dataset amazon-book --rw_alpha 0.40 --rw_beta 0.70 --rw_topk 16
```

Running `python main.py` uses Douban-book with `(alpha, beta, k) = (0.70, 0.30, 16)`.

To select a GPU, append `--gpu 0`. To use another dataset root, append `--dataset_path /path/to/dataset/`, including the trailing slash. Use `python main.py --help` to list arguments.

### Shared training defaults

| Argument | Default |
| --- | --- |
| `--seed` | `2023` |
| `--encoder` | `MF` |
| `--embedding_size` | `64` |
| `--learn_rate` | `0.001` |
| `--reg_lambda` | `0.0001` |
| `--tau` | `0.28` |
| `--ssl_lambda` | `1.0` |
| `--train_batch_size` | `2048` |
| `--test_batch_size` | `2048` |
| `--train_epoch` | `600` |
| `--early_stop` | `10` |

These values describe the current code defaults. The dataset-specific settings above configure the diffusion component; they do not change the shared training parameters.

### Ablations

Append any of the following switches to a dataset command:

| Switch | Effect |
| --- | --- |
| `--rw_alpha 0` | Disable structural diffusion and use base scores |
| `--rw_user_norm 0` | Construct affinities with `R^T R` |
| `--rw_pop_penalty 0` | Disable the target-item popularity penalty |
| `--rw_score_norm none` | Fuse raw diffusion scores without min–max normalization |

The parser also accepts legacy experimental options such as `--debias_na`, `--pop_bpr`, `--na_loss_type`, and `--neg_sample`. These options do not change the current model's loss or sampling path: training uses uniform negative sampling and the original neighborhood aggregation loss. `--hist_alpha` and `--rw_candidate_k` do not affect the current scoring path.

## Evaluation and outputs

The default evaluation reports Recall and NDCG at `[10, 20, 30, 40, 50]`. The training loop tracks best results and early stopping using the second cutoff, which is 20 by default.

- Logs: `log/stfrec/<dataset>/<timestamp>.log`
- Result summaries: `results/stfrec_results_<timestamp>.txt`

Logs include the resolved arguments, including the dataset-specific diffusion settings. Result summaries record the best Recall and NDCG epochs and their paired metrics. The current training loop does not save model checkpoints.

## Acknowledgments and base-method citation

This project builds upon LightCCF and retains its neighborhood aggregation objective. We acknowledge the original authors and their implementation.

The following citation refers to the underlying LightCCF work:

> Yu Zhang, Yiwen Zhang, Yi Zhang, Lei Sang, and Yun Yang. “Unveiling Contrastive Learning‘ Capability of Neighborhood Aggregation for Collaborative Filtering.” SIGIR 2025. [Paper](https://arxiv.org/pdf/2504.10113).

```bibtex
@inproceedings{Yu_Unveiling_2025,
  title = {Unveiling Contrastive Learning‘ Capability of Neighborhood Aggregation for Collaborative Filtering},
  author = {Zhang, Yu and Zhang, Yiwen and Zhang, Yi and Sang, Lei and Yang, Yun},
  booktitle = {Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  doi = {10.1145/3726302.3730111},
  pages = {1985--1994},
  numpages = {10},
  year = {2025}
}
```
