# MSF-Net: Multi-Scale Fusion Network for File Fragment Classification

Official implementation of the paper **"MSF-Net: Multi-Scale Fusion Network for File Fragment Classification"**.

## Description

MSF-Net is a multi-branch neural architecture for file fragment classification in digital forensics. It extracts and fuses four complementary byte-level feature representations from raw file fragments:

1. **Multi-scale bit-shifted n-gram tensors** (orders 4, 8, 16, 32), processed by a hierarchical branch built from Mobile Inverted Bottleneck Convolution (MBConv) blocks, squeeze-and-excitation (SE) attention, stochastic depth, and patch-based self-attention;
2. **Empirical byte-value distributions** (byte statistics branch);
3. **Shannon entropy characteristics** (entropy branch);
4. **Learned discrete cosine transform (DCT) frequency-domain projections** (DCT branch).

On Scenario #1 of the FiFTy (FFT-75) benchmark (75 file types, 512-byte fragments), MSF-Net achieves a validation accuracy of **73.26%**, outperforming state-of-the-art methods.

## Dataset Information

- **Name**: FiFTy / FFT-75 (File Fragment Type dataset), Scenario #1, 512-byte blocks
- **Source**: IEEE DataPort, DOI: [10.21227/kfxw-8084](https://doi.org/10.21227/kfxw-8084)
- **Scale**: 75 file types, 102,400 fragments per class; official split 80% train / 10% validation / 10% test
- **Format expected by the code**: NumPy `.npz` archives containing arrays `x` (uint8 byte sequences) and `y` (integer labels)

Expected directory layout (relative to `DATASET_PATH` in `config.py`):

```
<DATASET_PATH>/
└── 512_1/
    ├── train.npz
    ├── val.npz
    └── test.npz
```

## Code Information

| File | Content |
|---|---|
| `train.py` | Entry point: training/validation loop, model averaging, checkpointing, confusion-matrix export |
| `net.py` | Model definition (`FileFormatAnalyzer`): ByteAug, SE, MBConv, DropPath, PatchAttn, and the DCT / entropy / byte-statistics branches |
| `utils.py` | `FIFTY_DATASET`: `.npz` loading, bit-shift expansion, multi-scale n-gram tensor construction, DataLoader setup |
| `config.py` | All configuration: dataset/checkpoint paths, scenario, n-gram scales, hyperparameters |
| `TBFE.py` | Auxiliary experimental blocks (TBFE1D / TBFE1DPlus); not used by the final model |

## Requirements

- OS: Linux (validated on Ubuntu 24.04 LTS); a CUDA-capable GPU is recommended
- Python 3 with the following packages: `torch`, `torchvision`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`, `pillow`

Install the dependencies with:

```bash
pip install torch torchvision numpy scikit-learn matplotlib seaborn tqdm pillow
```

## Usage Instructions

1. Download the FFT-75 dataset (Scenario #1, 512-byte blocks) from IEEE DataPort (DOI: [10.21227/kfxw-8084](https://doi.org/10.21227/kfxw-8084)) and arrange the `.npz` files as shown above.
2. Edit `config.py`: set `DATASET_PATH` and `CHECKPOINT_PATH` to your local directories.
3. Start training:

```bash
python train.py
```

Outputs (written to `CHECKPOINT_PATH`):

- `best_model.pth` — checkpoint of the best model (model, averaged model, optimizer, and scheduler states)
- `best_confusion_matrix.png` — confusion matrix of the best model on the validation set

To resume training from a checkpoint, set `CONTINUE_FROM_CHECKPOINT = True` in `config.py`.

## Methodology

1. **Preprocessing**: each 512-byte fragment is expanded into 8 bit-shifted variants and converted into multi-scale n-gram tensors of orders {4, 8, 16, 32}; tensors are normalized to [-1, 1].
2. **Feature extraction**: four n-gram branches (stem convolution + 3 MBConv-SE blocks with stochastic depth + patch self-attention) and three lightweight auxiliary branches (byte statistics, entropy, DCT frequency) produce complementary embeddings.
3. **Fusion and classification**: branch outputs are concatenated (4,032 dimensions) and classified by a two-layer MLP head with dropout.
4. **Training**: cross-entropy loss; AdamW optimizer (weight decay 0.02); cosine annealing with warm restarts (learning rate 1e-4 → 1e-6, T0 = 10, Tmult = 2); batch size 128; up to 50 epochs with early stopping (patience 5); byte-level data augmentation (random masking with keep probability 0.8 plus additive Gaussian noise); a running average of model parameters is maintained for evaluation; random seed 42.

