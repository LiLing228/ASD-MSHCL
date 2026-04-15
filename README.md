# ABIDE Hypergraph Self-Supervised Learning

PyTorch Geometric training code for self-supervised hypergraph representation learning on ABIDE data.

## Requirements

- numpy
- torch
- torch-geometric
- torch-scatter
- scikit-learn

## Training

```bash
python train.py \
  --cuda 1 \
  --seed 921 \
  --kf_cv \
  --epochs 2000 \
  --path ./data/ABIDEI \
  --name ABIDEI \
  --save_dir ./results/ABIDEI
```