# SBAM: Salience-Based Adaptive Masking

> Official Implementation of
>
> **"Salience-Based Adaptive Masking: Revisiting Token Dynamics for Enhanced Pre-training"**  
> [ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10063.pdf)  
> Hyesong Choi\*, Hyejin Park\, Kwang Moo Yi, Sungmin Cha, Dongbo Min

---

## 🔍 Overview

**Salience-Based Adaptive Masking (SBAM)** is a simple yet effective method for **Masked Image Modeling (MIM)** that:

- Computes token salience using **outgoing attention**
- Applies **adaptive masking** robust to masking ratio variations
- Introduces **Adaptive Masking Ratio (AMR)** tailored to each sample

---

## SBM Code Snippet (Core Logic)

The following simplified code illustrates the core logic of **Salience-Based Masking (SBM)**:

```python
def saliency_guided_masking(self, x):
    N, L, D = x.shape

    aff = torch.matmul(x, x.permute(0, 2, 1))  # token-to-token attention
    aff = nn.functional.softmax(aff, dim=2)
    aff_sum = torch.sum(aff, dim=1)  # outgoing attention per token

    # Normalize salience scores to [0, 1]
    aff_sum_normalized = (aff_sum - aff_sum.min(dim=1, keepdim=True)[0]) / \
                         (aff_sum.max(dim=1, keepdim=True)[0] - aff_sum.min(dim=1, keepdim=True)[0])
```

This function computes token salience based on outgoing attention and normalizes it for use in salience-based masking.

## 🧠 Key Ideas

- **Token Salience**: Outgoing attention-based measure of token importance
- **Salience-Based Masking (SBM)**: Prioritize masking of salient tokens, with noise injection
- **Adaptive Masking Ratio (AMR)**: Dynamically adjusts the masking ratio per image

---

## 🚀 Performance Highlights

| Method       | Backbone | Fine-tune Top-1 |
|--------------|----------|------------------|
| MAE          | ViT-L    | 84.3%            |
| **+ SBAM**   | ViT-L    | **85.1%**        |
| MAE          | ViT-B    | 82.9%            |
| **+ SBAM**   | ViT-B    | **83.6%**        |

---

## 📦 Installation

This repo is a modification on the [MAE repo](https://github.com/facebookresearch/mae).  
Installation and preparation follow that repo.

---

## 🏃‍♀️ Quick Start

### Pretrain SBAM on ImageNet-1K with 8 GPUs

```bash
torchrun --nproc_per_node=8 main_pretrain.py \
  --data_path /path/to/imagenet \
  --output_dir ./outputs \
  --batch_size 256 \
  --accum_iter 2 \
  --model mae_vit_large_patch16 \
  --norm_pix_loss \
  --mask_ratio 0.75 \
  --epochs 400 \
  --warmup_epochs 40 \
  --blr 1.5e-4 \
  --weight_decay 0.05 \
  --resume ./output_reproduce/checkpoint-100.pth
```

> 📥 Download the warmup checkpoint (`checkpoint-100.pth`) [here](https://drive.google.com/file/d/1dkhpY8EwCtTkS7xw13dBNwzsUl5et3pj/view?usp=drive_link)

---

### Pretrained Model Checkpoints

| Model Version              | Epochs | Download Link                                                                 |
|---------------------------|--------|-------------------------------------------------------------------------------|
| SBM                       | 400    | [Download](https://drive.google.com/file/d/1LGPIMTxEdsA4b-rtQkKNpuS5KfWxnYVM/view?usp=drive_link) |
| SBM                       | 800    | [Download](https://drive.google.com/file/d/1smbobsinIhklcmJ_drog3t-dQRbJ0Tt6/view?usp=drive_link) |
| SBAM (w/ AMR)             | 800    | [Download](https://drive.google.com/file/d/18oYGqBdAPWoqmSCr-MMWps3swfF9geaB/view?usp=drive_link) |

---

### Finetune SBAM on ImageNet-1K with 8 GPUs

```bash
torchrun --nproc_per_node=8 main_finetune.py \
  --data_path /path/to/imagenet \
  --output_dir ./outputs \
  --batch_size 128 \
  --model vit_large_patch16 \
  --epochs 50 \
  --blr 1e-3 \
  --layer_decay 0.75 \
  --weight_decay 0.05 \
  --drop_path 0.2 \
  --reprob 0.25 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --finetune output/checkpoint-400.pth
```

---

## 📜 Citation

```bibtex
@article{choi2024sbam,
  title={Salience-Based Adaptive Masking: Revisiting Token Dynamics for Enhanced Pre-training},
  author={Choi, Hyesong and Park, Hyejin and Yi, Kwang Moo and Cha, Sungmin and Min, Dongbo},
  journal={arXiv preprint arXiv:2404.08327},
  year={2024}
}
```

---

## 📬 Contact

For questions or collaborations:
- Hyesong Choi: hyesongchoi2010@gmail.com
