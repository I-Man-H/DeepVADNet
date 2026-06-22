# DeepVADNet

> **Deep Learning Framework for Simultaneous Prediction of Quantitative and Qualitative Emotions using Visual and Bio-sensing Data**

[![Paper](https://img.shields.io/badge/Paper-Elsevier%20CVIU%202024-blue?style=flat-square&logo=read-the-docs)](https://doi.org/10.1016/j.cviu.2024.104121)
[![Python](https://img.shields.io/badge/Python-3.8-green?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

---

## Overview

**DeepVADNet** is a novel end-to-end multimodal deep learning framework for affective computing. It fuses **facial expression data** with **physiological signals** (EEG, ECG, GSR, respiration) to simultaneously predict:

- **Quantitative emotions** — continuous valence, arousal, and dominance (VAD) scores
- **Qualitative emotions** — discrete labels (happiness, sadness, fear, etc.)

Unlike prior work that addresses these as separate tasks, DeepVADNet handles both in a **single forward pass**, achieving state-of-the-art results on the DEAP and MAHNOB-HCI benchmarks.

📄 **Full paper:** [Computer Vision and Image Understanding, Elsevier (2024)](https://doi.org/10.1016/j.cviu.2024.104121)

---

## Architecture

The figure below illustrates the end-to-end DeepVADNet pipeline and the dual-task emotion recognition workflow.

![DeepVADNet Architecture-1](https://github.com/user-attachments/assets/9e51ed03-016e-4b9a-8cfb-f716bbb40640)

The model consists of:
- A **visual branch** that extracts facial appearance features using a deep CNN backbone
- A **bio-sensing branch** that processes multi-channel physiological signals
- A **fusion module** that combines both modality streams
- **Dual output heads** for simultaneous regression (VAD scores) and classification (discrete labels)

---

## Results

Classification accuracy (%) and Mean Squared Error (MSE) are shown for each VAD dimension. MSE values appear after the `/` symbol for `This work`.

### DEAP Dataset

| Study | Modalities | Features | Valence | Arousal | Dominance |
|---|---|---|---|---|---|
| Keoltra et al. (2012) | EEG, Peripheral | PSD, Statistic | 62.70% | 62.00% | — |
| Tang et al. (2017) | Bio-sensing | Differential entropy | 83.82% | 83.23% | — |
| Yang et al. (2018) | EEG | DL-based | 90.80% | 91.03% | — |
| Anubhav et al. (2020) | EEG | Band power | 94.69% | 93.13% | — |
| Zhang et al. (2022) | EEG, Peripheral | DL-based | 90.46% | 93.22% | — |
| Li et al. (2022) | EEG | DL-based | 97.41% | 97.25% | 98.35% |
| Gong et al. (2024) | EEG, Peripheral | Transformer-based | 97.97% | 98.02% | — |
| **This work** | **Vision, Bio-sensing** | **Face + Bio DL** | **98.89% / 5e-4** | **99.08% / 8e-4** | **98.82% / 4e-4** |

### MAHNOB-HCI Dataset

| Study | Modalities | Features | Valence | Arousal | Dominance |
|---|---|---|---|---|---|
| Soleymani et al. (2012) | Bio-sensing | PSD, Statistic | 57.00% | 52.40% | — |
| Siddharth et al. (2019) | Vision, Bio-sensing | Face + PSD | 85.49% | 82.93% | — |
| Kaur et al. (2022) | EEG | DL-based | 86.97% | 87.07% | — |
| Yuvaraj et al. (2023) | EEG, Peripheral | Statistical + spectral | 83.98% | 85.58% | — |
| **This work** | **Vision, Bio-sensing** | **Face + Bio DL** | **89.98% / 1.09** | **88.60% / 0.83** | **88.13% / 0.98** |

---

## Datasets

| Dataset | Subjects | Modalities | Labels | Access |
|---|---|---|---|---|
| [DEAP](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/) | 32 | EEG (32-ch), EMG, GSR, respiration, face video | Valence, arousal, dominance, liking (9-point scale) | [Request access](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/) |
| [MAHNOB-HCI](https://mahnob-db.eu/hci-tagging/) | 27 | EEG (32-ch), ECG, GSR, eye gaze, face video | Valence, arousal, dominance + 9 discrete emotions | [Request access](https://mahnob-db.eu/hci-tagging/) |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/I-Man-H/DeepVADNet.git
cd DeepVADNet

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
Python >= 3.8
torch
torchvision
numpy
pandas
Pillow
scipy
```

---

## Data Preprocessing

Run `data_preprocess.py` to prepare both datasets. A `preprocess_demo()` function is provided for the DEAP dataset.

After preprocessing, compress each subject's face and bio-sensing data into `.zip` format. The expected directory structure is:

```
./data/
├── DEAP/
│   ├── faces/
│   │   └── s{subject_id}.zip
│   ├── bio/
│   │   └── s{subject_id}.zip
│   └── labels/
│       └── participant_ratings.csv
└── MAHNOB/
    ├── faces/
    │   └── s{subject_id}.zip
    ├── bio/
    │   └── s{subject_id}.zip
    └── labels/
        └── mahnob_labels.npy
```

---

## Training & Evaluation

Run per-subject experiments using the following command:

```bash
python main.py --modal face_bio --dataset DEAP --task VADClassification --epoch 50 --lr 0.0005 --batch_size 64 --use_gpu True
```

### Arguments

| Argument | Description | Default |
|---|---|---|
| `--modal` | Data modality (`face_bio`, `face`, `bio`) | `face_bio` |
| `--dataset` | Dataset (`DEAP`, `MAHNOB`) | `DEAP` |
| `--task` | Task (`VADClassification`, `VADRegression`) | `VADClassification` |
| `--epoch` | Number of training epochs | `50` |
| `--lr` | Learning rate | `0.0005` |
| `--batch_size` | Batch size | `64` |
| `--face_feature_size` | Face feature embedding size | `16` |
| `--bio_feature_size` | Bio-sensing feature embedding size | `64` |
| `--use_gpu` | Enable GPU training | `False` |
| `--save_model` | Save trained model weights | `True` |
| `--mse_weight` | Weight for MSE loss term | `0.01` |

---

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{hosseini2024deep,
  title={Deep learning model for simultaneous recognition of quantitative and qualitative emotion using visual and bio-sensing data},
  author={Hosseini, Iman and Hossain, Md Zakir and Zhang, Yuhao and Rahman, Shafin},
  journal={Computer Vision and Image Understanding},
  volume={248},
  pages={104121},
  year={2024},
  publisher={Elsevier}
}
```

---

## References

[1] Koelstra, S. et al. (2012). DEAP: A database for emotion analysis using physiological signals. *IEEE Transactions on Affective Computing*, 3(1), 18–31.

[2] Soleymani, M. et al. (2012). A multimodal database for affect recognition and implicit tagging. *IEEE Transactions on Affective Computing*, 3(1), 42–55.

---

## Contact

**Iman Hosseini** — Data Scientist | ML Engineer
[LinkedIn](https://www.linkedin.com/in/i-man-hosseini/) · [Google Scholar](https://scholar.google.com/citations?user=ZBlw7J0AAAAJ) · [ResearchGate](https://www.researchgate.net/profile/Iman-Hosseini-6)
