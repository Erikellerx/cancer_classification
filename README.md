
# Cancer Classification

This repository contains code and resources for classifying cancer images into *Benign* or *Malignant* categories using various deep learning architectures. 

## Table of Contents
- [Project Structure](#project-structure)
- [Setup](#setup)
- [How to Run](#how-to-run)
- [Results](#results)

---

## Project Structure

```bash
cancer_classification/
├── checkpoints/
│   ├── convnext.pth
│   ├── efficientnet.pth
│   ├── resnext.pth
│   └── swin.pth
├── data/
│   ├── test/
│   │   ├── Benign/
│   │   └── Malignant/
│   └── train/
│       ├── Benign/
│       └── Malignant/
├── models/
├── .gitignore
├── analysis.ipynb
├── dataset.py
├── exp.py
├── README.md
├── requirements.txt
├── train.py
└── utils.py
```

- **checkpoints/**: Contains the trained model weights.
- **data/**: Dataset directory containing train/test splits of Benign and Malignant images.
- **models/**: Contains model definition scripts (if any).
- **analysis.ipynb**: Jupyter Notebook for inference, analysis, and visualization.
- **train.py**: Main script for training the model.
- **exp.py**: Script to run the entire experiment pipeline (training, evaluation, etc.).
- **dataset.py**: Data loading and preprocessing logic.
- **utils.py**: Utility functions for training, logging, metrics, etc.

---

## Setup

1. **Install required packages**:  
   ```bash
   pip install -r requirements.txt
   ```
2. **Download the dataset** and place your training/testing images under the `data/` directory in the following structure:
   ```
   data/
   ├── train/
   │   ├── Benign/
   │   └── Malignant/
   └── test/
       ├── Benign/
       └── Malignant/
   ```
3. **Pre-trained models** are located under the `checkpoints/` directory. These are necessary to run the analysis notebook (`analysis.ipynb`).

Checkpoints Link: https://drive.google.com/file/d/1QYoqQ0aQ_SNhI1MSYMVRYMHxUqXhWDDQ/view?usp=sharing

---

## How to Run

1. **Train a specific model** (e.g., ConvNeXt):
   ```bash
   python train.py --model convnext
   ```
2. **Rerun the entire experiment**:
   ```bash
   python exp.py
   ```
3. **Inference and analysis** can be found in:
   - `analysis.ipynb`

---

## Results

| Model           | Accuracy | F1 Score | Precision | Recall |
|-----------------|---------:|---------:|----------:|-------:|
| **SVM**         |   80.100 |   0.800  |    0.804  |  0.801 |
| **ResNeXt**     |   96.700 |   0.967  |    0.967  |  0.967 |
| **EfficientNet**|   96.700 |   0.967  |    0.967  |  0.967 |
| **Swin**        |   96.450 |   0.964  |    0.965  |  0.964 |
| **ConvNeXt**    |   **97.050** |   **0.970** |    **0.971**  |  **0.970**|
| **Ensemble**    |   96.900 |   0.968  |    0.969  |  0.969 |

> **Note**: The Ensemble combines ResNeXt, EfficientNet, Swin, and ConvNeXt models.

