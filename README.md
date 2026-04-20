# mnca-dino 

Thanks to [TALENT Library](https://github.com/LAMDA-Tabular/TALENT/tree/main), which provide the library for MNCA.

This research will be presented at PP-RAI 2026 (the 7th Polish Conference on Artificial Intelligence). For more information, please contact me: didihrizki@agh.edu.pl. The full code and instructions will be finished when the paper has been presented.

How to use this code:

# ModernNCA Classification Pipeline

This project implements a classification pipeline using a custom **Modern Neighborhood Components Analysis (ModernNCA)** model built with PyTorch. It supports optional dimensionality reduction using **PCA** or **NCA** before training.

---

## Features

- **ModernNCA model**
- Dimensionality reduction:
  - PCA (Principal Component Analysis)
  - NCA (Neighborhood Components Analysis)
- Automatic caching of transformed datasets
- Training with memory bank mechanism
- Evaluation metrics:
  - Accuracy
  - Precision (weighted)
  - Recall (weighted)
  - F1-score (weighted)

---

## Dataset Structure

Your dataset directory must contain:

dataset_name/

├── trainfeat.pth

├── trainlabels.pth

├── testfeat.pth

├── testlabels.pth


Optional (auto-generated if not present, dim:dimension):

dataset_name/

├── pca/dim/

├── nca/dim/

---

## Installation

Install required dependencies:

```bash
pip install torch scikit-learn numpy tqdm joblib
```

## Run the script with:
```bash
python modernNCA_classification_new.py \
  --dataset my_data \
  --dim 128 \
  --epoch 100 \
  --reduce NCA
```
for modernNCA_classification.py is focus without PCA/NCA and only using MLP.

Detail of Arguments for MNCA classification:

| Argument        | Type  | Default      | Description                               |
| --------------- | ----- | ------------ | ----------------------------------------- |
| `--dataset`     | str   | caltech256   | Dataset folder                            |
| `--dim`         | int   | 128          | Output embedding dimension                |
| `--dropout`     | float | 0.1          | Dropout rate                              |
| `--d_block`     | int   | 512          | Hidden size of MLP block                  |
| `--n_block`     | int   | 1            | Number of MLP blocks                      |
| `--temp`        | float | 0.5          | Temperature scaling                       |
| `--sample_rate` | float | 0.5          | Memory bank sampling rate                 |
| `--epoch`       | int   | 50           | Training epochs                           |
| `--batch_size`  | int   | 128          | Batch size                                |
| `--lr`          | float | 1e-3         | Learning rate                             |
| `--mode`        | int   | 0            | Model mode                                |
| `--folder_name` | str   | config_0     | Output folder                             |
| `--activation`  | str   | relu         | Activation function                       |
| `--reduce`      | str   | PCA          | Dimensionality reduction (`PCA` or `NCA`) |
