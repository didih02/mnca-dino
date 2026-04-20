# mnca-dino
### Modern Neighborhood Components Analysis (MNCA) using DINO-ViT Features for Image Classification with Feature Reduction

Thanks to [TALENT Library](https://github.com/LAMDA-Tabular/TALENT/tree/main) which provide the library for MNCA.

This research will be presented on PP-RAI 2026 (the 7th Polish Conference on Artificial Intelligence). For more information, please contact me: didihrizki@agh.edu.pl. The code will be uploaded when the paper has been presented.

## MNCA Classification Pipeline

This project implements a robust classification pipeline using a custom **Modern Neighborhood Components Analysis (ModernNCA)** model built with PyTorch. 

## Key Features
* **Feature Backbone:** Utilizes high-dimensional Transformer Features extracted via **DINO-ViT**.
* **Preprocessing:** Supports optional dimensionality reduction using **PCA** or standard **NCA** before training.
* **Core Model:** A modern PyTorch implementation of Neighborhood Components Analysis designed for seamless integration with deep learning workflows.

## Research Context
In this implementation, we focus exclusively on **DINO-ViT features**. These self-supervised representations provide rich semantic information that is ideal for distance-based classification like ModernNCA.

> **Note:** For implementation details regarding feature extraction, please refer to the official [DINO-ViT GitHub repository](https://github.com/facebookresearch/dino).

## Pipeline Workflow
1.  **Extraction:** Obtain features from a pre-trained DINO-ViT backbone.
2.  **Reduction (Optional):**
    * `PCA`: Standard variance-based dimensionality reduction.
    * `NCA`: Linear transformation to optimize nearest-neighbor classification.
3.  **Classification:** Training the `ModernNCA` model on the processed features.
---

## Features

- **ModernNCA model**
- Dimensionality reduction:
  - PCA (Principal Component Analysis)
  - NCA (Neighborhood Components Analysis)
- Automatic caching of transformed datasets using DINO-ViT Features
- Training with memory bank mechanism
- Evaluation metrics:
  - Accuracy
  - Precision (weighted)
  - Recall (weighted)
  - F1-score (weighted)

---

Your dataset directory must be organized as follows to be recognized by the pipeline:

```text
dataset_name/
├── trainfeat.pth    # DINO-ViT Training features
├── trainlabels.pth  # Training labels
├── testfeat.pth     # DINO-ViT Testing features
└── testlabels.pth   # Testing labels
```

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

### Dimensionality Reduction

If dim < 384, dimensionality reduction is applied.

PCA:
- Uses sklearn.decomposition.PCA
- Cached in: dataset/pca/[dim]/

NCA:
- Uses sklearn.neighbors.NeighborhoodComponentsAnalysis
- Cached in: dataset/nca/[dim]/

If cached files exist, they are loaded instead of recomputed.

### Training Setup
- Optimizer: AdamW
- Loss: Negative Log Likelihood (NLL Loss)
- Uses a memory bank (full or sampled training set)
- Evaluation

### Metrics computed on the test set:

- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)

### Output

Results are saved to:

```bash
<folder_name>/mnca_<mode>/results_<dataset>.csv
```

Each row format:

```bash
accuracy;precision;recall;f1;dim;dropout;d_block;n_blocks;temp;sample_rate;epoch;batch_size;lr
```

### Reproducibility
- Random seeds fixed using:
```bash set_seeds(42)```

- Deterministic CUDA behavior enabled

### Notes:

- Input features are standardized using StandardScaler
- Supports both NumPy arrays and PyTorch tensors
- Model runs on GPU (.cuda() required)

### Requirements:

- CUDA-enabled GPU recommended
- Ensure dataset tensors are compatible with PyTorch

## Classification of MLP and SVM : Running the code

In this project, we use also SVM and MLP classifier with combine PCA/NCA, the file is pca_svm.py, nca_svm.py, mlp_pca.py, and mlp_nca.py.

To run a simple classification using PCA reduction and an SVM classifier:

```bash
python svm_pca.py --dataset your_dataset_name --act_pca True --n_component 64 --float16 False
```
This code also can be used for SVM with NCA (svm_nca.py) and also for MLP with PCA/NCA (mlp_pca.py and mlp_nca.py). But for example, we use svm_pca.py.

### The Processing Pipeline
When you run any of the core scripts (svm_pca.py, svm_nca.py, mlp_pca.py and mlp_nca.py), the following automated workflow is triggered:

1. Data Loading: The script pulls trainfeat.pth and testfeat.pth from your dataset folder and converts them from PyTorch tensors to NumPy arrays.

2. Standardization: A StandardScaler is fitted to the training data and applied to both sets to ensure features have zero mean and unit variance.

3. Smart Caching Check:
   - The script looks for an existing PCA/NCA model for that specific dimension in the classify_... directory.
   - Cache Hit: Skips computation and loads pre-transformed .npy files.
   - Cache Miss: Runs the reduction algorithm (PCA or NCA), saves the model (.sav), and exports the transformed features for future use.

4. Metric Logging: A pca_report.txt is generated, documenting explained variance, file sizes, and SVD solver parameters.

5. Classification: The processed features are passed to the classifier (SVM, MLP, or ModernNCA) for final evaluation.

### Command Line Arguments

All scripts (svm_pca.py, svm_nca.py, mlp_pca.py, mlp_nca.py) utilize a standardized argument parser:

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--dataset` | `caltech256` | Name of the folder containing your `.pth` feature files. |
| `--act_pca` | `False` | Set to `True` to enable dimensionality reduction. |
| `--n_component` | `20` | The target number of dimensions (e.g., 32, 64, 128). |
| `--svd_solver` | `auto` | Solver for PCA. `randomized` is faster for high-dim features. |
| `--float16` | `False` | Casts features to FP16 to save significant disk space/RAM. |
| `--seed` | `None` | Set an integer (e.g., 42) for reproducible results. |

### Expected Outputs & Results
After execution, a result directory is created based on your chosen method (e.g., classify_pca_svd_solver_auto/). Inside, you will find:

```text
dataset_name/
└── [n_component]/
    ├── pca_model.sav                 # The trained Scikit-Learn PCA model
    ├── standard_scaler.sav           # The fitted scaler used for normalization
    ├── x_train_pca.npy               # Transformed training features (NumPy format)
    ├── pca_report.txt                # Metadata (Explained variance, component count, etc.)
    └── classification_report_svm.txt # Final metrics: Accuracy, Precision, Recall, and F1
```

### Evaluation Metrics
The pipeline automatically computes weighted metrics to provide a fair assessment, even if your dataset classes are imbalanced:

- Accuracy: Overall hit rate.
- Weighted Precision/Recall/F1-Score: Metrics adjusted for the number of instances in each class.

For more questions, please get in touch with me: didihrizki@agh.edu.pl. And more detailed results will be updated as soon as possible.
