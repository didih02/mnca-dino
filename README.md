# mnca-dino 

Thanks to [TALENT Library](https://github.com/LAMDA-Tabular/TALENT/tree/main) which provide the library for MNCA.

This research will be presented on PP-RAI 2026 (the 7th Polish Conference on Artificial Intelligence). For more information, please contact me: didihrizki@agh.edu.pl. The code will be uploaded when the paper has been presented.

How to use this code:

# ModernNCA Classification Pipeline

This project implements a classification pipeline using a custom **Modern Neighborhood Components Analysis (ModernNCA)** model built with PyTorch. It supports optional dimensionality reduction using **PCA** or **NCA** before training.

---

## 📌 Features

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

## 📂 Dataset Structure

Your dataset directory must contain:

dataset_name/

├── trainfeat.pth

├── trainlabels.pth

├── testfeat.pth

├── testlabels.pth


Optional (auto-generated if not present):

dataset_name/

├── pca/<dim>/

├── nca/<dim>/


---

## ⚙️ Installation

Install required dependencies:

```bash
pip install torch scikit-learn numpy tqdm joblib

Run the script with:
python your_script.py --dataset caltech256

