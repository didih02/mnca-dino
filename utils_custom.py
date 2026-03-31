import os
import time
import joblib
from matplotlib import pyplot as plt
import torch
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns
from sklearn.neural_network import MLPClassifier

def bool_flag(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
    raise ValueError('Boolean value expected.')

def get_file_size_in_kb(file_path):
    file_size_bytes = os.path.getsize(file_path)  # Get file size in bytes
    file_size_kb = file_size_bytes / 1024  # Convert bytes to kilobytes
    return file_size_kb

def calculate_topk_accuracy(classifier, output, target, topk=(1, 5)):
    # https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        res = []        
        check_class = torch.unique(target).size(0)

        if check_class > 4:
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
        else:
            print(f"Only two classes, not need top1 and top5 accuracy using {classifier}")
            res = torch.tensor([[0],[0]], dtype=torch.float)
        return res

def svm_classify(name_dataset, X_train, y_train, X_test, y_test, main_dir, _size, act_, n_component, config, results_dir, type_reduction="no_reduction", seed=None):
    #update 22 march 2026, adding new column: seed as random_state for SVM classifier
    print("Start SVM Classification")
    # Train SVM model
    clf = svm.SVC(kernel="linear", verbose=False, max_iter=10000000, random_state=seed) #default setting

    clf.fit(X_train, y_train.ravel())

    # Save the model
    with open(os.path.join(main_dir, 'svm_model.sav'), 'wb') as pickle_file:
        pickle.dump(clf, pickle_file)

    # Predict
    y_pred = clf.predict(X_test)
    # Calculate accuracy
    accuracy = (accuracy_score(y_test, y_pred))*100
    precission = (precision_score(y_test, y_pred, average='weighted'))*100
    recall = (recall_score(y_test, y_pred, average='weighted'))*100
    f1 = (f1_score(y_test, y_pred, average='weighted'))*100

    # Calculate top-1 and top-5 accuracy
    y_pred_prob = clf.decision_function(X_test)  # Get decision function scores
    y_test_tensor = torch.tensor(LabelEncoder().fit_transform(y_test))
    y_pred_prob_tensor = torch.tensor(y_pred_prob)

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot heatmap
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Save heatmap as JPG
    heatmap_path = os.path.join(main_dir, 'heatmap_confusion.jpg')
    plt.savefig(heatmap_path, dpi=300)
    plt.close()

    top1_acc, top5_acc = calculate_topk_accuracy("SVM", y_pred_prob_tensor, y_test_tensor)

    # Save classification report
    with open(os.path.join(main_dir, "classification_report_svm.txt"), 'w') as fd:
        fd.write(f'Size: {_size}\n')
        fd.write(f'Accuracy: {accuracy}%\n')
        fd.write(f'Top-1 Accuracy: {top1_acc.item():.2f}%\n')
        fd.write(f'Top-5 Accuracy: {top5_acc.item():.2f}%\n')
        fd.write(f'Precission: {precission}%\n')
        fd.write(f'Recall: {recall}%\n')
        fd.write(f'F1-score: {f1}%\n')
        fd.write(f'Classification report: \n{classification_report(y_test, y_pred)}\n')
        fd.write(f'Parameters: \n{clf.get_params()}\n\n\n')
    
    #combine all results in one csv file
    if act_:
        with open(os.path.join(results_dir, f"report_svm_{type_reduction}_{config}.csv"), 'a') as fd:
            fd.write(f'{name_dataset};{accuracy}%;{top1_acc.item():.2f}%;{top5_acc.item():.2f}%;{precission}%;{recall}%;{f1}%;{n_component};{_size};{time.time()};{seed}\n')   
    else:
        with open(os.path.join('classify', "report_svm.csv"), 'a') as fd:
            fd.write(f'{name_dataset};{accuracy}%;{top1_acc.item():.2f}%;{top5_acc.item():.2f}%;{precission}%;{recall}%;{f1}%;{_size};{time.time()};{seed}\n') 

def mlp_classify(name_dataset, X_train, y_train, X_test, y_test, main_dir, _size, act_, n_component, config, 
                 results_dir, type_reduction="no_reduction", seed=42):
    #update 22 march 2026, adding new column: seed as random_state for SVM classifier
    print("Start MLP Classification")
    # Train SVM model
    clf = MLPClassifier(
        hidden_layer_sizes=(512,256),
        activation='relu',
        solver='adam',
        max_iter=100,
        random_state=seed
    )

    clf.fit(X_train, y_train.ravel())

    # Save the model
    with open(os.path.join(main_dir, 'mlp_model.sav'), 'wb') as pickle_file:
        pickle.dump(clf, pickle_file)

    # Predict
    y_pred = clf.predict(X_test)
    # Calculate accuracy
    accuracy = (accuracy_score(y_test, y_pred))*100
    precission = (precision_score(y_test, y_pred, average='weighted'))*100
    recall = (recall_score(y_test, y_pred, average='weighted'))*100
    f1 = (f1_score(y_test, y_pred, average='weighted'))*100

    # Calculate top-1 and top-5 accuracy
    y_pred_prob = clf.predict_proba(X_test)  # Get decision function scores
    y_test_tensor = torch.tensor(LabelEncoder().fit_transform(y_test))
    y_pred_prob_tensor = torch.tensor(y_pred_prob)

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot heatmap
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Save heatmap as JPG
    heatmap_path = os.path.join(main_dir, 'heatmap_confusion.jpg')
    plt.savefig(heatmap_path, dpi=300)
    plt.close()

    top1_acc, top5_acc = calculate_topk_accuracy("MLP", y_pred_prob_tensor, y_test_tensor)

    # Save classification report
    with open(os.path.join(main_dir, "classification_report_mlp.txt"), 'w') as fd:
        fd.write(f'Size: {_size}\n')
        fd.write(f'Accuracy: {accuracy}%\n')
        fd.write(f'Top-1 Accuracy: {top1_acc.item():.2f}%\n')
        fd.write(f'Top-5 Accuracy: {top5_acc.item():.2f}%\n')
        fd.write(f'Precission: {precission}%\n')
        fd.write(f'Recall: {recall}%\n')
        fd.write(f'F1-score: {f1}%\n')
        fd.write(f'Classification report: \n{classification_report(y_test, y_pred)}\n')
        fd.write(f'Parameters: \n{clf.get_params()}\n\n\n')
    
    #combine all results in one csv file
    if act_:
        with open(os.path.join(results_dir, f"report_mlp_{type_reduction}_{config}.csv"), 'a') as fd:
            fd.write(f'{name_dataset};{accuracy}%;{top1_acc.item():.2f}%;{top5_acc.item():.2f}%;{precission}%;{recall}%;{f1}%;{n_component};{_size};{time.time()};{seed}\n')   
    else:
        with open(os.path.join('classify_mlp', "report_mlp.csv"), 'a') as fd:
            fd.write(f'{name_dataset};{accuracy}%;{top1_acc.item():.2f}%;{top5_acc.item():.2f}%;{precission}%;{recall}%;{f1}%;{_size};{time.time()};{seed}\n') 

def check_extensions_nca(dir):
    # Supported extensions
    extensions = [".sav", ".pkl", ".joblib"]

    # Check if any existing model file is present
    nca_file = None
    for ext in extensions:
        candidate = os.path.join(dir, f"nca_model{ext}")
        if os.path.exists(candidate):
            nca_file = candidate
            break  # stop at the first one found

    if nca_file:
        print(f"Loading NCA model from {nca_file} ...")

        if nca_file.endswith(".pkl") or nca_file.endswith(".joblib"):
            nca = joblib.load(nca_file)   # joblib
        elif nca_file.endswith(".sav"):
            with open(nca_file, "rb") as f:
                nca = pickle.load(f)  # pickle
    else:
        # Default if nothing exists → choose your preferred format for saving
        nca_file = os.path.join(dir, "nca_model.pkl")
        nca = None
        print("No existing NCA model found, will save future model as .pkl (joblib).")
    
    return nca, nca_file

def check_extensions_pca(dir):
    # Supported extensions
    extensions = [".sav", ".pkl", ".joblib"]

    # Check if any existing model file is present
    pca_file = None
    for ext in extensions:
        candidate = os.path.join(dir, f"pca_model{ext}")
        if os.path.exists(candidate):
            pca_file = candidate
            break  # stop at the first one found

    if pca_file:
        print(f"Loading pca model from {pca_file} ...")

        if pca_file.endswith(".pkl") or pca_file.endswith(".joblib"):
            pca = joblib.load(pca_file)   # joblib
        elif pca_file.endswith(".sav"):
            with open(pca_file, "rb") as f:
                pca = pickle.load(f)  # pickle
    else:
        # Default if nothing exists → choose your preferred format for saving
        pca_file = os.path.join(dir, "pca_model.pkl")
        pca = None
        print("No existing pca model found, will save future model as .pkl (joblib).")
    
    return pca, pca_file