import argparse
import os
import torch
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import utils_custom

def main(dataset, act_pca, n_component, svd_solver_args, fp16, seed):

    # Paths to data and results
    name_dataset=dataset
    print(name_dataset)
    file_pth = name_dataset

    x_train = torch.load(os.path.join(file_pth,'trainfeat.pth')).cpu().numpy()
    x_test = torch.load(os.path.join(file_pth,'testfeat.pth')).cpu().numpy()
    y_train = torch.load(os.path.join(file_pth,'trainlabels.pth')).numpy()
    y_test = torch.load(os.path.join(file_pth,'testlabels.pth')).numpy()

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    
    if fp16:    
        print("features use fp16")
        x_train = x_train.astype(np.float16)
        x_test = x_test.astype(np.float16)
    else:
        print("features use fp32")

    if act_pca:
        print(f"PCA Classifier with n_components = {n_component}")

        pca_dir = os.path.join(f"classify_mlp_pca_svd_solver_{svd_solver_args}", f"{name_dataset}/{n_component}")
        if not os.path.exists(pca_dir):
            os.makedirs(pca_dir)

        pca, pca_file = utils_custom.check_extensions_pca(pca_dir)

        if os.path.exists(pca_file):
            print("PCA model already exists. Skipping PCA computation...")
            
            # Load precomputed transformed datasets
            x_train = np.load(os.path.join(pca_dir, 'x_train_pca.npy'))
            x_test = np.load(os.path.join(pca_dir, 'x_test_pca.npy'))
            y_train = np.load(os.path.join(pca_dir, 'y_train_pca.npy'))
            y_test = np.load(os.path.join(pca_dir, 'y_test_pca.npy'))

            _size_pca = utils_custom.get_file_size_in_kb(os.path.join(pca_dir, 'x_train_pca.npy'))+utils_custom.get_file_size_in_kb(os.path.join(pca_dir, 'x_test_pca.npy'))

        else:
            print("Running PCA...")
            pickle.dump(sc, open(os.path.join(pca_dir, 'standard_scaler.sav'), 'wb'))

            # PCA
            if n_component is not None:
                if n_component < 1:
                    pca = PCA(n_component, svd_solver=svd_solver_args)
                else:
                    pca = PCA(min(n_component, x_train.shape[0], x_train.shape[1]), svd_solver=svd_solver_args)
            else:
                pca = PCA(svd_solver=svd_solver_args)

            x_train = pca.fit_transform(x_train)
            x_test = pca.transform(x_test)

            if fp16:    
                # print(".")
                x_train = x_train.astype(np.float16)
                x_test = x_test.astype(np.float16)
            else:
                print(".")

            pickle.dump(pca, open(pca_file, 'wb'))

            with open(os.path.join(pca_dir, 'x_train_pca.npy'),'wb') as npy_file:
                np.save(npy_file, x_train)
            
            with open(os.path.join(pca_dir, 'x_test_pca.npy'),'wb') as npy_file:
                np.save(npy_file, x_test)

            with open(os.path.join(pca_dir, 'y_train_pca.npy'),'wb') as npy_file:
                np.save(npy_file, y_train)
            
            with open(os.path.join(pca_dir, 'y_test_pca.npy'),'wb') as npy_file:
                np.save(npy_file, y_test)

            _size_pca = utils_custom.get_file_size_in_kb(os.path.join(pca_dir, 'x_train_pca.npy'))+utils_custom.get_file_size_in_kb(os.path.join(pca_dir, 'x_test_pca.npy'))
            
        with open(os.path.join(pca_dir, "pca_report.txt"),'w') as fd:
            fd.write(f'Size (kb): {_size_pca}\n')
            fd.write(f'Explained variance: {sum(pca.explained_variance_ratio_)}\n')
            fd.write(f'Number of components: {pca.n_components_}\n')
            fd.write(f'Number of features: {pca.n_features_in_}\n')
            fd.write(f'svd_solver: {svd_solver_args}\n')
            fd.write(f'Parameters: {pca.get_params()}\n')
            fd.write(f'Number of samples: {pca.n_samples_}\n\n\n')
        
        utils_custom.mlp_classify(name_dataset, x_train, y_train, x_test, y_test, pca_dir, _size_pca, act_pca, 
                                  n_component, svd_solver_args, f"classify_mlp_pca_svd_solver_{svd_solver_args}", "pca", seed=seed)

        print("Classify Done")

    else:
        print("Only Classifiers")  
        dir = os.path.join("classify_mlp", f"{name_dataset}")
    
        if not os.path.exists(dir):
            os.makedirs(dir) 
        
        with open(os.path.join(dir, 'x_train.npy'),'wb') as npy_file:
            np.save(npy_file, x_train)
        
        with open(os.path.join(dir, 'x_test.npy'),'wb') as npy_file:
            np.save(npy_file, x_test)

        with open(os.path.join(dir, 'y_train.npy'),'wb') as npy_file:
            np.save(npy_file, y_train)
        
        with open(os.path.join(dir, 'y_test.npy'),'wb') as npy_file:
            np.save(npy_file, y_test)

        
        _size = utils_custom.get_file_size_in_kb(os.path.join(dir, 'x_train.npy'))+utils_custom.get_file_size_in_kb(os.path.join(dir, 'x_test.npy'))
        utils_custom.mlp_classify(name_dataset, x_train, y_train, x_test, y_test, dir, _size, act_pca, n_component, svd_solver_args, None, None, seed=seed)

        print("Classify done")
       

if __name__ == "__main__":
    parser = argparse.ArgumentParser('PCA')
    parser.add_argument("--dataset", default="caltech101", type=str, help="""set your actual name of your dataset and folder which you save
                        your .pth file""")
    parser.add_argument("--act_pca", default=False, type=utils_custom.bool_flag, help="""set True if you want using PCA""")
    parser.add_argument("--n_component", default=20, type=int, help="""using this if you used PCA""")
    parser.add_argument("--svd_solver", default='auto',  help="""Using svd_solver randomized will increase accuracy up to 0.1 but for first run using auto is recomended""")
    parser.add_argument("--float16", default=False, type=utils_custom.bool_flag, help="""help to using floating point 16 on your results, 
                        basic extract features from all models is floating point 32""")
    parser.add_argument("--seed", default=42, type=int, help="""Set your random number""")
    args = parser.parse_args()
    main(args.dataset, args.act_pca, args.n_component, args.svd_solver, args.float16, args.seed)
