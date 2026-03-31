import argparse
import os
import torch
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NeighborhoodComponentsAnalysis
import utils_custom

def main(dataset, act_nca, n_component, init_args, fp16, seed):

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

    if act_nca:
        print(f"NCA Classifier with n_components = {n_component}")

        nca_dir = os.path.join(f"classify_mlp_nca_init_{init_args}", f"{name_dataset}/{n_component}")
        if not os.path.exists(nca_dir):
            os.makedirs(nca_dir)

        nca, nca_file = utils_custom.check_extensions_nca(nca_dir)

        if os.path.exists(nca_file):
            print("nca model already exists. Skipping nca computation...")
            
            # Load precomputed transformed datasets
            x_train = np.load(os.path.join(nca_dir, 'x_train_nca.npy'))
            x_test = np.load(os.path.join(nca_dir, 'x_test_nca.npy'))
            y_train = np.load(os.path.join(nca_dir, 'y_train_nca.npy'))
            y_test = np.load(os.path.join(nca_dir, 'y_test_nca.npy'))

            _size_nca = utils_custom.get_file_size_in_kb(os.path.join(nca_dir, 'x_train_nca.npy'))+utils_custom.get_file_size_in_kb(os.path.join(nca_dir, 'x_test_nca.npy'))
            
        else:
            print("Running nca...")
            pickle.dump(sc, open(os.path.join(nca_dir, 'standard_scaler.sav'), 'wb'))

            nca = NeighborhoodComponentsAnalysis(n_components=n_component, random_state=42, init=init_args)
            x_train = nca.fit_transform(x_train, y_train)
            x_test = nca.transform(x_test)

            if fp16:    
                print(".")
                x_train = x_train.astype(np.float16)
                x_test = x_test.astype(np.float16)
            else:
                print(".")

            pickle.dump(nca, open(nca_file, 'wb'))

            with open(os.path.join(nca_dir, 'x_train_nca.npy'),'wb') as npy_file:
                np.save(npy_file, x_train)
            
            with open(os.path.join(nca_dir, 'x_test_nca.npy'),'wb') as npy_file:
                np.save(npy_file, x_test)

            with open(os.path.join(nca_dir, 'y_train_nca.npy'),'wb') as npy_file:
                np.save(npy_file, y_train)
            
            with open(os.path.join(nca_dir, 'y_test_nca.npy'),'wb') as npy_file:
                np.save(npy_file, y_test)

            _size_nca = utils_custom.get_file_size_in_kb(os.path.join(nca_dir, 'x_train_nca.npy'))+utils_custom.get_file_size_in_kb(os.path.join(nca_dir, 'x_test_nca.npy'))
            
        with open(os.path.join(nca_dir, "nca_report.txt"),'w') as fd:
            fd.write(f'Size (kb): {_size_nca}\n')
            fd.write(f'Number of components: {nca.components_}\n')
            fd.write(f'Number of features: {nca.n_features_in_}\n')
            fd.write(f'init: {init_args}\n')
            fd.write(f'Parameters: {nca.get_params()}\n\n\n')
        
        utils_custom.mlp_classify(name_dataset, x_train, y_train, x_test, y_test, nca_dir, _size_nca, act_nca, n_component, init_args, f"classify_mlp_nca_init_{init_args}", "nca", seed=seed)

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
        utils_custom.mlp_classify(name_dataset, x_train, y_train, x_test, y_test, dir, _size, act_nca, n_component, init_args, None, None, seed=seed)

        print("Classify done")
       

if __name__ == "__main__":
    parser = argparse.ArgumentParser('nca')
    parser.add_argument("--dataset", default="caltech101", type=str, help="""set your actual name of your dataset and folder which you save
                        your .pth file""")
    parser.add_argument("--act_nca", default=False, type=utils_custom.bool_flag, help="""set True if you want using nca""")
    parser.add_argument("--n_component", default=20, type=int, help="""using this if you used nca""")
    parser.add_argument("--init_nca", default='auto',  help="""Using Initialization of the linear transformation""")
    parser.add_argument("--float16", default=False, type=utils_custom.bool_flag, help="""help to using floating point 16 on your results, 
                        basic extract features from all model is floating point 32""")
    parser.add_argument("--seed", default=42, type=int, help="""Set your random number""")
    args = parser.parse_args()
    main(args.dataset, args.act_nca, args.n_component, args.init_nca, args.float16, args.seed)