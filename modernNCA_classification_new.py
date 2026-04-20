import argparse
import os
import joblib
from sklearn.neighbors import NeighborhoodComponentsAnalysis
import torch
import torch.nn.functional as F
from TALENT.model.models.modernNCA_ import ModernNCA
from TALENT.model.methods.modernNCA import make_random_batches
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import numpy as np
from TALENT.model.utils import set_seeds
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
set_seeds(42)

def main_function(dataset, _dim, _dropout, _d_block, _n_blocks, _temp, 
                  _sample_rate, _epoch, _batch_size, _lr, _mode, _folder_name, _activation, _reduce):
    train_x = torch.load(f'{dataset}/trainfeat.pth')
    train_y = torch.load(f'{dataset}/trainlabels.pth')
    test_x = torch.load(f'{dataset}/testfeat.pth')
    test_y = torch.load(f'{dataset}/testlabels.pth')

    # train_x = F.normalize(train_x, dim=-1)
    # test_x  = F.normalize(test_x, dim=-1)

    results_folder = f"{_folder_name}/mnca_{_mode}"
    os.makedirs(results_folder, exist_ok=True)

    if torch.is_tensor(train_x):
         train_x = train_x.cpu().numpy()
         test_x = test_x.cpu().numpy()

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    if _dim<384:
        if _reduce == "PCA":
            # Define filenames
            prefix = f"{dataset}/pca/{_dim}"
            x_train_file = f"{prefix}/x_train_pca.npy"
            x_test_file = f"{prefix}/x_test_pca.npy"
            pca_file = f"{prefix}/pca_model.pkl"

            # Ensure all parent directories exist
            for fpath in [x_train_file, x_test_file, pca_file]:
                os.makedirs(os.path.dirname(fpath), exist_ok=True)

            # Check if files already exist
            if all(os.path.exists(f) for f in [x_train_file, x_test_file, pca_file]):
                print("Loading PCA-transformed data from files...")
                train_x = np.load(x_train_file)
                test_x = np.load(x_test_file)
                pca = joblib.load(pca_file)
            else:            
                print("PCA activated")
                pca = PCA(n_components=_dim, random_state=42)
                train_x = pca.fit_transform(train_x)
                test_x = pca.transform(test_x)
                
                # Save transformed data and PCA model
                np.save(x_train_file, train_x)
                np.save(x_test_file, test_x)
                joblib.dump(pca, pca_file)

        elif _reduce == "NCA":
            # Define filenames
            prefix = f"{dataset}/nca/{_dim}"
            x_train_file = f"{prefix}/x_train_nca.npy"
            x_test_file = f"{prefix}/x_test_nca.npy"
            nca_file = f"{prefix}/nca_model.pkl"

            # Ensure all parent directories exist
            for fpath in [x_train_file, x_test_file, nca_file]:
                os.makedirs(os.path.dirname(fpath), exist_ok=True)

            # Check if files already exist
            if all(os.path.exists(f) for f in [x_train_file, x_test_file, nca_file]):
                print("Loading NCA-transformed data from files...")
                train_x = np.load(x_train_file)
                test_x = np.load(x_test_file)
                nca = joblib.load(nca_file)
            else:
                print("NCA activated")
                nca = NeighborhoodComponentsAnalysis(n_components=_dim, random_state=42, init="auto")
                train_x = nca.fit_transform(train_x, train_y)
                test_x = nca.transform(test_x)
                
                # Save transformed data and NCA model
                np.save(x_train_file, train_x)
                np.save(x_test_file, test_x)
                joblib.dump(nca, nca_file)
        else:
            # Define filenames
            prefix = f"{dataset}/pca/{_dim}"
            x_train_file = f"{prefix}/x_train_pca.npy"
            x_test_file = f"{prefix}/x_test_pca.npy"
            pca_file = f"{prefix}/pca_model.pkl"

            # Ensure all parent directories exist
            for fpath in [x_train_file, x_test_file, pca_file]:
                os.makedirs(os.path.dirname(fpath), exist_ok=True)

            # Check if files already exist
            if all(os.path.exists(f) for f in [x_train_file, x_test_file, pca_file]):
                print("Loading PCA-transformed data from files...")
                train_x = np.load(x_train_file)
                test_x = np.load(x_test_file)
                pca = joblib.load(pca_file)
            else:            
                print("PCA activated")
                pca = PCA(n_components=_dim, random_state=42)
                train_x = pca.fit_transform(train_x)
                test_x = pca.transform(test_x)
                # Save transformed data and PCA model
                np.save(x_train_file, train_x)
                np.save(x_test_file, test_x)
                joblib.dump(pca, pca_file)

    train_x = torch.from_numpy(train_x).float()
    test_x  = torch.from_numpy(test_x).float()

    train_y = torch.from_numpy(train_y).long() if isinstance(train_y, np.ndarray) else train_y.long()
    test_y  = torch.from_numpy(test_y).long() if isinstance(test_y, np.ndarray) else test_y.long()

    D = train_x.shape[1]
    num_classes = int(train_y.max())+1

    model = ModernNCA(
        d_in=D, #feature original
        d_num=0, #not Tabular, must set 0
        d_out=num_classes, #for classification, must set >1, if not = regression model
        dim=_dim, #set as feature, can set as you wish
        num_embeddings=None, #no tabular
        dropout=_dropout, 
        d_block=_d_block,
        n_blocks=_n_blocks,
        temperature=_temp,
        sample_rate=_sample_rate,
        mode=_mode,
        activation=_activation
    ).cuda()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=_lr,
        weight_decay=0.0002
        )

    model.train()

    for epoch in tqdm(range(_epoch)):

        total_loss = 0.0

        for i in (make_random_batches(len(train_x), _batch_size)):

            x = train_x[i].cuda()
            y = train_y[i].cuda()

            logits = model(
                x=x,
                y=y,
                candidate_x=train_x.cuda(), #memory bank, help for training or testing process, common process for method like KNN
                candidate_y=train_y.cuda(),
                is_train=True
            )

            loss = F.nll_loss(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

    model.eval()

    with torch.no_grad():
        logits = model(
            x=test_x.cuda(),
            y=None,
            candidate_x=train_x.cuda(),
            candidate_y=train_y.cuda(),
            is_train=False
        )

    pred = logits.argmax(dim=-1).cpu() #find highest value from computation, and set as prediction
    acc = ((pred == test_y).float().mean())*100

    y_true = test_y.cpu().numpy()
    y_pred = pred.numpy()

    precission = (precision_score(y_true, y_pred, average='weighted'))*100
    recall = (recall_score(y_true, y_pred, average='weighted'))*100
    f1 = (f1_score(y_true, y_pred, average='weighted'))*100

    print(f"Accuracy: {acc:.4f}") #in percentage (%)

    dataset = dataset.replace("/", "_")
    with open(f'{results_folder}/results_{dataset}.csv', 'a') as f:
            f.write(
                # the sequence ==> 
                f"{acc:.4f};{precission};{recall};{f1};{_dim};{_dropout};{_d_block};{_n_blocks};{_temp};{_sample_rate};{_epoch};{_batch_size};{_lr}\n"
            )
    print(f"Results saved to: {results_folder}/results_{dataset}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser('ModernNCA Classification with PCA or NCA')
    parser.add_argument("--dataset", default="diagset_tiny", type=str, help="""set your actual name of your dataset which has been extracted""")
    parser.add_argument("--dim", default=128, type=int, help="Output dimension" )
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout rate inside MLP block")
    parser.add_argument("--d_block", default=512, type=int, help="Hidden dimension each MLP block")
    parser.add_argument("--n_block", default=1, type=int, help="Set how much block for MLP after encoder, if 0 means only using Linear Encoder without MLP block, only works on MNCA default")
    parser.add_argument("--temp", default=0.5, type=float, help="For distance scalling, lower temp means confident neighbour, higher more neighbour influence")
    parser.add_argument("--sample_rate", default=0.5, type=float, help="Percentage of training data used as memory bank per batch, if set 1 means full of memory bank, 0.5 means faster, stable and <0.3 is noisy")
    parser.add_argument("--epoch", default=50, type=int, help="Number of training epoch")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size for query sample of x")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate for optimizer")
    parser.add_argument("--mode", default=0, type=int, help="Mode")
    parser.add_argument("--folder_name", default="config_0", type=str, help="Folder Name")
    parser.add_argument("--activation", default="relu", type=str, help="Activation")
    parser.add_argument("--reduce", default="PCA", type=str, help="Set PCA or NCA")

    args = parser.parse_args()
    main_function(args.dataset, args.dim, args.dropout, args.d_block, args.n_block, args.temp, args.sample_rate, args.epoch,
                  args.batch_size, args.lr, args.mode, args.folder_name, args.activation, args.reduce
                  )
