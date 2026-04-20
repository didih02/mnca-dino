import torch
import torch.nn as nn
import torch.nn.functional as F
from TALENT.model.lib.tabr.utils import make_module
from typing import Optional

# latent dimension impact on calculation distance, if latent more higher, the processing can be more noisy
# but if smaller, it can be less noise
# https://pdfs.semanticscholar.org/446a/09cfd212400d7c97032c98c4c9ed3fac4eb2.pdf
# https://medium.com/data-science/understanding-latent-space-in-machine-learning-de5a7c687d8d

#this is an original code for the MLP block and encoder of ModernNCA, you can replace it with any block and encoder you want, such as CNN, RNN, Transformer, etc. The MLP block is a simple feedforward neural network with one hidden layer and dropout, and the encoder is a simple feedforward neural network with two hidden layers and dropout. You can modify the architecture of the MLP block and encoder according to your needs.
class MLP_Block(nn.Module):
    def __init__(self, d_in: int, d: int, dropout: float, activation="relu"):
        super().__init__()
        print(activation)
        act = get_activation(activation)
        self.block = nn.Sequential(
            # nn.BatchNorm1d(d_in),
            nn.Linear(d_in, d),
            # act,

            nn.ReLU(inplace=True), 
            # https://discuss.pytorch.org/t/whats-the-difference-between-nn-relu-and-nn-relu-inplace-true/948
            #inplace in ReLU = True ==> not recomended, can modify an original input, but can decreased memory usage, 
            #it means trade off, sometimes can increased accuracy, not recomended, because original value can be destroyed
            
            nn.Dropout(dropout),
            nn.Linear(d, d_in)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

# this is the code for the encoder of ModernNCA, you can replace it with any encoder you want, such as MLP, CNN, RNN, Transformer, etc.
class mnca_1(nn.Module):
    def __init__(self, d_in: int, dropout: float, d: int, latent_dim=128, activation="relu"):
        super().__init__()
        h1 = 256
        h2 = 128
        act = get_activation(activation)
        self.encoder = nn.Sequential(
            nn.Linear(d_in,h1), 
            act,
            # nn.Dropout(dropout),
            nn.Linear(h1,h2), 
            act,
            # nn.Dropout(dropout),
            nn.Linear(h2,latent_dim)
        )
        self.activation_name = activation
    def forward(self,x):
        z = self.encoder(x)
        return z

def get_activation(name):
    name = name.lower()
    if name=="relu": return nn.ReLU(True)
    elif name=="leakyrelu": return nn.LeakyReLU(0.2, True)
    elif name=="gelu": return nn.GELU()
    elif name=="tanh": return nn.Tanh()
    else: return nn.ReLU(True) #set to ReLU as default

class ModernNCA(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        d_num: int,
        d_out: int,
        dim: int,
        dropout: int,
        d_block: int,
        n_blocks: int,
        num_embeddings: Optional[dict],
        temperature: float = 1.0,
        sample_rate: float = 0.8,
        mode: int,
        activation: str
    ) -> None:

        super().__init__()
        self.d_in = d_in if num_embeddings is None else d_num * num_embeddings['d_embedding'] + d_in - d_num
        self.d_out = d_out
        self.d_num = d_num
        self.dim = dim
        self.dropout = dropout
        self.d_block = d_block
        self.n_blocks = n_blocks
        self.T = temperature
        self.sample_rate = sample_rate
        self.mode = mode
        self.activation = activation

        if self.mode==0:
            print(self.mode)
            #----------interesting----------------------
            #remove when using AE
            if n_blocks > 0:
                self.post_encoder = nn.Sequential(*[
                    MLP_Block(dim, d_block, dropout, activation)
                    for _ in range(n_blocks)
                ], 
                # nn.BatchNorm1d(dim) 
                )
            self.encoder = nn.Linear(self.d_in, dim)
        #-------------------------------------------
        else:
            self.encoder = mnca_1(d_in=self.d_in, latent_dim=self.dim, dropout=self.dropout, d=self.d_block, activation=self.activation)

        self.num_embeddings = (
            None
            if num_embeddings is None
            else make_module(num_embeddings, n_features=d_num)
        )

    def make_layer(self):
        block = MLP_Block(self.dim, self.d_block, self.dropout)
        return block
            
    def forward(self, x, y, candidate_x, candidate_y, is_train):
        if is_train:
            data_size = candidate_x.shape[0]
            retrival_size = int(data_size * self.sample_rate)
            sample_idx = torch.randperm(data_size)[:retrival_size]
            candidate_x = candidate_x[sample_idx]
            candidate_y = candidate_y[sample_idx]
            # print(sample_idx)
        
        if self.num_embeddings is not None and self.d_num > 0:
            x_num, x_cat = x[:, :self.d_num], x[:, self.d_num:]
            candidate_x_num, candidate_x_cat = candidate_x[:, :self.d_num], candidate_x[:, self.d_num:]
            x_num = self.num_embeddings(x_num).flatten(1)
            candidate_x_num = self.num_embeddings(candidate_x_num).flatten(1)
            x = torch.cat([x_num, x_cat], dim=-1)
            candidate_x = torch.cat([candidate_x_num, candidate_x_cat], dim=-1)
        # x = x.double()
        # candidate_x = candidate_x.double()
        x = self.encoder(x)
        candidate_x = self.encoder(candidate_x)

        #------------------------------------------------
        #remove when using AE
        # if self.n_blocks > 0:
        #     x = self.post_encoder(x)
        #     candidate_x = self.post_encoder(candidate_x)
        #------------------------------------------------
        # print(len(x))
        if is_train: 
            assert y is not None
            candidate_x = torch.cat([x, candidate_x])
            candidate_y = torch.cat([y, candidate_y])
        
        if self.d_out > 1:
            candidate_y = F.one_hot(candidate_y, self.d_out).to(x.dtype)
        elif len(candidate_y.shape) == 1:
            candidate_y=candidate_y.unsqueeze(-1)

        # calculate distance
        # default we use euclidean distance, however, cosine distance is also a good choice for classification.
        # Using cosine distance, you need to tune the temperature. You can add "temperature":["loguniform",1e-5,1] in the configs/opt_space/modernNCA.json file.
        distances = torch.cdist(x, candidate_x, p=2)
        # following is the code for cosine distance
        # x = F.normalize(x, p=2, dim=-1)
        # candidate_x = F.normalize(candidate_x, p=2, dim=-1)
        # distances = torch.mm(x, candidate_x.T)
        # distances = -distances
        distances = distances / self.T
        # remove the label of training index
        if is_train:
            distances = distances.fill_diagonal_(torch.inf)     
        distances = F.softmax(-distances, dim=-1)
        logits = torch.mm(distances, candidate_y)
        if self.d_out > 1:
            # if task type is classification, since the logit is already normalized, we calculate the log of the logit
            # and use nll_loss to calculate the loss
            eps = 1e-7
            logits = torch.log(logits + eps)
        return logits.squeeze(-1)
