import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from scipy.sparse import coo_matrix, csr_matrix
from scipy import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import torch
!pip install torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv
import torch_geometric.transforms as T
import torch.nn.functional as F
!pip install scanpy
import scanpy as sc
import anndata

    #Read Data
X = io.mmread('matrix.mtx')
#gene names
with open('gene_names.csv', 'r') as f:
    gene_names = f.read().splitlines()

#metadata
metadata = pd.read_csv('metadata.csv')

#Integrated Embeddings
harmony = pd.read_csv('harmony.csv')
harmony.index = adata.obs.index
adata.obsm['X_harmony'] = harmony.to_numpy()

adata = anndata.AnnData(X=X.transpose().tocsr())
adata.obs = metadata
adata.obs.index = adata.obs['barcodes']
adata.var.index = gene_names
adata.var_names_make_unique()
adata.obs_names_make_unique()
    #QC Metrics
adata.var['mt'] = adata.var_names.str.startswith('MT')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
#Visalize coding genes, total genes and mt genes
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True, save='.pdf')
sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', save='mt_pct.pdf')
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', save='coding_genes.pdf')
    #Slicing using quantiles
#coding genes
upper_lim = np.quantile(adata.obs.n_genes_by_counts.values, .98)
lower_lim = np.quantile(adata.obs.n_genes_by_counts.values, .02)
adata=adata[(adata.obs.n_genes_by_counts < upper_lim) & (adata.obs.n_genes_by_counts > lower_lim)]
#total genes
upper_lim = np.quantile(adata.obs.total_counts.values, .98)
lower_lim = np.quantile(adata.obs.total_counts.values, .02)
adata=adata[(adata.obs.total_counts < upper_lim1) & (adata.obs.total_counts > lower_lim1)]
#mt genes
lower_lim = np.quantile(adata.obs.pct_counts_mt.values, .01)
adata=adata[(adata.obs.pct_counts_mt < 20) & (adata.obs.pct_counts_mt > lower_lim2)]

    #Normalizea and Scale
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
print('Normalized and log transformed')
sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
print('Regressed out')
sc.pp.scale(adata, max_value=10)
print('Scaled')

#Connectivities
sc.pp.neighbors(adata, n_neighbors = 15, use_rep = 'X_harmony')
connectivities = adata.obsp['connectivities']
sparse.save_npz("connectivities.npz", connectivities)
#connectivities = sparse.load_npz('connectivities.npz')

G = nx.from_scipy_sparse_array(connectivities)
print("Graph is weighted: ", nx.is_weighted(G))
print("Graph is directed: ", nx.is_directed(G))
print("number of nodes: ", int(G.number_of_nodes()))
print("number of edges: ", int(G.number_of_edges()))

  #Data Prep
counts = torch.tensor(adata.raw.X.transpose(), dtype=torch.float)
features_log = torch.log1p(counts)
    #Min-Max Scaling
#min_max_scaler = MinMaxScaler()
#features_log_norm = min_max_scaler.fit_transform(features_log.detach().numpy())
    #Standard Scaling
std_scaler = StandardScaler()
features_log_std = std_scaler.fit_transform(features_log.detach().numpy())

edge_list = torch.tensor(np.array(list(G.edges), dtype=np.int64).transpose(), dtype=torch.int64)
edge_weight = torch.tensor(np.array(list(dict(nx.get_edge_attributes(G, "weight")).values())), dtype=torch.float)
node_features = torch.tensor(features_log_std, dtype=torch.float)
y = torch.tensor(features_log_std, dtype=torch.float)

print(edge_list.size())
print(edge_weight.size())
print(node_features.size())
print(y.size())

data = Data(x=features_log_std, edge_index=edge_list, edge_attr=edge_weight, y=features_log_std)

transform = T.RandomNodeSplit(split = 'train_rest', num_val=5000, num_test=5000)
data = transform(data)

      ################## Model Architecture ##################

class AutoEncoder(torch.nn.Module):
    def __init__(self, data, n_dim = 128):
        input_size = data.num_node_features
        super().__init__()
        
        # encoder (input -> latent)
        self.conv1 = GATv2Conv(input_size, (n_dim*8))
        self.conv2 = GATv2Conv((n_dim*8), (n_dim*2))
        self.conv3 = GATv2Conv((n_dim*2), n_dim)
      
      # decoder (latent -> reconstructed)
        self.conv4 = GATv2Conv(n_dim, (n_dim*2))
        self.conv5 = GATv2Conv((n_dim*2), (n_dim*8))
        self.conv6 = GATv2Conv((n_dim*8), input_size)


    def forward(self, data):
        latent, _,_,_  = self.run_encoder(data)
        reconstructed, _,_,_ = self.run_decoder(latent, data.edge_index, data.edge_weight)
        return reconstructed
    
    def run_encoder(self, data, return_embeddings = False):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        
        x1, (att_edge_index1, att_coeff1) = self.conv1(x, edge_index, edge_weight, return_attention_weights=True)
        x2, (att_edge_index2, att_coeff2) = self.conv2(x1, edge_index, edge_weight, return_attention_weights=True)
        latent, (att_edge_index3, att_coeff3) = self.conv3(x2, edge_index, edge_weight, return_attention_weights=True)
      
        if return_embeddings == False:
          return latent, (att_edge_index1, att_coeff1), (att_edge_index2, att_coeff2), (att_edge_index3, att_coeff3)
        else:
            return x1, x2, latent


    def run_decoder(self, latent, edge_index, edge_weight, return_embeddings = False):
        x4, (att_edge_index4, att_coeff4) = self.conv4(latent, edge_index, edge_weight, return_attention_weights=True)
        x5, (att_edge_index5, att_coeff5) = self.conv5(x4, edge_index, edge_weight, return_attention_weights=True)        
        reconstructed, (att_edge_index6, att_coeff6) = self.conv6(x5, edge_index, edge_weight, return_attention_weights=True)
      
        if return_embeddings == False:
          return reconstructed, (att_edge_index4, att_coeff4), (att_edge_index5, att_coeff5), (att_edge_index6, att_coeff6)
        else:
            return x4, x5, reconstructed

                              ######################################################################

    #Initialize model
model = AutoEncoder(data)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



kld = torch.nn.KLDivLoss(reduction = 'batchmean', log_target = False)
mse = torch.nn.MSELoss()
cel = torch.nn.CrossEntropyLoss()

#Trainig
def train(data):
    optimizer.zero_grad()
    model.train()
    reconstructed = model(data)
    reconstructed = F.log_softmax(reconstructed, dim = 1)
    features_soft = F.log_softmax(data.y, dim = 1)
    
    # Compute loss for training nodes only
    mask = data.train_mask
    loss = kld(reconstructed[mask], features_soft[mask])
    loss.backward()
    optimizer.step()

#Testing
def test(data):
    with torch.inference_mode():
        reconstructed = model(data)
        reconstructed = F.log_softmax(reconstructed, dim = 1)
        features_soft = F.log_softmax(data.y, dim = 1)
        
        losses = []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            loss = kld(reconstructed[mask], features_soft[mask])
            losses.append(loss)
        return losses

#Train for 1000 Epoch
epoch_count = []
train_values = []
validation_values = []
test_values = []

for epoch in range(1000):
    train(data)
    train_loss, validation_loss, test_loss = test(data)
    
    log = 'Epoch: {:03d},\t Train Loss: {:.3f},\t Validation Loss: {:.3f},\t Test Loss: {:.3f}'
    print(log.format(epoch+1, train_loss, validation_loss, test_loss))
    
    epoch_count.append(epoch)
    train_values.append(train_loss)
    validation_values.append(validation_loss)
    test_values.append(test_loss)
    

    #Plot Losses
plt.plot(epoch_count, np.array(train_values), label="Train Loss")
plt.plot(epoch_count, np.array(validation_values), label="Validation Loss")
plt.plot(epoch_count, np.array(test_values), label="Test Loss")

plt.title("Train Validation and Test KL Div Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

#Attention Weights
with torch.inference_mode():
    latent, (att_index1, w1), (att_index2, w2), (att_index3, w3) = model.run_encoder(data)
    out, (att_index4, w4), (att_index5, w5), (att_index6, w6) = model.run_decoder(latent, data.edge_index, data.edge_weight)

#Put att index and coeff in a Dataframe
attention_weights_1 = []
for i in range(len(w1)):
    source_node = att_index1[0, i].item()
    target_node = att_index1[1, i].item()
    coefficient = w1[i].item()
    source_node_name = gene_names[source_node]
    target_node_name = gene_names[target_node]
    attention_weights.append([source_node_name, target_node_name, coefficient])
# Save to CSV
df = pd.DataFrame(attention_weights, columns=['Source Node', 'Target Node', 'Attention Coefficient'])
df.to_csv('attention_coefficients.csv', index=False)

# Extract Embeddings
with torch.inference_mode():
    x1, x2, latent = model.run_encoder(data, return_embeddings = True)
    x4, x5, reconstructed = model.run_decoder(latent, data.edge_index, data.edge_weight, return_embeddings = True)

