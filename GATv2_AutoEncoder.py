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
!pip install umap
import umap


    #Read Data
X = io.mmread('matrix.mtx')
connectivities = sparse.load_npz('connectivities.npz')
with open('gene_names.csv', 'r') as f:
    gene_names = f.read().splitlines()
metadata = pd.read_csv('metadata.csv')

#Encode Labels
le = LabelEncoder().fit(metadata['Annotations'])
labels = le.transform(metadata['Annotations'])


G = nx.from_scipy_sparse_array(connectivities)
print("Graph is weighted: ", nx.is_weighted(G))
print("Graph is directed: ", nx.is_directed(G))
print("number of nodes: ", int(G.number_of_nodes()))
print("number of edges: ", int(G.number_of_edges()))

  #Data Prep
counts = torch.tensor(X.transpose().tocsr(), dtype=torch.float)
features_log = torch.log1p(counts)
    #Min-Max Norm
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

#Define Trainig
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

#Define Testing
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

#Embeddings
with torch.inference_mode():
    x1, x2, latent = model.run_encoder(data, return_embeddings = True)
    x4, x5, reconstructed = model.run_decoder(latent, data.edge_index, data.edge_weight, return_embeddings = True)


reducer = umap.UMAP()
  #Put each Embeddings in a Dataframe
original_space = reducer.fit_transform(features_log_std)
original_space = pd.DataFrame(original_space, columns=['UMAP1', 'UMAP2'])
embedding_1 = reducer.fit_transform(x1)
embedding_1 = pd.DataFrame(embedding_1, columns=['UMAP1', 'UMAP2'])
embedding_2 = reducer.fit_transform(x2)
embedding_2 = pd.DataFrame(embedding_2, columns=['UMAP1', 'UMAP2'])
latent_embeddeing = reducer.fit_transform(latent)
latent_embeddeing = pd.DataFrame(latent_embeddeing, columns=['UMAP1', 'UMAP2'])
embedding_4 = reducer.fit_transform(x4)
embedding_4 = pd.DataFrame(embedding_4, columns=['UMAP1', 'UMAP2'])
embedding_5 = reducer.fit_transform(x5)
embedding_5 = pd.DataFrame(embedding_5, columns=['UMAP1', 'UMAP2'])
reconstructed_embeddeing = reducer.fit_transform(x5)
reconstructed_embeddeing = pd.DataFrame(reconstructed_embeddeing, columns=['UMAP1', 'UMAP2'])

original_space['Label'], embedding_1['Label'], embedding_2['Label'], latent_embeddeing['Label'], embedding_4['Label'], embedding_5['Label'], reconstructed_embeddeing['Label'] = le.inverse_transform(labels)

plt.figure(figsize=(10, 7))
sns.scatterplot(data=latent_embeddeing, x='UMAP1', y='UMAP2', hue='Label', palette='Set1', s=100)
plt.title('Latent Embeddings')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.legend(title='Annotations')
plt.show()

#Plot Encoding
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sns.scatterplot(data=original_space, x='UMAP1', y='UMAP2', hue='Label', palette='Set1', s=100, ax=axes[0])
axes[0].set_title('Original')
axes[0].set_xlabel('UMAP1')
axes[0].set_ylabel('UMAP2')
axes[0].grid(True)

sns.scatterplot(data=embedding_1, x='UMAP1', y='UMAP2', hue='Label', palette='Set1', s=100, ax=axes[1])
axes[1].set_title('Encoder Embeddings 1')
axes[1].set_xlabel('UMAP1')
axes[1].set_ylabel('UMAP2')
axes[1].grid(True)

sns.scatterplot(data=embedding_2, x='UMAP1', y='UMAP2', hue='Label', palette='Set1', s=100, ax=axes[2])
axes[2].set_title('Encoder Embeddings 2')
axes[2].set_xlabel('UMAP1')
axes[2].set_ylabel('UMAP2')
axes[2].grid(True)
plt.legend(title='Annotations')
plt.show()

#Plot Decoding
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sns.scatterplot(data=embedding_4, x='UMAP1', y='UMAP2', hue='Label', palette='Set1', s=100, ax=axes[0])
axes[0].set_title('Decoder Embeddings 1')
axes[0].set_xlabel('UMAP1')
axes[0].set_ylabel('UMAP2')
axes[0].grid(True)

sns.scatterplot(data=embedding_5, x='UMAP1', y='UMAP2', hue='Label', palette='Set1', s=100, ax=axes[1])
axes[1].set_title('Decoder Embeddings 2')
axes[1].set_xlabel('UMAP1')
axes[1].set_ylabel('UMAP2')
axes[1].grid(True)

sns.scatterplot(data=reconstructed_embeddeing, x='UMAP1', y='UMAP2', hue='Label', palette='Set1', s=100, ax=axes[2])
axes[2].set_title('Reconstructed Embeddings')
axes[2].set_xlabel('UMAP1')
axes[2].set_ylabel('UMAP2')
axes[2].grid(True)
plt.legend(title='Annotations')
plt.show()
