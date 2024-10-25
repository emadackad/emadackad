import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import seaborn as sns
import torch
from scipy.sparse import coo_matrix, csr_matrix
!pip install umap
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from scipy import io

X = io.mmread('matrix.mtx')
metadata = pd.read_csv('metadata.csv')
with open('/kaggle/input/rawgliods/gene_names.csv', 'r') as f:
    gene_names = f.read().splitlines()

    #Log Transform and Scale
counts = torch.tensor(X.transpose().tocsr(), dtype=torch.float)
features_log = torch.log1p(counts)    
scaler = StandardScaler()
features_log_std = scaler.fit_transform(features_log.detach().numpy())

#Define Label
def label(x):
    if x == 'Lymphoid':
        return 1
    else:
        return 0
y = metadata['Annotations'].map(label)

  #Train Test Split
x_train, x_test, y_train, y_test = train_test_split(features_log_std, y, test_size = 0.2)
#Create Random Forest
rf = RandomForestClassifier(n_jobs = 10, n_estimators=10)
rf.fit(x_train, y_train)

pd.DataFrame(rf.feature_importances_,
                  index = gene_names).sort_values(0, ascending = False)

predictions = rf.predict(x_test)

reducer = umap.UMAP()
umap_0 = reducer.fit_transform(features_log_std)
umap_0 = pd.DataFrame(umap_0, columns=['UMAP1', 'UMAP2'])
umap_0['Label'] = y

umap_1 = umap_0 = reducer.fit_transform(x_test)
umap_1 = pd.DataFrame(umap_1, columns=['UMAP1', 'UMAP2'])
umap_1['Label'] = predictions

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
sns.scatterplot(data=umap_0, x='UMAP1', y='UMAP2', hue='Label', palette='Set1', s=100, ax=axes[0])
axes[0].set_title('Original Labels')
axes[0].set_xlabel('UMAP1')
axes[0].set_ylabel('UMAP2')
axes[0].grid(True)

sns.scatterplot(data=umap_1, x='UMAP1', y='UMAP2', hue='Label', palette='Set1', s=100, ax=axes[1])
axes[1].set_title('Model Predictions')
axes[1].set_xlabel('UMAP1')
axes[1].set_ylabel('UMAP2')
axes[1].grid(True)
