#!pip install celltypist
!pip install scanpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import scanpy as sc
import anndata
from scipy.sparse import coo_matrix, csr_matrix
import os
from scipy import io
import matplotlib.pyplot as plt


X = io.mmread('matrix.mtx')
adata = anndata.AnnData(X=X.transpose().tocsr())
metadata = pd.read_csv('metadata.csv')
with open('gene_names.csv', 'r') as f:
    gene_names = f.read().splitlines()


adata.obs = metadata
adata.obs.index = adata.obs['barcodes']
adata.var.index = gene_names

sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.scale(adata, max_value=10)


harmony = pd.read_csv('harmony.csv')
harmony.index = adata.obs.index 
adata.obsm['X_harmony'] = harmony.to_numpy()
sc.pp.neighbors(adata, use_rep="X_harmony")

import celltypist
from celltypist import models

models.models_description()

predictions = celltypist.annotate(adata, model = 'Immune_All_High.pkl', majority_voting = True)

predictions.predicted_labels
adata_predicted = predictions.to_adata()


sc.pl.umap(adata_predicted, color = 'level_2')
sc.pl.umap(adata_predicted, color = 'majority_voting')


sc.pl.umap(adata_predicted, color = 'level_3')
sc.pl.umap(adata_predicted, color = 'predicted_labels')
