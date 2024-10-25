import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import scanpy as sc
import anndata
import leidenalg
from scipy import io
from scipy.sparse import coo_matrix, csr_matrix
import os


#Preprocessing

sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')

#results_file = '/Users/emad/PycharmProjects/SCMelanoma/out/results/results.h5ad'

#Read Data
X = io.mmread('/Users/emad/RProjects/Leukemia/data/AcuteMyeloid/GSE227122/GSE227122_counts.mtx')
adata = anndata.AnnData(X=X.transpose().tocsr())
with open('/Users/emad/RProjects/Leukemia/data/AcuteMyeloid/GSE227122/GSE227122_gene_names.csv', 'r') as f:
    gene_names = f.read().splitlines()
metadata = pd.read_csv('/Users/emad/RProjects/Leukemia/data/AcuteMyeloid/GSE227122/GSE227122_metadata.csv')

adata.obs = metadata
adata.obs.index = adata.obs['barcodes']
adata.var.index = gene_names
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
print('Filtering Done')


print(adata)
print("\n")

adata.var_names_make_unique()

print(adata.var_names)
print("\n")

sc.pl.highest_expr_genes(adata, n_top=20, save='.pdf')

#Filter cells with fewer than 200 genes and filter genes that are found in less than 3 cells
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

print(adata)
print("\n")

#Filtering cells that have too many mitochondrial genes expressed or too many total counts:

    #Visualize mt_genes
adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'

print(adata.var[adata.var.mt == True])
print("\n")


print("\nRunning QC Metrics\n")
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

print(adata.obs)
print("\n")

sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True, save='_before_slicing.pdf')

sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')

    #Slicing to remove outliers
#adata = adata[adata.obs.n_genes_by_counts < 5500, :]
adata = adata[adata.obs.pct_counts_mt < 18, :]
#adata = adata[adata.obs.total_counts < 28500, :]
    #Slicing using quantiles
upper_lim = np.quantile(adata.obs.n_genes_by_counts.values, .97)
lower_lim = np.quantile(adata.obs.n_genes_by_counts.values, .02)
adata=adata[(adata.obs.n_genes_by_counts < upper_lim) & (adata.obs.n_genes_by_counts > lower_lim)]

upper_lim1 = np.quantile(adata.obs.total_counts.values, .95)
lower_lim1 = np.quantile(adata.obs.total_counts.values, .05)
adata=adata[(adata.obs.total_counts < upper_lim1) & (adata.obs.total_counts > lower_lim1)]


    #Visualize mt-genes without outliers
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True, save='_after_slicing.pdf')

sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', save='_totalcounts_mt_genes.pdf')
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', save='_totalcounts_features.pdf')

print(adata)
print(adata.var)
print("\n")

#Normalize Data to 10,000 reads per cell, so that counts become comparable among cells.
print("\nOriginal data")
print(adata.X[1,:].sum())

sc.pp.normalize_total(adata, target_sum=1e4)

print("\nNormalized data")
print(adata.X[1,:].sum())

sc.pp.log1p(adata) #change to log counts

#Identify highly-variable genes
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc.pl.highly_variable_genes(adata, save='_highly_variable.pdf')

#Save Raw Data
adata.raw = adata
#Filter
adata = adata[:, adata.var.highly_variable]

#Use linear regression to remove unwanted variations
sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])

#Scale each gene to unit variance. Remove values with standard deviation > 10.
sc.pp.scale(adata, max_value=10)

#Principal component analysis shows 2 main axes of variation and can be visualized
sc.tl.pca(adata, svd_solver='arpack')

#sc.pl.pca(adata, color='CST3', save='.pdf')
sc.pl.pca_variance_ratio(adata, log=True, save='_variance.pdf')

#adata.write(results_file)



#Leiden/Louvain graph-clustering method and visualization using UMAP scatter plot
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
sc.tl.louvain(adata, resolution=0.1)
sc.tl.paga(adata)
sc.pl.paga(adata, plot=True)
sc.tl.umap(adata, init_pos='paga')
sc.tl.umap(adata)
sc.pl.umap(adata, color=['louvain'], save='.pdf')

#adata.write(results_file)

#Finding marker genes
#Compute a ranking for the highly differential genes in each cluster
    #t-test
#sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
#sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, save=True)

    #Wilcoxon rank-sum test
sc.tl.rank_genes_groups(adata, 'louvain', method='wilcoxon')
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, save=True)

#adata.write(results_file)

    #multi-variate appraoch logistic regression
#sc.tl.rank_genes_groups(adata, 'leiden', method='logreg')
#sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, save=True)


#Table with the scores and groups.
result = adata.uns['rank_genes_groups']
out = np.array([[0,0,0,0,0]])

for group in result['names'].dtype.names:
    out = np.vstack((out, np.vstack((result['names'][group],
                                     result['scores'][group],
                                     result['pvals_adj'][group],
                                     result['logfoldchanges'][group],
                                     np.array([group] * len(result['names'][group])).astype('object'))).T))

markers = pd.DataFrame(out[1:], columns=['names','scores', 'pVals_Adj', 'logFoldChange', 'Cluster'])
markers = markers[(markers.pVals_Adj < 0.005) & (abs(markers.logFoldChange) > 1.5)]

file_name = '/markers.xlsx'

with pd.ExcelWriter(file_name, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
    markers[markers.Cluster == '0'].to_excel(writer, sheet_name='Sheet_1')
    markers[markers.Cluster == '1'].to_excel(writer, sheet_name='Sheet_2')
    markers[markers.Cluster == '2'].to_excel(writer, sheet_name='Sheet_3')
    markers[markers.Cluster == '3'].to_excel(writer, sheet_name='Sheet_4')
    markers[markers.Cluster == '4'].to_excel(writer, sheet_name='Sheet_5')
    markers[markers.Cluster == '5'].to_excel(writer, sheet_name='Sheet_6')
    markers[markers.Cluster == '6'].to_excel(writer, sheet_name='Sheet_7')
    markers[markers.Cluster == '7'].to_excel(writer, sheet_name='Sheet_8')
    markers[markers.Cluster == '8'].to_excel(writer, sheet_name='Sheet_9')
    markers[markers.Cluster == '9'].to_excel(writer, sheet_name='Sheet_10')
    markers[markers.Cluster == '10'].to_excel(writer, sheet_name='Sheet_11')
    markers[markers.Cluster == '11'].to_excel(writer, sheet_name='Sheet_12')
    markers[markers.Cluster == '12'].to_excel(writer, sheet_name='Sheet_13')
  
#Compare single cluster in violin plot for more details
sc.pl.rank_genes_groups_violin(adata, groups='0', n_genes=8, save=True)

#Reload the object with the computed differential expression
#adata = sc.read(results_file)

#compare single gene across clusters
#sc.pl.rank_genes_groups_violin(adata, groups='0', n_genes=8, save=True)

marker_genes = ['IL32', 'HBB', 'CD1A', 'NFKB1', 'HLA-A', 'CD99', 'CDK6', 'GAPDH', 'TRGC2', 'CST3', 'NAP1L1', 'CD82', 'CD74']
sc.pl.violin(adata, marker_genes, groupby='louvain', save='.pdf')
sc.pl.dotplot(adata, marker_genes, groupby='louvain', save='_marker_per_cluster.pdf')

mapping = {
    '0': 'Myeloid',
    '1': 'Myeloid',
    '2': 'Stem-like',
    '3': 'Lymphoid',
    '4': 'Myeloid',
    '5': 'Differentiated-like',
    '6': 'Stem-like',
    '7': 'Lymphoid',
    '8': 'Lymphoid',
    '9': 'Myeloid',
    '10': 'Differentiated-like',
    '11': 'Differentiated-like',
    '12': 'Lymphoid'
}


adata.obs['level_1'] = adata.obs['louvain'].map(mapping)

sc.pl.umap(adata, color='level_1', legend_loc='on data', title='', frameon=False, save='_level_1.pdf')


adata.obs.to_csv('/metadata.csv')
#visualize the marker genes with annotated cell type

#sc.pl.stacked_violin(adata, marker_genes, groupby='leiden', rotation=90)

#adata.write(results_file, compression='gzip')
