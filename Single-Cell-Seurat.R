library(Matrix)
library(Seurat)
library(CellChat)
library(patchwork)
library(tidy-verse)
library(BiocManager)
BiocManager::install('multtest')
install.packages('metap')


#Read Data
raw_data <- ReadMtx(mtx = 'matrix.mtx', features = 'features.tsv', cells = 'barcodes.tsv')

raw_data[1:5,1:5]

#Check for nulls
sum(is.na(raw_data))
#If result of previous line > 0
raw_data[is.na(raw_data)] = 0

#Check for duplicates
raw_data <- raw_data[!duplicated(raw_data@Dimnames[1]),]

#drop first column with gene names and make it rownames
genes <- raw_data@Dimnames[1]
raw_data <- raw_data[,-1]
rownames(raw_data) <- genes

#Create Seurat Object
seurob <- CreateSeuratObject(counts = raw_data, project = "Project Name")

#Check for Normalization
sum(seurob@assays$RNA@counts[,1])
sum(seurob@assays$RNA@data[,1])

#Preprocessing

#MT percent Count
seurob[["percent.mt"]] <- PercentageFeatureSet(seurob, pattern = "^MT")
View(seurob@meta.data)

seurob[["percent.rb"]] <- PercentageFeatureSet(seurob, pattern = "^RP[SL]")
View(seurob@meta.data)

#Visualize percent MT and RB
#Violin Plot
VlnPlot(seurob, features = c("nCount_RNA", "percent.mt", "nFeature_RNA"), ncol = 3)

#Scatter Plot
FeatureScatter(seurob, feature1 = "nCount_RNA", feature2 = "percent.mt", pt.size = 1)+ geom_smooth(method = 'lm')
FeatureScatter(seurob, feature1 = "nCount_RNA", feature2 = "nFeature_RNA", pt.size = 1) + geom_smooth(method = 'lm')
FeatureScatter(seurob, feature1 = "nCount_RNA", feature2 = "percent.rb", pt.size = 1) + geom_smooth(method = 'lm')

#Filter to remove outliers
seurob<- subset(seurob, subset = nFeature_RNA < 4000 & nCount_RNA < 9000 & percent.mt < 16)

#Normalize
seurob <- NormalizeData(seurob)

#Find Variable features
seurob <- FindVariableFeatures(seurob, nfeatures = 2000)

#Identify 10 most variable genes
vf <- VariableFeatures(seurob)
top10 <- head(VariableFeatures(seurob),10)

#Plot variable genes with labels
plot1 <- VariableFeaturePlot(seurob)
LabelPoints(plot = plot1, points = top10, repel = TRUE, xnudge = 0, ynudge = 0)

#Run Principle Component Analysis
seurob <- RunPCA(seurob, assay = "RNA", features = VariableFeatures(seurob), npcs = 30)

#Visualize PCs
DimPlot(seurob, reduction = "pca")

DimHeatmap(seurob, dims = 1:30, cells = 30, balanced = TRUE, reduction = "pca")
VizDimLoadings(seurob, dims = 1:30, reduction = "pca") & 
    theme(axis.text=element_text(size=5), axis.title=element_text(size=8,face="bold"))

#Determine Relevant PCs (n.components)
ElbowPlot(seurob)

#UMAP
seurob <- RunUMAP(seurob, dims = 1:30, reduction = "pca")
DimPlot(seurob, reduction = "umap", label = TRUE, pt.size = 0.5)

#Clustering
seurob <- FindNeighbors(seurob, dims = 1:30, k.param = 20)
seurob <- FindClusters(seurob, resolution = 0.25)

#Visualize Clusters
DimPlot(seurob, group.by = "seurat_clusters", label = TRUE, pt.size = 1, repel = TRUE)

#Check Cluster Identities
head(Idents(seurob))

#Get markers for each cluster
markers_cluster <- FindMarkers(seurob, ident.1 = '0', features = VariableFeatures(seurob))

#Check markers
head(markers_cluster)


#Rename Clusters
seurob <- RenameIdents(seurob, '0'= "Naïve T-Cells",
                               '1'= "CD8 T-Cells",
                               '2'= "B-Cells",
                               '3'= "Firbroblasts",
                               '4'= "Melanoma" ,
                               '5'= "Keratinocytes",
                               '6'= "Endothelial Cells",
                               '7'= "Glial-Neuronal")



DimPlot(seurob, reduction = "umap", label = TRUE, pt.size = 0.5)


#Save Plot as pdf file
pdf(file='UMAP.pdf',
   width=10,
    height=10)

DimPlot(seurob, reduction = "umap", group.by = "seurat_clusters", cols = DiscretePalette(17, palette = 'glasbey', shuffle = FALSE))
