library(Seurat)
library(tidyverse)
library(ggplot2)

query_downsample <- readRDS("your file path")

#Check Normalization
sum(query_downsample@assays$RNA@data[,1])
sum(query_downsample@assays$RNA@counts[,1])

#extract raw counts and metadata
raw_counts = query_downsample@assays$RNA@counts
meta_data = query_downsample@meta.data

#Sanity Check
identical(x=rownames(query_downsample@meta.data), y= colnames(raw_counts))

#Create Seurat Object
raw_obj<- CreateSeuratObject(counts = raw_counts, meta.data = meta_data)

#Split Object and Combine into list
obj.list <- SplitObject(raw_obj, split.by = "study_id")

#Normailze and Find Variable Features
obj.list <- lapply(X = obj.list, FUN = function(x){
    x <- NormalizeData(x)
    x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = 2000)
})

#Selecting Integration Features 
features <- SelectIntegrationFeatures(object.list = obj.list)

#Scaling and PCA

obj.list <- lapply(X = obj.list, FUN = function(x) {
    x <- ScaleData(x, features = features, verbose = FALSE)
    x <- RunPCA(x, features = features, verbose = FALSE)
})

#Find Integration Anchors
anchors <- FindIntegrationAnchors(object.list = obj.list, anchor.features = features, reduction = "rpca")

#Integrate Data
integrated <-  IntegrateData(anchorset = anchors, k.weight = 50)
DefaultAssay(integrated) <- "integrated"

# Run the standard workflow for visualization and clustering
integrated <- ScaleData(integrated, verbose = FALSE)
integrated <- RunPCA(integrated, npcs = 30, verbose = FALSE)
integrated <- RunUMAP(integrated, reduction = "pca", dims = 1:30)
integrated <- FindNeighbors(integrated, reduction = "pca", dims = 1:30)
integrated <- FindClusters(integrated, resolution = 0.5)

#Check meta.data names to visualize by
names(integrated@meta.data)

#Dimentional Plot
DimPlot(integrated, reduction = "umap", group.by = "study.id")

#Check reductions slot
integrated@reductions

#extract the reduction embeddings
rpca.embeddings <- Embeddings(integrated, "umap")

#Sanity Check
rpca.embeddings[1:5,1:2]

write.csv(integrated@reductions$rpca@cell.embeddings, file = 'rPCA.csv', quote = F, row.names = F)
