library(Seurat)
library(tidyverse)
library(ggplot2)
install.packages("harmony")
library(harmony)

#Read Data that you want to integrate
Obj1 <- readRDS("data1.rds")
Obj2 <- readRDS("data2.rds")
Obj3 <- readRDS("data3.rds")

#Merge Objects into single Seurat Object
MergedObj <- merge(Obj1, y = c(Obj2, Obj3), add.cell.ids = c("1", "2", "3"), project = "Project Name")

#Check meta.data names to integrate by
names(MergedObj@meta.data)

#Integrate Data
integrated <- MergedObj %>% RunHarmony(group.by.vars = 'study.id', plot.convergence = FALSE)

#Run UMAP on integrated Data
integrated <- RunUMAP(integrated, reduction = "harmony", dims = 1:30)

#Check meta.data names to visualize by
names(integrated@meta.data)

#Dimentional Plot
DimPlot(integrated, reduction = "umap", pt.size = 0.5, group.by = 'study.id')
DimPlot(integrated, reduction = "umap", pt.size = 0.5, group.by = 'Annotations')

#Check reductions slot
integrated@reductions

#extract the reduction embeddings
harmony.embeddings <- Embeddings(integrated, "harmony")

#Sanity Check
harmony.embeddings[1:5,1:5]

#Save Data to files
write.csv(integrated@reductions$harmony@cell.embeddings, file = 'harmony.csv', quote = F, row.names = F)
write.csv(integrated@reductions$umap@cell.embeddings, file = 'umap.csv', quote = F, row.names = F)

integrated_mtx <- GetAssayData(integrated, assay='RNA', slot='counts')
writeMM(integrated_mtx, file=paste0(file='integrated.mtx'))

write.table(data.frame('gene'=rownames(integrated_mtx)),
            file='integrated_gene_names.csv',
            quote=F,row.names=F,col.names=F)

integrated$barcodes = colnames(query_downsample)
write.csv(integrated@meta.data, file='integrated_metadata.csv', 
          quote=F, row.names=F)
