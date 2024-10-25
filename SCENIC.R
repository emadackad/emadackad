BiocManager::install("RcisTarget")
devtools::install_github("aertslab/SCENIC")
library(RcisTarget)
library(SCENIC)
library(Seurat)
library(tidyverse)
library(Matrix)
library(readr)
library(stats)
library(reshape2)
library(dplyr)
library(GENIE3)
library(doMC)
library(R2HTML)


#Load Data
Glio <- readRDS('query_downsample.rds')

DimPlot(Glio, reduction = 'umap', label = TRUE)

Glio <- RunTSNE(Glio, reduction = 'pca')

DimPlot(Glio, reduction = 'tsne', label = TRUE)

#Get Expression Matrix
var_genes <- VariableFeatures(Glio)
counts <- GetAssayData(Glio)[var_genes,]

counts <- as.matrix(counts)


#Initialize
org <- "hgnc"
dbDir <- '/DB/'
data(defaultDbNames)
dbs <- defaultDbNames[[org]]

motifAnnotations_hgnc <- motifAnnotations

scenicOptions <- initializeScenic(org=org, 
                                 dbDir = dbDir, 
                                 dbs = dbs)

scenicOptions@inputDatasetInfo$cellInfo <- cellInfo
scenicOptions@inputDatasetInfo$colVars <- colVars


#Filter Genes
genesKept <- geneFiltering(counts, scenicOptions=scenicOptions,
                           minCountsPerGene=3*.01*ncol(counts),
                           minSamples=ncol(counts)*.01)

# filter the expression matrix to contain only those genes. 
# Matrix ready for co-expression analysis.

### 2. Correlation
counts_filtered <- counts[genesKept,]
runCorrelation(counts_filtered, scenicOptions)

# Run GENIE3
runGenie3(counts_filtered, scenicOptions, nTrees=10)

weightMat <- readRDS('/int/1.3_GENIE3_weightMatrix_part_10.Rds')
linkList <- getLinkList(weightMat, reportMax=120)


linkList$NormalizedWeights <- with(linkList, ave(weight, TF, FUN = function(x) x / sum(x)))


RidgePlot(Glio, features = head(linkList$TF, 20), ncol = 2)
VlnPlot(Glio, features = unique(head(linkList$TF, 30)))
DotPlot(Glio, features = unique(head(linkList$TF, 20))) + RotatedAxis()

library(igraph)

linkList <- subset(linkList, weight > 0.25)
g <- graph.data.frame(linkList, directed = TRUE)

regulator_genes <- linkList$TF
V(g)$color <- ifelse(V(g)$name %in% regulator_genes, "skyblue", "orange")
V(g)$size <- 15

plot(
  g,
  layout = layout.auto(g),
  vertex.label.cex = 0.7,
  edge.width = E(g)$weights*10,  # Adjust the edge width based on weight
  main = "Gene Regulatory Network",
  vertex.label.color = "black",
  vertex.frame.color = "white",
  vertex.frame.width = 2,
)




### Build the gene regulatory network: 
# 1. Get co-expression modules
scenicOptions <- runSCENIC_1_coexNetwork2modules(scenicOptions)

# 2. Get GRN (with RcisTarget)
scenicOptions <- runSCENIC_2_createRegulons(scenicOptions,
                                            coexMethod=c("top5perTarget"))

# 3. Score GRN in the cells (with AUCell) 
scenicOptions <- runSCENIC_3_scoreCells(scenicOptions, counts)

library(shiny)
library(rbokeh)
# 4.1 Look at results in shinyAPP
aucellApp <- plotTsne_AUCellApp(scenicOptions, counts)
savedSelections <- shiny::runApp(aucellApp)


features1 <- c("SOX6", "SOX9", "SOX4", "EBF1")
features2 <- c("TCF7L1", "EPAS1", "GATA3", "KLF4")

FeaturePlot(Glio, features = features1)
FeaturePlot(Glio, features = features2)

RidgePlot(Glio, features = features2, ncol = 2)
VlnPlot(Glio, features = features1, ncol = 2)
VlnPlot(Glio, features = features2, ncol = 2)

DotPlot(Glio, features = features1) + RotatedAxis()


# 4.2 Binarize the network activity (regulon on/off)
scenicOptions <- runSCENIC_4_aucell_binarize(scenicOptions, exprMat = counts)


fileNames <- tsneAUC(scenicOptions, counts, aucType="AUC", nPcs=5, perpl=5)

plotTsne_compareSettings("/Users/emad/int/tSNE_AUC_50pcs_30perpl.Rds", scenicOptions,
                         showLegend=TRUE, cex=.5)



