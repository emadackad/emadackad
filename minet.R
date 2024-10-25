library(Seurat)
library(tidyverse)
library(Matrix)
library(readr)
library(stats)
library(reshape2)
library(dplyr)
library(minet)
library(GENIE3)

Glio <- readRDS('/Users/emad/RProjects/Glioblastoma/query_downsample.rds')

#Get Expression Matrix
var_genes <- VariableFeatures(Glio)
counts <- GetAssayData(Glio)[var_genes,]
counts <- as.matrix(counts)
counts <- t(counts)
counts <- as.data.frame(counts)

mi_matrix <- build.mim(counts)

ar_matrix <- aracne(mim = mi_matrix)
clr_matrix <- clr(mim = mi_matrix)
mr_matrix <- mrnet(mim = mi_matrix)


aracne_linkList <- getLinkList(ar_matrix)
clr_linkList <- getLinkList(clr_matrix)
mr_linkList <- getLinkList(mr_matrix)

##Get Normalized Weights
aracne_linkList$NormalizedWeights <- with(linkList, ave(weight, TF, FUN = function(x) x / sum(x)))

library(igraph)

aracne_filtered <- subset(aracne_linkList, weight > 0.4)
clr_filtered <- subset(clr_linkList, weight > 35)
mr_filtered <- subset(mr_linkList, weight > 0.4)

g <- graph.data.frame(clr_filtered, directed = FALSE)


V(g)$color <- ifelse(V(g)$name %in% regulator_genes, "skyblue", "orange")
V(g)$size <- 6

plot(
  g,
  layout = layout_nicely(g),
  vertex.label.cex = 0.8,
  edge.width = E(g)$weight/8,  # Adjust the edge width based on weight
  edge.arrow.size = 0.1,
  main = "Gene Regulatory Network",
  vertex.label.color = "black",
  vertex.frame.color = "white",
  vertex.frame.width = 2,
)

