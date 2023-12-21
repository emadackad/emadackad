install.packages('devtools')
devtools::install_github('YuLab-SMU/clusterProfiler')

install.packages('BiocManager')
install.packages('msigdbr')
BiocManager::install("clusterProfiler")
BiocManager::install("limma")

library(tidyverse)
library(dplyr)
library(limma)
library(ggplot2)
library(tidyr)
library(msigdbr)
library(clusterProfiler)
library(fgsea)

gene.set = c('TXNIP', 'BLCAP', 'FLJ90757', 'OR', 'PCID2', 'DNAH9', 'P4HB', 'DAZAP1','FLJ90757', 'P4HB',
            'BAIAP2-AS1', 'DQX1', 'DAXX', 'TCF', 'NELF', 'PFKFB2', 'CACNA2D2', 'HK1', 'LGR6', 'TCF3',
            'MUS81', 'DHRS2', 'SLC30A2', 'KIAA1257', 'THBS4', 'PPP1R3E', 'TNRC6C', 'FRMD4B','PBRM1',
            'SPRED2', 'PEF1', 'SERPINF1', 'P4HB', 'NR5A1', 'OCIAD1', 'ARHGEF7', 'HDAC5', 'PFKFB2',
            'LMTK2', 'P4HB', 'POU2F2', 'TFF3', 'S100A7A', 'IDO2', 'LOC100288637', 'PRKAR2A', 'LOC101929524',
            'TXNIP', 'NBPF20', 'NBPF10', 'RPRD1B', 'LOC55908', 'DOCK6', 'KIAA1211L', 'DOCK10', 'HAUS8', 'TPCN1')

length(gene.set)

DB.info <- (msigdbr_collections())

#dplyr::filter(gs_cat == "H") #Can be used to subset DB

#Get Database (Hallmark)
H <- msigdbr(species = "Homo sapiens", category = "H")
H_geneSymbol <- select(H, gs_name, gene_symbol, gs_description)

#Get Database (Curated)
C <- msigdbr(species = "Homo sapiens", category = "C2")
C_geneSymbol <- select(C, gs_name, gene_symbol, gs_description)

#Run Enrichment
enrich_c2 <- enricher(gene = gene.set, TERM2GENE = C_geneSymbol)

enrich_h <- enricher(gene = gene.set, TERM2GENE = H_geneSymbol)

#Extract Results
head(enrich_c2@result)
class(enrich_result@result$GeneRatio)


head(enrich_h@result)
class(enrich_h@result$GeneRatio)


#Format Results

enrich_h_df <- enrich_h@result %>%
  #seperate ratios into 2 columns
  separate(BgRatio, into =c("size.term", "size.category"), sep = "/") %>%
  separate(GeneRatio, into = c("size.overlap.term", "size.overlap.category"), sep = "/") %>%
  #conert. to numeric
  mutate_at(vars("size.term", "size.category", "size.overlap.term", "size.overlap.category"), as.numeric) %>%
  #calculate k/K
  mutate("k.K" = size.overlap.term/size.term)

enrich_C2_df <- enrich_c2@result
  #seperate ratios into 2 columns
  #separate(BgRatio, into =c("size.term", "size.category"), sep = "/") %>%
  #separate(GeneRatio, into = c("size.overlap.term", "size.overlap.category"), sep = "/") %>%
  #conert. to numeric
  #mutate_at(vars("size.term", "size.category", "size.overlap.term", "size.overlap.category"), as.numeric) %>%
  #calculate k/K
  #mutate("k.K" = size.overlap.term/size.term)



#Visualize the result
enrich_h_df %>%
  filter(pvalue <= 0.35) %>%
  #Rename plot description
  mutate(Description = gsub("HALLMARK_", "", Description),
         Description = gsub("_", " ", Description)) %>%
  
  ggplot(aes(x = reorder(Description, k.K),
             y = k.K)) +
  geom_col()+
  theme_minimal()+
  #Flip x and y to make readable
  coord_flip()+
  #fix labels
  labs(y = "Significant genes in list / Total genes in list \nk/K",
       x = "Gene Set",
       title = "Enriched Hallmark")

  
  #With C2 Db
enrich_C2_df %>%
  filter(pvalue <= 0.001) %>%
  #Rename plot description
  mutate(Description = gsub("_", " ", Description)) %>%
    
  ggplot(aes(x = reorder(Description, k.K),
               y = k.K)) +
  geom_col()+
  theme_minimal()+
  #Flip x and y to make readable
  coord_flip()+
  #fix labels
  labs(y = "Significant genes in list / Total genes in list \nk/K",
       x = "Gene Set",
       title = "Enriched Curated Database")


#------------GSEA---------------------

##### Format GSEA #####

H.ensemble.ls <- H %>%
  select(gs_name, ensembl_gene) %>%
  group_by(gs_name) %>%
  summarise(all.genes = list(ensembl_gene)) %>%
  deframe()

C.geneSymbol.ls <- C %>%
  select(gs_name, gene_symbol) %>%
  group_by(gs_name) %>%
  summarise(all.genes = list(gene_symbol)) %>%
  deframe()

#use dim if it is df (without deframe)
length(C.geneSymbol.ls)

#load DGE Matrix
dat <- get(load("/Users/emad/RProjects/Enrichment Analysis/RSTR_data_clean_subset.RData"))
head(dat)
model.results <- read_csv("/Users/emad/RProjects/Enrichment Analysis/RSTR.Mtb.model.subset.csv")

C.ensemble.ls <- C %>%
  select(gs_name, ensembl_gene) %>%
  group_by(gs_name) %>%
  summarise(all.genes = list(ensembl_gene)) %>%
  deframe()

#calculate foldchange
#Extract expression data
FC <- as.data.frame(dat$E) %>% 
  #Move gene IDs from rownames to a column
  rownames_to_column("ensembl_gene_id") %>% 
  #Make long format with all expression values in 1 column
  pivot_longer(-ensembl_gene_id, 
               names_to = "libID", values_to = "expression") %>% 
  #Extract RSID and TB condition from libID
  #If this info was not in the libID, we could get it by joining
  # with dat$targets
  separate(libID, into = c("RSID","condition"), sep="_") %>% 
  #Make wide with media and tb expression in separate columns
  pivot_wider(names_from = condition, values_from = expression) %>% 
  #Calculate tb minus media fold change (delta for change)
  #Because expression values are log2, subtraction is the same as division
  mutate(delta = TB-MEDIA) %>% 
  #Calculate mean fold change per gene
  group_by(ensembl_gene_id) %>% 
  summarise(mean.delta = mean(delta, na.rm=TRUE)) %>% 
  #Arrange by descending fold change
  arrange(desc(mean.delta))


FC.vec <- FC$mean.delta
names(FC.vec) <- FC$ensembl_gene_id

min(FC.vec)
max(FC.vec)

#Run GSEA
gsea.C <- fgseaSimple(pathways = C.ensemble.ls,
                      stats = FC.vec,
                      scoreType = "std",
                      nperm=1000)

gsea.H <- fgseaSimple(pathways = H.ensemble.ls,
                     stats = FC.vec,
                     scoreType = "std",
                     nperm=1000)


gsea.C %>% 
  filter(pval <= 0.001015) %>% 
  #Beautify descriptions by removing _ and HALLMARK
  mutate(#pathway = gsub("HALLMARK_","", pathway),
         pathway = gsub("_"," ", pathway)) %>% 
  
  ggplot(aes(x=reorder(pathway, NES), #Reorder gene sets by NES values
             y=NES)) +
  geom_col() +
  theme_classic() +
  #Force equal max min
  lims(y=c(-3.2,3.2)) +
  #Some more customization to pretty it up
  #Flip x and y so long labels can be read
  coord_flip() +
  #fix labels
  labs(y="Normalized enrichment score",
       x="Gene set",
       title = "GSEA\n     Downregulated <--            --> Upregulated")

gsea.H %>% 
  filter(padj <= 0.05) %>% 
  #Beautify descriptions by removing _ and HALLMARK
  mutate(pathway = gsub("HALLMARK_","", pathway),
    pathway = gsub("_"," ", pathway)) %>% 
  
  ggplot(aes(x=reorder(pathway, NES), #Reorder gene sets by NES values
             y=NES)) +
  geom_col() +
  theme_classic() +
  #Force equal max min
  lims(y=c(-3.2,3.2)) +
  #Some more customization to pretty it up
  #Flip x and y so long labels can be read
  coord_flip() +
  #fix labels
  labs(y="Normalized enrichment score",
       x="Gene set",
       title = "GSEA\n     Downregulated <--            --> Upregulated")

