#%%
import numpy as np
import pandas as pd
import os.path
import gzip
import re



outpath = "/mnt/home/zhouyuqi/bert/out/nlp/biomedbert/datasets"
datapath = "/mnt/home/zhouyuqi/bert/data/nlp"


ASD_dataset1_fn = os.path.join(datapath, "SFARI", "SFARI-Gene_genes_10-09-2024release_11-07-2024export.csv")
trait = "Autism Spectrum Disorder (ASD)"
df_SFARI_data = pd.read_csv(ASD_dataset1_fn, sep=",") 
out = []
for _, row in df_SFARI_data.iterrows():
    occurence = int(row['number-of-reports'])  #column: 'number-of-reports'
    out += [(row['gene-symbol'], trait) for j in range(occurence)]   #column: 'gene-symbol'

df_SFARI_data = pd.DataFrame(out, columns=['GENE', 'TRAIT'])



ASD_dataset2_fn = os.path.join(datapath, "SFARI", "102_genes_from_large_scale_WES.txt")
with open(ASD_dataset2_fn, 'r') as f:
    genes = [line.strip() for line in f.readlines()]

out = []
for gene in genes:
    out += [(gene, trait)]

df_WES102_data = pd.DataFrame(out, columns=['GENE', 'TRAIT'])



GWAS_data_fn = os.path.join(outpath, "case", "gene_trait_assoc.GWAS.20250428.tsv") 

df_GWAS_data = pd.read_csv(GWAS_data_fn,sep='\t')

df_out = df_GWAS_data
df_out = df_out[['GENE', 'TRAIT']]




df_out1 = pd.concat([df_out,
                    df_SFARI_data, 
                    df_WES102_data],
                    ignore_index=True)
df_out2 = df_out1.drop_duplicates()

GWAS_ASD_data_fn = os.path.join(outpath, "case", "all_gene_trait_assoc.GWAS_ASD.tsv")
df_out2.to_csv(GWAS_ASD_data_fn, sep='\t', index=False)