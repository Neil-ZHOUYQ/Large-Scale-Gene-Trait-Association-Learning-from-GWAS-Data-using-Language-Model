#%%
import numpy as np
import pandas as pd
import os.path
import gzip
import re

# GWAS_dbpath="/mnt/home/kenneth/data/share/gwas_catalog/associations_v1.0.2.tsv"
# datapath = "/mnt/home/kenneth/data/nlp"
# outpath="/mnt/home/kenneth/out/nlp/biomedbert/datasets"

GWAS_dbpath = "/mnt/home/zhouyuqi/bert/data/share/gwas_catalog/associations_v1.0.2.tsv"
datapath = "/mnt/home/zhouyuqi/bert/data/nlp"
outpath = "/mnt/home/zhouyuqi/bert/out/nlp/biomedbert/datasets"

ASD_dataset1_fn = os.path.join(datapath, "SFARI", "SFARI-Gene_genes_10-09-2024release_11-07-2024export.csv")
ASD_dataset2_fn = os.path.join(datapath, "SFARI", "102_genes_from_large_scale_WES.txt")

GWAS_data_fn = os.path.join(outpath, "case", "gene_trait_assoc.GWAS.20250428.tsv") #convert from GWAS_db, containing information for GENE, PLVAL_MLOG, TRAIT
GWAS_ASD_data_fn = os.path.join(outpath, "case", "gene_trait_assoc.GWAS_GTR5.3_ASD.tsv") #combine GWAS_db, ASD, columns containing GENE, TRAIT, only keeping genes with PLVAL_MLOG more than suggestive assocciate s

GWAS_train_dataset_fn = os.path.join(outpath, "case", "case_train_dataset.GWAS.20250428.txt.gz")
ASD_train_dataset_fn = os.path.join(outpath, "case", "case_train_dataset.ASD.20250428.txt.gz")
ALL_train_dataset_fn = os.path.join(outpath, "case", "case_train_dataset.GWAS_ASD.20250428.txt.gz") #combine the upper two, with sentences

#FIELDS_LIST = ['DISEASE/TRAIT', 'INITIAL SAMPLE SIZE', 'MAPPED_GENE', 'PVALUE_MLOG', 'MAPPED_TRAIT']
FIELDS_LIST = ['DISEASE/TRAIT', 'MAPPED_GENE', 'PVALUE_MLOG']

SUGGESTIVE_ASSOC_THRESHOLD = 5.3




print("code is running")


def build_idx_file(df_assoc):

    gene_idx = {}

    for i in df_assoc.index:
        # Extract gene names from MAPPED_GENE, then combined them with duplicate removed
        genes = str(df_assoc['MAPPED_GENE'][i])
        
        # Skip this record if no mapped gene is found (SNP may be fallen within intergenic region)
        if genes == 'nan': 
            continue

        # Remove space and split the string by ' - ' symbol (i.e. GeneA - GeneB means SNP fallen within region between GeneA and GeneB)
        genes = genes.strip().split(' - ')
        
        for gene_set in genes:
            gene_list = re.split(r'[x;, ]', gene_set.replace(' ', ''))
            gene_list = [g for g in gene_list if g != 'Nomappedgenes']
            for g in list(set(gene_list)):
                gene_idx[g] = gene_idx.get(g, []) + [i]

    # Create the Gene idx
    gene_idx = { k: list(set(v)) for k,v in gene_idx.items()}

    return gene_idx


# Preprocess the case data (include p-values)
def categorize_association(p_mlog):
    if p_mlog > 7.3 :   # p-value < 5e-08
        return "strongly"
    elif p_mlog > 5.3:  # p-value < 5e-06
        return "moderately"
    else:
        return "weakly"


# Build a 1-to-many mapping from trait to genes for the case data
def trait_to_genes(df):
    """
    Convert a DataFrame with 'GENE' and 'TRAIT' columns into a dictionary.
    
    Args:
    df (pd.DataFrame): The input DataFrame with 'GENE' and 'TRAIT' columns.
    
    Returns:
    
    pd.DataFrame: remove dupilicates and convert the original df from long format to wide format(just a description, not strict codes meaning), with 'TRAIT'and  'GENES' 
    """   
    trait_dict = {}
    for trait, group in df.groupby('TRAIT'):
        trait_dict[trait] = list(set(group['GENE']))
    
    data = [(trait, ', '.join(map(str, genes))) for trait, genes in trait_dict.items()]
    return pd.DataFrame(data, columns=['TRAIT', 'GENES'])


def create_sentences_gene2trait(row, PVAL_MLOG=False):
    gene = row['GENE']
    trait = row['TRAIT']
    if PVAL_MLOG:
        pval_mlog = row['PVAL_MLOG']
        association = categorize_association(pval_mlog)
        out_str = f"Gene {gene} is {association} associated with {trait}."
    else:
        out_str = f"Gene {gene} is associated with {trait}."
    return out_str


def create_sentences_trait2genes(row):
    trait = row['TRAIT']
    genes = row['GENES']

    return f"{trait} is associated with genes {genes}."


# %%
# RLoad the GWAS Database file
df_assoc = pd.read_csv(GWAS_dbpath, delimiter='\t', usecols=FIELDS_LIST, dtype={'PVALUE_MLOG':np.float16 } ) 

#FIELDS_LIST = ['DISEASE/TRAIT', 'MAPPED_GENE', 'PVALUE_MLOG']

# Convert to dictionary for further processing
gene_idx = build_idx_file(df_assoc)

# Generate the dataframe containing GENE, PVAL_MLOG and TRAIT column
df_GWAS_data = pd.DataFrame()
for gene, indexes in gene_idx.items():
    #gene = 'LY86'
    #indexes = gene_idx[gene]

    rows = df_assoc.iloc[indexes]
    rows = rows.drop('MAPPED_GENE', axis=1)

    rows = rows.groupby('DISEASE/TRAIT')['PVALUE_MLOG'].max().reset_index()

    assoc = [{'GENE': gene,
             'PVAL_MLOG': row['PVALUE_MLOG'],
             'TRAIT': row['DISEASE/TRAIT']}
             for _, row in rows.iterrows()] # a list of dictionaries with same gene
    
    df_GWAS_data = pd.concat([df_GWAS_data, pd.DataFrame(assoc)], ignore_index=True) # a dataframe with 3 column: GENE, PLVAL_MLOG, TRAIT.

# Save the dataframe to file
df_GWAS_data.to_csv(GWAS_data_fn, sep='\t', index=False, float_format='%.1f')



#%%
# Create the sentences from dictionary of GWAS
# 1-to-1 gene-trait mapping
df_out1 = df_GWAS_data.apply(create_sentences_gene2trait, PVAL_MLOG=True, axis=1).to_frame(name='text') #convert to df, one text column, content has sentence as "Gene BRCA1 is strongly associated with Breast Cancer." 

# 1-to-many trait-genes mapping
df_trait2genes = trait_to_genes(df_GWAS_data)
df_out2 = df_trait2genes.apply(create_sentences_trait2genes, axis=1).to_frame(name='text') #Trait1 is associated with genes GeneA, GeneB.

# Combine both mapping to a dataframe
df_GWAS = pd.concat([df_out1, df_out2], ignore_index=True)

# Save the combined dataframe as the GWAS training dataset to a TSV file
with gzip.open(GWAS_train_dataset_fn, 'wt', encoding='utf-8') as f:
    df_GWAS.to_csv(f, sep="\t", index=False, )



#%%
# Load the SFARI dataset
trait = "Autism Spectrum Disorder (ASD)"
df_SFARI_data = pd.read_csv(ASD_dataset1_fn, sep=",")   #columns: 'gene-symbol', 'number-of-reports'

out = []
for _, row in df_SFARI_data.iterrows():
    occurence = int(row['number-of-reports'])  #column: 'number-of-reports'
    out += [(row['gene-symbol'], trait) for j in range(occurence)]   #column: 'gene-symbol'

df_SFARI_data = pd.DataFrame(out, columns=['GENE', 'TRAIT'])


# Load the WES 102 ASD gene list
with open(ASD_dataset2_fn, 'r') as f:
    genes = [line.strip() for line in f.readlines()]

out = []
for gene in genes:
    out += [(gene, trait)]

df_WES102_data = pd.DataFrame(out, columns=['GENE', 'TRAIT'])

# Combine datasets into one case training dataset
df_out = pd.concat([df_SFARI_data, df_WES102_data],
                 ignore_index=True)   # 'GENE', 'TRAIT'

# Shuffle the dataframe
df_out = df_out.sample(frac=1).reset_index(drop=True)

# 1-to-1 gene-trait assoc. mapping
df_ASD = df_out.apply(create_sentences_gene2trait, PVAL_MLOG=False, axis=1).to_frame(name='text') #Gene BRCA1 is strongly associated with Breast Cancer

# Write to a file
with gzip.open(ASD_train_dataset_fn, 'wt', encoding='utf-8') as f:
    df_ASD.to_csv(f, sep='\t', index=False)

# Concat all case training set together and write to a file
df_all = pd.concat([df_GWAS, df_ASD], ignore_index=True)
with gzip.open(ALL_train_dataset_fn, 'wt', encoding='utf-8') as f:
    df_all.to_csv(f, sep='\t', index=False)


#%%
# Prepare a gene-trait pairs and save to a tsv file
df_out = df_GWAS_data[df_GWAS_data.PVAL_MLOG > SUGGESTIVE_ASSOC_THRESHOLD]
df_out = df_out[['GENE', 'TRAIT']]

df_out1 = pd.concat([df_out,
                    df_SFARI_data, 
                    df_WES102_data],
                    ignore_index=True)

df_out2 = df_out1.drop_duplicates()

# Save to files
df_out2.to_csv(GWAS_ASD_data_fn, sep='\t', index=False)
