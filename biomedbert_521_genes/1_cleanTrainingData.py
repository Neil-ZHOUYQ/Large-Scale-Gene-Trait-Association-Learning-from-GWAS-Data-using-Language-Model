#%%
import numpy as np
import pandas as pd
import os.path
import gzip
import re

# <<< MODIFICATION START >>>
# Define the path to your known_genes.txt file
# Please adjust this path if it's different
KNOWN_GENES_FILE = "/mnt/home/zhouyuqi/bert02/known_genes.txt"

# Load known genes into a set for efficient lookup
try:
    with open(KNOWN_GENES_FILE, 'r') as f:
        known_genes_set = {line.strip().upper() for line in f if line.strip()} # Convert to uppercase for case-insensitive matching
    print(f"Successfully loaded {len(known_genes_set)} unique genes from {KNOWN_GENES_FILE}")
except FileNotFoundError:
    print(f"Error: {KNOWN_GENES_FILE} not found. Please ensure the file exists in the correct location.")
    known_genes_set = set() # Initialize as empty set if file not found to avoid further errors
# <<< MODIFICATION END >>>







GWAS_dbpath="/mnt/home/zhouyuqi/bert02/data/share/gwas_catalog/associations_v1.0.2.tsv"
datapath = "/mnt/home/zhouyuqi/bert02/data/nlp"
outpath="/mnt/home/zhouyuqi/bert02/out/nlp/biomedbert/datasets"

ASD_dataset1_fn = os.path.join(datapath, "SFARI", "SFARI-Gene_genes_10-09-2024release_11-07-2024export.csv")
ASD_dataset2_fn = os.path.join(datapath, "SFARI", "102_genes_from_large_scale_WES.txt")

GWAS_data_fn = os.path.join(outpath, "case", "gene_trait_assoc.GWAS.20250508.tsv") #convert from GWAS_db, containing information for GENE, PLVAL_MLOG, TRAIT
GWAS_ASD_data_fn = os.path.join(outpath, "case", "gene_trait_assoc.GWAS_GTR5.3_ASD.tsv") #combine GWAS_db, ASD, columns containing GENE, TRAIT, only keeping genes with PLVAL_MLOG more than suggestive assocciate s

GWAS_train_dataset_fn = os.path.join(outpath, "case", "case_train_dataset.GWAS.20250508.txt.gz")
ASD_train_dataset_fn = os.path.join(outpath, "case", "case_train_dataset.ASD.20250508.txt.gz")
ALL_train_dataset_fn = os.path.join(outpath, "case", "case_train_dataset.GWAS_ASD.20250508.txt.gz") #combine the upper two, with sentences

#FIELDS_LIST = ['DISEASE/TRAIT', 'INITIAL SAMPLE SIZE', 'MAPPED_GENE', 'PVALUE_MLOG', 'MAPPED_TRAIT']
FIELDS_LIST = ['DISEASE/TRAIT', 'MAPPED_GENE', 'PVALUE_MLOG']

SUGGESTIVE_ASSOC_THRESHOLD = 5.3

# ... existing code ...
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
                # <<< MODIFICATION START >>>
                # Only consider genes that are in our known_genes_set (case-insensitive)
                if known_genes_set and g.upper() in known_genes_set:
                    gene_idx[g] = gene_idx.get(g, []) + [i]
                elif not known_genes_set: # If known_genes.txt was not found or empty, process all genes
                    gene_idx[g] = gene_idx.get(g, []) + [i]
                # <<< MODIFICATION END >>>

    # Create the Gene idx
    gene_idx = { k: list(set(v)) for k,v in gene_idx.items()}

    return gene_idx


# Preprocess the case data (include p-values)
def categorize_association(p_mlog):
    if p_mlog > 7.3 :   # p-value < 5 * 10^-8
        return "strongly"
    elif p_mlog > 5.3:  # p-value < 5 * 10^-6
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
    #PVAL_MLOG: determine whether use description: "strongly", "moderately", "weakly" associated with trait.
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
# This will now only include genes from known_genes_set if the file was loaded
gene_idx = build_idx_file(df_assoc) 

# Generate the dataframe containing GENE, PVAL_MLOG and TRAIT column
df_GWAS_data = pd.DataFrame()
for gene, indexes in gene_idx.items(): # gene_idx already filtered by known_genes_set
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
if not df_GWAS_data.empty:
    df_GWAS_data.to_csv(GWAS_data_fn, sep='\t', index=False, float_format='%.1f')
    print(f"Saved df_GWAS_data with {len(df_GWAS_data)} rows to {GWAS_data_fn}")
else:
    print(f"Skipping saving GWAS_data_fn as df_GWAS_data is empty (possibly due to gene filtering or no GWAS data).")


#%%
# Create the sentences from dictionary of GWAS
df_GWAS = pd.DataFrame(columns=['text']) # Initialize empty DataFrame

if not df_GWAS_data.empty:
    # 1-to-1 gene-trait mapping
    df_out1 = df_GWAS_data.apply(create_sentences_gene2trait, PVAL_MLOG=True, axis=1).to_frame(name='text') #convert to df, one text column, content has sentence as "Gene BRCA1 is strongly associated with Breast Cancer." 

    # 1-to-many trait-genes mapping
    df_trait2genes = trait_to_genes(df_GWAS_data)
    if not df_trait2genes.empty:
        df_out2 = df_trait2genes.apply(create_sentences_trait2genes, axis=1).to_frame(name='text') #Trait1 is associated with genes GeneA, GeneB.
        # Combine both mapping to a dataframe
        df_GWAS = pd.concat([df_out1, df_out2], ignore_index=True)
    else: # If no traits left after filtering (e.g. all genes for a trait were filtered out)
        df_GWAS = df_out1 
else:
    print("df_GWAS_data is empty, so df_GWAS (GWAS sentences) will be empty.")


# Save the combined dataframe as the GWAS training dataset to a TSV file
if not df_GWAS.empty:
    with gzip.open(GWAS_train_dataset_fn, 'wt', encoding='utf-8') as f:
        df_GWAS.to_csv(f, sep="\t", index=False, )
    print(f"Saved df_GWAS (GWAS sentences) to {GWAS_train_dataset_fn}")
else:
    print(f"Skipping saving GWAS_train_dataset_fn as df_GWAS is empty.")


#%%
# Load the SFARI dataset
trait = "Autism Spectrum Disorder (ASD)"
df_SFARI_data_raw = pd.read_csv(ASD_dataset1_fn, sep=",")   #columns: 'gene-symbol', 'number-of-reports'

out = []
for _, row in df_SFARI_data_raw.iterrows():
    gene_symbol = row['gene-symbol']
    # <<< MODIFICATION START >>>
    if known_genes_set and gene_symbol.upper() not in known_genes_set:
        continue # Skip this gene if not in the known list
    # <<< MODIFICATION END >>>
    occurence = int(row['number-of-reports'])  #column: 'number-of-reports'
    out += [(gene_symbol, trait) for j in range(occurence)]   #column: 'gene-symbol'

df_SFARI_data = pd.DataFrame(out, columns=['GENE', 'TRAIT'])
# <<< MODIFICATION START >>>
if known_genes_set:
    print(f"Filtered/Constructed df_SFARI_data with {len(df_SFARI_data)} entries based on known_genes.txt")
# <<< MODIFICATION END >>>


# Load the WES 102 ASD gene list
df_WES102_data = pd.DataFrame(columns=['GENE', 'TRAIT']) # Initialize empty
try:
    with open(ASD_dataset2_fn, 'r') as f:
        genes_from_file = [line.strip() for line in f.readlines()]
    
    out_wes = []
    for gene in genes_from_file:
        # <<< MODIFICATION START >>>
        if known_genes_set and gene.upper() not in known_genes_set:
            continue # Skip this gene if not in the known list
        # <<< MODIFICATION END >>>
        out_wes += [(gene, trait)]
    
    df_WES102_data = pd.DataFrame(out_wes, columns=['GENE', 'TRAIT'])
    # <<< MODIFICATION START >>>
    if known_genes_set:
        print(f"Filtered/Constructed df_WES102_data with {len(df_WES102_data)} entries based on known_genes.txt")
    # <<< MODIFICATION END >>>
except FileNotFoundError:
    print(f"Warning: {ASD_dataset2_fn} not found. df_WES102_data will be empty.")


# Combine datasets into one case training dataset
df_ASD_combined = pd.DataFrame(columns=['GENE', 'TRAIT'])
if not df_SFARI_data.empty or not df_WES102_data.empty:
    df_ASD_combined = pd.concat([df_SFARI_data, df_WES102_data], ignore_index=True)
else:
    print("Both SFARI and WES102 data are empty after filtering (or file not found). df_ASD_combined will be empty.")


df_ASD_sentences = pd.DataFrame(columns=['text']) # Initialize empty
if not df_ASD_combined.empty:
    # Shuffle the dataframe
    df_ASD_combined = df_ASD_combined.sample(frac=1).reset_index(drop=True)
    # 1-to-1 gene-trait assoc. mapping
    df_ASD_sentences = df_ASD_combined.apply(create_sentences_gene2trait, PVAL_MLOG=False, axis=1).to_frame(name='text') 
else:
    print("df_ASD_combined is empty. df_ASD_sentences (ASD sentences) will be empty.")

# Write to a file
if not df_ASD_sentences.empty:
    with gzip.open(ASD_train_dataset_fn, 'wt', encoding='utf-8') as f:
        df_ASD_sentences.to_csv(f, sep='\t', index=False)
    print(f"Saved df_ASD_sentences (ASD sentences) to {ASD_train_dataset_fn}")
else:
    print(f"Skipping saving ASD_train_dataset_fn as df_ASD_sentences is empty.")


# Concat all case training set together and write to a file
df_all = pd.DataFrame(columns=['text']) # Initialize empty
if not df_GWAS.empty or not df_ASD_sentences.empty:
    df_all = pd.concat([df_GWAS, df_ASD_sentences], ignore_index=True)
else:
    print("Both GWAS and ASD sentence dataframes are empty. df_all will be empty.")

if not df_all.empty:
    with gzip.open(ALL_train_dataset_fn, 'wt', encoding='utf-8') as f:
        df_all.to_csv(f, sep='\t', index=False)
    print(f"Saved df_all (all sentences) to {ALL_train_dataset_fn}")
else:
     print(f"Skipping saving ALL_train_dataset_fn as df_all is empty.")


#%%
# Prepare a gene-trait pairs and save to a tsv file
df_final_out = pd.DataFrame(columns=['GENE', 'TRAIT']) # Initialize

# df_GWAS_data is already filtered by known_genes_set if it was loaded
if not df_GWAS_data.empty:
    df_gwas_filtered_pval = df_GWAS_data[df_GWAS_data.PVAL_MLOG > SUGGESTIVE_ASSOC_THRESHOLD]
    if not df_gwas_filtered_pval.empty:
        df_gwas_filtered_pval = df_gwas_filtered_pval[['GENE', 'TRAIT']]
        df_final_out = pd.concat([df_final_out, df_gwas_filtered_pval], ignore_index=True)

# df_SFARI_data and df_WES102_data are already filtered by known_genes_set if it was loaded
if not df_SFARI_data.empty:
     df_final_out = pd.concat([df_final_out, df_SFARI_data], ignore_index=True)

if not df_WES102_data.empty:
     df_final_out = pd.concat([df_final_out, df_WES102_data], ignore_index=True)


if not df_final_out.empty:
    df_final_out_unique = df_final_out.drop_duplicates()
    # Save to files
    df_final_out_unique.to_csv(GWAS_ASD_data_fn, sep='\t', index=False)
    print(f"Saved {len(df_final_out_unique)} unique gene-trait pairs to {GWAS_ASD_data_fn}")
else:
    print(f"Skipping saving {GWAS_ASD_data_fn} as the final combined dataframe is empty.")
