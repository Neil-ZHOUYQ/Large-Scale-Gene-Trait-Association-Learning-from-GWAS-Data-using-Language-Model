import pandas as pd

df = pd.read_csv("/mnt/home/zhouyuqi/bert/out/nlp/biomedbert/datasets/case/gene_trait_assoc.GWAS.20250428.tsv", sep="\t")

genes = df['GENE']

unique_genes = genes.dropna().astype(str).unique()

outputfilepath="/mnt/home/zhouyuqi/bert02/codes/all_genes.txt"
with open(outputfilepath, "w") as f:
    for gene in unique_genes:
        f.write(gene + '\n')

print("Done!")