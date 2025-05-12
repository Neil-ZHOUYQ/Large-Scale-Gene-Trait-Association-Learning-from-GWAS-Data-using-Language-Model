from transformers import AutoTokenizer
import pandas as pd
import os

def check_gene_tokenization(tokenizer_name, gene_list_file):
    """
    Checks how many gene names from a list can be tokenized without using UNK tokens.
    
    Args:
        tokenizer_name (str): The name of the pretrained tokenizer to use
        gene_list_file (str): Path to file containing the list of gene names
    
    Returns:
        tuple: (total_genes, tokenized_genes, percentage)
    """
    # Load the tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return 0, 0, 0
    
    # Load gene list
    if not os.path.exists(gene_list_file):
        print(f"Gene list file not found: {gene_list_file}")
        return 0, 0, 0
    
    # Determine file type and read accordingly
    if gene_list_file.endswith('.csv'):
        try:
            df = pd.read_csv(gene_list_file)
            # Assuming the first column contains gene names
            genes = df.iloc[:, 0].tolist()
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return 0, 0, 0
    elif gene_list_file.endswith('.txt'):
        try:
            with open(gene_list_file, 'r') as f:
                genes = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading text file: {e}")
            return 0, 0, 0
    else:
        print(f"Unsupported file format: {gene_list_file}")
        return 0, 0, 0
    
    total_genes = len(genes)
    if total_genes == 0:
        print("No gene names found in the file.")
        return 0, 0, 0
    
    # Get UNK token information
    unk_token = tokenizer.unk_token
    unk_token_id = tokenizer.unk_token_id
    
    # Count genes that can be tokenized without UNK tokens
    tokenized_genes = 0
    problematic_genes = []
    
    for gene in genes:
        tokens = tokenizer.tokenize(gene)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # Check if UNK token is present
        if unk_token_id is not None and unk_token_id in token_ids:
            problematic_genes.append(gene)
        elif unk_token is not None and unk_token in tokens:
            problematic_genes.append(gene)
        else:
            tokenized_genes += 1
    
    percentage = (tokenized_genes / total_genes) * 100 if total_genes > 0 else 0
    
    return total_genes, tokenized_genes, percentage, problematic_genes

if __name__ == "__main__":
    # Configuration
    tokenizer_name = "Qwen/Qwen3-30B-A3B"  # Example: BioBERT or any BERT model you're using
    gene_list_file = "/mnt/home/zhouyuqi/bert02/codes/all_genes.txt"  # Replace with your gene list file
    
    # Check tokenization
    total, tokenized, percentage, problematic = check_gene_tokenization(tokenizer_name, gene_list_file)
    
    # Print results
    print(f"\n--- Gene Tokenization Analysis with {tokenizer_name} ---")
    print(f"Total gene names: {total}")
    print(f"Tokenized without UNK: {tokenized}")
    print(f"Percentage: {percentage:.2f}% ({tokenized}/{total})")
    
    # Print some examples of problematic genes
    if len(problematic) > 0:
        print(f"\nSample of genes with UNK tokens (showing up to 10):")
        for gene in problematic[:10]:
            print(f"  - {gene}")
        
        if len(problematic) > 10:
            print(f"  ... and {len(problematic) - 10} more")