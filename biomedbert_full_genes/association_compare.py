import os
import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine
from tqdm.auto import tqdm # For progress bar

# --- Configuration ---

# 1. Paths
#    INPUT: Path to the TSV file containing gene names
#    (Adjust based on your actual structure relative to where you run this script)
gwas_data_fn = "/mnt/home/zhouyuqi/bert/out/nlp/biomedbert/datasets/case/gene_trait_assoc.GWAS.20250428.tsv" 

#    INPUT: Path to your domain-adapted model directory
#    (Use the final checkpoint directory if applicable)
model_dir = "/mnt/home/zhouyuqi/bert/out/nlp/biomedbert/DAP/250429_113915_E9_B32_LR2e-05_WD0.01" 

#    OUTPUT: Path to save the ranked gene list
output_ranking_fn = os.path.join(os.path.dirname(model_dir), "ranked_genes_by_asd_similarity.csv")

# 2. Prompts (Based on previous discussion)
ASD_PROMPT = "Synaptic function, neuronal connectivity, excitation/inhibition balance, and genetic factors contributing to Autism Spectrum Disorder."
GENE_PROMPT_TEMPLATE = "The role of {gene_name} in Autism Spectrum Disorder."

# 3. Model & Tokenizer Info (assuming it's the same base as DAP)
BASE_MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

# --- End Configuration ---


def get_cls_embedding(text: str, tokenizer, model, device, max_length: int = 512) -> np.ndarray:
    """
    Gets the [CLS] token embedding for a given text.
    """
    inputs = tokenizer(text, 
                       return_tensors="pt", 
                       truncation=True, 
                       max_length=max_length,
                       padding=True) # Padding added for consistency
    inputs = {k: v.to(device) for k, v in inputs.items()} # Move input to device
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Extract the last hidden state of the [CLS] token (index 0)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
    
    return cls_embedding.cpu().numpy()


def main():
    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model and Tokenizer ---
    print(f"Loading tokenizer from: {BASE_MODEL_NAME}")
    # It's generally recommended to load the tokenizer corresponding to the *base* model
    # unless the DAP specifically saved a modified tokenizer (less common).
    # Using the DAP model path ensures consistency if it *did* save one.
    try:
         tokenizer = BertTokenizer.from_pretrained(model_dir) 
         print(f"Loaded tokenizer from domain-adapted model directory: {model_dir}")
    except OSError:
         print(f"Could not find tokenizer in {model_dir}, loading from base: {BASE_MODEL_NAME}")
         tokenizer = BertTokenizer.from_pretrained(BASE_MODEL_NAME)
         
    print(f"Loading domain-adapted BERT model from: {model_dir}")
    # Load the base BertModel for embeddings, not BertForMaskedLM
    model = BertModel.from_pretrained(model_dir) 
    model.eval() # Set to evaluation mode
    model.to(device) # Move model to GPU if available
    print("Model and tokenizer loaded successfully.")

    # --- Load and Process Gene List ---
    print(f"Loading genes from: {gwas_data_fn}")
    if not os.path.exists(gwas_data_fn):
        print(f"Error: GWAS data file not found at {gwas_data_fn}")
        return

    try:
        df_gwas = pd.read_csv(gwas_data_fn, sep='\t')
        if "GENE" not in df_gwas.columns:
             print(f"Error: 'GENE' column not found in {gwas_data_fn}")
             return
             
        # Get unique, non-null gene names
        unique_genes = df_gwas["GENE"].dropna().unique().tolist()
        print(f"Found {len(unique_genes)} unique gene names.")
        if not unique_genes:
             print("Error: No gene names extracted.")
             return
             
    except Exception as e:
        print(f"Error reading or processing GWAS file: {e}")
        return

    # --- Calculate ASD Embedding ---
    print(f"Calculating embedding for ASD prompt: '{ASD_PROMPT}'")
    try:
        asd_embedding = get_cls_embedding(ASD_PROMPT, tokenizer, model, device)
        print(f"ASD embedding calculated (shape: {asd_embedding.shape})")
    except Exception as e:
        print(f"Error calculating ASD embedding: {e}")
        return

    # --- Calculate Gene Similarities ---
    print("Calculating gene embeddings and similarities...")
    gene_similarities = []
    skipped_genes = []

    for gene in tqdm(unique_genes, desc="Processing Genes"):
        # Check if gene is known to tokenizer (optional but good practice)
        # Use lower() because BiomedBERT is uncased
        gene_lower = gene.lower() 
        token_id = tokenizer.convert_tokens_to_ids(gene_lower)
        if token_id == tokenizer.unk_token_id:
            # print(f"  Warning: Gene '{gene}' not in tokenizer vocabulary. Skipping similarity calculation.")
            skipped_genes.append(gene)
            continue # Skip if gene itself is unknown

        try:
            # Create the prompt for the specific gene
            gene_prompt = GENE_PROMPT_TEMPLATE.format(gene_name=gene)
            
            # Get the embedding for the gene prompt
            gene_embedding = get_cls_embedding(gene_prompt, tokenizer, model, device)
            
            # Calculate cosine similarity (1 - cosine distance)
            similarity = 1 - cosine(asd_embedding, gene_embedding)
            
            # Ensure similarity is a float (sometimes can be numpy float)
            if isinstance(similarity, np.floating):
                 similarity = float(similarity)
            
            gene_similarities.append({"GENE": gene, "ASD_Similarity": similarity})
            
        except Exception as e:
            print(f"  Error processing gene '{gene}': {e}. Skipping.")
            skipped_genes.append(gene)

    print(f"Finished calculating similarities for {len(gene_similarities)} genes.")
    if skipped_genes:
        print(f"Skipped {len(skipped_genes)} genes (e.g., not in vocab or error during processing).")


    # --- Rank Genes and Save ---
    if not gene_similarities:
        print("Error: No similarity scores were calculated.")
        return
        
    print("Ranking genes by similarity...")
    ranked_df = pd.DataFrame(gene_similarities)
    ranked_df = ranked_df.sort_values(by="ASD_Similarity", ascending=False)

    print(f"Saving ranked gene list to: {output_ranking_fn}")
    try:
        ranked_df.to_csv(output_ranking_fn, index=False)
        print("Ranking complete and saved.")
    except Exception as e:
        print(f"Error saving ranked list: {e}")
        
    # Optionally print top N genes
    print("\nTop 10 Genes by ASD Similarity:")
    print(ranked_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()


