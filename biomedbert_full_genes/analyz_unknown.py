import os
import pandas as pd
from transformers import BertTokenizer
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# --- Configuration ---

# 1. Paths
#    INPUT: Path to the TSV file containing gene names
gwas_data_fn = "/mnt/home/zhouyuqi/bert/out/nlp/biomedbert/datasets/case/gene_trait_assoc.GWAS.20250428.tsv" 

#    INPUT: Tokenizer name/path (Using the base uncased model is appropriate here)
TOKENIZER_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

#    OUTPUT: Directory to save analysis results (plots, unknown genes list)
#    It will be created relative to the directory containing the GWAS file
output_analysis_dir = os.path.join(os.path.dirname(gwas_data_fn), "gene_tokenization_analysis")

# --- End Configuration ---


def analyze_gene_tokenization():
    """
    Analyzes and visualizes the tokenization status of genes from a TSV file.
    """
    print("Starting gene tokenization analysis...")

    # --- Create Output Directory ---
    os.makedirs(output_analysis_dir, exist_ok=True)
    print(f"Analysis results will be saved to: {output_analysis_dir}")

    # --- Load Tokenizer ---
    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    try:
        # Use AutoTokenizer for robustness, handles uncased automatically
        tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME) 
        unk_token_id = tokenizer.unk_token_id
        if unk_token_id is None:
             print("Error: Could not determine the [UNK] token ID for this tokenizer.")
             return
        print(f"Tokenizer loaded. [UNK] token ID: {unk_token_id}")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

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
             
        # Get unique, non-null gene names, convert to string just in case
        unique_genes = df_gwas["GENE"].dropna().astype(str).unique().tolist()
        total_unique_genes = len(unique_genes)
        print(f"Found {total_unique_genes} unique gene names.")
        if not unique_genes:
             print("Error: No gene names extracted.")
             return
             
    except Exception as e:
        print(f"Error reading or processing GWAS file: {e}")
        return

    # --- Check Tokenization Status ---
    print("Checking tokenization status for each unique gene...")
    unknown_genes = []
    known_genes = []

    for gene in tqdm(unique_genes, desc="Checking Genes"):
        # Convert to lowercase for the uncased tokenizer
        gene_lower = gene.lower()
        
        # Get the token ID(s). Using encode gives list of IDs, check if UNK is the *only* result
        # or if convert_tokens_to_ids directly gives UNK
        token_id = tokenizer.convert_tokens_to_ids(gene_lower) 

        if token_id == unk_token_id:
            unknown_genes.append(gene)
        else:
            # Optional: Further check if it tokenizes into multiple pieces vs single known token
            # tokens = tokenizer.tokenize(gene_lower)
            # if len(tokens) > 1 and any(tok.startswith('##') for tok in tokens):
            #     # It's known but tokenized into subwords
            #     pass
            known_genes.append(gene)

    # --- Calculate Statistics ---
    num_unknown = len(unknown_genes)
    num_known = len(known_genes)
    percent_unknown = (num_unknown / total_unique_genes) * 100 if total_unique_genes > 0 else 0

    print("\n--- Tokenization Statistics ---")
    print(f"Total unique genes analyzed: {total_unique_genes}")
    print(f"Number of known genes (tokenizable): {num_known}")
    print(f"Number of unknown genes ([UNK]): {num_unknown}")
    print(f"Percentage of unknown genes: {percent_unknown:.2f}%")

    if num_unknown > 0:
        print(f"\nFirst few unknown genes: {unknown_genes[:10]}...")
        # Save the full list of unknown genes
        unknown_genes_file = os.path.join(output_analysis_dir, "unknown_genes.txt")
        try:
            with open(unknown_genes_file, 'w') as f:
                for gene in unknown_genes:
                    f.write(f"{gene}\n")
            print(f"Full list of {num_unknown} unknown genes saved to: {unknown_genes_file}")
        except Exception as e:
            print(f"Error saving unknown genes list: {e}")


    # --- Generate Visualizations ---
    if total_unique_genes == 0:
        print("Skipping visualization as no genes were processed.")
        return
        
    print("\nGenerating visualization plots...")
    plt.style.use('ggplot')

    # 1. Pie Chart
    labels = ['Known Genes', 'Unknown [UNK] Genes']
    sizes = [num_known, num_unknown]
    colors = ['#66b3ff','#ff9999'] # Blueish and Reddish
    explode = (0, 0.1) if num_unknown > 0 and num_known > 0 else (0, 0) # Explode the 'Unknown' slice if it exists

    fig_pie, ax_pie = plt.subplots(figsize=(7, 7))
    ax_pie.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90)
    ax_pie.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(f'Tokenization Status of {total_unique_genes} Unique Genes\n(Tokenizer: {os.path.basename(TOKENIZER_NAME)})', pad=20)
    pie_chart_path = os.path.join(output_analysis_dir, 'gene_tokenization_pie_chart.png')
    try:
        plt.savefig(pie_chart_path, dpi=300, bbox_inches='tight')
        print(f"✓ Pie chart saved to {pie_chart_path}")
    except Exception as e:
        print(f"Error saving pie chart: {e}")
    plt.close(fig_pie)


    # 2. Bar Chart
    fig_bar, ax_bar = plt.subplots(figsize=(6, 5))
    categories = ['Known', 'Unknown [UNK]']
    counts = [num_known, num_unknown]
    bars = ax_bar.bar(categories, counts, color=colors)

    ax_bar.set_ylabel('Number of Unique Genes')
    ax_bar.set_title(f'Gene Tokenization Counts (Total: {total_unique_genes})')
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)

    # Add counts on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2.0, yval + counts[0]*0.01 , int(yval), va='bottom', ha='center') # Add text label

    bar_chart_path = os.path.join(output_analysis_dir, 'gene_tokenization_bar_chart.png')
    try:
        plt.tight_layout()
        plt.savefig(bar_chart_path, dpi=300)
        print(f"✓ Bar chart saved to {bar_chart_path}")
    except Exception as e:
        print(f"Error saving bar chart: {e}")
    plt.close(fig_bar)

    print("\nAnalysis complete.")


# --- Execution ---
if __name__ == "__main__":
    analyze_gene_tokenization()
# --- End Execution ---