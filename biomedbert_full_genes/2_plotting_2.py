# visualize_dap.py - Scientific visualization for Domain Adaptive Pretraining
import os
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches

# Set style parameters
plt.style.use('ggplot')
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

# Paths - using same paths as in 2_DomainAdaptivePretaining.py
root_path = "/mnt/home/zhouyuqi/bert/out/nlp/biomedbert"
model_dir = os.path.join(root_path, "DAP", "250429_113915_E9_B32_LR2e-05_WD0.01")
output_dir = os.path.join(model_dir, "visualization")

os.makedirs(output_dir, exist_ok=True)

def plot_learning_curve():
    """
    Plot training loss curve by extracting data from trainer_state.json
    """
    model01_dir = os.path.join(model_dir, "models", "checkpoint-66834")
    state_file = os.path.join(model01_dir, "trainer_state.json")
    
    if not os.path.exists(state_file):
        print(f"No trainer_state.json found at {state_file}")
        return False
    
    with open(state_file, 'r') as f:
        trainer_state = json.load(f)
    
    # Extract logs
    logs = trainer_state.get('log_history', [])
    
    steps = []
    train_losses = []
    eval_losses = []
    perplexities = []
    
    for entry in logs:
        if 'loss' in entry:
            steps.append(entry.get('step'))
            train_losses.append(entry.get('loss'))
        if 'eval_loss' in entry:
            eval_losses.append(entry.get('eval_loss'))
            # Calculate perplexity from loss (perplexity = exp(loss))
            perplexities.append(math.exp(entry.get('eval_loss')))
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, label='Training Loss', color='blue')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Domain Adaptive Pretraining Loss Over Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'training_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot evaluation loss if available
    if eval_losses:
        eval_steps = np.linspace(min(steps), max(steps), len(eval_losses))
        
        plt.figure(figsize=(10, 6))
        plt.plot(eval_steps, eval_losses, label='Validation Loss', color='red', marker='o')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Domain Adaptive Pretraining Validation Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'validation_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot perplexity
        plt.figure(figsize=(10, 6))
        plt.plot(eval_steps, perplexities, label='Validation Perplexity', color='purple', marker='o')
        plt.xlabel('Training Steps')
        plt.ylabel('Perplexity')
        plt.title('Domain Adaptive Pretraining Perplexity')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'perplexity.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    return True

def visualize_masked_word_embeddings():
    """
    Visualize word embeddings before and after domain pretraining
    """
    try:
        # Load pre-trained model and tokenizer
        original_model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
        pretrained_model_name = model_dir
        
        # Load original and domain-adapted models
        original_model = BertForMaskedLM.from_pretrained(original_model_name)
        pretrained_model = BertForMaskedLM.from_pretrained(pretrained_model_name)
        tokenizer = BertTokenizer.from_pretrained(original_model_name)
        
        # Create a list of domain-specific words (genes and traits)
        # domain_words = [
        #     "gene", "allele", "mutation", "variant", "snp", "polymorphism",
        #     "autism", "diabetes", "cancer", "alzheimer", "schizophrenia",
        #     "pathway", "receptor", "promoter", "intron", "exon",
        #     "chromosome", "genomic", "proteomic", "methylation", "expression"
        # ]
        domain_words = [
                        "SHANK3","MECP2","SCN2A","NRXN1","CHD8","PTEN","SYNGAP1","SCN1A","CNTNAP2","ARID1B","GRIN2B","ADNP","DYRK1A","CHD2","FOXP1","STXBP1","ANKRD11","TCF4","FMR1","AUTS2","GRIN2A","PCDH19","RELN",
                        "CACNA1C","POGZ","SCN8A","CDKL5","FOXG1","KCNQ2","FOXP2","TSC2","IQSEC2" 
            #             "gene", "allele", "mutation", "variant", "snp", "polymorphism",
            # "autism", "diabetes", "cancer", "alzheimer", "schizophrenia",
            # "pathway", "receptor", "promoter", "intron", "exon",
            # "chromosome", "genomic", "proteomic", "methylation", "expression"
]

# You can now use this 'domain_words' list in your Python code.
# For example, to print the first item:
# print(domain_words[0])  # Output: SHANK3
        # Get word embeddings from both models
        original_embeddings = {}
        pretrained_embeddings = {}
        
        for word in domain_words:
            # Get token ID
            token_id = tokenizer.convert_tokens_to_ids(word.lower())
            if token_id == tokenizer.unk_token_id:
                continue  # Skip words not in vocabulary
                
            # Get embeddings from original model
            original_embedding = original_model.bert.embeddings.word_embeddings.weight[token_id].detach().numpy()
            original_embeddings[word] = original_embedding
            
            # Get embeddings from pretrained model
            pretrained_embedding = pretrained_model.bert.embeddings.word_embeddings.weight[token_id].detach().numpy()
            pretrained_embeddings[word] = pretrained_embedding
        
        # Combine embeddings for visualization
        words = list(original_embeddings.keys())
        original_vecs = np.array([original_embeddings[word] for word in words])
        pretrained_vecs = np.array([pretrained_embeddings[word] for word in words])
        
        # Calculate embedding differences
        embedding_diffs = {}
        for word in words:
            diff = np.linalg.norm(pretrained_embeddings[word] - original_embeddings[word])
            embedding_diffs[word] = diff
        
        # Sort words by embedding difference
        sorted_diffs = sorted(embedding_diffs.items(), key=lambda x: x[1], reverse=True)
        
        # Plot embedding differences
        plt.figure(figsize=(12, 8))
        words_sorted = [item[0] for item in sorted_diffs]
        diffs_sorted = [item[1] for item in sorted_diffs]
        
        plt.bar(words_sorted, diffs_sorted, color='teal')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Domain-Specific Terms')
        plt.ylabel('Embedding Change Magnitude')
        plt.title('Word Embedding Changes After Domain Adaptive Pretraining')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'embedding_changes.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Use dimensionality reduction to visualize embeddings in 2D
        for method, reducer in [('PCA', PCA(n_components=2)), ('t-SNE', TSNE(n_components=2, random_state=42))]:
            # Reduce dimensions of combined embeddings
            combined_vecs = np.vstack([original_vecs, pretrained_vecs])
            reduced_vecs = reducer.fit_transform(combined_vecs)
            
            original_reduced = reduced_vecs[:len(words)]
            pretrained_reduced = reduced_vecs[len(words):]
            
            # Plot
            plt.figure(figsize=(12, 10))
            
            # Plot original embeddings
            plt.scatter(original_reduced[:, 0], original_reduced[:, 1], 
                       color='blue', marker='o', alpha=0.7)
            
            # Plot pretrained embeddings
            plt.scatter(pretrained_reduced[:, 0], pretrained_reduced[:, 1], 
                       color='red', marker='o', alpha=0.7)
            
            # Draw lines connecting the same words
            for i in range(len(words)):
                plt.plot([original_reduced[i, 0], pretrained_reduced[i, 0]],
                        [original_reduced[i, 1], pretrained_reduced[i, 1]],
                        'k-', alpha=0.3)
                
                # Add word labels
                plt.annotate(words[i], 
                            (original_reduced[i, 0], original_reduced[i, 1]),
                            fontsize=8, alpha=0.8, color='blue',
                            xytext=(5, 5), textcoords='offset points')
                
                plt.annotate(words[i], 
                            (pretrained_reduced[i, 0], pretrained_reduced[i, 1]),
                            fontsize=8, alpha=0.8, color='red',
                            xytext=(5, 5), textcoords='offset points')
            
            # Create legend
            blue_patch = mpatches.Patch(color='blue', label='Original Model')
            red_patch = mpatches.Patch(color='red', label='Domain-Adapted Model')
            plt.legend(handles=[blue_patch, red_patch])
            
            plt.title(f'Word Embeddings Visualization Using {method}')
            plt.savefig(os.path.join(output_dir, f'embeddings_{method.lower()}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        return True
    except Exception as e:
        print(f"Error visualizing word embeddings: {e}")
        return False

def visualize_attention_for_gene_trait_pairs():
    """
    Visualize attention patterns between genes and traits 
    using the domain-adapted model
    """
    try:
        # Load pretrained model and tokenizer
        model_name = model_dir
        model = BertModel.from_pretrained(model_name, output_attentions=True)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Sample gene-trait pairs
        gene_trait_pairs = [
            ("MECP2", "Rett syndrome"),
            ("BRCA1", "breast cancer"),
            ("APP", "Alzheimer's disease"),
            ("CFTR", "cystic fibrosis"),
            ("PTEN", "autism spectrum disorder")
        ]
        
        for gene, trait in gene_trait_pairs:
            # Create an input text that includes both gene and trait
            text = f"The gene {gene} is associated with {trait}."
            
            # Tokenize the input
            inputs = tokenizer(text, return_tensors="pt")
            
            # Get model outputs with attention weights
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Get attention weights from the last layer (shape: [batch, heads, seq_len, seq_len])
            # We use the last layer as it typically captures the most semantic relations
            attentions = outputs.attentions[-1].squeeze(0)  # Remove batch dimension
            
            # Average across attention heads
            avg_attention = attentions.mean(dim=0)
            
            # Get token IDs and map back to tokens
            input_ids = inputs["input_ids"][0]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            
            # Find positions of gene and trait in token list
            gene_tokens = tokenizer.tokenize(gene)
            trait_tokens = tokenizer.tokenize(trait)
            
            gene_indices = []
            trait_indices = []
            
            for i, token in enumerate(tokens):
                # Check if this token is part of the gene
                if i < len(tokens) - len(gene_tokens) + 1:
                    if tokens[i:i+len(gene_tokens)] == gene_tokens:
                        gene_indices.extend(list(range(i, i+len(gene_tokens))))
                
                # Check if this token is part of the trait
                if i < len(tokens) - len(trait_tokens) + 1:
                    if tokens[i:i+len(trait_tokens)] == trait_tokens:
                        trait_indices.extend(list(range(i, i+len(trait_tokens))))
            
            # Create attention heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(avg_attention.numpy(), 
                        xticklabels=tokens, 
                        yticklabels=tokens,
                        cmap="YlOrRd")
            
            plt.title(f'Attention Map for Gene-Trait Pair: {gene}-{trait}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'attention_map_{gene}_{trait.replace(" ", "_")}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create focused attention visualization between gene and trait
            if gene_indices and trait_indices:
                # Extract sub-matrix of attention between gene and trait tokens
                gene_to_trait_attention = avg_attention[gene_indices, :][:, trait_indices].numpy()
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(gene_to_trait_attention, 
                            xticklabels=[tokens[i] for i in trait_indices], 
                            yticklabels=[tokens[i] for i in gene_indices],
                            cmap="YlGnBu", annot=True)
                
                plt.title(f'Gene-to-Trait Attention: {gene} → {trait}')
                plt.xlabel(f'Tokens for "{trait}"')
                plt.ylabel(f'Tokens for "{gene}"')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'gene_to_trait_attention_{gene}_{trait.replace(" ", "_")}.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                # Create the reverse direction attention as well
                trait_to_gene_attention = avg_attention[trait_indices, :][:, gene_indices].numpy()
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(trait_to_gene_attention, 
                            xticklabels=[tokens[i] for i in gene_indices], 
                            yticklabels=[tokens[i] for i in trait_indices],
                            cmap="YlGnBu", annot=True)
                
                plt.title(f'Trait-to-Gene Attention: {trait} → {gene}')
                plt.xlabel(f'Tokens for "{gene}"')
                plt.ylabel(f'Tokens for "{trait}"')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'trait_to_gene_attention_{gene}_{trait.replace(" ", "_")}.png'), dpi=300, bbox_inches='tight')
                plt.close()
        
        return True
    except Exception as e:
        print(f"Error visualizing attention patterns: {e}")
        return False

# Main execution
if __name__ == "__main__":
    print("Generating scientific visualizations for Domain Adaptive Pretraining...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all plots
    learning_curve_success = plot_learning_curve()
    print(f"Learning curve plots: {'SUCCESS' if learning_curve_success else 'FAILED'}")
    
    embedding_success = visualize_masked_word_embeddings()
    print(f"Word embedding visualization: {'SUCCESS' if embedding_success else 'FAILED'}")
    
    attention_success = visualize_attention_for_gene_trait_pairs()
    print(f"Attention visualization for gene-trait pairs: {'SUCCESS' if attention_success else 'FAILED'}")
    
    print(f"All visualizations saved to: {output_dir}")