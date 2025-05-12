import os
import json
import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForMaskedLM
import gzip
from matplotlib.ticker import MaxNLocator
from collections import Counter

# Set high-quality plot settings for scientific publication
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'figure.dpi': 300,
})

# Path setup based on 2_DomainAdaptivePretraining.py
root_path = "/mnt/home/zhouyuqi/bert02/out/nlp/biomedbert"  # Adjust this to your path
run_name = "250508_022350_E9_B32_LR2e-05_WD0.01"  

# Set up directories
logging_dir = os.path.join(root_path, "DAP", run_name)
training_output_dir = os.path.join(root_path, "DAP", run_name)
model_output_dir = os.path.join(training_output_dir, "models")
plots_dir = os.path.join(logging_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# Path to training data
dataset_fn = os.path.join(root_path, "datasets", "case", "case_train_dataset.GWAS_ASD.20250508.txt.gz")
trainingloss_file = os.path.join(logging_dir, 'training_loss.json')
test_metrics_file = os.path.join(logging_dir, "test_model_metrics.txt")

def read_test_metrics():
    """Read test perplexity from metrics file"""
    test_perplexity = None
    try:
        with open(test_metrics_file, 'r') as file:
            for line in file:
                if "Perplexity:" in line:
                    test_perplexity = float(line.split(":")[1].strip())
                    break
        return test_perplexity
    except FileNotFoundError:
        print(f"Test metrics file not found: {test_metrics_file}")
        return None

def plot_training_loss():
    """Plot the training loss curve with scientific formatting"""
    try:
        with open(trainingloss_file, 'r') as file:
            log_history = json.load(file)
        
        # Filter out entries that don't have 'loss' or 'step', and the final summary
        loss_entries = [x for x in log_history if 'loss' in x and 'step' in x and 'train_runtime' not in x]
        
        if not loss_entries:
            print("No valid loss entries found in training_loss.json")
            return None

        train_loss = [x['loss'] for x in loss_entries]
        steps = [x['step'] for x in loss_entries]
        
        if not train_loss or not steps:
            print("Could not extract training loss or steps.")
            return None
            
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps, train_loss, color='#1f77b4', linewidth=2, label='Training Loss')
        
        # Add moving average for trend visualization
        # Ensure steps are sorted for correct rolling average calculation if they aren't already
        df = pd.DataFrame({'steps': steps, 'loss': train_loss}).sort_values(by='steps')
        
        window_size = min(100, max(10, len(df) // 20)) # Ensure window_size is at least 1
        if len(df) >= window_size and window_size > 0 : # Check if window_size is valid
            df['moving_avg'] = df['loss'].rolling(window=window_size).mean()
            # Plot moving average only for points where it's calculable
            ax.plot(df['steps'][window_size-1:], df['moving_avg'][window_size-1:], 
                   color='#d62728', linewidth=2, linestyle='-', label=f'{window_size}-step Moving Average')
        
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Masked Language Modeling Loss')
        ax.set_title('Domain Adaptive Pretraining Loss')
        
        # Attempt to add epoch markers based on 'epoch' field in log_history
        # This assumes 'epoch' is a float and changes mark epoch boundaries roughly
        epoch_changes = []
        last_epoch_int = -1
        for entry in loss_entries:
            if 'epoch' in entry and 'step' in entry:
                current_epoch_int = int(entry['epoch'])
                if current_epoch_int > last_epoch_int and last_epoch_int != -1 : # Mark after the first full epoch completes
                    # Mark the step *before* the epoch number changed if possible, or current step
                    # This logic might need refinement based on how epoch numbers are logged
                    epoch_marker_step = entry['step'] 
                    # Check previous entry's step if available
                    prev_entry_idx = loss_entries.index(entry) -1
                    if prev_entry_idx >=0:
                        epoch_marker_step = loss_entries[prev_entry_idx]['step']

                    if epoch_marker_step not in [ec[0] for ec in epoch_changes]: # Avoid duplicate markers for same step
                         epoch_changes.append((epoch_marker_step, last_epoch_int))
                last_epoch_int = current_epoch_int
        
        # Add the end of the last epoch if it's a whole number
        if loss_entries and 'epoch' in loss_entries[-1] and loss_entries[-1]['epoch'] == float(last_epoch_int) and last_epoch_int > 0:
             if loss_entries[-1]['step'] not in [ec[0] for ec in epoch_changes]:
                epoch_changes.append((loss_entries[-1]['step'], last_epoch_int))


        # Sort changes by step to draw lines correctly
        epoch_changes.sort(key=lambda x: x[0])
        
        # Keep track of labeled epochs to avoid duplicate labels
        labeled_epochs = set()

        for step_val, epoch_num in epoch_changes:
            if epoch_num not in labeled_epochs:
                 ax.axvline(x=step_val, color='gray', linestyle='--', alpha=0.7)
                 ax.text(step_val, ax.get_ylim()[1]*0.95, f'End of Epoch {epoch_num}', 
                        ha='right', va='top', alpha=0.7, rotation=45, fontsize=10)
                 labeled_epochs.add(epoch_num)

        ax.legend()
        plt.tight_layout()
        loss_plot_path = os.path.join(plots_dir, 'DAP_loss_curve.png')
        plt.savefig(loss_plot_path)
        plt.close()
        print(f"✓ Training loss plot saved to {loss_plot_path}")
        
        # Return the actual loss values for other functions that might use them
        return [x['loss'] for x in loss_entries] 
        
    except FileNotFoundError:
        print(f"Training loss file not found: {trainingloss_file}")
        return None
    except Exception as e:
        print(f"An error occurred in plot_training_loss: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_perplexity(train_loss):
    """Plot perplexity curve"""
    if train_loss is None:
        return
    
    # Calculate perplexity (e^loss)
    perplexity = [math.exp(loss) for loss in train_loss]
    steps = range(1, len(perplexity) + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, perplexity, color='#2ca02c', linewidth=2)
    
    # Add test perplexity if available
    test_perplexity = read_test_metrics()
    if test_perplexity:
        ax.axhline(y=test_perplexity, color='#d62728', linestyle='-', 
                  label=f'Test Perplexity: {test_perplexity:.2f}')
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Perplexity')
    ax.set_title('Model Perplexity During Domain Adaptation')
    
    # Use log scale for perplexity
    ax.set_yscale('log')
    
    # Add epochs as vertical lines if we can infer them
    if 'num_train_epochs' in run_name:
        # Try to extract epochs from run name
        parts = run_name.split('_')
        for part in parts:
            if part.startswith('E') and part[1:].isdigit():
                epochs = int(part[1:])
                steps_per_epoch = len(perplexity) // epochs
                for e in range(1, epochs):
                    ax.axvline(x=e*steps_per_epoch, color='gray', linestyle='--', alpha=0.7)
    
    # Add initial and final perplexity annotations
    ax.annotate(f'Initial: {perplexity[0]:.2f}', 
               xy=(1, perplexity[0]), xytext=(10, 0), 
               textcoords="offset points", ha='left', va='center')
    ax.annotate(f'Final: {perplexity[-1]:.2f}', 
               xy=(len(perplexity), perplexity[-1]), xytext=(-10, 0), 
               textcoords="offset points", ha='right', va='center')
    
    if test_perplexity:
        ax.legend()
    
    plt.tight_layout()
    perplexity_plot_path = os.path.join(plots_dir, 'DAP_perplexity.png')
    plt.savefig(perplexity_plot_path)
    plt.close()
    print(f"✓ Perplexity plot saved to {perplexity_plot_path}")

def plot_learning_rate():
    """Plot learning rate schedule"""
    try:
        with open(trainingloss_file, 'r') as file:
            log_history = json.load(file)
        
        # Extract learning rate
        entries_with_lr = [x for x in log_history if 'learning_rate' in x]
        if not entries_with_lr:
            print("No learning rate data found in log history")
            return
            
        learning_rates = [x['learning_rate'] for x in entries_with_lr]
        steps = [x.get('step', i) for i, x in enumerate(entries_with_lr)]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps, learning_rates, color='#9467bd', linewidth=2)
        
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        
        # Detect warmup phase
        for i in range(1, len(learning_rates)):
            if learning_rates[i] < learning_rates[i-1]:
                warmup_end = steps[i-1]
                ax.axvline(x=warmup_end, color='#d62728', linestyle='--', alpha=0.7)
                ax.text(warmup_end, max(learning_rates), 'Warmup End', 
                       ha='center', va='bottom', alpha=0.7)
                break
        
        plt.tight_layout()
        lr_plot_path = os.path.join(plots_dir, 'DAP_learning_rate.png')
        plt.savefig(lr_plot_path)
        plt.close()
        print(f"✓ Learning rate plot saved to {lr_plot_path}")
    except (FileNotFoundError, KeyError) as e:
        print(f"Error plotting learning rate: {e}")

# def plot_epoch_metrics():
#     """Plot metrics by epoch"""
#     train_loss = None
#     test_perplexity = read_test_metrics()
    
#     try:
#         with open(trainingloss_file, 'r') as file:
#             log_history = json.load(file)
#         train_loss = [x['loss'] for x in log_history if 'loss' in x]
#     except FileNotFoundError:
#         print(f"Training loss file not found: {trainingloss_file}")
#         return
    
#     if train_loss is None:
#         return
        
#     # Try to infer number of epochs from run name
#     epochs = None
#     parts = run_name.split('_')
#     for part in parts:
#         if part.startswith('E') and part[1:].isdigit():
#             epochs = int(part[1:])
#             break
    
#     if epochs is None:
#         print("Could not determine number of epochs from run name")
#         return
        
#     # Calculate average loss and perplexity per epoch
#     steps_per_epoch = len(train_loss) // epochs
#     epoch_avg_loss = []
#     epoch_perplexity = []
    
#     for i in range(epochs):
#         start_idx = i * steps_per_epoch
#         end_idx = (i + 1) * steps_per_epoch if i < epochs - 1 else len(train_loss)
#         epoch_loss = sum(train_loss[start_idx:end_idx]) / (end_idx - start_idx)
#         epoch_avg_loss.append(epoch_loss)
#         epoch_perplexity.append(math.exp(epoch_loss))
    
#     # Create epoch-wise plot
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
#     # Loss plot
#     ax1.plot(range(1, epochs + 1), epoch_avg_loss, 'o-', color='#1f77b4', linewidth=2)
#     ax1.set_xlabel('Epoch')
#     ax1.set_ylabel('Average MLM Loss')
#     ax1.set_title('Training Loss by Epoch')
#     ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
#     # Annotate percentage improvement
#     initial_loss = epoch_avg_loss[0]
#     final_loss = epoch_avg_loss[-1]
#     pct_improvement = (initial_loss - final_loss) / initial_loss * 100
#     ax1.annotate(f'{pct_improvement:.1f}% decrease', 
#                 xy=(epochs, final_loss), 
#                 xytext=(epochs-1, (initial_loss + final_loss)/2),
#                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    
#     # Perplexity plot
#     ax2.plot(range(1, epochs + 1), epoch_perplexity, 'o-', color='#2ca02c', linewidth=2)
#     if test_perplexity:
#         ax2.axhline(y=test_perplexity, color='#d62728', linestyle='-', 
#                    label=f'Test Perplexity: {test_perplexity:.2f}')
#         ax2.legend()
#     ax2.set_xlabel('Epoch')
#     ax2.set_ylabel('Perplexity')
#     ax2.set_title('Perplexity by Epoch')
#     ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    
#     # Annotate percentage improvement
#     initial_perp = epoch_perplexity[0]
#     final_perp = epoch_perplexity[-1]
#     pct_improvement = (initial_perp - final_perp) / initial_perp * 100
#     ax2.annotate(f'{pct_improvement:.1f}% decrease', 
#                 xy=(epochs, final_perp), 
#                 xytext=(epochs-1, (initial_perp + final_perp)/2),
#                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    
#     plt.tight_layout()
#     epoch_plot_path = os.path.join(plots_dir, 'DAP_epoch_metrics.png')
#     plt.savefig(epoch_plot_path)
#     plt.close()
#     print(f"✓ Epoch metrics plot saved to {epoch_plot_path}")

def plot_token_prediction():
    """Visualize token prediction for gene-trait sentences"""
    print("Loading model for token prediction analysis...")
    try:
        # Load tokenizer and model
        tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
        model = BertForMaskedLM.from_pretrained(os.path.join(root_path, "DAP", run_name))
        model.eval()
        
        # Sample sentences for visualization
        examples = [
            "Gene BRCA1 is associated with breast cancer.",
            "Autism spectrum disorder is associated with SHANK3.",
            "Schizophrenia has been linked to various genes including DISC1."
        ]
        
        key_terms = {
            "Gene terms": ["gene", "brca1", "shank3", "disc1"],
            "Relation terms": ["associated", "disassociated"],
            "Disease terms": ["cancer", "autism", "schizophrenia"]
        }
        
        # Compute prediction probabilities for key terms
        results = {}
        
        for category, terms in key_terms.items():
            term_probs = []
            
            for term in terms:
                # Choose a relevant example
                for sentence in examples:
                    if term.lower() in sentence.lower():
                        break
                else:
                    sentence = f"Gene {term} is associated with disease."
                
                # Tokenize sentence
                tokens = tokenizer.tokenize(sentence)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                
                # Find the term tokens
                term_tokens = tokenizer.tokenize(term)
                
                for i in range(len(tokens) - len(term_tokens) + 1):
                    if tokens[i:i+len(term_tokens)] == term_tokens:
                        # Create masked version of sentence
                        masked_tokens = tokens.copy()
                        for j in range(len(term_tokens)):
                            masked_tokens[i+j] = '[MASK]'
                        
                        masked_text = tokenizer.convert_tokens_to_string(masked_tokens)
                        
                        # Get model prediction
                        inputs = tokenizer(masked_text, return_tensors="pt")
                        with torch.no_grad():
                            outputs = model(**inputs)
                        
                        # Calculate probability of original tokens
                        total_prob = 0
                        for j in range(len(term_tokens)):
                            mask_idx = torch.where(inputs['input_ids'][0] == tokenizer.mask_token_id)[0][j]
                            logits = outputs.logits[0, mask_idx]
                            probs = torch.nn.functional.softmax(logits, dim=0)
                            original_token_id = input_ids[i+j]
                            original_token_prob = probs[original_token_id].item()
                            total_prob += original_token_prob
                        
                        # Average probability across tokens
                        avg_prob = total_prob / len(term_tokens)
                        term_probs.append((term, avg_prob))
                        break
            
            results[category] = term_probs
        
        # Plot results
        fig, axs = plt.subplots(len(results), 1, figsize=(10, 3*len(results)), squeeze=False)
        
        for i, (category, term_probs) in enumerate(results.items()):
            ax = axs[i, 0]
            terms = [tp[0] for tp in term_probs]
            probs = [tp[1] for tp in term_probs]
            
            ax.bar(terms, probs, color='#1f77b4')
            ax.set_title(f"{category} Prediction Probabilities")
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 1)
            
            # Add value labels
            for j, v in enumerate(probs):
                ax.text(j, v + 0.02, f"{v:.2f}", ha='center')
            
            # Add reference line at 0.5
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        token_plot_path = os.path.join(plots_dir, 'DAP_token_prediction.png')
        plt.savefig(token_plot_path)
        plt.close()
        print(f"✓ Token prediction plot saved to {token_plot_path}")
    except Exception as e:
        print(f"Error in token prediction analysis: {e}")

def analyze_domain_vocabulary():
    """Analyze domain-specific vocabulary from training data"""
    try:
        # Load dataset to extract domain-specific terms
        print("Analyzing domain vocabulary...")
        tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
        
        # Open and read the gzipped dataset
        with gzip.open(dataset_fn, 'rt', encoding='utf-8') as f:
            df = pd.read_csv(f, sep="\t")
        
        # Process each text example
        domain_terms = Counter()
        
        # Common words to filter out
        stopwords = set(['[cls]', '[sep]', '.', ',', 'is', 'the', 'a', 'an', 'with', 'and', 'of', 'to', 
                         'in', 'by', 'for', 'has', 'been', 'are', 'have', 'that', 'this', 'it', '##s'])
        
        for text in df['text']:
            # Tokenize
            tokens = tokenizer.tokenize(text.lower())
            # Count terms (excluding stopwords)
            for token in tokens:
                if token not in stopwords:
                    domain_terms[token] += 1
        
        # Get top terms
        top_terms = domain_terms.most_common(30)
        
        # Create bar plot of domain terms
        plt.figure(figsize=(12, 8))
        terms = [term for term, count in top_terms]
        counts = [count for term, count in top_terms]
        
        # Create color mapping for different term types
        colors = []
        gene_markers = ['gene', '##1', '##2', '##3', '##4', '##k', 'shank', 'brc', 'mec', 'disc']
        disease_markers = ['cancer', 'disorder', 'disease', 'autism', 'spectrum', 'schizophrenia']
        relation_markers = ['associated', 'linked', 'related', 'connection']
        
        for term in terms:
            if any(marker in term for marker in gene_markers):
                colors.append('#1f77b4')  # Blue for genes
            elif any(marker in term for marker in disease_markers):
                colors.append('#d62728')  # Red for diseases
            elif any(marker in term for marker in relation_markers):
                colors.append('#2ca02c')  # Green for relations
            else:
                colors.append('#7f7f7f')  # Gray for others
        
        # Create plot
        plt.bar(range(len(terms)), counts, color=colors)
        plt.xticks(range(len(terms)), terms, rotation=45, ha='right')
        plt.title('Top 30 Domain-Specific Terms in Training Data')
        plt.xlabel('Terms')
        plt.ylabel('Frequency')
        
        # Create legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1f77b4', label='Gene-related'),
            Patch(facecolor='#d62728', label='Disease-related'),
            Patch(facecolor='#2ca02c', label='Relation-related'),
            Patch(facecolor='#7f7f7f', label='Other')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        vocab_plot_path = os.path.join(plots_dir, 'DAP_domain_vocabulary.png')
        plt.savefig(vocab_plot_path)
        plt.close()
        print(f"✓ Domain vocabulary plot saved to {vocab_plot_path}")
    except Exception as e:
        print(f"Error analyzing domain vocabulary: {e}")

# Main execution
def main():
    print(f"\nGenerating scientific visualization plots for domain adaptive pretraining...\n")
    print(f"Results will be saved to: {plots_dir}")
    
    # Execute all visualization functions
    train_loss = plot_training_loss()
    plot_perplexity(train_loss)
    plot_learning_rate()
    # plot_epoch_metrics()
    
    # These require loading the model which might be heavyweight
    
    plot_token_prediction()
    analyze_domain_vocabulary()
    
    print(f"\nVisualization complete! All plots saved to: {plots_dir}")

if __name__ == "__main__":
    main()