import os
import torch
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForMaskedLM
import numpy as np
import re # For creating safe filenames

# --- Configuration ---
# Assume these are defined globally or passed as arguments
# Example paths (replace with your actual paths)
root_path = "/mnt/home/zhouyuqi/bert02/out/nlp/biomedbert" # Path to base output directory
run_name = "250508_022350_E9_B32_LR2e-05_WD0.01" # Specific DAP run directory name
plots_dir = os.path.join(root_path, "DAP", run_name, "visualization_detailed") # Output directory for plots

os.makedirs(plots_dir, exist_ok=True)

# Style settings for plots
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
# --- End Configuration ---


def plot_token_prediction_per_sentence():
    """
    Visualize token prediction probability for each key term
    within the specific context of each example sentence.
    Generates one plot per example sentence.
    """
    print("Loading model for detailed token prediction analysis...")
    try:
        # Load tokenizer and model
        # Using the original tokenizer is standard practice
        tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
        # Load the domain-adapted model for masked LM
        model_path = os.path.join(root_path, "DAP", run_name)
        print(f"Loading domain-adapted model from: {model_path}")
        model = BertForMaskedLM.from_pretrained(model_path)
        model.eval() # Set model to evaluation mode
        # If using GPU: model.to('cuda')

        print("Model loaded successfully.")

        # Sample sentences for visualization
        examples = [
            "BRCA1 is associated with breast cancer.",
            "ASD is associated with SHANK3.",
            "Schizophrenia has been linked to various genes including DISC1."
        ]

        # Define the key terms to look for within the examples
        key_terms_dict = {
            "Gene terms": [ "brca1", "shank3", "disc1"],
            "Relation terms": ["associated", "linked"],
            "Disease terms": ["cancer", "ASD", "schizophrenia"] # Added 'disorder'
        }
        # Flatten the dictionary into a set for easier checking
        all_key_terms = set(term for sublist in key_terms_dict.values() for term in sublist)

        # --- Data Collection ---
        # Store results per sentence: {sentence: {term: probability}}
        sentence_results = {}

        print("Analyzing sentences...")
        for sentence_idx, sentence in enumerate(examples):
            print(f"  Processing sentence {sentence_idx+1}/{len(examples)}: '{sentence}'")
            sentence_results[sentence] = {}
            tokens = tokenizer.tokenize(sentence)
            # Convert sentence tokens to IDs (needed for finding original IDs later)
            sentence_input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # Find which key terms are in this sentence
            terms_in_sentence = []
            # Iterate through all defined key terms
            for term in all_key_terms:
                 # Check if term (as a whole word or part) is in the tokenized sentence
                 # This requires careful handling if terms can be substrings of tokens
                 # A simple check might be sufficient for these examples:
                 if term.lower() in sentence.lower():
                     # More robust check: see if the tokenized term exists in the sentence tokens
                     term_tokens = tokenizer.tokenize(term)
                     for i in range(len(tokens) - len(term_tokens) + 1):
                         if tokens[i:i+len(term_tokens)] == term_tokens:
                             terms_in_sentence.append((term, i, len(term_tokens))) # Store term, start index, length
                             break # Found the first occurrence

            if not terms_in_sentence:
                print(f"    No key terms found in sentence: {sentence}")
                continue

            # Perform prediction for each found key term *in this sentence's context*
            for term, term_start_idx, term_len in terms_in_sentence:
                print(f"      Testing term: '{term}'")
                # Create masked version of the *original* sentence tokens
                masked_tokens = tokens.copy()
                original_term_token_ids = []
                for j in range(term_len):
                    token_index = term_start_idx + j
                    masked_tokens[token_index] = '[MASK]'
                    original_term_token_ids.append(sentence_input_ids[token_index]) # Store original ID

                # Prepare input for the model
                # Need to add [CLS] and [SEP] for BERT
                input_tokens = ['[CLS]'] + masked_tokens + ['[SEP]']
                masked_input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                input_tensor = torch.tensor([masked_input_ids])
                # If using GPU: input_tensor = input_tensor.to('cuda')

                # Get model prediction
                with torch.no_grad():
                    outputs = model(input_tensor) # Shape: [batch, seq_len, vocab_size]

                # Calculate probability of the original term tokens
                total_prob = 0
                mask_indices_in_model_input = [k for k, token in enumerate(input_tokens) if token == '[MASK]']

                if len(mask_indices_in_model_input) != term_len:
                     print(f"      Warning: Mismatch in mask count for term '{term}'. Skipping.")
                     continue # Should not happen with this logic, but safety check

                for j in range(term_len):
                    # Get the index of the j-th mask token in the model's input sequence
                    mask_model_idx = mask_indices_in_model_input[j]
                    # Get logits for this specific masked position
                    logits = outputs.logits[0, mask_model_idx] # Shape: [vocab_size]
                    # Convert logits to probabilities
                    probs = torch.nn.functional.softmax(logits, dim=0)
                    # Get the probability of the original token
                    original_token_id = original_term_token_ids[j]
                    original_token_prob = probs[original_token_id].item()
                    total_prob += original_token_prob

                # Average probability across the term's tokens
                avg_prob = total_prob / term_len if term_len > 0 else 0
                sentence_results[sentence][term] = avg_prob
                print(f"        Avg probability for '{term}': {avg_prob:.4f}")

        # --- Plotting ---
        print("\nGenerating plots...")
        for i, sentence in enumerate(examples):
            predictions = sentence_results.get(sentence)

            if not predictions or not predictions.items():
                print(f"  Skipping plot for sentence {i+1} (no valid predictions).")
                continue

            plot_terms = list(predictions.keys())
            plot_probs = list(predictions.values())

            # Create a safe filename from the sentence
            safe_sentence_part = re.sub(r'[^a-zA-Z0-9]+', '_', sentence[:30]).strip('_')
            plot_filename = f'token_prediction_sentence_{i+1}_{safe_sentence_part}.png'
            plot_path = os.path.join(plots_dir, plot_filename)

            plt.figure(figsize=(max(6, len(plot_terms) * 1.5), 5)) # Adjust width based on terms
            bars = plt.bar(plot_terms, plot_probs, color='#1f77b4', zorder=2)
            plt.ylabel("Prediction Probability of Original Term")
            plt.title(f"Prediction Probabilities in:\n'{sentence}'", fontsize=MEDIUM_SIZE)
            plt.ylim(0, 1.05) # Extend slightly for labels
            plt.xticks(rotation=15, ha='right') # Rotate labels slightly if needed

            # Add value labels on top of bars
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}", va='bottom', ha='center') # Draw text slightly above bar

            # Add reference line
            plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, zorder=1, label='0.5 Threshold')
            plt.legend(loc='upper right')

            plt.grid(axis='y', linestyle='--', alpha=0.6) # Keep horizontal grid lines
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"  ✓ Plot saved to {plot_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

# --- Execution ---
if __name__ == "__main__":
    plot_token_prediction_per_sentence()
# --- End Execution ---