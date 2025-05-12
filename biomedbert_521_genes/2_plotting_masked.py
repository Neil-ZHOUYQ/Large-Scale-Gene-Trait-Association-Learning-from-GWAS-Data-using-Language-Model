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
plots_dir = os.path.join(root_path, "DAP", run_name, "visualization_top10") # Output directory for plots

os.makedirs(plots_dir, exist_ok=True)

# Style settings for plots
plt.style.use('ggplot')
SMALL_SIZE = 9 # Slightly smaller font sizes might be needed
MEDIUM_SIZE = 11
BIGGER_SIZE = 13

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE - 1) # Make legend even smaller
plt.rc('figure', titlesize=BIGGER_SIZE)
# --- End Configuration ---


def plot_top10_token_predictions():
    """
    Visualize the top 10 predicted tokens and their probabilities for each
    masked position corresponding to key terms within example sentences.
    Generates one plot per masked term per sentence.
    """
    print("Loading model for Top-10 token prediction analysis...")
    try:
        # Load tokenizer and model
        tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
        model_path = os.path.join(root_path, "DAP", run_name)
        print(f"Loading domain-adapted model from: {model_path}")
        model = BertForMaskedLM.from_pretrained(model_path)
        model.eval() # Set model to evaluation mode
        # If using GPU: model.to('cuda')
        print("Model loaded successfully.")

        # Sample sentences for visualization
        examples = [
            "Gene BRCA1 is associated with breast cancer.",
            "ASD is associated with SHANK3.",
            "Schizophrenia has been linked to various genes including DISC1."
        ]

        # Define the key terms to look for
        key_terms_dict = {
            "Gene terms": ["gene", "brca1", "shank3", "disc1"],
            "Relation terms": ["associated", "linked"],
            "Disease terms": ["cancer", "ASD", "schizophrenia", "breast"]
        }
        all_key_terms = set(term for sublist in key_terms_dict.values() for term in sublist)

        # --- Data Collection ---
        # Store results: {sentence: {term_key: [ {token_pos_info}, ... ]}}
        # token_pos_info = { "original_token": str,
        #                    "top_preds": [(tok_str, prob), ...],
        #                    "original_details": (orig_tok_str, orig_prob) }
        sentence_results = {}

        print("Analyzing sentences and predicting top 10 tokens...")
        for sentence_idx, sentence in enumerate(examples):
            print(f"  Processing sentence {sentence_idx+1}/{len(examples)}: '{sentence}'")
            sentence_results[sentence] = {}
            tokens = tokenizer.tokenize(sentence)
            sentence_input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # Find key terms in this sentence
            terms_in_sentence = []
            processed_indices = set()
            for term in sorted(list(all_key_terms), key=len, reverse=True):
                 term_tokens = tokenizer.tokenize(term)
                 term_len = len(term_tokens)
                 for i in range(len(tokens) - term_len + 1):
                     if tokens[i:i+term_len] == term_tokens and all(idx not in processed_indices for idx in range(i, i+term_len)):
                         terms_in_sentence.append({"term": term, "start": i, "len": term_len})
                         for k in range(i, i+term_len):
                             processed_indices.add(k)

            if not terms_in_sentence:
                print(f"    No key terms found in sentence: {sentence}")
                continue

            terms_in_sentence.sort(key=lambda x: x["start"])

            # Perform prediction for each found key term
            for term_info in terms_in_sentence:
                term = term_info["term"]
                term_start_idx = term_info["start"]
                term_len = term_info["len"]
                term_key = f"{term}_{term_start_idx}" # Unique key for this occurrence

                print(f"      Testing term: '{term}' starting at index {term_start_idx}")
                sentence_results[sentence][term_key] = []

                masked_tokens = tokens.copy()
                original_term_token_ids = []
                original_term_tokens_str = []
                for j in range(term_len):
                    token_index = term_start_idx + j
                    masked_tokens[token_index] = '[MASK]'
                    original_term_token_ids.append(sentence_input_ids[token_index])
                    original_term_tokens_str.append(tokens[token_index])

                input_tokens = ['[CLS]'] + masked_tokens + ['[SEP]']
                masked_input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                input_tensor = torch.tensor([masked_input_ids])
                # If using GPU: input_tensor = input_tensor.to('cuda')

                with torch.no_grad():
                    outputs = model(input_tensor)

                mask_indices_in_model_input = [k for k, token in enumerate(input_tokens) if token == '[MASK]']
                term_mask_model_indices = []
                current_mask_idx_in_sentence = 0
                for k, token in enumerate(input_tokens):
                    if token == '[MASK]':
                        sentence_token_idx = k - 1
                        if sentence_token_idx >= term_start_idx and sentence_token_idx < term_start_idx + term_len:
                           term_mask_model_indices.append(k)
                        current_mask_idx_in_sentence += 1

                if len(term_mask_model_indices) != term_len:
                     print(f"      Warning: Mismatch in mask count for term '{term}' at index {term_start_idx}. Skipping term.")
                     if term_key in sentence_results[sentence]:
                         del sentence_results[sentence][term_key]
                     continue

                for j in range(term_len):
                    mask_model_idx = term_mask_model_indices[j]
                    logits = outputs.logits[0, mask_model_idx]
                    probs = torch.nn.functional.softmax(logits, dim=0)

                    # --- CHANGE HERE: Get Top 10 predictions ---
                    top_k_probs, top_k_indices = torch.topk(probs, 10)
                    top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_indices.tolist())
                    top_10_preds = list(zip(top_k_tokens, [p.item() for p in top_k_probs]))

                    original_token_id = original_term_token_ids[j]
                    original_token_str = original_term_tokens_str[j]
                    original_prob = probs[original_token_id].item()
                    original_details = (original_token_str, original_prob)

                    token_pos_result = {
                        "original_token": original_token_str,
                        "top_preds": top_10_preds, # Store top 10
                        "original_details": original_details
                    }
                    sentence_results[sentence][term_key].append(token_pos_result)
                    print(f"        Processed mask position {j+1}/{term_len} (Original: '{original_token_str}')")


        # --- Plotting ---
        print("\nGenerating Top-10 prediction plots...")
        for sentence, term_predictions in sentence_results.items():
            if not term_predictions: continue

            for term_key, token_pos_results in term_predictions.items():
                if not token_pos_results: continue

                term, term_start_index_str = term_key.rsplit('_', 1)
                num_token_positions = len(token_pos_results)

                all_labels = []
                all_bar_data = []

                for j, pos_result in enumerate(token_pos_results):
                    all_labels.append(f"Pos {j}\nOrig: '{pos_result['original_token']}'")
                    combined_preds = {}
                    for token, prob in pos_result['top_preds']: # Now top 10
                        combined_preds[token] = prob
                    orig_token, orig_prob = pos_result['original_details']
                    combined_preds[orig_token] = orig_prob
                    sorted_preds = sorted(combined_preds.items(), key=lambda item: item[1], reverse=True)
                    all_bar_data.append(sorted_preds) # Can have up to 11 unique entries

                legend_tokens = set()
                original_tokens_set = set()
                for j, pos_data in enumerate(all_bar_data):
                    original_tokens_set.add(token_pos_results[j]['original_token'])
                    for token, prob in pos_data:
                        legend_tokens.add(token)

                sorted_legend_tokens = sorted(list(legend_tokens), key=lambda t: next((p[1] for pos_data in all_bar_data for p in pos_data if p[0] == t), 0), reverse=True)

                # Plotting adjustments for potentially more bars
                num_groups = len(all_labels)
                # Max bars can be up to 11 (top 10 + original if not in top 10)
                max_bars_per_group = max(len(pos_data) for pos_data in all_bar_data)
                # Make bars narrower if there are many
                bar_width = 0.8 / max_bars_per_group if max_bars_per_group > 0 else 0.8
                x = np.arange(num_groups)

                # Increase figure width based on total potential bars
                fig_width = max(8, num_groups * max_bars_per_group * 0.25) # Adjusted factor
                fig, ax = plt.subplots(figsize=(fig_width, 6))

                # Use a colormap with more colors if needed, or cycle through tab10/tab20
                color_map = plt.get_cmap('tab20')
                token_colors = {token: color_map(i % 20) for i, token in enumerate(sorted_legend_tokens)}

                plotted_legend_handles = {}

                for j, pos_data in enumerate(all_bar_data):
                    group_bar_count = len(pos_data)
                    start_offset = -bar_width * (group_bar_count - 1) / 2
                    orig_token_for_pos = token_pos_results[j]['original_token']

                    for k, (token, prob) in enumerate(pos_data):
                        bar_pos = x[j] + start_offset + k * bar_width
                        color = token_colors.get(token, 'grey')
                        is_original = (token == orig_token_for_pos)
                        rect = ax.bar(bar_pos, prob, bar_width, label=token if token not in plotted_legend_handles else "", color=color,
                                      edgecolor='black' if is_original else None,
                                      linewidth=1.5 if is_original else 0.5,
                                      zorder= 3 if is_original else 2)
                        if token not in plotted_legend_handles:
                             plotted_legend_handles[token] = rect[0]

                ax.set_ylabel("Prediction Probability")
                ax.set_title(f"Top-10 Predictions for Masked Term: '{term}'\nin: '{sentence}'", fontsize=MEDIUM_SIZE)
                ax.set_xticks(x)
                ax.set_xticklabels(all_labels)
                ax.set_ylim(0, 1.05)
                # Place legend outside, adjust font size further if needed
                ax.legend(handles=plotted_legend_handles.values(), labels=plotted_legend_handles.keys(), bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=SMALL_SIZE - 2)
                ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

                safe_sentence_part = re.sub(r'[^a-zA-Z0-9]+', '_', sentence[:30]).strip('_')
                safe_term_part = re.sub(r'[^a-zA-Z0-9]+', '_', term).strip('_')
                plot_filename = f'top10_preds_sent{sentence_idx+1}_{safe_sentence_part}_term_{safe_term_part}_idx{term_start_index_str}.png'
                plot_path = os.path.join(plots_dir, plot_filename)

                plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjusted rect slightly for potentially wider legend
                plt.savefig(plot_path, dpi=300)
                plt.close(fig)
                print(f"    ✓ Top-10 plot saved to {plot_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

# --- Execution ---
if __name__ == "__main__":
    plot_top10_token_predictions()
# --- End Execution ---