import json
import matplotlib.pyplot as plt
import os
import math

# Path setup based on 2_DomainAdaptivePretaining.py
root_path = "/mnt/home/zhouyuqi/bert02/out/nlp/biomedbert"
run_name = "250508_022350_E9_B32_LR2e-05_WD0.01"  # e.g. "241112_082615_E9_B32_LR2e-05_WD0.01"bert02/out/nlp/biomedbert/DAP/250508_022350_E9_B32_LR2e-05_WD0.01
logging_dir = f"{root_path}/DAP/{run_name}/"
trainingloss_file = os.path.join(logging_dir, 'training_loss.json')


#1. Training Loss Curve
# Load training history
with open(trainingloss_file, 'r') as file:
    log_history = json.load(file)

# Extract training loss
train_loss = [x['loss'] for x in log_history if 'loss' in x]
steps = range(1, len(train_loss) + 1)

plt.figure(figsize=(12, 6))
plt.plot(steps, train_loss, color='blue', linewidth=2)
plt.xlabel('Training Steps')
plt.ylabel('Masked Language Modeling Loss')
plt.title('Domain Adaptive Pretraining Loss')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(logging_dir, 'DAP_loss_curve.png'), dpi=300)
plt.close()

#2. Perplexity Plot (Key Metric for Language Models)
# Calculate perplexity (e^loss)
perplexity = [math.exp(loss) for loss in train_loss]

plt.figure(figsize=(12, 6))
plt.plot(steps, perplexity, color='green', linewidth=2)
plt.xlabel('Training Steps')
plt.ylabel('Perplexity')
plt.title('Model Perplexity During Domain Adaptation')
plt.yscale('log')  # Log scale makes perplexity trends clearer
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(logging_dir, 'DAP_perplexity.png'), dpi=300)
plt.close()



#3. Learning Rate Schedule
# Extract learning rate
learning_rates = [x['learning_rate'] for x in log_history if 'learning_rate' in x]
lr_steps = range(1, len(learning_rates) + 1)

plt.figure(figsize=(12, 4))
plt.plot(lr_steps, learning_rates, color='purple', linewidth=2)
plt.xlabel('Training Steps')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(logging_dir, 'DAP_learning_rate.png'), dpi=300)
plt.close()





#4. Gene-Trait Token Prediction Visualization
from transformers import BertTokenizer, BertForMaskedLM
import torch
import numpy as np

model_path = os.path.join(root_path, "DAP", run_name)
tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
model = BertForMaskedLM.from_pretrained(model_path)
model.eval()

# Sample sentences for visualization
examples = [
    "Gene BRCA1 is associated with breast cancer.",
    "Autism spectrum disorder is associated with SHANK3.",
    "Schizophrenia has been linked to various genes including DISC1."
]

plt.figure(figsize=(15, 10))

for idx, sentence in enumerate(examples):
    # Tokenize sentence
    tokens = tokenizer.tokenize(sentence)
    
    # Create a visualization for each token being masked
    masks = []
    predictions = []
    
    for i in range(len(tokens)):
        if tokens[i] not in ['[CLS]', '[SEP]', '.']:
            # Create masked version of sentence
            masked_tokens = tokens.copy()
            masked_tokens[i] = '[MASK]'
            masked_text = tokenizer.convert_tokens_to_string(masked_tokens)
            
            # Get model prediction
            inputs = tokenizer(masked_text, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get prediction for masked token
            mask_idx = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero()[0, 1]
            probs = torch.nn.functional.softmax(outputs.logits[0, mask_idx], dim=0)
            top_5 = torch.topk(probs, 5)
            
            original_token_id = tokenizer.convert_tokens_to_ids(tokens[i])
            original_token_prob = probs[original_token_id].item()
            
            masks.append(tokens[i])
            predictions.append(original_token_prob)
    
    # Plot for this sentence
    plt.subplot(len(examples), 1, idx+1)
    plt.bar(masks, predictions, color='skyblue')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
    plt.title(f"Example {idx+1}: {sentence}")
    plt.ylabel("Prediction Probability")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

plt.suptitle("Token Prediction Probabilities After Domain Adaptation", fontsize=16)
plt.subplots_adjust(top=0.9)
plt.savefig(os.path.join(logging_dir, 'DAP_token_prediction.png'), dpi=300)
plt.close()



# 5. Training vs Validation Loss Comparison (Epochs)
# Load test metrics
test_metrics_file = os.path.join(logging_dir, "test_model_metrics.txt")
test_perplexity = None

with open(test_metrics_file, 'r') as file:
    for line in file:
        if "Perplexity:" in line:
            test_perplexity = float(line.split(":")[1].strip())

# Group training loss by epoch
steps_per_epoch = len(train_loss) // 9  # Assuming 9 epochs as in your script
epoch_avg_loss = []
epoch_perplexity = []

for i in range(9):
    start_idx = i * steps_per_epoch
    end_idx = (i + 1) * steps_per_epoch if i < 8 else len(train_loss)
    epoch_loss = sum(train_loss[start_idx:end_idx]) / (end_idx - start_idx)
    epoch_avg_loss.append(epoch_loss)
    epoch_perplexity.append(math.exp(epoch_loss))

epochs = range(1, 10)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(epochs, epoch_perplexity, 'bo-', label='Training Perplexity')
if test_perplexity:
    plt.axhline(y=test_perplexity, color='r', linestyle='-', label=f'Test Perplexity: {test_perplexity:.2f}')
plt.xlabel('Epochs')
plt.ylabel('Perplexity')
plt.title('Training vs Test Perplexity')
plt.xticks(epochs)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(logging_dir, 'DAP_epoch_perplexity.png'), dpi=300)
plt.close()



# 6. Domain-Specific Vocabulary Analysis
import pandas as pd
from collections import Counter

# Load dataset to extract domain-specific terms
dataset_fn = os.path.join(root_path, "datasets", "case", "case_train_dataset.GWAS_ASD.20241107.txt.gz")
import gzip

# Function to extract terms
def extract_domain_terms(tokenizer, dataset_fn):
    domain_terms = Counter()
    
    with gzip.open(dataset_fn, 'rt', encoding='utf-8') as f:
        df = pd.read_csv(f, sep="\t")
    
    # Process each text example
    for text in df['text']:
        # Tokenize
        tokens = tokenizer.tokenize(text)
        # Count terms (excluding punctuation, common words)
        for token in tokens:
            if token not in ['[CLS]', '[SEP]', '.', ',', 'is', 'the', 'a', 'an', 'with']:
                domain_terms[token] += 1
    
    return domain_terms

# Get domain terms
domain_terms = extract_domain_terms(tokenizer, dataset_fn)
top_30_terms = dict(domain_terms.most_common(30))

# Plot domain vocabulary distribution
plt.figure(figsize=(15, 8))
plt.bar(top_30_terms.keys(), top_30_terms.values(), color='teal')
plt.xlabel('Domain Terms')
plt.ylabel('Frequency')
plt.title('Top 30 Domain-Specific Terms in Training Data')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(logging_dir, 'DAP_domain_vocabulary.png'), dpi=300)
plt.close()



#7. Attention Visualization for Gene-Trait Pairs
from bertviz import head_view
import torch

# Function to visualize attention
def visualize_attention(model, tokenizer, sentence, layer=11, heads=None):
    inputs = tokenizer(sentence, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Get attention matrices
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Select attention heads and layer
    attention = outputs.attentions[layer]
    if heads is not None:
        attention = attention[:, heads]
    
    # Save attention matrices and tokens for visualization
    attention_data = {
        'all': {
            'attn': attention.cpu().numpy(),
            'left_text': tokens,
            'right_text': tokens
        }
    }
    
    # Use bertviz to visualize (or create your own heatmap)
    # For simple script, we'll create our own heatmap
    
    plt.figure(figsize=(12, 10))
    plt.imshow(attention[0, 0].cpu(), cmap='viridis')
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    plt.title(f"Attention Pattern for: {sentence}")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(logging_dir, f'DAP_attention_{layer}.png'), dpi=300)
    plt.close()

# Visualize attention for a sample sentence
visualize_attention(
    model, 
    tokenizer, 
    "Gene BRCA1 is strongly associated with breast cancer."
)

