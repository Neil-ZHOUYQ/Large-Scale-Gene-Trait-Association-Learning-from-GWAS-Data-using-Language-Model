# compare_model.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, BertTokenizer, Trainer
from torch.utils.data import Dataset
import glob


model_dir1 = "/mnt/home/zhouyuqi/bert02/out/nlp/biomedbert/finetune/250509_100029_E3_B64_LR2e-05/models/checkpoint-690"
model_dir2 = "/mnt/home/zhouyuqi/bert02/out/nlp/biomedbert/finetune/250510_024241_E4_B64_LR2e-05/models/checkpoint-920"
comparison_dir = "/mnt/home/zhouyuqi/bert02/codes"


# Set paths based on project structure
root_path = "/mnt/home/zhouyuqi/bert02/out/nlp/biomedbert"
# finetune_path = os.path.join(root_path, "finetune")

# # Find the most recent model directory
# model_dirs = [d for d in os.listdir(finetune_path) if os.path.isdir(os.path.join(finetune_path, d))]
# latest_model_dir = sorted(model_dirs)[-1]  # Get most recent by name
# training_dir = os.path.join(finetune_path, latest_model_dir)
# model_dir = os.path.join(training_dir, "models")

# # Create output directory for comparison plots
# comparison_dir = os.path.join(training_dir, "model_comparison")
# os.makedirs(comparison_dir, exist_ok=True)

# print(f"Analyzing model: {training_dir}")
# print(f"Saving comparison plots to: {comparison_dir}")

# # Check for checkpoints directories for the finetuned model
# checkpoint_dirs = sorted(glob.glob(os.path.join(model_dir, "checkpoint-*")))
# if checkpoint_dirs:
#     print(f"Found {len(checkpoint_dirs)} checkpoint directories")
#     # The latest checkpoint directory should contain the finetuned model
#     finetuned_model_dir = checkpoint_dirs[-1]
# else:
#     print("No checkpoint directories found")
#     finetuned_model_dir = os.path.join(training_dir)

# Original BiomedBERT model name
original_model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
# Define datasets paths
case_dataset_path = os.path.join(root_path, "datasets", "case", "gene_trait_assoc.GWAS_GTR5.3_ASD.tsv")
ctrl_dataset_path = os.path.join(root_path, "datasets", "ctrl", "parallel_run", "ctrl_train_dataset.GWAS_ASD.20250509.txt")


def create_sentences(row):
    gene = row['GENE']
    trait = row['TRAIT']
    return f"Gene {gene} is associated with {trait}."


def compute_metrics(predictions, labels):
    predictions = np.argmax(predictions, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    return {
        'accuracy': accuracy, 
        'precision': precision, 
        'recall': recall, 
        'f1': f1
    }


class StatementDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        statement, label = self.data[idx]
        encoding = self.tokenizer(statement, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def prepare_test_dataset():
    """Prepare the test dataset for evaluation."""
    try:
        # Load case data
        case_data = pd.read_csv(case_dataset_path, sep="\t")
        case_data['text'] = case_data.apply(create_sentences, axis=1)
        case_data['label'] = 1

        # Load control data
        lines = []
        with open(ctrl_dataset_path, 'r') as file:
            for line in file:
                lines.append(line.strip())
        ctrl_data = pd.DataFrame(lines, columns=['text'])
        ctrl_data['label'] = 0

        # Combine and shuffle with same seed
        data = pd.concat([case_data, ctrl_data], ignore_index=True)
        data = data.sample(frac=1, random_state=42)
        
        # Split data same way as in 4_finetuning.py
        train_texts, eval_texts, train_labels, eval_labels = train_test_split(
            data['text'].tolist(), data['label'].tolist(), test_size=0.1, random_state=42)
        
        validation_texts, test_texts, validation_labels, test_labels = train_test_split(
            eval_texts, eval_labels, test_size=0.5, random_state=42)
        
        return test_texts, test_labels
        
    except Exception as e:
        print(f"Error preparing test dataset: {e}")
        return None, None


def evaluate_model(model, tokenizer, test_texts, test_labels, model_name="model"):
    """Evaluate a model on the test dataset."""
    try:
        # Create test dataset
        test_dataset = StatementDataset(list(zip(test_texts, test_labels)), tokenizer)
        
        # Initialize trainer for evaluation
        trainer = Trainer(model=model)
        
        # Run evaluation
        results = trainer.predict(test_dataset)
        
        # Extract predictions and metrics
        predictions = results.predictions
        computed_metrics = compute_metrics(predictions, test_labels)
        
        # Get probabilities for positive class
        probabilities = torch.nn.functional.softmax(torch.tensor(predictions), dim=1)[:, 1].numpy()
        
        return {
            "predictions": np.argmax(predictions, axis=1).tolist(),
            "probabilities": probabilities.tolist(),
            "metrics": computed_metrics
        }
        
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return None


def generate_comparison_plots(original_results, finetuned_results, test_texts, test_labels):
    """Generate comparison plots between original and finetuned models."""
    try:
        # Extract data
        original_probs = original_results["probabilities"]
        finetuned_probs = finetuned_results["probabilities"]
        original_preds = original_results["predictions"]
        finetuned_preds = finetuned_results["predictions"]
        
        # 1. Confusion Matrices
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Original model confusion matrix
        cm_original = confusion_matrix(test_labels, original_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_original, display_labels=["Not Associated", "Associated"])
        disp.plot(ax=axes[0], cmap='Blues')
        axes[0].set_title('Model1 BiomedBERT Confusion Matrix')    #Modified!
        
        # Finetuned model confusion matrix
        cm_finetuned = confusion_matrix(test_labels, finetuned_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_finetuned, display_labels=["Not Associated", "Associated"])
        disp.plot(ax=axes[1], cmap='Blues')
        axes[1].set_title('Model2 BiomedBERT Confusion Matrix')   #Modified!
        
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, 'confusion_matrix_comparison.png'), dpi=300)
        plt.close()
        print("✓ Confusion matrix comparison saved")
        
        # 2. ROC Curves
        plt.figure(figsize=(10, 8))
        
        # Original model ROC curve
        fpr_original, tpr_original, _ = roc_curve(test_labels, original_probs)
        roc_auc_original = auc(fpr_original, tpr_original)
        plt.plot(fpr_original, tpr_original, color='blue', lw=2, 
                 label=f'Model1 BiomedBERT (AUC = {roc_auc_original:.3f})')                            #Modified!
        
        # Finetuned model ROC curve
        fpr_finetuned, tpr_finetuned, _ = roc_curve(test_labels, finetuned_probs)
        roc_auc_finetuned = auc(fpr_finetuned, tpr_finetuned)
        plt.plot(fpr_finetuned, tpr_finetuned, color='red', lw=2, 
                 label=f'Model2 BiomedBERT (AUC = {roc_auc_finetuned:.3f})')                            #Modified!
        
        # Reference line
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(comparison_dir, 'roc_curve_comparison.png'), dpi=300)
        plt.close()
        print("✓ ROC curve comparison saved")
        
        # 3. Precision-Recall Curves
        plt.figure(figsize=(10, 8))
        
        # Original model PR curve
        precision_original, recall_original, _ = precision_recall_curve(test_labels, original_probs)
        pr_auc_original = auc(recall_original, precision_original)
        plt.plot(recall_original, precision_original, color='blue', lw=2,
                 label=f'Model1 BiomedBERT (AUC = {pr_auc_original:.3f})')                       #Modified!
        
        # Finetuned model PR curve
        precision_finetuned, recall_finetuned, _ = precision_recall_curve(test_labels, finetuned_probs)
        pr_auc_finetuned = auc(recall_finetuned, precision_finetuned)
        plt.plot(recall_finetuned, precision_finetuned, color='red', lw=2,
                 label=f'Model2 BiomedBERT (AUC = {pr_auc_finetuned:.3f})')                      #Modified!
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve Comparison')
        plt.legend(loc='lower left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(comparison_dir, 'precision_recall_curve_comparison.png'), dpi=300)
        plt.close()
        print("✓ Precision-Recall curve comparison saved")
        
        # 4. Prediction Distribution Comparison
        plt.figure(figsize=(12, 8))
        
        # Original model prediction distributions
        plt.hist([original_probs[i] for i in range(len(test_labels)) if test_labels[i] == 1], 
                alpha=0.4, bins=20, color='blue', label='Model1: True Associations')                      #Modified!
        plt.hist([original_probs[i] for i in range(len(test_labels)) if test_labels[i] == 0],               #Modified!
                alpha=0.4, bins=20, color='cyan', label='Model1: False Associations')
        
        # Finetuned model prediction distributions
        plt.hist([finetuned_probs[i] for i in range(len(test_labels)) if test_labels[i] == 1], 
                alpha=0.4, bins=20, color='red', label='Model2: True Associations')                        #Modified!
        plt.hist([finetuned_probs[i] for i in range(len(test_labels)) if test_labels[i] == 0], 
                alpha=0.4, bins=20, color='orange', label='Model2: False Associations')                     #Modified!
        
        plt.xlabel('Prediction Probability')
        plt.ylabel('Count')
        plt.title('Distribution of Model Predictions')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(comparison_dir, 'prediction_distribution_comparison.png'), dpi=300)
        plt.close()
        print("✓ Prediction distribution comparison saved")
        
        # 5. Metrics Comparison Bar Chart
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        original_values = [original_results['metrics'][m] for m in metrics]
        finetuned_values = [finetuned_results['metrics'][m] for m in metrics]
        
        plt.figure(figsize=(10, 8))
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, original_values, width, label='Model1 BiomedBERT', color='blue', alpha=0.7)              #Modified!
        plt.bar(x + width/2, finetuned_values, width, label='Model2 BiomedBERT', color='red', alpha=0.7)              #Modified!
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Performance Metrics Comparison')
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Add value labels on top of bars
        for i, v in enumerate(original_values):
            plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        for i, v in enumerate(finetuned_values):
            plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.ylim(0, 1.15)
        plt.savefig(os.path.join(comparison_dir, 'metrics_comparison.png'), dpi=300)
        plt.close()
        print("✓ Metrics comparison bar chart saved")
        
        # Save comparison results to JSON
        comparison_data = {
            "original_biomedbert": original_results,
            "finetuned_biomedbert": finetuned_results
        }
        
        with open(os.path.join(comparison_dir, 'model_comparison_results.json'), 'w') as f:
            json.dump(comparison_data, f, indent=4)
        
        print("✓ Model comparison results saved to JSON")
        
        return True
        
    except Exception as e:
        print(f"Error generating comparison plots: {e}")
        return False


def compare_models():
    """Compare the original BiomedBERT model with the finetuned model."""
    print("\nComparing original BiomedBERT with finetuned BiomedBERT model...\n")
    
    # Prepare test dataset
    test_texts, test_labels = prepare_test_dataset()
    if test_texts is None or test_labels is None:
        print("Failed to prepare test dataset. Aborting comparison.")
        return
    
    print(f"Prepared test dataset with {len(test_texts)} examples")
    
    # Load tokenizer (same for both models)
    tokenizer = BertTokenizer.from_pretrained(original_model_name)
    
    # 1. Evaluate original BiomedBERT
    print("\nEvaluating original BiomedBERT model...")
    original_model = BertForSequenceClassification.from_pretrained(model_dir1, num_labels=2)
    original_results = evaluate_model(original_model, tokenizer, test_texts, test_labels, "original")
    
    if original_results is None:
        print("Failed to evaluate original BiomedBERT model. Aborting comparison.")
        return
    
    print("\nOriginal BiomedBERT Metrics:")
    for key, value in original_results["metrics"].items():
        print(f"  {key}: {value:.4f}")
    
    # 2. Evaluate finetuned BiomedBERT
    print("\nEvaluating finetuned BiomedBERT model...")
    finetuned_model = BertForSequenceClassification.from_pretrained(model_dir2, num_labels=2)
    finetuned_results = evaluate_model(finetuned_model, tokenizer, test_texts, test_labels, "finetuned")
    
    if finetuned_results is None:
        print("Failed to evaluate finetuned BiomedBERT model. Aborting comparison.")
        return
    
    print("\nFinetuned BiomedBERT Metrics:")
    for key, value in finetuned_results["metrics"].items():
        print(f"  {key}: {value:.4f}")
    
    # 3. Generate comparison plots and save results
    success = generate_comparison_plots(original_results, finetuned_results, test_texts, test_labels)
    
    if success:
        print("\nModel comparison completed successfully!")
        print(f"\nAll comparison plots and results saved to: {comparison_dir}")
    else:
        print("\nFailed to generate some or all comparison plots.")


# Main execution
if __name__ == "__main__":
    compare_models()