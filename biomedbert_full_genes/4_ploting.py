# visualize_finetuning.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
import glob

# Set paths based on 4_finetuning.py structure
root_path = "/mnt/home/zhouyuqi/bert/kenneth_data/out/nlp/biomedbert"
finetune_path = os.path.join(root_path, "finetune")

# Find the most recent model directory
model_dirs = [d for d in os.listdir(finetune_path) if os.path.isdir(os.path.join(finetune_path, d))]
latest_model_dir = sorted(model_dirs)[-1]  # Get most recent by name
training_dir = os.path.join(finetune_path, latest_model_dir)
model_dir = os.path.join(training_dir, "models")


# Create output directory for plots
plots_dir = os.path.join(training_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

print(f"Analyzing model: {training_dir}")
print(f"Saving plots to: {plots_dir}")

# Check for checkpoints directories
checkpoint_dirs = sorted(glob.glob(os.path.join(model_dir, "checkpoint-*")))
if checkpoint_dirs:
    print(f"Found {len(checkpoint_dirs)} checkpoint directories")
    # The latest checkpoint directory should contain trainer_state.json
    trainer_state_file = os.path.join(checkpoint_dirs[-1], "trainer_state.json")
else:
    print("No checkpoint directories found")
    trainer_state_file = None

# 1. Plot Training and Validation Loss (if available)
def plot_training_progress():
    if trainer_state_file and os.path.exists(trainer_state_file):
        print(f"Found trainer_state.json at {trainer_state_file}")
        with open(trainer_state_file, 'r') as f:
            trainer_state = json.load(f)
            
        # Extract training history
        log_history = trainer_state.get('log_history', [])
        
        if log_history:
            # Extract training loss
            train_loss = []
            train_steps = []
            eval_loss = []
            eval_steps = []
            
            for entry in log_history:
                if 'loss' in entry:
                    train_loss.append(entry['loss'])
                    train_steps.append(entry['step'])
                if 'eval_loss' in entry:
                    eval_loss.append(entry['eval_loss'])
                    eval_steps.append(entry['step'])
            
            if train_loss:
                plt.figure(figsize=(10, 6))
                plt.plot(train_steps, train_loss, label='Training Loss', color='blue')
                if eval_loss:
                    plt.plot(eval_steps, eval_loss, label='Validation Loss', color='red', marker='o')
                plt.xlabel('Training Steps')
                plt.ylabel('Loss')
                plt.title('Training and Validation Loss')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.savefig(os.path.join(plots_dir, 'training_loss.png'), dpi=300)
                plt.close()
                print("✓ Training loss plot saved")
            
            # Plot evaluation metrics
            metrics = ['eval_accuracy', 'eval_precision', 'eval_recall', 'eval_f1']
            available_metrics = {}
            for metric in metrics:
                values = []
                steps = []
                for entry in log_history:
                    if metric in entry:
                        values.append(entry[metric])
                        steps.append(entry['step'])
                if values:
                    available_metrics[metric] = (steps, values)
            
            if available_metrics:
                plt.figure(figsize=(12, 6))
                for metric, (steps, values) in available_metrics.items():
                    plt.plot(steps, values, marker='o', label=metric.replace('eval_', ''))
                plt.xlabel('Training Steps')
                plt.ylabel('Score')
                plt.title('Evaluation Metrics')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.savefig(os.path.join(plots_dir, 'evaluation_metrics.png'), dpi=300)
                plt.close()
                print("✓ Evaluation metrics plot saved")
        else:
            print("No training history found in trainer_state.json")
    else:
        print("trainer_state.json not found, skipping training progress plots")

# 2. Test set evaluation and visualization
def evaluate_test_set():
    from transformers import BertForSequenceClassification, BertTokenizer, Trainer
    import pandas as pd
    from torch.utils.data import Dataset
    
    # Load the model and tokenizer
    model_dir_real = checkpoint_dirs[-1]
    model = BertForSequenceClassification.from_pretrained(model_dir_real)
    tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    
    # Define datasets paths (from 4_finetuning.py)
    case_dataset_path = os.path.join(root_path, "datasets", "case", "gene_trait_assoc.GWAS_GTR5.3_ASD.tsv")
    ctrl_dataset_path = os.path.join(root_path, "datasets", "ctrl", "parallel_run", "ctrl_train_dataset.GWAS_ASD.20241107.txt")
    
    # Define the dataset class (same as in 4_finetuning.py)
    class StatementDataset(torch.utils.data.Dataset):
        def __init__(self, data):
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
    
    # Create the same test dataset as in 4_finetuning.py
    def create_sentences(row):
        gene = row['GENE']
        trait = row['TRAIT']
        return f"Gene {gene} is associated with {trait}."
    
    # Prepare test data
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
        from sklearn.model_selection import train_test_split
        data = pd.concat([case_data, ctrl_data], ignore_index=True)
        data = data.sample(frac=1, random_state=42)
        
        # Split data same way as in 4_finetuning.py
        train_texts, eval_texts, train_labels, eval_labels = train_test_split(
            data['text'].tolist(), data['label'].tolist(), test_size=0.1, random_state=42)
        
        validation_texts, test_texts, validation_labels, test_labels = train_test_split(
            eval_texts, eval_labels, test_size=0.5, random_state=42)
        
        # Use test set for evaluation
        test_dataset = StatementDataset(list(zip(test_texts, test_labels)))
        
        # Initialize trainer for evaluation
        trainer = Trainer(model=model)
        
        # Run evaluation
        results = trainer.predict(test_dataset)
        
        # Extract predictions and metrics
        predictions = np.argmax(results.predictions, axis=1)
        probabilities = torch.nn.functional.softmax(torch.tensor(results.predictions), dim=1)[:, 1].numpy()
        
        # Save prediction data
        prediction_data = {
            "texts": test_texts,
            "labels": test_labels,
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(),
            "metrics": results.metrics
        }
        
        with open(os.path.join(plots_dir, 'test_predictions.json'), 'w') as f:
            json.dump(prediction_data, f)
        
        # Create confusion matrix
        cm = confusion_matrix(test_labels, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Associated", "Associated"])
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(ax=ax, cmap='Blues')
        plt.title('Confusion Matrix on Test Set')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()
        print("✓ Confusion matrix saved")
        
        # Create ROC curve
        fpr, tpr, _ = roc_curve(test_labels, probabilities)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (area = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(plots_dir, 'roc_curve.png'), dpi=300)
        plt.close()
        print("✓ ROC curve saved")
        
        # Create Precision-Recall curve
        precision, recall, _ = precision_recall_curve(test_labels, probabilities)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 8))
        plt.plot(recall, precision, color='green', lw=2,
                 label=f'PR curve (area = {pr_auc:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.savefig(os.path.join(plots_dir, 'precision_recall_curve.png'), dpi=300)
        plt.close()
        print("✓ Precision-Recall curve saved")
        
        # Create prediction distribution
        plt.figure(figsize=(10, 6))
        plt.hist([probabilities[i] for i in range(len(test_labels)) if test_labels[i] == 1], 
                alpha=0.5, bins=20, label='True Associations')
        plt.hist([probabilities[i] for i in range(len(test_labels)) if test_labels[i] == 0], 
                alpha=0.5, bins=20, label='False Associations')
        plt.xlabel('Prediction Probability')
        plt.ylabel('Count')
        plt.title('Distribution of Model Predictions')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, 'prediction_distribution.png'), dpi=300)
        plt.close()
        print("✓ Prediction distribution saved")
        
        # Print test metrics
        print("\nTest Metrics:")
        for key, value in results.metrics.items():
            print(f"  {key}: {value:.4f}")
        
        return True
    
    except Exception as e:
        print(f"Error evaluating test set: {e}")
        return False

# Main execution
if __name__ == "__main__":
    print("\nGenerating visualization plots for fine-tuning...\n")
    
    # Plot training progress
    plot_training_progress()
    
    # Evaluate on test set
    if evaluate_test_set():
        print("\nTest evaluation and visualization completed successfully")
    
    print(f"\nAll plots saved to: {plots_dir}")