#%%

import os.path
import datetime, pytz

# Set local time and directories
timezone = pytz.timezone("Asia/Hong_Kong")
model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext" 
root_path = "/mnt/home/zhouyuqi/bert02/out/nlp/biomedbert"
#case_dataset_path = os.path.join(root_path, "datasets", "case", "gene_trait_assoc.20241009.tsv")
case_dataset_path = os.path.join(root_path, "datasets", "case", "gene_trait_assoc.GWAS_GTR5.3_ASD.tsv")
#ctrl_dataset_path = os.path.join(root_path, "datasets", "ctrl", "five_fold", "ctrl_assoc.txt")
#ctrl_dataset_path = os.path.join(root_path, "datasets", "ctrl", "parallel_run", "ctrl_assoc.sentence.txt")
ctrl_dataset_path = os.path.join(root_path, "datasets", "ctrl", "parallel_run", "ctrl_train_dataset.GWAS_ASD.20250509.txt")
#model_path =  os.path.join(root_path, "DAP/241015_170728_E3_B32_LR2e-05")
#model_path =  os.path.join(root_path, "DAP/241023_105320_E3_B32_LR2e-05")
model_path =  os.path.join(root_path, "DAP/250508_022350_E9_B32_LR2e-05_WD0.01")


# number gpus
num_gpus = "2"
# batch size for training and eval
batch_size = 64
# max learning rate
max_lr = 2e-05
# learning schedule
#lr_schedule_fn = "linear"
# warmup steps
warmup_steps = 1500
# number of epochs
epochs = 4
# weight_decay
weight_decay = 0.01
# Metric for best model
best_metric ="eval_loss"
# Seed for model finetuning
train_seed=777


early_stopping_patience = 2  # Training stops if no improvement for this many evaluations
early_stopping_threshold = 0.01  # Minimum relative improvement to be considered significant (1%)


current_date = datetime.datetime.now(tz=timezone)
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}_{current_date.strftime('%X').replace(':','')}"
#run_name = f"{datestamp}_geneformer_30M_L{num_layers}_emb{num_embed_dim}_SL{max_input_size}_E{epochs}_B{geneformer_batch_size}_LR{max_lr}_LS{lr_schedule_fn}_WU{warmup_steps}_O{optimizer}_DS{num_gpus}"
run_name = f"{datestamp}_E{epochs}_B{batch_size}_LR{max_lr}"
training_output_dir = f"{root_path}/finetune/{run_name}/"
logging_dir = f"{root_path}/finetune/{run_name}/"
model_output_dir = os.path.join(training_output_dir, "models/")



# %%
# Read data from GWAS catalog database and create the training dataset.
import pandas as pd
from sklearn.model_selection import train_test_split

def create_sentences(row):
    gene = row['GENE']
    trait = row['TRAIT']
    sentence = f"Gene {gene} is associated with {trait}."
    return sentence

# Prepare the training data for case item (label = 1, i.e. TRUE)
data = pd.read_csv(case_dataset_path, sep="\t")
#data = data.head(3000)
print(data.head(3))
#%%
case_data = pd.DataFrame()
case_data['text'] = data.apply(create_sentences, axis=1)
case_data['label'] = 1
print(case_data.head(3))

# Prepare the training data for ctrl item (label = 0, i.e. FALSE)
lines = []
with open(ctrl_dataset_path, 'r') as file:
    for line in file:
        # Strip newline characters and append to the list
        lines.append(line.strip())

ctrl_data = pd.DataFrame(lines, columns=['text'])
#ctrl_data = ctrl_data.head(3000)
ctrl_data['label'] = 0
print(ctrl_data.head(3))

print(f"Total case #: {len(case_data)}")
print(f"Total ctrl #: {len(ctrl_data)}")

# Merge two dataframe into one and shuffle
data = pd.concat([case_data, ctrl_data], ignore_index=True)
data = data.sample(frac=1)
print(f"Total training sample #: {len(data)}")    #data: a df with text, label

# Free the memory
del case_data, ctrl_data

# Split the data into train and test sets
train_texts, eval_texts, train_labels, eval_labels = train_test_split(data['text'].tolist(), 
                                                                      data['label'].tolist(),
                                                                      test_size=0.1, 
                                                                      random_state=42)

# Further split the test sets into validation and test tests
validation_texts, test_texts, validation_labels, test_labels = train_test_split(eval_texts,
                                                                      eval_labels,
                                                                      test_size=0.5,
                                                                      random_state=42)


# %%
# Create training and validation dataset

import torch
from transformers import BertTokenizer
from datasets import Dataset

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)

class StatementDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        statement, label = self.data[idx]
        encoding = self.tokenizer(statement, padding='max_length', truncation=True, max_length=512, return_tensors='pt') #the output of tokenizer
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)                 #for every data, stored in these 3 parts
        }

tokenized_train_dataset = StatementDataset(list(zip(train_texts, train_labels)))
tokenized_validation_dataset = StatementDataset(list(zip(validation_texts, validation_labels)))
tokenized_test_dataset = StatementDataset(list(zip(test_texts, test_labels)))

#print("Tokenized Train dataset:")
#print(tokenized_train_dataset[0])

print("\n")
print(f"Total item in train dataset: {len(tokenized_train_dataset)}")
print(f"Total item in validation dataset: {len(tokenized_validation_dataset)}")
print(f"Total item in test dataset: {len(tokenized_test_dataset)}")

#%%
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}



#%%
# Model fine-tuning

import torch
from transformers import BertForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback

#model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2,
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2,
                                                      attn_implementation="sdpa")
                                                      #torch_dtype=torch.float16, 


# Unfreeze the model's weights before fine-tuning
for param in model.parameters():
    param.requires_grad = True

training_args = TrainingArguments(
    output_dir=model_output_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=warmup_steps,
    learning_rate=max_lr,
    weight_decay=weight_decay,
    logging_dir=logging_dir,
    logging_strategy='steps',
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model=best_metric, #
    greater_is_better=False,
    #fp16=True,
    save_total_limit=2,
    seed=train_seed,
)

early_stopping_callback =  EarlyStoppingCallback(
    early_stopping_patience = early_stopping_patience,
    early_stopping_threshold = early_stopping_threshold
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset, #tokenized , token, attention mask, label
    eval_dataset=tokenized_validation_dataset,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback],
)

# Fine-tune the model
trainer.train()


#%%
import json

# Access the metrics for each epoch
print(trainer.state.log_history)

# Save to a JSON file
trainingloss_file = os.path.join(logging_dir, 'training_loss.json')
with open(os.path.join(trainingloss_file), 'w') as file:
    json.dump(trainer.state.log_history, file, indent=4)

# Save the fine-tuned model
trainer.save_model(training_output_dir)

print(f"Training loss saved to {trainingloss_file}")
print(f"Pre-trained model saved to {training_output_dir} folder")

#%%

# Evaluate the model performance by independent test dataset
# Evaluate the model on the test set
test_result = trainer.evaluate(eval_dataset=tokenized_test_dataset)
print(f"Accuracy: {test_result['eval_accuracy']:.4f}\n")
print(f"Precision: {test_result['eval_precision']:.4f}\n")
print(f"Recall: {test_result['eval_recall']:.4f}\n")
print(f"F1-score: {test_result['eval_f1']:.4f}\n")

# Print evaluation metrics
with open(os.path.join(logging_dir, "test_model_metrics.txt"), 'w') as file:
    file.write(f"Accuracy: {test_result['eval_accuracy']:.4f}\n")
    file.write(f"Precision: {test_result['eval_precision']:.4f}\n")
    file.write(f"Recall: {test_result['eval_recall']:.4f}\n")
    file.write(f"F1-score: {test_result['eval_f1']:.4f}\n")

