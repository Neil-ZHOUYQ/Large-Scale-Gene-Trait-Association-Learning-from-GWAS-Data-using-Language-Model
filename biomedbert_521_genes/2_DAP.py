#%%
import datetime, pytz
import gzip
import os

# set local time/directories
timezone = pytz.timezone("Asia/Hong_Kong")

model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

root_path = "/mnt/home/zhouyuqi/bert02/out/nlp/biomedbert"
dataset_fn = os.path.join(root_path, "datasets", "case", "case_train_dataset.GWAS_ASD.20250508.txt.gz")
#model_path = os.path.join(root_path, "DAP", "241023_105320_E3_B32_LR2e-05")

# set training parameters
# total number of examples in adaption training dataset:
# num_examples = 300
# number gpus
num_gpus = "2"
# number cpus
# num_cpus = "40"
# batch size for training and eval
batch_size = 32
# max learning rate
max_lr = 2e-05
# learning schedule
#lr_schedule_fn = "linear"
# warmup steps (default is ~10-20% of total training steps)
warmup_steps = 1500 # 15% * (500,000 samples/32 batch_size) * 3 Epochs / 2 GPU - (1500 warmup steps in scheduler)
# number of epochs
epochs = 9
# weight_decay (default is 0.01, higher value can be useful for more complex datasets or when overfitting is a significant concern.)
weight_decay = 0.01
# Metric for best model
#best_metric ="accuracy"
# Seed for model finetuning
train_seed=777

current_date = datetime.datetime.now(tz=timezone)
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}_{current_date.strftime('%X').replace(':','')}"
#run_name = f"{datestamp}_geneformer_30M_L{num_layers}_emb{num_embed_dim}_SL{max_input_size}_E{epochs}_B{geneformer_batch_size}_LR{max_lr}_LS{lr_schedule_fn}_WU{warmup_steps}_O{optimizer}_DS{num_gpus}"
run_name = f"{datestamp}_E{epochs}_B{batch_size}_LR{max_lr}_WD{weight_decay}"
training_output_dir = f"{root_path}/DAP/{run_name}/"
logging_dir = f"{root_path}/DAP/{run_name}/"
model_output_dir = os.path.join(training_output_dir, "models/")

# Create directories if they don't exist
os.makedirs(model_output_dir, exist_ok=True)
os.makedirs(logging_dir, exist_ok=True)


#%%
import pandas as pd

# Read the tab-separated file (with p-values)
with gzip.open(dataset_fn, 'rt', encoding='utf-8') as f:
    df_all = pd.read_csv(f, sep="\t")

print(df_all.head(3))
print(f"......\nTotal no. of sentences loaded: {len(df_all):,}\n")
# For fast testing purpose
#df_all = df_all.head(500)



#%%

df_train = df_all.sample(frac=0.95, random_state=777)
df_test = df_all.drop(df_train.index)
#df_tmp = df_all.drop(df_train.index)
#df_validate = df_tmp.sample(frac=0.5, random_state=777)
#df_test = df_tmp.drop(df_validate.index)

print(f"Size of train dataset: {len(df_train)}, Size of test dataset: {len(df_test)}")

del df_all
#del df_tmp, df_all


#%%
# Prepare the dataset for Domain Adaptive Pretraining (DAP) using gene-trait
# asscocation from GWAS catalog 
from datasets import Dataset, DatasetDict

dataset = DatasetDict({
    'train': Dataset.from_pandas(df_train[['text']]),
    #'validate': Dataset.from_pandas(df_validate[['text']]),
    'test': Dataset.from_pandas(df_test[['text']])
})                                                             # DatasetDict: dict-like structure so store train, test, validate datasets
                                                #Hugging Face datasets are designed to work seamlessly with their transformers library, allowing direct input into tokenizers and models.

#%%
from transformers import BertTokenizer, DataCollatorForLanguageModeling

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)

# Fix: Proper tokenization without applying DataCollator in the map function
def tokenize_function(examples):
    return tokenizer(
        examples['text'], 
        padding='max_length', 
        truncation=True, 
        max_length=512, 
        return_special_tokens_mask=True
    )

tokenized_dataset = dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=['text']
)

# Create data collator for MLM - to be used by the Trainer
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=True, 
    mlm_probability=0.15
)


#%%

# Define a function to compute metrics
import numpy as np
import itertools
import evaluate

# Define the evaluation function
def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    labels = list(itertools.chain(*labels))
    predictions = list(itertools.chain(*predictions))

    return metric.compute(predictions=predictions, references=labels)



#%%
import torch
from transformers import Trainer, TrainingArguments
from transformers import BertForMaskedLM
from transformers import get_scheduler
from torch.optim import AdamW

# Initialize the model
model = BertForMaskedLM.from_pretrained(model_name, 
                                        attn_implementation="sdpa")
                                        #torch_dtype=torch.float16, 

# Fix: Check for CUDA availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()
model.to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir=model_output_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    #per_device_eval_batch_size=batch_size,  # Fixed: Added eval batch size
    warmup_steps=warmup_steps,
    learning_rate=max_lr,
    weight_decay=weight_decay,
    logging_dir=logging_dir,
    logging_steps=500,
    eval_strategy="no",  # do evaluate to save conda space
    save_strategy="epoch",  # Save at the end of each epoch
    #load_best_model_at_end=True,  # Load the best model at the end of training
    #metric_for_best_model=best_metric,  # Use accuracy to determine the best model
    #fp16=True,
    seed=train_seed,  # Fixed: Enable seed for reproducibility
    save_total_limit=2,
    report_to="none",  # Disable wandb logging
)

# Set up the optimizer
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=max_lr)

# Set up the learning rate scheduler
lr_scheduler = get_scheduler(
    'linear',
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,  # Fixed: Use the variable instead of hard-coded value
    num_training_steps=(len(tokenized_dataset['train']) // training_args.per_device_train_batch_size) * training_args.num_train_epochs
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    #eval_dataset=tokenized_dataset["test"]
    data_collator=data_collator,  
    #compute_metrics=compute_metrics,  
    optimizers=(optimizer, lr_scheduler),
)

# Start the fine-tuning process
train_result = trainer.train()



#%%
import json

# Access the metrics for each epoch
print(trainer.state.log_history)

# Save to a JSON file
trainingloss_file = os.path.join(logging_dir, 'training_loss.json')
with open(trainingloss_file, 'w') as file:  # Fixed: Removed redundant os.path.join
    json.dump(trainer.state.log_history, file, indent=4)

# Save the adaptive trained model
trainer.save_model(training_output_dir)

print(f"Training loss saved to {trainingloss_file}")
print(f"Pre-trained model saved to {training_output_dir} folder")


#%%
import math
# Evaluate the model performance by independent test dataset
# Evaluate the model on the test set
test_result = trainer.evaluate(eval_dataset=tokenized_dataset['test'])

# Print evaluation metrics
with open(os.path.join(logging_dir, "test_model_metrics.txt"), 'w') as file:
    # Fixed: Check if eval_accuracy exists in the results
    if 'eval_accuracy' in test_result:
        file.write(f"Accuracy: {test_result['eval_accuracy']:.4f}\n")
    file.write(f"Perplexity: {math.exp(test_result['eval_loss']):.2f}")



#%%
# To visualize the training progress, you can plot the metrics
import matplotlib.pyplot as plt

# Extract training and validation loss
train_loss = [x['loss'] for x in trainer.state.log_history if 'loss' in x]

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
plt.xlabel('Log steps')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

# Save the figure to a file
plt.savefig(os.path.join(logging_dir, 'Train_loss_plot.png'))
plt.close()  # Fixed: Close the plot after saving