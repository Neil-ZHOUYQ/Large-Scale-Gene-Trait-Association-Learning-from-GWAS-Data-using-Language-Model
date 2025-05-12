from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from transformers import DebertaV2Tokenizer
import torch
import torch.nn.functional as F

# Replace 'biobert-base-cased-v1.1' with the actual model name
#model_name = "dmis-lab/biobert-base-cased-v1.2"
#model_name = "dmis-lab/biobert-v1.1"
model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
#model_name = "KISTI-AI/Scideberta-full"

# Train for 3 epoch with 1-to-1 trait-gene assoc. (GWAS)
#model_path = "/mnt/home/kenneth/out/nlp/biomedbert/DAP/241015_170728_E3_B32_LR2e-05"
# Train for 3 epoch with 1-to-1 and 1-to-many trait-gene assoc (GWAS).
#model_path = "/mnt/home/kenneth/out/nlp/biomedbert/DAP/241023_105320_E3_B32_LR2e-05"
# Train for 3 epoch with 1-to-1 and 1-to-many trait-gene assoc (ASD + GWAS).
#model_path = "/mnt/home/kenneth/out/nlp/biomedbert/DAP/241111_090405_E3_B32_LR2e-05_WD0.01"
# Train for 9 epoch with 1-to-1 and 1-to-many trait-gene assoc (ASD + GWAS).
model_path = "/mnt/home/kenneth/out/nlp/biomedbert/DAP/241112_082615_E9_B32_LR2e-05_WD0.01"

# Train for 3+6 epoch with the same dataset as model 241023_105320_E3_B32_LR2e-05
#model_path = "/mnt/home/kenneth/out/nlp/biomedbert/DAP/241030_092855_E6_B32_LR2e-05"
# Fine-tune for classification based on DAP model (241023_105320_E3_B32_LR2e-05)
# model_path = "/mnt/home/kenneth/out/nlp/biomedbert/finetune/241026_225932_E3_B64_LR2e-05"



# Function declaration
def get_embedding(word, model, tokenizer):
    # Tokenize the word
    inputs = tokenizer(word, return_tensors="pt").to("cuda")
    
    # Get the model output
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Extract the embedding of the first token (which represents the word)
    last_hidden_state = outputs.hidden_states[-1]
    word_embedding = last_hidden_state[0, 1:-1].mean(dim=0)
    
    return word_embedding


def get_word_embedding(sentence, word, model, tokenizer):
    # Tokenize the sentence
    inputs = tokenizer(sentence, return_tensors="pt").to("cuda")

    # Find the position of the word in the tokenized sentence
    word_tokens = tokenizer.tokenize(word)
    input_ids = inputs['input_ids'][0]
    word_start = None
    for i in range(len(input_ids)):
        if tokenizer.convert_ids_to_tokens(input_ids[i:i+len(word_tokens)]) == word_tokens:
            word_start = i
            break

    if word_start is None:
        raise ValueError(f"Word '{word}' not found in the sentence")

    # Get the model output
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Extract the embeddings of the word tokens
    last_hidden_state = outputs.hidden_states[-1]
    word_embedding = last_hidden_state[0, word_start:word_start+len(word_tokens)].mean(dim=0)

    return word_embedding

# Calculate the similarity score between embeddings
def compare_embeddings(emb1, emb2):
    return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()


# Define the function to measure the similarity between gene and trait using cosine similarity
def get_cosine_similarity(feature_vec_1, feature_vec_2):    
    return cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0]


# Load the tokenizer and model from HuggingFace repository
# Pre-requesties : pip install tiktoken sentencepiece

tokenizer = AutoTokenizer.from_pretrained(model_name)
#tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_path).to("cuda")
#model = AutoModel.from_pretrained(model_name).to("cuda")


# Save model to investigate the vocabulary
#tokenizer.save_pretrained("/mnt/home/kenneth/out/nlp/foundation_model/sciDeBERtav2")

# Load the tokenizer and model from local folder
#tokenizer = BertTokenizer.from_pretrained(model_path)
#model = BertForMaskedLM.from_pretrained(model_path)

# Tokenize a trait
#trait="schizophrenia"
#trait="lipid measurement"
trait="Autism Spectrum Disorder (ASD)"

# Genes associated with SCZ (Suggested by Claude)
#case_genes = ['DISC1', 'NRG1', 'DTNBP1', 'COMT', 'RGS4', 'GRM3', 'PRODH', 'ZNF804A', 'AKT1', 'DAOA']
# Genes not likely to be associated with SCZ
#ctrl_genes = ['HBB', 'BRCA1', 'CFTR', 'LEP', 'PAH', 'GJB2', 'F8', 'APOE', 'HEXA', 'HFE']

# Genes associated with SCZ (Suggested by ChatGPT)
#case_genes = ["DISC1", "NRG1", "DTNBP1","DAOA","COMT","ZNF804A","CACNA1C","ANK3","GRIN2A","TCF4"]
# Genes not likely to be associated with SCZ
#ctrl_genes = ["ACTB", "GAPDH", " MYH7", "ALB", "INS", "CFTR", "BRCA1", "COL1A1", "TTN", "HBB"]

# Genes associated with lipid measurement (Suggested by ChatGPT)
#case_genes = ["APOB", "LDLR", "PCSk9", "LPA", "SORT1", "ANGPTL3", "CETP", "APOE", "HMGCR", "ABCA1"]
#ctrl_genes = ["BRCA1", "TP53", "CFTR", "HTT", "FMR1", "DMD", "HBB", "PAH", "G6PD", "F8"]

# Case genes extract from SFARI Gene (Top genes ranked by no. of reports)
# Ctrl genes associated with SCZ (Suggested by ChatGPT)
case_genes = ["SHANK3", "MECP2", "SCN2A","NRXN1", "CHD8", "PTEN", "SYNGAP1", "SCN1A", "CNTNAP2", "ARID1B"]
ctrl_genes = ["TP53", "BRCA1", "CFTR", "HBB", "FBN1", "G6PC", "MYH7", "PKD1", "COL1A1", "DMD"]

#genes = case_genes + ctrl_genes
genes = case_genes + ctrl_genes

# Get the embeddings of the trait
#trait_embeddings = get_embedding(trait, model, tokenizer)
context = f"{trait} is associated with some genes in a GWAS study."
trait_embeddings = get_word_embedding(context, trait, model, tokenizer)


scores = []
for g in genes:
    context = f"Gene {g} is associated with {trait}."
    #context = f"Gene {g} is significantly associated with {trait} in a GWAS study."
    #gene_embeddings = get_embedding(g, model, tokenizer)
    print(context)
    gene_embeddings = get_word_embedding(context, g, model, tokenizer)
    

    # Compare the similarity between gene and trait embedding
    #similar_score = get_cosine_similarity(gene_embeddings.detach().numpy(), trait_embeddings.detach().numpy())
    similar_score = compare_embeddings(gene_embeddings, trait_embeddings)
    scores.append(similar_score)
    
results = dict((key, value) for key, value in zip(genes, scores))
results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

print(f"Similarity score between the embedding of trait {trait} and the following genes\n")
print(results)
print("")

# Performance evaluation

# T.test of similarity score between the two groups
case_scores = []     # contains similarity scores of case genes and the trait 
for key in case_genes:
    case_scores.append(results[key])
ctrl_scores = []     # contains similarity scores of control genes and the trait
for key in ctrl_genes:
    ctrl_scores.append(results[key])
print(f"case gene score: {case_scores}")
print(f"ctrl gene score: {ctrl_scores}\n")

from scipy import stats

t_stat, p_value = stats.ttest_ind(case_scores, ctrl_scores)
print(f"P-value of rejecting the null hypothesis (mean scores between two groups are the same) = {p_value}\n") #should be greatly significant 


from sklearn import metrics

# Accuracy of classification

predict_case_gene = list(results.keys())[0:10]
predict_ctrl_gene = list(results.keys())[10:20]

print(f"predict_case_gene={predict_case_gene}")
print(f"predict_ctrl_gene={predict_ctrl_gene}\n")

tot_gene = len(genes)
predicted = [0 for i in range(tot_gene)]
for i in range(tot_gene):
    if genes[i] in predict_case_gene:
        predicted[i] = 1     #predicted is a list aligning with all the genes, if the gene is case-predicted, the predicted list[] will be 1



actual = [1 for i in range(int(tot_gene/2))]
actual = actual + [0 for i in range(int(tot_gene/2))] #actual list is half 1 and half 2, representing actual case/ctrl statistics

print(f"Actual    = {actual}")
print(f"Predicted = {predicted}\n")

confusion_matrix = metrics.confusion_matrix(actual, predicted)
print("Confusion Matrix [[TP, FP], [FN, TN]]")
print(confusion_matrix)
print("")

rocauc = metrics.roc_auc_score(actual, predicted) #quatifiable evaluate the prediction
Accuracy = metrics.accuracy_score(actual, predicted)
Precision = metrics.precision_score(actual, predicted)
Sensitivity = metrics.recall_score(actual, predicted)
Specificity = metrics.recall_score(actual, predicted, pos_label=0)
F1_score = metrics.f1_score(actual, predicted)

print("Classification performance based on the ranking of similarity scores")
print(f"AUC={rocauc:.2f}\nAccuracy={Accuracy}\nPrecision={Precision}\nSensitivity={Sensitivity}\nSpecificity={Specificity}\nF1_score={F1_score}\n")

