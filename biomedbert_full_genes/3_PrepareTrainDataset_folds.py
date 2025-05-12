#%%
import os.path
import torch
import sys


# Global constants
model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext" 
rootpath = "/mnt/home/zhouyuqi/bert/out/nlp/biomedbert"
datapath = os.path.join(rootpath, "datasets", "case")
outpath = os.path.join(rootpath, "datasets", "ctrl", "parallel_run")

case_data_fn = os.path.join(datapath, "gene_trait_assoc.GWAS_GTR5.3_ASD.tsv")
ctrl_file_prefix = "ctrl_train_dataset.GWAS_ASD.20250430"
#out_fn = os.path.join(outpath, f"{ctrl_file_prefix}.{fold_no}.txt")

#model_path =  os.path.join(rootpath, "DAP/241015_170728_E3_B32_LR2e-05")
#model_path =  os.path.join(rootpath, "DAP/241023_105320_E3_B32_LR2e-05")
model_path =  os.path.join(rootpath, "DAP/250429_113915_E9_B32_LR2e-05_WD0.01")   #pretrained model!

#CTRL_PER_TRAIT = 17
NONASSOC_THRESHOLD = 0.91
ASSOC_THRESHOLD = 0.96




# Create input texts by combining gene and trait information
def create_text(row):
    gene = row['GENE']
    trait = row['TRAIT']
    return f"Gene {gene} is associated with {trait}."

# Build a 1-to-many mapping from trait to genes
def trait_to_genes(df):
    """
    Convert a DataFrame with 'GENE' and 'TRAIT' columns into a dictionary.
    
    Args:
    df (pd.DataFrame): The input DataFrame with 'GENE' and 'TRAIT' columns.
    
    Returns:
    dict: A dictionary with traits as keys and lists of genes as values.
    """   
    trait_dict = {}

    for trait, group in df.groupby('TRAIT'):
        trait_dict[trait] = list(set(group['GENE']))

    return trait_dict

def gene_to_traits(df):
    """
    Convert a DataFrame with 'GENE' and 'TRAIT' columns into a dictionary.
    
    Args:
    df (pd.DataFrame): The input DataFrame with 'GENE' and 'TRAIT' columns.
    
    Returns:
    dict: A dictionary with gene as keys and lists of traits as values.
    """   
    gene_dict = {}

    for gene, group in df.groupby('GENE'):
        gene_dict[gene] = list(set(group['TRAIT']))

    return gene_dict







# Compare similarity of meaning between traits
from sklearn.metrics.pairwise import cosine_similarity

def get_word_embedding(sentence, word, model, tokenizer): 
    '''
    Get the word embeding in a sentence(here, only one sentence not batch) passsing through
    the adaptive BiomedBert 
    '''
    # Tokenize the sentence
    inputs = tokenizer(sentence, return_tensors="pt").to("cuda")
    #Tokenizes the sentence and converts it into token IDs.
    #Adds special tokens like [CLS] and [SEP].

    # Find the position of the word in the tokenized sentence
    word_tokens = tokenizer.tokenize(word)  #Tokenizes the target word into subwords as per the model's vocabulary.
    input_ids = inputs['input_ids'][0]        #the first row of ids, which should be the first sentence of the input argument sentence
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


def trait_similarity(m, tkz, trait1, trait2,):
    context = f"{trait1} is associated with some genes in a GWAS study."
    trait1_embeddings = get_word_embedding(context, trait1, m, tkz)

    #inputs = tkz(trait1, return_tensors="pt").to("cuda")
    #outputs = m(**inputs)
    #trait1_embeddings = outputs.last_hidden_state.mean(dim=1)
    
    context = f"{trait2} is associated with some genes in a GWAS study."
    trait2_embeddings = get_word_embedding(context, trait2, m, tkz)

    #inputs = tkz(trait2, return_tensors="pt").to("cuda")
    #outputs = m(**inputs)
    #trait2_embeddings = outputs.last_hidden_state.mean(dim=1)

    # Compare the similarity between gene and trait embedding
    feature_vec_1 = trait1_embeddings.cuda().cpu().detach().numpy()
    feature_vec_2 = trait2_embeddings.cuda().cpu().detach().numpy()

    return cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0]


def split_dict_into_folds(data_dict, num_folds):

    '''
    
    '''
    # Ensure the number of folds is at least 1
    if num_folds < 1:
        raise ValueError("Number of folds must be at least 1")
    
    # Convert dictionary items to a list
    items = list(data_dict.items())
    
    # Calculate the size of each fold
    fold_size = len(items) // num_folds
    remainder = len(items) % num_folds
    
    # Create the folds
    folds = []  #folds is a list and every element is a dictionary of gene-trait
    start_index = 0
    for i in range(num_folds):
        end_index = start_index + fold_size + (1 if i < remainder else 0)
        fold = dict(items[start_index:end_index])
        folds.append(fold)
        start_index = end_index    #separate the remainder(n) to 0:remainder(n) interations.
     
    return folds



#%%
import sys
import pandas as pd

# if len(sys.argv) != 3:
#     print(f"Usage: python {sys.argv[0]}.py <n_fold> <fold_no>")
#     exit(0)

# n_fold = int(sys.argv[1])
# fold_no = int(sys.argv[2])

n_fold = 5
fold_no = 1

print(f"Total no. of fold: {n_fold}")
print(f"Fold ID: {fold_no}")                           #pipeline, support run python file from command line, and specify the fold number and which fold to use


# Load the training case dataset
data = pd.read_csv(case_data_fn, sep="\t")
#df_GWAS = df_GWAS.sample(frac=1, random_state=777)
#df_GWAS = df_GWAS.head(491)


#%%
t2g_dict = trait_to_genes(data)
g2t_dict = gene_to_traits(data)

# Split t2g_dict into n_fold
folds = split_dict_into_folds(t2g_dict, n_fold)
onefold = folds[fold_no-1]
print(f"Total records in fold {fold_no}: {len(onefold)} / {len(t2g_dict)}")     #onefold: use one fold of trait_to_genes dictionaries


# %%
from random import shuffle
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_path).to("cuda")
#model = AutoModel.from_pretrained(model_name).to("cuda")

trait_list = list(t2g_dict.keys())  #traits in all the folds

n_trait=0
ctrl_assoc_list = []

out_fn = os.path.join(outpath, f"{ctrl_file_prefix}.{fold_no}.txt")
#for trait, genes in list(onefold.items())[323:330]:
for trait, genes in onefold.items():

    print(f"{n_trait}: Processing trait [{trait}] with {len(genes)} genes.....") # 1: Processing trait [Breast cancer] with 5 genes.....
    n_trait += 1

    #n_candidate_trait = CTRL_PER_TRAIT/len(genes)
    #n_candidate_trait = int(n_candidate_trait) if n_candidate_trait > 1 else 1

    #i = 0
    #j=1
    shuffle(trait_list)
    for candidate_trait in trait_list:
        
        # Skip the same trait
        if candidate_trait == trait: 
            continue

        similarity_score = trait_similarity(model, 
                                            tokenizer, 
                                            trait,
                                            candidate_trait)    #compute the similarity of eone trait in onefold with all the other trait in all-folds
        
        #print(f"{j}: Similarity Score = {similarity_score:.2f}, {trait}  VS  {candidate_trait}")
        #j += 1

        if similarity_score < NONASSOC_THRESHOLD:
            # Successfully find a candidate trait not similar in meaning with the given trait
            #i += 1
            #print(f"Find the {i} candidate trait - [{candidate_trait}] which is unlikely to match with given trait")

            for g in genes:  #the genes of the checking trait in onefold

                # Check if the given gene is not highly associated with a trait that is also
                # associated with that gene.

                traits = g2t_dict.get(g, None)
                scores = [0]

                if traits is not None:
                    scores = [int(trait_similarity(model, tokenizer, candidate_trait, t) > ASSOC_THRESHOLD) for t in traits]

                if (traits is None) or (sum(scores) == 0): 
                    ctrl_assoc_list.append(f"Gene {g} is associated with {candidate_trait}.")
                    #print(f"Successfully create an example - {g} x {candidate_trait}") 

            break

        #if i > n_candidate_trait:
        #    # Finish searching the candidate non-assoc. traits for that given trait and genes.
        #    # Go to the next given trait
        #    break

    if n_trait == 1000:     #save every 1000 traits, to avoid ctrl_assoc_list overload
        with open(out_fn, 'w') as file:
            file.write('\n'.join(map(str, ctrl_assoc_list))+'\n')
        ctrl_assoc_list = []
    elif n_trait%1000 == 0:
        with open(out_fn, 'a') as file:
            file.write('\n'.join(map(str, ctrl_assoc_list))+'\n')
        ctrl_assoc_list = []

# Flush the remaining results to file.
with open(out_fn, 'a') as file:
    file.write('\n'.join(map(str, ctrl_assoc_list))+'\n')

print('Process is done.')

# %%