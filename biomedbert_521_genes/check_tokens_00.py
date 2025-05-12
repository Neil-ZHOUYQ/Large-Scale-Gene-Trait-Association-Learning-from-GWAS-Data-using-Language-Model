from transformers import AutoTokenizer

def analyze_word_tokenization(tokenizer_name: str, word_to_check: str):
    """
    Analyzes how a Hugging Face tokenizer processes a specific word.

    Args:
        tokenizer_name (str): The name of the tokenizer model (e.g., "gpt2").
        word_to_check (str): The word to analyze.
    """
    try:
        # Load the tokenizer
        # For GPT-2, add_prefix_space=True is often recommended for tokenizing
        # individual words or segments not at the beginning of a sentence,
        # as GPT-2's BPE merges are sensitive to leading spaces.
        # However, for analyzing a single word's composition,
        # not adding it can sometimes be more illustrative of its raw breakdown.
        # Let's test with both to be clear or stick to default for now.
        # For simplicity and to match original intent, let's use default.
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        print(f"\n--- Analyzing word: '{word_to_check}' with tokenizer: '{tokenizer_name}' ---")

        # Tokenize the word to get token strings
        tokens = tokenizer.tokenize(word_to_check)

        # Encode the word to get input IDs
        # Note: encode() adds special tokens by default if the model uses them
        # for sentence context (e.g., BOS/EOS). For a single word, this might
        # not be desired for pure subword analysis, but tokenize() is cleaner for that.
        # Let's get the IDs corresponding *only* to the word's tokens.
        encoded_ids = tokenizer.convert_tokens_to_ids(tokens)


        print(f"Word: {word_to_check}")
        print(f"Tokens: {tokens}")
        print(f"Encoded IDs: {encoded_ids}")

        # UNK token details
        unk_token = tokenizer.unk_token
        unk_token_id = tokenizer.unk_token_id
        print(f"Tokenizer's UNK token: {unk_token}")
        print(f"Tokenizer's UNK token ID: {unk_token_id}")

        is_unk_present = False
        if unk_token_id is not None and unk_token_id in encoded_ids:
            is_unk_present = True
        # Redundant check if already checking IDs, but good for completeness
        # elif unk_token is not None and unk_token in tokens:
        #     is_unk_present = True

        print(f"Is UNK token present in the tokenization of this word? {is_unk_present}")
        print(f"Number of tokens generated for this word: {len(tokens)}")

        # For BBPE (like gpt2's), a high number of tokens for a single word
        # often indicates novelty or rarity relative to the tokenizer's training data.
        # It doesn't mean an error, but rather a fine-grained breakdown.
        # The threshold of 7 is arbitrary and for demonstration.
        if len(tokens) > 7:
            print(f"Note: This word was broken into many pieces ({len(tokens)} tokens). "
                  f"For a Byte-Level BPE tokenizer like '{tokenizer_name}', "
                  f"this typically suggests the word is rare, novel, or complex "
                  f"compared to its training data, rather than being an 'unknown' that "
                  f"couldn't be processed at all.")
        elif len(tokens) == 0 and len(word_to_check) > 0:
             print(f"Warning: The word was tokenized into an empty list of tokens, "
                   f"which is unusual for non-empty input.")
        elif len(tokens) == 1 and tokens[0] == word_to_check:
            print(f"Note: This word was tokenized as a single unit, suggesting it's likely "
                  f"common or a learned token in the tokenizer's vocabulary.")
        else:
            print(f"Note: This word was broken into {len(tokens)} subword units.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    #tokenizer_name_to_use = "openai-community/gpt2"
    tokenizer_name_to_use ="microsoft/biogpt"

    words_to_test = [
        "SHANK3"
    ]

    for word in words_to_test:
        analyze_word_tokenization(tokenizer_name_to_use, word)