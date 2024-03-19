
def encode_target(df):
    """
    Encodes the target labels to integers
    """
    df["polarity"] = df["polarity"].map({"positive": 0, "neutral": 1, "negative": 2})
    return df

def encode_category(df, mapping=None):
    """
    Encodes the category labels to integers
    """
    if mapping:
        df["category"] = df["category"].map(mapping)
        return df
    unique_categories = df["category"].unique()
    mapping = {category: i for i, category in enumerate(unique_categories)}

    df["category"] = df["category"].map(mapping)
    return df, mapping

def encode_sentence(tokenizer, sentence, max_len):
    return tokenizer(sentence,
                     return_tensors="pt", 
                     truncation=True, 
                     max_length=max_len,
                     padding="max_length",
                     add_special_tokens=False,)
