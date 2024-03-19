from torch.utils.data import Dataset
import pandas as pd
from preprocessing import encode_target, encode_category, encode_sentence
import torch

class TermPolarityDataset(Dataset):
    header=["polarity", "category", "subject", "offset", "sentence"]

    def __init__(self, file_path, tokenizer):
        self.tokenizer = tokenizer
    
        self.df = pd.read_csv(file_path, delimiter='\t', header=None)
        self.df.columns = self.header
        self.df = encode_target(self.df)
        self.df, self.mapping = encode_category(self.df)
        self.max_len_sentence = self.get_max_len_sentence()
        self.max_len_subject = self.get_max_len_subject()

        self.df["sentence_input"] = self.df.apply(lambda row: encode_sentence(self.tokenizer, row.sentence, self.max_len_sentence), axis=1)
        self.df["subject_input"] = self.df.apply(lambda row: encode_sentence(self.tokenizer, row.subject, self.max_len_subject), axis=1)

    def get_max_len_sentence(self):
        return max([len(self.tokenizer.encode(sentence, add_special_tokens=True)) for sentence in self.df.sentence.values])
    
    def get_max_len_subject(self):
        return max([len(self.tokenizer.encode(subject, add_special_tokens=True)) for subject in self.df.subject.values])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, item):
        return {
            "sentence_input": {
                "input_ids": self.df.sentence_input[item]["input_ids"].flatten(),
                "attention_mask": self.df.sentence_input[item]["attention_mask"].flatten()
            },
            "subject_input": 
            {
                "input_ids": self.df.subject_input[item]["input_ids"].flatten(),
                "attention_mask": self.df.subject_input[item]["attention_mask"].flatten()
            },
            "category": self.df.category[item],
            "polarity": self.df.polarity[item]
        }
