import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn

def load_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased')

def load_model():
    model = BertModel.from_pretrained('bert-base-uncased')
    return model, 768

class Baseline(nn.Module):
    def __init__(self, n_categories, n_classes, embed_dim=128, hidden_dim=512):
        super(Baseline, self).__init__()
        self.model, model_hidden_dim = load_model()
        self.embeddings = nn.Embedding(n_categories, embed_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + model_hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, sentence_input, subject_input, category):
        output_sentence = self.model(sentence_input["input_ids"], sentence_input["attention_mask"]).last_hidden_state[:, 0, :]
        output_target = self.model(subject_input["input_ids"], subject_input["attention_mask"]).last_hidden_state[:, 0, :]
        category = self.embeddings(category)
        x = torch.cat([output_sentence, output_target, category], dim=1)

        return self.classifier(x)
