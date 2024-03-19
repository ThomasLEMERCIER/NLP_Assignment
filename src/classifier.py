from typing import List

import torch
import torch.nn as nn
import pandas as pd

from model import Baseline, load_tokenizer
from train import train
from dataset import TermPolarityDataset
from torch.utils.data import DataLoader


class Classifier:
    """
    The Classifier: complete the definition of this class template by providing a constructor (i.e. the
    __init__() function) and the 2 methods train() and predict() below. Please do not change the signature
    of these methods
     """



    ############################################# complete the classifier class below
    
    def __init__(self):
        """
        This should create and initilize the model. Does not take any arguments.
        
        """
        self.tokenizer = load_tokenizer() 
  
    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """

        # Load the data
        train_dataset = TermPolarityDataset(train_filename, self.tokenizer)
        train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Define the model
        model = Baseline(len(train_dataset.mapping), 3, embed_dim=128, hidden_dim=512)
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)

        train(model, train_dl, optimizer, criterion, scheduler, device, 10)

    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """
