import pandas as pd
import os
import numpy as np

from transformers import AutoTokenizer, AutoModel

import torch.nn as nn
import torch
class ClinicalCausalBert(nn.Module):
    def __init__(self,args):
        super().__init__()
        # Load baseline model
        self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.hidden_size = 768
        self.treatment_num_labels = 2
        self.outcome_num_labels = 1
        if args.outcome_label == 'READMISSION':
            self.outcome_num_labels = 1
        self.propensity_estimator = nn.Sequential(nn.Linear(self.hidden_size,self.treatment_num_labels))
        #+1 for the treatment
        self.outcome_estimator = nn.Sequential(nn.Linear(self.hidden_size+1,self.outcome_num_labels))



    def forward(self,text,t):

        note_input = self.tokenizer(str(text), return_tensors="pt", truncation=True, max_length=500)
        embedding = self.model(**note_input)['last_hidden_state'][0, 0, :].detach().numpy()
        embedding = torch.FloatTensor(embedding)
        propensity_prob = self.propensity_estimator(embedding)
        outcome = self.outcome_estimator(torch.cat([embedding,t]))
        return propensity_prob, outcome