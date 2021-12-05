import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from transformers import BertTokenizer, BertModel


class KREModel(nn.Module):
    """ Model for Multi-label classification for Korean Relation Extraction Dataset.
    """
    def __init__(self, args, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        
        self.args = args
        self.pretrained_model = 'datawhales/korean-relation-extraction'
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        
        self.bert = BertModel.from_pretrained(self.pretrained_model, return_dict=True)
        
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model)
        # # entity markers tokens
        # special_tokens_dict = {'additional_special_tokens': ['[E1]', '[/E1]', '[E2]', '[/E2]']}
        # num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)   # num_added_toks: 4
        
        # self.bert.resize_token_embeddings(len(self.tokenizer))
        
        if self.args.mode == "ALLCC":
            self.scale = 4
        elif self.args.mode == "ENTMARK":
            self.scale = 2
            
        self.classifier = nn.Linear(self.bert.config.hidden_size * self.scale, args.n_class)
        
        self.criterion = nn.BCELoss()
        
    def forward(self, input_ids, attention_mask, labels=None):
        batch_size = input_ids.size()[0]
        
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = bert_outputs.last_hidden_state
        
        # 모든 entity marker의 hidden states를 concat
        if self.args.mode == "ALLCC":
            h_start_pos_tensor = (input_ids == 20000).nonzero()
            h_end_pos_tensor = (input_ids == 20001).nonzero()
            t_start_pos_tensor = (input_ids == 20002).nonzero()
            t_end_pos_tensor = (input_ids == 20003).nonzero()
            
            h_start_list = h_start_pos_tensor.tolist()
            h_end_list = h_end_pos_tensor.tolist()
            t_start_list = t_start_pos_tensor.tolist()
            t_end_list = t_end_pos_tensor.tolist()
            
            special_token_idx = []
            
            # special_token_idx example: [[1, 9, 11, 19], [3, 5, 8, 12], ..]
            for h_start, h_end, t_start, t_end in zip(h_start_list, h_end_list, t_start_list, t_end_list):
                special_token_idx.append([h_start[1], h_end[1], t_start[1], t_end[1]])
            
            # concat_state shape: [batch size, hidden size * 4]
            for i, idx_list in enumerate(special_token_idx):
                if i == 0:
                    concat_state = last_hidden_state[i, idx_list].flatten().unsqueeze(0)
                else:
                    concat_state = torch.cat([concat_state, last_hidden_state[i, idx_list].flatten().unsqueeze(0)], dim=0)
            
        elif self.args.mode == "ENTMARK":
            h_start_pos_tensor = (input_ids == 20000).nonzero()
#             h_end_pos_tensor = (input_ids == 20001).nonzero()
            t_start_pos_tensor = (input_ids == 20002).nonzero()
#             t_end_pos_tensor = (input_ids == 20003).nonzero()
            
            h_start_list = h_start_pos_tensor.tolist()
#             h_end_list = h_end_pos_tensor.tolist()
            t_start_list = t_start_pos_tensor.tolist()
#             t_end_list = t_end_pos_tensor.tolist()
            
            special_token_idx = []
        
            # special_token_idx example: [[1, 11], [3, 8], ..]
            for h_start, t_start in zip(h_start_list, t_start_list):
                special_token_idx.append([h_start[1], t_start[1]])
            
            # concat_state shape: [batch size, hidden size * 2]
            for i, idx_list in enumerate(special_token_idx):
                if i == 0:
                    concat_state = last_hidden_state[i, idx_list].flatten().unsqueeze(0)
                else:
                    concat_state = torch.cat([concat_state, last_hidden_state[i, idx_list].flatten().unsqueeze(0)], dim=0)
        
        output = self.classifier(concat_state)
        output = torch.sigmoid(output)
        
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output
