import os
import sys
import torch
import numpy as np
import pandas as pd
import easydict
import argparse
import json
import requests
import wget
import math

import warnings
warnings.filterwarnings('ignore')

from pororo import Pororo
from itertools import permutations
from transformers import BertTokenizer
from transformers import logging
from .model import KREModel

class KorRE:
    def __init__(self):
        self.args = easydict.EasyDict({'bert_model': 'datawhales/korean-relation-extraction', 'mode': 'ALLCC', 
                                        'n_class': 97, 'max_token_len': 512, 'max_acc_threshold': 0.6})
        self.ner_module = Pororo(task='ner', lang='ko')
        
        logging.set_verbosity_error()

        self.tokenizer = BertTokenizer.from_pretrained(self.args.bert_model)
        
        # # entity markers tokens
        # special_tokens_dict = {'additional_special_tokens': ['[E1]', '[/E1]', '[E2]', '[/E2]']}
        # num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)   # num_added_toks: 4
        
        self.trained_model = self.__get_model()
        
        # relation id to label
        r = requests.get('https://raw.githubusercontent.com/datawhales/Korean_RE/main/data/relation/relid2label.json')
        self.relid2label = json.loads(r.text)
        
        # relation list
        self.relation_list = list(self.relid2label.keys())

        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trained_model = self.trained_model.to(self.device)
        
    def __get_model(self):
        """ 사전학습된 한국어 관계 추출 모델을 로드하는 함수.
        """
        if not os.path.exists('./pretrained_weight'):
            os.mkdir('./pretrained_weight')

        pretrained_weight = './pretrained_weight/pytorch_model.bin'

        if not os.path.exists(pretrained_weight):
            url = 'https://huggingface.co/datawhales/korean-relation-extraction/resolve/main/pytorch_model.bin'
            wget.download(url, out=pretrained_weight)

        trained_model = KREModel(self.args)
        
        trained_model.load_state_dict(torch.load(pretrained_weight))
        trained_model.eval()

        return trained_model
    
    def __idx2relid(self, idx_list):
        """ onehot label에서 1인 위치 인덱스 리스트를 relation id 리스트로 변환하는 함수.
        
        Example:
            relation_list = ['P17', 'P131', 'P530', ...] 일 때,
            __idx2relid([0, 2]) => ['P17', 'P530'] 을 반환.
        """
        label_out = []

        for idx in idx_list:
            label = self.relation_list[idx]
            label_out.append(label)
            
        return label_out

    def pororo_ner(self, sentence: str):
        """ pororo의 ner 모듈을 이용하여 그대로 반환하는 함수.
        """
        return self.ner_module(sentence)
        
    def ner(self, sentence: str):
        """ 주어진 문장에서 pororo의 ner 모듈을 이용해 개체명 인식을 수행하고 각 개체의 인덱스 위치를 함께 반환하는 함수.
        """
        ner_result = self.ner_module(sentence)

        # 인식된 각 개체명의 range 계산
        ner_result = [(item[0], item[1], len(item[0])) for item in ner_result]
        
        modified_list = []
        tmp_cnt = 0

        for item in ner_result:
            modified_list.append((item[0], item[1], [tmp_cnt, tmp_cnt + item[2]]))
            tmp_cnt += item[2]
        
        ent_list = [item for item in modified_list if item[1] != 'O']
        
        return ent_list
    
    def get_all_entity_pairs(self, sentence: str) -> list:
        """ 주어진 문장에서 개체명 인식을 통해 모든 가능한 [문장, subj_range, obj_range]의 리스트를 반환하는 함수.
        
        Example:
            sentence = '모토로라 레이저 M는 모토로라 모빌리티에서 제조/판매하는 안드로이드 스마트폰이다.'
            
        Return: 
            [(('모토로라 레이저 M', 'ARTIFACT', [0, 10]), ('모토로라 모빌리티', 'ORGANIZATION', [12, 21])),
             (('모토로라 레이저 M', 'ARTIFACT', [0, 10]), ('안드로이드', 'TERM', [32, 37])),
             (('모토로라 레이저 M', 'ARTIFACT', [0, 10]), ('스마트폰', 'TERM', [38, 42])),
             (('모토로라 모빌리티', 'ORGANIZATION', [12, 21]), ('모토로라 레이저 M', 'ARTIFACT', [0, 10])),
             (('모토로라 모빌리티', 'ORGANIZATION', [12, 21]), ('안드로이드', 'TERM', [32, 37])),
             (('모토로라 모빌리티', 'ORGANIZATION', [12, 21]), ('스마트폰', 'TERM', [38, 42])),
             (('안드로이드', 'TERM', [32, 37]), ('모토로라 레이저 M', 'ARTIFACT', [0, 10])),
             (('안드로이드', 'TERM', [32, 37]), ('모토로라 모빌리티', 'ORGANIZATION', [12, 21])),
             (('안드로이드', 'TERM', [32, 37]), ('스마트폰', 'TERM', [38, 42])),
             (('스마트폰', 'TERM', [38, 42]), ('모토로라 레이저 M', 'ARTIFACT', [0, 10])),
             (('스마트폰', 'TERM', [38, 42]), ('모토로라 모빌리티', 'ORGANIZATION', [12, 21])),
             (('스마트폰', 'TERM', [38, 42]), ('안드로이드', 'TERM', [32, 37]))]
        """
        # 너무 긴 문장의 경우 500자 이내로 자름
        if len(sentence) >= 500:
            sentence = sentence[:499]
        
        ent_list = self.ner(sentence)

        pairs = list(permutations(ent_list, 2))
        
        return pairs

    def get_all_inputs(self, sentence: str) -> list:
        """ 주어진 문장에서 관계 추출 모델에 통과시킬 수 있는 모든 input의 리스트를 반환하는 함수.
        
        Example:
            sentence = '모토로라 레이저 M는 모토로라 모빌리티에서 제조/판매하는 안드로이드 스마트폰이다.'
            
        Return:
            [['모토로라 레이저 M는 모토로라 모빌리티에서 제조/판매하는 안드로이드 스마트폰이다.', [0, 10], [12, 21]],
            ['모토로라 레이저 M는 모토로라 모빌리티에서 제조/판매하는 안드로이드 스마트폰이다.', [0, 10], [32, 37]],
            ..., ]
        """
        pairs = self.get_all_entity_pairs(sentence)
        return [[sentence, ent_subj[2], ent_obj[2]] for ent_subj, ent_obj in pairs]

    def entity_markers_added(self, sentence: str, subj_range: list, obj_range: list) -> str:
        """ 문장과 관계를 구하고자 하는 두 개체의 인덱스 범위가 주어졌을 때 entity marker token을 추가하여 반환하는 함수.
        
        Example:
            sentence = '모토로라 레이저 M는 모토로라 모빌리티에서 제조/판매하는 안드로이드 스마트폰이다.'
            subj_range = [0, 10]   # sentence[subj_range[0]: subj_range[1]] => '모토로라 레이저 M'
            obj_range = [12, 21]   # sentence[obj_range[0]: obj_range[1]] => '모토로라 모빌리티'
            
        Return:
            '[E1] 모토로라 레이저 M [/E1] 는  [E2] 모토로라 모빌리티 [/E2] 에서 제조/판매하는 안드로이드 스마트폰이다.'
        """
        result_sent = ''
        
        for i, char in enumerate(sentence):
            if i == subj_range[0]:
                result_sent += ' [E1] '
            elif i == subj_range[1]:
                result_sent += ' [/E1] '
            if i == obj_range[0]:
                result_sent += ' [E2] '
            elif i == obj_range[1]:
                result_sent += ' [/E2] '
            result_sent += sentence[i]
        if subj_range[1] == len(sentence):
            result_sent += ' [/E1]'
        elif obj_range[1] == len(sentence):
            result_sent += ' [/E2]'
        
        return result_sent.strip()

    def infer(self, sentence: str, subj_range=None, obj_range=None, entity_markers_included=False):
        """ 입력받은 문장에 대해 관계 추출 태스크를 수행하는 함수.
        """
        # entity marker token이 포함된 경우
        if entity_markers_included:
            # subj, obj name 구하기
            tmp_input_ids = self.tokenizer(sentence)['input_ids']

            if tmp_input_ids.count(20000) != 1 or tmp_input_ids.count(20001) != 1 or \
            tmp_input_ids.count(20002) != 1 or tmp_input_ids.count(20003) != 1:
                raise Exception("Incorrect number of entity marker tokens('[E1]', '[/E1]', '[E2]', '[/E2]').")

            subj_start_id, subj_end_id = tmp_input_ids.index(20000), tmp_input_ids.index(20001)
            obj_start_id, obj_end_id = tmp_input_ids.index(20002), tmp_input_ids.index(20003)

            subj_name = self.tokenizer.decode(tmp_input_ids[subj_start_id+1:subj_end_id])
            obj_name = self.tokenizer.decode(tmp_input_ids[obj_start_id+1:obj_end_id])

            encoding = self.tokenizer.encode_plus(
                             sentence,
                             add_special_tokens=True,
                             max_length=self.args.max_token_len,
                             return_token_type_ids=False,
                             padding='max_length',
                             truncation=True,
                             return_attention_mask=True,
                             return_tensors="pt")

            input_ids = encoding['input_ids'].to(self.device)
            mask = encoding['attention_mask'].to(self.device)

            _, prediction = self.trained_model(input_ids, mask)

            predictions = [prediction.flatten()]
            predictions = torch.stack(predictions).detach().cpu()

            y_pred = predictions.numpy()
            upper, lower = 1, 0
            y_pred = np.where(y_pred > self.args.max_acc_threshold, upper, lower)

            preds_list = []

            for i in range(len(y_pred)):
                class_pred = self.__idx2relid(np.where(y_pred[i]==1)[0])
                preds_list.append(class_pred)

            preds_list = preds_list[0]

            pred_rel_list = [self.relid2label[pred] for pred in preds_list]               

            return [(subj_name, obj_name, pred_rel) for pred_rel in pred_rel_list]

        # entity_markers_included=False인 경우
        else:
            # entity marker가 문장에 포함된 경우
            tmp_input_ids = self.tokenizer(sentence)['input_ids']
            
            if tmp_input_ids.count(20000) >= 1 or tmp_input_ids.count(20001) >= 1 or \
            tmp_input_ids.count(20002) >= 1 or tmp_input_ids.count(20003) >= 1:
                raise Exception("Entity marker tokens already exist in the input sentence. Try 'entity_markers_included=True'.")
            
            # subj range와 obj range가 주어진 경우
            if subj_range is not None and obj_range is not None:
                # add entity markers
                converted_sent = self.entity_markers_added(sentence, subj_range, obj_range)

                encoding = self.tokenizer.encode_plus(
                             converted_sent,
                             add_special_tokens=True,
                             max_length=self.args.max_token_len,
                             return_token_type_ids=False,
                             padding='max_length',
                             truncation=True,
                             return_attention_mask=True,
                             return_tensors="pt")
                
                input_ids = encoding['input_ids'].to(self.device)
                mask = encoding['attention_mask'].to(self.device)
                
                _, prediction = self.trained_model(input_ids, mask)

                predictions = [prediction.flatten()]
                predictions = torch.stack(predictions).detach().cpu()

                y_pred = predictions.numpy()
                upper, lower = 1, 0
                y_pred = np.where(y_pred > self.args.max_acc_threshold, upper, lower)

                preds_list = []

                for i in range(len(y_pred)):
                    class_pred = self.__idx2relid(np.where(y_pred[i]==1)[0])
                    preds_list.append(class_pred)

                preds_list = preds_list[0]

                pred_rel_list = [self.relid2label[pred] for pred in preds_list]

                return [(sentence[subj_range[0]:subj_range[1]], sentence[obj_range[0]:obj_range[1]], pred_rel) for pred_rel in pred_rel_list]

            # 문장만 주어진 경우: 모든 경우에 대해 inference 수행
            else:
                input_list = self.get_all_inputs(sentence)

                converted_sent_list = [self.entity_markers_added(*input_list[i]) for i in range(len(input_list))]

                encoding_list = []

                for i, converted_sent in enumerate(converted_sent_list):
                    tmp_encoding = self.tokenizer.encode_plus(
                                            converted_sent,
                                            add_special_tokens=True,
                                             max_length=self.args.max_token_len,
                                             return_token_type_ids=False,
                                             padding='max_length',
                                             truncation=True,
                                             return_attention_mask=True,
                                             return_tensors="pt"
                                        )
                    encoding_list.append(tmp_encoding)

                predictions = []

                for i, item in enumerate(encoding_list):
                    _, prediction = self.trained_model(
                        item['input_ids'].to(self.device),
                        item['attention_mask'].to(self.device)
                    )

                    predictions.append(prediction.flatten())

                if predictions:
                    predictions = torch.stack(predictions).detach().cpu()

                    y_pred = predictions.numpy()
                    upper, lower = 1, 0
                    y_pred = np.where(y_pred > self.args.max_acc_threshold, upper, lower)

                    preds_list = []
                    for i in range(len(y_pred)):
                        class_pred = self.__idx2relid(np.where(y_pred[i]==1)[0])
                        preds_list.append(class_pred)

                    result_list = []
                    for i, input_i in enumerate(input_list):
                        tmp_subj_range, tmp_obj_range = input_i[1], input_i[2]
                        result_list.append((sentence[tmp_subj_range[0]:tmp_subj_range[1]], sentence[tmp_obj_range[0]:tmp_obj_range[1]], preds_list[i]))

                    final_list = []
                    for tmp_subj, tmp_obj, tmp_list in result_list:
                        for i in range(len(tmp_list)):
                            final_list.append((tmp_subj, tmp_obj, tmp_list[i]))

                    return [(item[0], item[1], self.relid2label[item[2]]) for item in final_list]

                else: return []
