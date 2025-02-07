# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.sentences = []
        # 加载 BERT 分词器
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.schema = self.load_schema(config["schema_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentence = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentence.append(char)
                    labels.append(self.schema[label])
                self.sentences.append("".join(sentence))
                input_ids, attention_mask, new_labels = self.encode_sentence(sentence, labels)
                input_ids = self.padding(input_ids)
                attention_mask = self.padding(attention_mask)
                new_labels = self.padding(new_labels, -1)
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(attention_mask), torch.LongTensor(new_labels)])
        return

    def encode_sentence(self, text, labels, padding=True):
        # 使用 BERT 分词器编码句子，自动添加 [CLS] 和 [SEP]
        encoded_input = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.config["max_length"],
            padding='max_length' if padding else False,
            truncation=True,
            return_attention_mask=True
        )
        input_id = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']

        # 处理标签序列以匹配输入序列
        new_labels = [-1]  # 对应 [CLS] 的标签
        for char, label in zip(text, labels):
            sub_tokens = self.tokenizer.tokenize(char)
            new_labels.append(label)
            if len(sub_tokens) > 1:
                # 对于一个字符被分成多个子词的情况，除第一个子词外，其他子词标签设为 -1
                new_labels.extend([-1] * (len(sub_tokens) - 1))
        new_labels.append(-1)  # 对应 [SEP] 的标签

        return input_id, attention_mask, new_labels

    # 补齐或截断输入的序列，使其可以在一个 batch 内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)

# 加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0 留给 padding 位置，所以从 1 开始
    return token_dict

# 用 torch 自带的 DataLoader 类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../ner_data/train.txt", Config)
