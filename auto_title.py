## 自动摘要的例子
import sys
import torch 
from tqdm import tqdm
import torch.nn as nn 
from torch.optim import Adam
import pandas as pd
import numpy as np
import os
import json
import time
import bert_seq2seq
from torch.utils.data import Dataset, DataLoader
from bert_seq2seq.tokenizer import Tokenizer, load_chinese_base_vocab, T5PegasusTokenizer
from bert_seq2seq.utils import load_bert


vocab_path = "./state_dict/data/vocab"  # roberta模型字典的位置
#word2idx, keep_tokens = load_chinese_base_vocab(vocab_path=vocab_path)
word2idx = load_chinese_base_vocab(vocab_path=vocab_path)
model_name = "roberta"  # 选择模型名字
#model_path = "./state_dict/roberta_wwm_pytorch_model.bin"  # 模型位置
recent_model_path = "./state_dict/bert_auto_title_model.bin"   # 用于把已经训练好的模型继续训练
model_save_path = "./state_dict/bert_auto_title_model_"
batch_size = 50
lr = 1e-5
train_epoches = 1000


class BertDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """
    def __init__(self, sents_src, sents_tgt) :
        ## 一般init函数是加载所有数据
        super(BertDataset, self).__init__()
        # 读原始数据
        # self.sents_src, self.sents_tgt = read_corpus(poem_corpus_dir)
        self.sents_src = sents_src
        self.sents_tgt = sents_tgt
        
        self.idx2word = {k: v for v, k in word2idx.items()}
        self.tokenizer = T5PegasusTokenizer(word2idx)

    def __getitem__(self, i):
        ## 得到单个数据
        # print(i)
        src = self.sents_src[i]
        tgt = self.sents_tgt[i]
        token_ids, token_type_ids = self.tokenizer.encode(src, tgt, max_length=512)
        output = {
            "token_ids": token_ids,
            "token_type_ids": token_type_ids,
        }
        return output

    def __len__(self):

        return len(self.sents_src)
        
def collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """

    def padding(indice, max_length, pad_idx=0):
        """
        pad 函数
        """
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    target_ids_padded = token_ids_padded[:, 1:].contiguous()

    return token_ids_padded, token_type_ids_padded, target_ids_padded

class Trainer:
    def __init__(self):
        self.sents_src = open("./corpus/auto_title/train_clean_src.txt", "r").readlines()
        self.sents_tgt = open("./corpus/auto_title/train_clean_tgt.txt", "r").readlines()

        self.test_data = open("./corpus/auto_title/test_clean_src.txt", "r").readlines()
        print('Testing data ground truth:')
        print(open("./corpus/auto_title/test_clean_tgt.txt", "r").readlines())

        # 判断是否有可用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: " + str(self.device))
        # 定义模型
        self.bert_model = load_bert(word2idx, model_name=model_name)
        self.bert_model.set_device(self.device)
        ## 加载预训练的模型参数～  
        #self.bert_model.load_pretrain_params(model_path, keep_tokens=keep_tokens)

        # 声明需要优化的参数
        self.optim_parameters = list(self.bert_model.parameters())
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=lr, weight_decay=1e-3)
        # 声明自定义的数据加载器
        dataset = BertDataset(self.sents_src, self.sents_tgt)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    def train(self, epoch):
        # 一个epoch的训练
        self.bert_model.train()
        self.iteration(epoch, dataloader=self.dataloader, train=True)
    
    def save(self, save_path):
        """
        保存模型
        """
        self.bert_model.save_all_params(save_path)
        print("{} saved!".format(save_path))

    def iteration(self, epoch, dataloader, train=True):
        total_loss = 0
        start_time = time.time()
        step = 0
        report_loss = 0
        for token_ids, token_type_ids, target_ids in tqdm(dataloader,position=0, leave=True):
            step += 1

            token_ids = token_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            # 因为传入了target标签，因此会计算loss并且返回
            predictions, loss = self.bert_model(token_ids,
                                                token_type_ids,
                                                labels=target_ids,
                                                )
            report_loss += loss.item()
            # 反向传播
            if train:
                # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播, 获取新的梯度
                loss.backward()
                # 用获取的梯度更新模型参数
                self.optimizer.step()

            # 为计算当前epoch的平均loss
            total_loss += loss.item()

        end_time = time.time()
        spend_time = end_time - start_time
        # 打印训练信息
        print("\nEpoch is " + str(epoch)+". Loss is " + str(total_loss) + ". Spent time is " + str(spend_time))

        self.bert_model.eval()

        #for text in self.test_data:
        #    print(self.bert_model.generate(text, beam_size=3))
        if (epoch+1) % 10 == 0:
            print('Testing result:')
            print(self.bert_model.generate(self.test_data[0], beam_size=3))
            print(self.bert_model.generate(self.test_data[1], beam_size=3))
            # 保存模型
            self.save(model_save_path + str(epoch) + '.bin')
        # print("loss is " + str(report_loss))
        report_loss = 0
        # self.eval(epoch)
        self.bert_model.train()


if __name__ == '__main__':

    trainer = Trainer()
    for epoch in range(train_epoches):
        # 训练一个epoch
        trainer.train(epoch)
    trainer.save(model_save_path)
