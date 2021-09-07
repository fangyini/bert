'''
数据预处理，读取data_file目录下的所有txt文件，然后删去时间信息，最后在corpus里产生两个txt文件
'''



import csv
import os
import re
import collections
import jieba
import jieba.posseg as pseg
jieba_counter = collections.Counter()
data_file = 'data/train/'
entries = os.listdir(data_file)
length = len(entries)
print('Data length=' + str(length))
f1 = open("../corpus/auto_title/train_clean_src.txt", "w", encoding='utf-8')
for ind in range(length):
    text = entries[ind]
    #print(text)
    try:
        with open(data_file + text, encoding="GB2312") as f:
            data = f.read()
            replaced = re.sub('....年..月..日............', '', data)
            l = replaced.split('\n')
            l = set(l)
            replaced = ''.join(list(l))

            replaced = re.sub('\n', '', replaced)
            l = replaced.split('\t')
            jieba_counter.update(l)
            replaced = re.sub('\t0\t', '', replaced)
            replaced = re.sub('\t', '', replaced)
            f1.write(replaced+'\n')
    except:
        continue
f1.close()

f2=open("../corpus/auto_title/train_clean_tgt.txt", "w", encoding='utf-8')
for ind in range(length):
    text = entries[ind]
    f2.write(text[:-4]+'\n')
f2.close()

data_file = 'data/test/'
entries = os.listdir(data_file)
length = len(entries)
f1 = open("../corpus/auto_title/test_clean_src.txt", "w", encoding='utf-8')
for ind in range(length):
    text = entries[ind]
    try:
        with open(data_file + text, encoding="GB2312") as f:
            data = f.read()
            replaced = re.sub('....年..月..日............', '', data)
            l = replaced.split('\n')
            l = set(l)
            replaced = ''.join(list(l))

            replaced = re.sub('\n', '', replaced)
            l = replaced.split('\t')
            jieba_counter.update(l)
            replaced = re.sub('\t0\t', '', replaced)
            replaced = re.sub('\t', '', replaced)
            f1.write(replaced+'\n')
    except:
        continue
f1.close()

f2=open("../corpus/auto_title/test_clean_tgt.txt", "w", encoding='utf-8')
for ind in range(length):
    text = entries[ind]
    f2.write(text[:-4]+'\n')
f2.close()

with open("data/vocab_to_add_manually.txt", encoding='utf-8') as f:
    manual_token = f.readlines()
manual_token = [re.sub('\n', '', t) for t in manual_token]
jieba_counter.update(manual_token)
with open("data/jieba_dic.txt", "w", encoding='utf-8') as f:
    for k, v in jieba_counter.most_common():
        f.write("{}\n".format(k))
jieba.load_userdict("data/jieba_dic.txt")


vocab_path = "data/vocab"
VOCAB_SIZE = 10000
vocab_counter = collections.Counter()

with open("../corpus/auto_title/train_clean_tgt.txt") as f:
    data = f.read()
    tokens = list(jieba.cut(data, cut_all=False))
    tokens = [t for t in tokens if t != "\n"]  # remove empty
    vocab_counter.update(tokens)

with open("../corpus/auto_title/train_clean_src.txt") as f:
    data = f.read()
    tokens = list(jieba.cut(data, cut_all=False))
    tokens = [t for t in tokens if (t != "\n") and (t != '')]  # remove empty
    vocab_counter.update(tokens)

with open(vocab_path, 'w', encoding='utf-8') as writer:
    for word in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']:
        writer.write(word + '\n')
    for word, count in vocab_counter.most_common(VOCAB_SIZE):
        # writer.write(word + ' ' + str(count) + '\n')
        writer.write(word + '\n')
    print('vocab saved')



