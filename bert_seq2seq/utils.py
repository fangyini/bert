import torch
from bert_seq2seq.seq2seq_model import Seq2SeqModel

def load_bert(word2ix, tokenizer=None, model_name="roberta", model_class="seq2seq", target_size=0):
    if model_class == "seq2seq":
        bert_model = Seq2SeqModel(word2ix, model_name=model_name, tokenizer=tokenizer)
        return bert_model
    else :
        raise Exception("model_name_err")


