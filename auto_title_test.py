import torch
from bert_seq2seq.tokenizer import load_chinese_base_vocab
from bert_seq2seq.utils import load_bert

auto_title_model = "./state_dict/bert_auto_title_model_639.bin"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    vocab_path = "./state_dict/data/vocab"  # roberta模型字典的位置
    model_name = "roberta"  # 选择模型名字
    word2idx = load_chinese_base_vocab(vocab_path=vocab_path)
    # 定义模型
    bert_model = load_bert(word2idx, model_name=model_name)
    bert_model.set_device(device)
    bert_model.eval()
    bert_model.load_all_params(model_path=auto_title_model, device=device)
    test_data = open("./corpus/auto_title/test_clean_src.txt", "r").readlines()#[:3]
    gt = open("./corpus/auto_title/test_clean_tgt.txt", "r").readlines()#[:3]
    for ind in range(len(test_data)):
        print('Ground truth: ' + str(gt[ind][:-1]))
        text = test_data[ind]
        with torch.no_grad():
            res = bert_model.generate(text, beam_size=3)
            print('Testing result: ' + str(res) + '\n')
