import jieba
from opencc import OpenCC
from tqdm import tqdm
import torch
import os

opencc = OpenCC("t2s")


class Tokenizer(object):
    """
    定义分词器
    """

    def __init__(self, file="./data.txt", saved_dict="./dicts.bin"):
        """
        初始化
        """
        self.file = file
        self.saved_dict = saved_dict
        self.input_word2idx = {}
        self.input_idx2word = {}
        self.input_dict_len = None
        self.input_embed_dim = 256
        self.input_hidden_size = 256

        self.output_word2idx = {}
        self.output_idx2word = {}
        self.output_dict_len = None

        # 推理时，输出的最大长度
        self.output_max_len = 100
        self.output_embed_dim = 256
        self.output_hidden_size = 256

        # 英文的标点符号
        self.punctuations = [".", ",", "?", "!"]

    def build_dict(self):
        """
        构建字典
            file: 训练数据集的文件
        """
        if os.path.exists(self.saved_dict):
            self.load()
            print("加载本地字典成功")
            return

        input_words = {"<UNK>", "<PAD>"}
        output_words = {"<UNK>", "<PAD>", "<SOS>", "<EOS>"}

        with open(file=self.file, mode="r", encoding="utf8") as f:
            for line in tqdm(f.readlines()):
                if line:
                    input_sentence, output_sentence = line.strip().split("\t")
                    input_sentence_words = self.split_input(input_sentence)
                    output_sentence_words = self.split_output(output_sentence)
                    input_words = input_words.union(set(input_sentence_words))
                    output_words = output_words.union(set(output_sentence_words))
        # 输入字典
        self.input_word2idx = {word: idx for idx, word in enumerate(input_words)}
        self.input_idx2word = {idx: word for word, idx in self.input_word2idx.items()}
        self.input_dict_len = len(self.input_word2idx)

        # 输出字典
        self.output_word2idx = {word: idx for idx, word in enumerate(output_words)}
        self.output_idx2word = {idx: word for word, idx in self.output_word2idx.items()}
        self.output_dict_len = len(self.output_word2idx)

        # 保存
        self.save()
        print("保存字典成功")

    def split_input(self, sentence):
        """
        预处理
            输入：I'm a student.
            输出：['i', 'm', 'a', 'student', '.']
        """
        # 英文变小写
        sentence = sentence.lower()
        # 把缩写拆开为两个词
        sentence = sentence.replace("'", " ")
        # 把标点符号和单词分开
        sentence = "".join(
            [
                " " + char + " " if char in self.punctuations else char
                for char in sentence
            ]
        )
        # 切分单词
        words = [word for word in sentence.split(" ") if word]
        # 返回结果（列表形式）
        return words

    def split_output(self, sentence):
        """
        切分汉语
            输入：我爱北京天安门
            输出：['我', '爱', '北京', '天安门']
        """
        # 繁体转简体
        sentence = opencc.convert(sentence)
        # jieba 分词
        words = jieba.lcut(sentence)
        # 返回结果（列表形式）
        return words

    def encode_input(self, input_sentence, input_sentence_len):
        """
        将输入的句子，转变为指定长度的序列号
        输入：["i", "m", "a", "student"]
        输出：[5851, 4431, 6307, 1254, 2965]
        """
        # 变索引号
        input_idx = [
            self.input_word2idx.get(word, self.input_word2idx.get("<UNK>"))
            for word in input_sentence
        ]
        # 填充PAD
        input_idx = (
                            input_idx + [self.input_word2idx.get("<PAD>")] * input_sentence_len
                    )[:input_sentence_len]

        return input_idx

    def encode_output(self, output_sentence, output_sentence_len):
        """
        将输出的句子，转变为指定长度的序列号
        输入：["我", "爱", "北京", "天安门"]
        输出：[11642, 10092, 5558, 3715, 10552, 1917]
        """
        # 添加结束标识符 <EOS>
        output_sentence = ["<SOS>"] + output_sentence + ["<EOS>"]
        output_sentence_len += 2
        # 变 索引号
        output_idx = [
            self.output_word2idx.get(word, self.output_word2idx.get("<UNK>"))
            for word in output_sentence
        ]
        # 填充 PAD
        output_idx = (
                             output_idx + [self.output_word2idx.get("<PAD>")] * output_sentence_len
                     )[:output_sentence_len]
        return output_idx

    def decode_output(self, pred):
        """
        把预测结果转换为输出文本
        输入：[6360, 7925, 8187, 7618, 1653, 4509]
        输出：['我', '爱', '北京', '<UNK>']
        """
        results = []
        for idx in pred:
            if idx == self.output_word2idx.get("<EOS>"):
                break
            results.append(self.output_idx2word.get(idx))
        return results

    @classmethod
    def subsequent_mask(cls, size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
        return subsequent_mask == 0

    @classmethod
    def make_std_mask(cls, tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Tokenizer.subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask

    def __repr__(self):
        return f"[ Tokenizer ]: 输入字典长度: {self.input_dict_len}, 输出字典长度: {self.output_dict_len}"

    def save(self):
        """
        保存字典
        """
        state_dict = {
            "input_word2idx": self.input_word2idx,
            "input_idx2word": self.input_idx2word,
            "input_dict_len": self.input_dict_len,
            "output_word2idx": self.output_word2idx,
            "output_idx2word": self.output_idx2word,
            "output_dict_len": self.output_dict_len,
        }
        # 保存到文件
        torch.save(obj=state_dict, f=self.saved_dict)

    def load(self):
        """
        加载字典
        """
        if os.path.exists(self.saved_dict):
            state_dict = torch.load(f=self.saved_dict)
            self.input_word2idx = state_dict.get("input_word2idx")
            self.input_idx2word = state_dict.get("input_idx2word")
            self.input_dict_len = state_dict.get("input_dict_len")
            self.output_word2idx = state_dict.get("output_word2idx")
            self.output_idx2word = state_dict.get("output_idx2word")
            self.output_dict_len = state_dict.get("output_dict_len")


def get_tokenizer(file="./data.txt", saved_dict="./dicts.bin"):
    """
    获取分词器
    """
    # 定义分词器
    tokenizer = Tokenizer(file="./data.txt", saved_dict="./dicts.bin")
    tokenizer.build_dict()
    return tokenizer


if __name__ == "__main__":
    tokenizer = get_tokenizer()
    print(tokenizer)
