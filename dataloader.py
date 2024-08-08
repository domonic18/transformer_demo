import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os


class Seq2SeqDataset(Dataset):
    """
    自定义数据集
    """

    def __init__(self, tokenizer, file="./data.txt", part="train"):
        self.tokenizer = tokenizer
        self.file = file
        self.part = part
        self.data = None
        self._load_data()

    def _load_data(self):
        if os.path.exists(fr"./{self.part}.bin"):
            self.data = torch.load(f=fr"./{self.part}.bin")
            print("加载本地数据集成功")
            return

        data = []
        with open(file=self.file, mode="r", encoding="utf-8") as f:
            for line in tqdm(f.readlines()):
                if line:
                    input_sentence, output_sentence = line.strip().split("\t")
                    input_sentence = self.tokenizer.split_input(input_sentence)
                    output_sentence = self.tokenizer.split_output(output_sentence)
                    data.append([input_sentence, output_sentence])
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=0)
        if self.part == "train":
            self.data = train_data
        else:
            self.data = test_data

        # 保存数据
        torch.save(obj=self.data, f=fr"./{self.part}.bin")

    def __getitem__(self, idx):
        """
        返回一个样本
            - 列表格式
            - 内容 + 实际长度
        """

        input_sentence, output_sentence = self.data[idx]
        return (
            input_sentence,
            len(input_sentence),
            output_sentence,
            len(output_sentence),
        )

    def __len__(self):
        return len(self.data)


def collate_fn(batch, tokenizer):
    # 根据 x 的长度来 倒序排列
    # batch = sorted(batch, key=lambda ele: ele[1], reverse=True)
    # 合并整个批量的每一部分
    input_sentences, input_sentence_lens, output_sentences, output_sentence_lens = zip(
        *batch
    )

    # 转索引【按本批量最大长度来填充】
    input_sentence_len = max(input_sentence_lens)
    input_idxes = []
    for input_sentence in input_sentences:
        input_idxes.append(tokenizer.encode_input(input_sentence, input_sentence_len))

    # 转索引【按本批量最大长度来填充】
    output_sentence_len = max(output_sentence_lens)
    output_idxes = []
    for output_sentence in output_sentences:
        output_idxes.append(
            tokenizer.encode_output(output_sentence, output_sentence_len)
        )
    # 转张量 [batch_size, seq_len]  src
    input_idxes = torch.LongTensor(input_idxes)
    # src_mask [batch_size, 1, seq_len]
    input_mask = (input_idxes != tokenizer.input_word2idx.get("<PAD>")).unsqueeze(-2)
    # tgt [batch_size, seq_len]
    output_idxes = torch.LongTensor(output_idxes)
    # tgt [batch_size, seq_len - 1] 去掉最后一个
    output_idxes_in = output_idxes[:, :-1]
    # tgt_y [batch_size, seq_len - 1] 去掉开头 的 SOS
    output_idxes_out = output_idxes[:, 1:]
    # tgt_mask [batch_size, seq_len-1, seq_len-1]
    output_mask = tokenizer.make_std_mask(output_idxes_in, tokenizer.output_word2idx.get("<PAD>"))
    # 记录生成的有效字符
    ntokens = (output_idxes_out != tokenizer.output_word2idx.get("<PAD>")).data.sum()
    # src, src_mask, tgt, tgt_mask, tgt_y, ntokens
    return input_idxes, input_mask, output_idxes_in, output_mask, output_idxes_out, ntokens


def get_dataloader(tokenizer,
                   file=r"./data.txt",
                   part="train",
                   batch_size=1024):
    dataset = Seq2SeqDataset(file=file, tokenizer=tokenizer, part=part)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True if part == "train" else False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )
    return dataloader


if __name__ == "__main__":
    from tokenizer import get_tokenizer
    tokenizer = get_tokenizer()
    dataloader = get_dataloader(tokenizer=tokenizer)
    for src, src_mask, tgt, tgt_mask, tgt_y, ntokens in dataloader:
        # [batch_size, src_max_seq_len]
        print(src.shape)
        # [batch_size, 1, src_max_seq_len]
        print(src_mask.shape)
        # [batch_size, tgt_max_seq_len-1]
        print(tgt.shape)
        # [batch_size, tgt_max_seq_len-1, tgt_max_seq_len-1]
        print(tgt_mask.shape)
        # [batch_size, tgt_max_seq_len-1]
        print(tgt_y.shape)
        # 7784
        print(ntokens)
        break
