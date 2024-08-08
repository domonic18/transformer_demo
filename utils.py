import torch
from torch import nn
import time


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
            model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


class SimpleLossCompute(object):
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
                self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))
                / norm
        )
        return sloss.data * norm, sloss


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    # 获取中间表达
    memory = model.encode(src, src_mask)
    # 启动信号
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    # 自回归式生成
    for _ in range(max_len - 1):
        # 获取结果
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        # print(out)
        # 取出最后一步的结果
        prob = model.generator(out[:, -1])
        # 获取概率最大的值
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        # 拼接起来，准备生成下一个词
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


def get_real_input(x, tokenizer):
    """
    将输入转换为字符
    """
    x = x.tolist()
    results = []
    for s in x:
        results.append(
            [
                tokenizer.input_idx2word.get(idx)
                for idx in s
                if idx not in [tokenizer.input_word2idx.get("<PAD>")]
            ]
        )
    return results


def get_real_output(y, tokenizer):
    """
    将预测结果转换为真实结果
    """
    y = y.tolist()
    raw_results = []
    final_results = []
    # 原始输出
    for s in y:
        raw_results.append(
            [tokenizer.output_idx2word.get(idx) for idx in s]
        )
    # 去除 <EOS> <PAD>
    for s in y:
        result = []
        for idx in s:
            if idx == tokenizer.output_word2idx.get("<EOS>"):
                # 遇到 EOS 直接结束
                break
            elif idx == tokenizer.output_word2idx.get("<PAD>"):
                # 遇到 PAD 跳过
                continue
            result.append(tokenizer.output_idx2word.get(idx))

        final_results.append(result)

    return raw_results, final_results


class TrainState(object):
    """
    Track number of steps, examples, and tokens processed
    """

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


def run_epoch(
        data_iter,
        model,
        loss_compute,
        optimizer,
        scheduler,
        mode="train",
        accum_iter=1,
        train_state=TrainState(),
        device="cpu"
):
    """
    Train a single epoch
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, (src, src_mask, tgt, tgt_mask, tgt_y, ntokens) in enumerate(data_iter):
        #
        src = src.to(device=device)
        tgt = tgt.to(device=device)
        tgt_y = tgt_y.to(device=device)
        # src = src.to(device=device)
        out = model.forward(src, tgt, src_mask, tgt_mask)
        loss, loss_node = loss_compute(out, tgt_y, ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += src.shape[0]
            train_state.tokens += ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += ntokens
        tokens += ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                        "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                        + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state
