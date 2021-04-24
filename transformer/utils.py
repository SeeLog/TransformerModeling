import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np
import os
import shutil


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
        true_dist = torch.zeros(x.size()).to(x.device)
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


def attention(query: torch.tensor, key: torch.tensor, value: torch.tensor, mask=None, dropout=None) -> torch.tensor:
    "Compute Simple 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def src_mask(src: torch.tensor, padding_idx=0):
    src_mask = (src != padding_idx).unsqueeze(-2)
    return src_mask


def tgt_mask(tgt: torch.tensor, padding_idx=0):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != padding_idx).unsqueeze(-2)
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
    return tgt_mask


def greedy_decode(model, src: torch.tensor, src_mask: torch.tensor, max_len: torch.tensor, start_symbol: int):
    """
    Greedy Decoding

    :param model: encode()とdecode()が実装されたモデル
    :param src: Source Tensor
    :param src_mask: Source Mask
    :param max_len: Max Length
    :param start_symbol: SOSシンボル

    :return: Greedy DecodingされたTensor
    """
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm


def save_checkpoint(model: nn.Module, filepath: str, is_best: bool, epoch_i: int):
    os.makedirs(filepath, exist_ok=True)
    model_save_path = os.path.join(filepath, 'model_epoch_{}.pt'.format(epoch_i))
    torch.save(model.state_dict(), model_save_path)
    if is_best:
        best_save_path = os.path.join(filepath, 'best_model.pt')
        shutil.copyfile(model_save_path, best_save_path)


def load_checkpoint(model, model_path, device, is_eval=True, is_file=False):
    if is_file:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model.to(device)

    if is_eval:
        model.load_state_dict(torch.load(os.path.join(model_path, 'best_model.pt')))
        model.eval()
        return model.to(device=device)

    model = model.load_state_dict(torch.load(os.path.join(model_path, 'last_model.pt')))
    global_step = torch.load(os.path.join(model_path, 'global_step.pt'))
    return model.to(device=device), global_step