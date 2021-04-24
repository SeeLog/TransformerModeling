import torch
from torch.utils.data import DataLoader

from .models import EncoderDecoder
from .optimizer import NoamOpt
from .utils import SimpleLossCompute, LabelSmoothing, get_src_mask, get_tgt_mask

from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict
from tqdm import tqdm as tqdm_orig


class Trainer():
    def __init__(self, model, dataloader, writer):
        pass


class SimpleTransformerTrainer(Trainer):
    def __init__(
        self,
        model: EncoderDecoder,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        writer: Optional[SummaryWriter] = None,
        labelsmooth: Optional[LabelSmoothing] = None,
        tqdm=tqdm_orig,
        device: str="cpu"
    ):
        self.model = model

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.writer = writer

        self.tqdm = tqdm
        self.train_global_step = 0

        self.device = device

        self.model_opt = NoamOpt(model.src_embed[0].d_model, 1, 16000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

        if labelsmooth is None:
            self.labelsmooth = LabelSmoothing(size=model.tgt_embed[0].vocab_size, padding_idx=0, smoothing=0.1)

    def loss_compute(self, x: torch.tensor, y: torch.tensor, norm: int):
        x = self.model.generator(x)
        loss = self.labelsmooth(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        loss.backward()
        if self.model_opt is not None:
            self.model_opt.step()
            self.model_opt.optimizer.zero_grad()
        return loss.item() * norm

    def valid_loss_compute(self, x: torch.tensor, y: torch.tensor, norm: int):
        x = self.model.generator(x)
        loss = self.labelsmooth(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        return loss.item() * norm

    def run_train_epoch(self, epoch_i: int, refresh: int = 100):
        "Standard Training and Logging Method"
        self.model.train()
        total_tokens = 0
        total_loss = 0
        tokens = 0

        with self.tqdm(self.train_dataloader, leave=False, desc="[Train] Epoch: {}".format(epoch_i)) as pber:
            for i, batch in enumerate(pber):
                # batch = MiniBatch(batch.src, batch.tgt, 1)
                src, tgt = batch
                src = src.to(self.device)
                tgt = tgt.to(self.device)

                src_mask = get_src_mask(src=src, padding_idx=0).to(self.device)
                tgt_mask = get_tgt_mask(tgt=tgt, padding_idx=0).to(self.device)
                tgt_y = tgt[:, 1:]
                # 0 == padding_idx
                ntokens = (tgt_y != 0).data.sum()

                out = self.model(src, tgt, src_mask, tgt_mask)
                loss = self.loss_compute(out, tgt_y, ntokens)
                total_loss += loss
                total_tokens += ntokens
                tokens += ntokens
                self.train_global_step += 1

                if i % refresh == 0:
                    pber.set_postfix(ordered_dict={"loss": (float(total_loss) / float(total_tokens))})
                    if self.writer is not None:
                        self.writer.add_scalar('optimizer/learning_rate', self.model_opt._rate, self.train_global_step)
                        self.writer.add_scalar('loss/train_step_loss', float(loss) / float(ntokens), self.train_global_step)

        return total_loss / total_tokens

    def run_valid_epoch(self, epoch_i: int, refresh: int = 10):
        self.model.eval()

        total_tokens = 0
        total_loss = 0
        tokens = 0

        with torch.no_grad():
            with self.tqdm(self.valid_dataloader, leave=False, desc="[Valid] Epoch: {}".format(epoch_i)) as pber:
                for i, batch in enumerate(pber):
                    # batch = MiniBatch(batch.src, batch.tgt, 1)
                    src, tgt = batch
                    src.to(self.device)
                    tgt.to(self.device)

                    src_mask = get_src_mask(src=src, padding_idx=0).to(self.device)
                    tgt_mask = get_tgt_mask(tgt=tgt, padding_idx=0).to(self.device)
                    tgt_y = tgt[:, 1:]
                    # 0 == padding_idx
                    ntokens = (tgt_y != 0).data.sum()

                    out = self.model(src, tgt, src_mask, tgt_mask)
                    loss = self.valid_loss_compute(out, tgt_y, ntokens)
                    total_loss += loss
                    total_tokens += ntokens
                    tokens += ntokens

                    if i % refresh == 0:
                        pber.set_postfix(ordered_dict={"loss": (float(total_loss) / float(total_tokens))})

        return total_loss / total_tokens