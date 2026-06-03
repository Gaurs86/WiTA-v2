"""
train.py — paper's training loop + Stage 13B joint CTC+attention loss.

What changed vs the original Kim et al. file:
  1. Imports torch.nn.functional and GestureTranslator (for SOS/EOS/PAD).
  2. `pad_collate` builds attention I/O tensors when the joint head is on.
  3. `run_one_epoch` switches between CTC-only and joint loss based on
     opts.loss_type / opts.use_joint_decoder.
  4. `validate` decodes via the attention head when joint mode is on.
"""

import os
import sys
import time
import json
import logging
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from model   import GestureTranslator
from data    import AirTypingDataset
from options import AirTypingOptions
from utils   import (WarmupMultiStepLR, Lamb, sec_to_hm_str, cer,
                     calc_seq_len, calc_seq_len_mc3, calc_seq_len_rmc3,
                     seq_len_r3d_kor, seq_len_mc3_kor, seq_len_rmc3_kor,
                     calc_seq_len_2d_eng, calc_seq_len_2d_kor)


options = AirTypingOptions()
opts = options.parse()


# ----------------------------------------------------------------------
# Stage 13B helper: build teacher-forcing attention I/O
# ----------------------------------------------------------------------

def build_attn_io(labels, label_lens,
                  sos=GestureTranslator.SOS_TOKEN,
                  eos=GestureTranslator.EOS_TOKEN,
                  pad=GestureTranslator.PAD_TOKEN):
    """
    labels     : LongTensor [B, L_max]  (zero-padded by pad_sequence)
    label_lens : LongTensor [B]

    Returns:
      attn_input  : [B, L_max+1]  -- prefixes SOS, then L_max-many label tokens.
      attn_target : [B, L_max+1]  -- shifted by 1: ends with EOS at position L,
                                     PAD everywhere past EOS.
    """
    B, L_max = labels.size()
    attn_input  = torch.full((B, L_max + 1), pad, dtype=torch.long, device=labels.device)
    attn_target = torch.full((B, L_max + 1), pad, dtype=torch.long, device=labels.device)
    for b in range(B):
        L = int(label_lens[b].item())
        attn_input[b, 0]       = sos
        if L > 0:
            attn_input[b, 1:1 + L] = labels[b, :L]
            attn_target[b, :L]     = labels[b, :L]
        attn_target[b, L]      = eos
    return attn_input, attn_target


# ----------------------------------------------------------------------

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    if opts.data_type == 'english' or (opts.data_type == 'korean' and opts.num_res_layer == 1):
        if   opts.model_type in ('r3d', 'twoplusone'):
            x_lens = torch.LongTensor([calc_seq_len(len(x)) for x in xx])
        elif opts.model_type == 'rmc3':
            x_lens = torch.LongTensor([calc_seq_len_rmc3(len(x)) for x in xx])
        elif opts.model_type == 'mc3':
            x_lens = torch.LongTensor([calc_seq_len_mc3(len(x)) for x in xx])
        elif opts.model_type == 'r2d':
            x_lens = torch.LongTensor([calc_seq_len_2d_eng(len(x)) for x in xx])
    elif opts.data_type == 'korean' and opts.num_res_layer == 2:
        if   opts.model_type == 'r3d':         x_lens = torch.LongTensor([seq_len_r3d_kor(len(x)) for x in xx])
        elif opts.model_type == 'twoplusone':  x_lens = torch.LongTensor([calc_seq_len(len(x)) for x in xx])
        elif opts.model_type == 'rmc3':        x_lens = torch.LongTensor([seq_len_rmc3_kor(len(x)) for x in xx])
        elif opts.model_type == 'mc3':         x_lens = torch.LongTensor([seq_len_mc3_kor(len(x)) for x in xx])
        elif opts.model_type == 'r2d':         x_lens = torch.LongTensor([calc_seq_len_2d_kor(len(x)) for x in xx])
    y_lens = torch.LongTensor([len(y) for y in yy])
    return xx_pad, yy_pad, x_lens, y_lens


# ----------------------------------------------------------------------

class Trainer:
    def __init__(self):
        self.opts = options.parse()
        self.save_dir = os.path.join(self.opts.save_dir, self.opts.model_name)
        os.makedirs(self.save_dir, exist_ok=True)

        logging.basicConfig(
            filename=os.path.join(self.save_dir, "log-train.log"),
            format='%(asctime)s %(message)s',
            datefmt='%m/%d/%Y %p %I:%M:%S',
            level=logging.INFO,
        )
        logging.getLogger().setLevel(logging.INFO)
        self.logger = logging.getLogger("trainLogger")
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        use_cuda = torch.cuda.is_available() and not self.opts.no_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # ---- data ----
        data_train = AirTypingDataset(self.opts, self.opts.data_path_train)
        data_val   = AirTypingDataset(self.opts, self.opts.data_path_val)
        # data_test held out -- evaluated EXACTLY ONCE post-training by eval_test.py.
        #
        # PERF: with AMP the fp16 GPU computes a batch in ~0.4s but then
        # starves on the 4-vCPU JPEG-decode pipeline (effective throughput
        # measured at ~6-9 ex/s vs ~19 ex/s instantaneous compute).  These
        # knobs keep the GPU fed:
        #   pin_memory          -> faster host->device copies
        #   persistent_workers  -> don't respawn workers each epoch
        #   prefetch_factor     -> each worker buffers several batches ahead
        # num_workers can exceed the 4 cores because per-frame file reads are
        # latency-bound (workers overlap I/O waits).
        _nw = max(int(self.opts.num_workers), 0)
        _dl = dict(collate_fn=pad_collate, pin_memory=True)
        if _nw > 0:
            _dl.update(persistent_workers=True, prefetch_factor=6)
        self.data_loader_train = DataLoader(
            data_train, batch_size=self.opts.batch_size, shuffle=True,
            num_workers=_nw, drop_last=True, **_dl,
        )
        self.data_loader_val = DataLoader(
            data_val, batch_size=self.opts.batch_size, shuffle=False,
            num_workers=_nw, drop_last=True, **_dl,
        )

        # ---- model ----
        self.model = GestureTranslator(self.opts).to(self.device)
        self.use_joint = bool(self.opts.use_joint_decoder) and self.opts.loss_type == 'joint'
        self.logger.info(f"loss_type={self.opts.loss_type}  use_joint_decoder={self.opts.use_joint_decoder}")
        self.logger.info(f"vocab_ctc={self.model.vocab_ctc}  vocab_attn={self.model.vocab_attn}")
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"trainable params: {n_trainable/1e6:.2f} M")

        if torch.cuda.device_count() > 1:
            self.logger.info(f"using {torch.cuda.device_count()} GPUs")
            self.model = torch.nn.DataParallel(self.model)
            self.model_without_dp = self.model.module
        else:
            self.model_without_dp = self.model

        # ---- optimizer ----
        self.params_to_train = list(self.model_without_dp.parameters())
        if   self.opts.optimizer_type == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.params_to_train, lr=self.opts.learning_rate)
        elif self.opts.optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(self.params_to_train, lr=self.opts.learning_rate)
        elif self.opts.optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(self.params_to_train, lr=self.opts.learning_rate)
        elif self.opts.optimizer_type == 'adamW':
            self.optimizer = torch.optim.AdamW(self.params_to_train, lr=self.opts.learning_rate)
        elif self.opts.optimizer_type == 'lamb':
            self.optimizer = Lamb(self.params_to_train, lr=self.opts.learning_rate)

        # ---- scheduler ----
        iter_size = len(self.data_loader_train)
        scheduler_step_size = self.opts.scheduler_step_size * iter_size
        if self.opts.scheduler_type == 'warmup':
            self.scheduler = WarmupMultiStepLR(
                self.optimizer,
                [scheduler_step_size * (i + 1) for i in range(self.opts.num_epochs)],
                gamma=self.opts.scheduler_gamma, warmup_iters=500,
            )
        elif self.opts.scheduler_type == 'steplr':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=scheduler_step_size, gamma=self.opts.scheduler_gamma,
            )
        else:
            self.scheduler = None

        if self.opts.load_dir:
            self.load_model()

        # ---- losses ----
        self.ctc_loss = torch.nn.CTCLoss(reduction='mean', zero_infinity=True)

        # ---- mixed precision ----
        # r3d_18 in fp32 on a single GPU is ~2x slower than fp16 tensor
        # cores can do.  AMP roughly halves memory (enabling batch=8 even
        # with the frame cap) and ~2x throughput.  CTC + CE losses are
        # computed in fp32 (outside autocast) for numerical stability.
        self.use_amp = bool(getattr(self.opts, 'use_amp', True)) and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.logger.info(f"AMP enabled: {self.use_amp}")

        self.epoch = 0
        self.step = 0
        self.start_step = 0
        self.start_time = time.time()
        self.num_total_steps = iter_size * self.opts.num_epochs
        self.best_val_cer = float('Inf')
        self.best_val_epoch = -1

        self.save_opts()

    # ------------------------------------------------------------------

    def train(self):
        for self.epoch in range(self.opts.num_epochs):
            self.logger.info(f"epoch: {self.epoch}")
            self.model.train()
            self.run_one_epoch()
            is_best = self.validate(self.data_loader_val, False)
            if is_best or (self.opts.save_frequency > 0 and (self.epoch + 1) % self.opts.save_frequency == 0):
                self.save_model(is_best)

    def run_one_epoch(self):
        losses = []
        for batch_idx, (xx_pad, yy_pad, x_lens, y_lens) in enumerate(self.data_loader_train):
            t_before = time.time()
            xx_pad, yy_pad = xx_pad.to(self.device), yy_pad.to(self.device)
            x_lens, y_lens = x_lens.to(self.device), y_lens.to(self.device)
            self.optimizer.zero_grad()

            # Forward in mixed precision; losses in fp32 (outside autocast)
            # for CTC/CE numerical stability.
            if self.use_joint:
                attn_in, attn_tgt = build_attn_io(
                    yy_pad, y_lens,
                    sos=GestureTranslator.SOS_TOKEN,
                    eos=GestureTranslator.EOS_TOKEN,
                    pad=GestureTranslator.PAD_TOKEN,
                )
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    ctc_logits, attn_logits = self.model(xx_pad, x_lens, attn_input=attn_in)
                ctc_log_probs = ctc_logits.float().permute(1, 0, 2).log_softmax(-1)
                ctc_loss = self.ctc_loss(ctc_log_probs, yy_pad, x_lens, y_lens)
                attn_loss = F.cross_entropy(
                    attn_logits.float().reshape(-1, attn_logits.size(-1)),
                    attn_tgt.reshape(-1),
                    ignore_index=GestureTranslator.PAD_TOKEN,
                    label_smoothing=self.opts.label_smoothing,
                )
                loss = self.opts.lambda_ctc * ctc_loss + (1 - self.opts.lambda_ctc) * attn_loss
            else:
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    ctc_logits, _ = self.model(xx_pad, x_lens)
                ctc_log_probs = ctc_logits.float().permute(1, 0, 2).log_softmax(-1)
                loss = self.ctc_loss(ctc_log_probs, yy_pad, x_lens, y_lens)
                ctc_loss = loss
                attn_loss = torch.tensor(0.0, device=self.device)

            losses.append(loss.item())
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler:
                self.scheduler.step()

            if (batch_idx + 1) % self.opts.log_interval == 0:
                duration = abs(t_before - time.time())
                self.log_time(duration, batch_idx, loss.item(),
                              ctc_loss.item(), attn_loss.item())
                # CTC argmax preview (matches paper's logging shape).
                y_pred = torch.max(ctc_log_probs, 2)[1]
                self.logger.info('\tGT  : {}'.format(
                    self.data_loader_train.dataset.converter.decode(yy_pad[0, :y_lens[0]], y_lens[:1])))
                self.logger.info('\tCTC : {}'.format(
                    self.data_loader_train.dataset.converter.decode(y_pred[:x_lens[0], 0], x_lens[:1])))
                if self.scheduler:
                    self.logger.info(f"\tLR  : {self.scheduler.get_lr()[0]:.6f}")
            self.step += 1
        return losses

    def validate(self, data_loader, load=False):
        self.logger.info('--------------- Validation ----------------')
        if load:
            self.load_model()
        self.model.eval()
        losses = []
        with torch.no_grad():
            total_err, total_len = 0, 0
            for batch_idx, (xx_pad, yy_pad, x_lens, y_lens) in enumerate(data_loader):
                xx_pad, yy_pad = xx_pad.to(self.device), yy_pad.to(self.device)
                x_lens, y_lens = x_lens.to(self.device), y_lens.to(self.device)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    ctc_logits, _ = self.model(xx_pad, x_lens)
                ctc_log_probs = ctc_logits.float().permute(1, 0, 2).log_softmax(-1)
                loss = self.ctc_loss(ctc_log_probs, yy_pad, x_lens, y_lens)
                losses.append(loss.item())

                if self.use_joint:
                    # Decode via attention head (greedy autoregressive).
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        pred_tokens = self.model_without_dp.decode_attention(
                            xx_pad, x_lens, max_len=self.opts.attn_max_len,
                        )
                    gt_b = self._decode_label(yy_pad[0, :y_lens[0]],
                                              data_loader.dataset.converter)
                    pred_b = self._decode_attn(pred_tokens[0].tolist(),
                                               data_loader.dataset.converter)
                else:
                    y_pred = torch.max(ctc_log_probs, 2)[1]
                    gt_b = data_loader.dataset.converter.decode(
                        yy_pad[0, :y_lens[0]], y_lens[:1])
                    pred_b = data_loader.dataset.converter.decode(
                        y_pred[:x_lens[0], 0], x_lens[:1])

                err, length = cer(gt_b, pred_b)
                if err > length:
                    err = length
                total_err += err
                total_len += length

                if (batch_idx + 1) % self.opts.log_interval == 0:
                    self.logger.info(f"\tGT  : {gt_b}")
                    self.logger.info(f"\tPRED: {pred_b}")

            cur_val_loss = sum(losses) / max(len(losses), 1)
            cur_val_cer  = total_err / max(total_len, 1)
            self.logger.info(f"\tvalidation_loss  : {cur_val_loss:.4f}")
            self.logger.info(f"\tvalidation_cer   : {cur_val_cer:.4f}")
            if self.best_val_cer > cur_val_cer:
                self.best_val_cer = cur_val_cer
                self.best_val_epoch = self.epoch
                return True
            return False

    # ------------------------------------------------------------------

    @staticmethod
    def _decode_label(ids_tensor, converter):
        """Convert a 1-D label tensor (CTC-encoded GT) into a string."""
        chars = []
        prev = 0
        for t in ids_tensor.tolist():
            if t != 0 and t != prev and 1 <= t <= len(converter.alphabet):
                chars.append(converter.alphabet[t - 1])
            prev = t
        return ''.join(chars)

    @staticmethod
    def _decode_attn(token_ids, converter):
        """Convert attention-decoded ids (CTC vocab + SOS/EOS/PAD) into a string."""
        out = []
        for t in token_ids:
            if t == GestureTranslator.EOS_TOKEN or t == GestureTranslator.PAD_TOKEN:
                break
            if t == GestureTranslator.SOS_TOKEN:
                continue
            if 1 <= t <= len(converter.alphabet):
                ch = converter.alphabet[t - 1]
                # Drop the StrLabelConverter's '-' repeat-separator just like
                # the CTC decode path does: it never appears in ground truth.
                if ch != '-':
                    out.append(ch)
        return ''.join(out)

    # ------------------------------------------------------------------

    def log_time(self, duration, batch_idx, loss, ctc_loss=None, attn_loss=None):
        samples_per_sec = self.opts.batch_size / max(duration, 1e-6)
        time_sofar = time.time() - self.start_time
        time_left = ((self.num_total_steps - self.step)
                     / max(self.step - self.start_step, 1)) * time_sofar if self.step > 0 else 0
        msg = (f"epoch {self.epoch:>3} | batch [{batch_idx * self.opts.batch_size:>4}/"
               f"{len(self.data_loader_train.dataset):>4}] | "
               f"ex/s: {samples_per_sec:5.1f} | loss: {loss:.4f}")
        if ctc_loss is not None:
            msg += f" | ctc: {ctc_loss:.4f}"
        if attn_loss is not None:
            msg += f" | attn: {attn_loss:.4f}"
        msg += f" | elapsed: {sec_to_hm_str(time_sofar)} | left: {sec_to_hm_str(time_left)}"
        self.logger.info(msg)

    def save_opts(self):
        to_save = self.opts.__dict__.copy()
        with open(os.path.join(self.save_dir, 'opts.json'), 'w') as f:
            json.dump(to_save, f, indent=2, default=str)

    def save_model(self, is_best):
        save_folder = os.path.join(
            self.save_dir, "models",
            f"weights_{self.epoch}{'_best' if is_best else ''}",
        )
        os.makedirs(save_folder, exist_ok=True)
        torch.save(self.model_without_dp.state_dict(),
                   os.path.join(save_folder, "model.pth"))
        torch.save(self.optimizer.state_dict(),
                   os.path.join(save_folder, "optimizer.pth"))
        if self.scheduler:
            torch.save(self.scheduler.state_dict(),
                       os.path.join(save_folder, "scheduler.pth"))
        # Also update a stable "best/" symlink-style folder for the test eval.
        if is_best:
            best_folder = os.path.join(self.save_dir, "models", "best")
            os.makedirs(best_folder, exist_ok=True)
            torch.save(self.model_without_dp.state_dict(),
                       os.path.join(best_folder, "model.pth"))
            with open(os.path.join(best_folder, "best_meta.json"), 'w') as f:
                json.dump({"best_epoch": self.best_val_epoch,
                           "best_val_cer": float(self.best_val_cer)}, f, indent=2)

    def load_model(self):
        self.opts.load_dir = os.path.expanduser(self.opts.load_dir)
        assert os.path.isdir(self.opts.load_dir), f"Cannot find directory {self.opts.load_dir}"
        self.logger.info(f"loading model from {self.opts.load_dir}")
        state = torch.load(os.path.join(self.opts.load_dir, "model.pth"),
                           map_location=self.device)
        # strict=False so adding the joint decoder to a CTC-only checkpoint works.
        missing, unexpected = self.model_without_dp.load_state_dict(state, strict=False)
        if missing:    self.logger.info(f"  missing keys    : {len(missing)}")
        if unexpected: self.logger.info(f"  unexpected keys : {len(unexpected)}")
        opt_path = os.path.join(self.opts.load_dir, "optimizer.pth")
        if os.path.isfile(opt_path):
            self.optimizer.load_state_dict(torch.load(opt_path, map_location=self.device))


if __name__ == "__main__":
    os.environ['PYTHONHASHSEED'] = str(opts.seed_number)
    random.seed(opts.seed_number)
    np.random.seed(opts.seed_number)
    torch.manual_seed(opts.seed_number)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    Trainer().train()
