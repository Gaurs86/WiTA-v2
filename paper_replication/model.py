"""
model.py — paper's GestureTranslator + Stage 13B joint attention decoder.

Vocab convention (English, CTC):
  0  = blank (CTC)
  1..26  = a..z
  27 = '-' (StrLabelConverter's repeat-separator)
  vocab_ctc = 28

Attention decoder vocab adds:
  28 = SOS
  29 = EOS
  30 = PAD
  vocab_attn = 31

The encoder + CTC head are identical to the paper's recipe.  The
attention branch is only constructed when `use_joint_decoder=True`.
"""

import utils
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from resnet3d import r2d, r3d, mc3, rmc3, r2plus1d


class GestureTranslator(nn.Module):
    """Receives a sequence of images (video) and returns logits for CTC
    and (optionally) an attention head."""

    # Stage 13B special tokens (only used when use_joint_decoder=True).
    SOS_TOKEN = 28
    EOS_TOKEN = 29
    PAD_TOKEN = 30

    def __init__(self, opts):
        super().__init__()
        self.opts = opts

        # ----- vocab sizing -----
        # The paper's StrLabelConverter has alphabet+'-' (27 chars) with
        # indices 1..27; index 0 is blank.  vocab_ctc=28 covers blank+a..z+'-'.
        if self.opts.data_type == 'english':
            base = len(utils.ALPHABET)              # 26
        elif self.opts.data_type == 'korean':
            base = len(utils.HANGUL)
        else:
            base = len(utils.ALPHA_HAN)
        # "+2" matches the paper's `len(ALPHABET) + 2` for the CTC case.
        self.vocab_ctc  = base + 2                  # English: 28
        self.vocab_attn = self.vocab_ctc + 3        # English: 31 (SOS, EOS, PAD)
        # Kept for backward-compat with the paper's checkpoint loader.
        self.num_class  = self.vocab_ctc

        # ----- encoder -----
        if   self.opts.model_type == "r3d":         self.encoder_module = r3d()
        elif self.opts.model_type == "mc3":         self.encoder_module = mc3()
        elif self.opts.model_type == "rmc3":        self.encoder_module = rmc3()
        elif self.opts.model_type == "twoplusone":  self.encoder_module = r2plus1d()
        elif self.opts.model_type == "r2d":         self.encoder_module = r2d()
        else: raise ValueError(f"unknown model_type {self.opts.model_type}")

        # Backward-compat: the original code compared `opts.pretrained == "True"`,
        # which never fires when argparse converts the value to a real bool.
        # Use the proper bool here.
        if bool(self.opts.pretrained):
            self._load_pretrained()

        # ----- optional recurrent layer (paper's flag, default 'none') -----
        self.recurrent_module = None
        if self.opts.recurrent_type.lower() == 'lstm':
            self.recurrent_module = nn.LSTM(
                opts.input_size, opts.hidden_size, opts.num_layers,
                batch_first=True, bidirectional=True,
            )
        elif self.opts.recurrent_type.lower() == 'gru':
            self.recurrent_module = nn.GRU(
                opts.input_size, opts.hidden_size, opts.num_layers,
                batch_first=True, bidirectional=True,
            )
        elif self.opts.recurrent_type.lower() == 'transformers':
            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=opts.d_model, nhead=opts.nhead, batch_first=True,
            )
            self.recurrent_module = nn.TransformerEncoder(
                encoder_layer=self.encoder_layer, num_layers=opts.num_encoder_layer,
            )

        # ----- CTC head -----
        if self.recurrent_module and self.opts.recurrent_type.lower() != 'transformers':
            self.fc1 = nn.Linear(2 * opts.hidden_size, opts.hidden_size_fc)
            self.fc2 = nn.Linear(opts.hidden_size_fc, self.vocab_ctc)
        elif self.recurrent_module and self.opts.recurrent_type.lower() == 'transformers':
            self.fc1 = nn.Linear(opts.d_model, opts.hidden_size_fc)
            self.fc2 = nn.Linear(opts.hidden_size_fc, self.vocab_ctc)
        else:
            self.fc1 = None
            self.fc2 = nn.Linear(opts.input_size, self.vocab_ctc)

        # ----- Stage 13B: attention decoder branch -----
        self.use_joint = bool(getattr(opts, 'use_joint_decoder', False))
        # d_model dim for the decoder's cross-attention to encoder memory.
        # The encoder output after the optional recurrent module:
        #   * none / no rnn  -> opts.input_size (256)
        #   * lstm/gru       -> 2 * hidden_size
        #   * transformers   -> opts.d_model
        if self.recurrent_module is None or self.opts.recurrent_type.lower() == 'transformers':
            mem_dim = opts.d_model if self.opts.recurrent_type.lower() == 'transformers' else opts.input_size
        else:
            mem_dim = 2 * opts.hidden_size
        self.mem_dim = mem_dim

        if self.use_joint:
            self.token_embed = nn.Embedding(self.vocab_attn, mem_dim,
                                            padding_idx=self.PAD_TOKEN)
            # 64 covers anything the WiTA labels can throw at us (max ~24 chars).
            self.pos_embed = nn.Embedding(64, mem_dim)
            dec_layer = nn.TransformerDecoderLayer(
                d_model         = mem_dim,
                nhead           = opts.attn_decoder_heads,
                dim_feedforward = 4 * mem_dim,
                dropout         = opts.attn_decoder_dropout,
                batch_first     = True,
            )
            self.attn_decoder = nn.TransformerDecoder(
                dec_layer, num_layers=opts.attn_decoder_layers,
            )
            self.attn_head = nn.Linear(mem_dim, self.vocab_attn)

    # ------------------------------------------------------------------

    def _load_pretrained(self):
        if self.opts.model_type == "r3d":
            model_ft = torchvision.models.video.r3d_18(pretrained=True)
        elif self.opts.model_type == "mc3":
            model_ft = torchvision.models.video.mc3_18(pretrained=True)
        elif self.opts.model_type == "twoplusone":
            model_ft = torchvision.models.video.r2plus1d_18(pretrained=True)
        else:
            return
        pre_dict = model_ft.state_dict()
        del pre_dict['fc.weight']; del pre_dict['fc.bias']
        if self.opts.data_type == "korean":
            pre_dict['stem.0.weight'] = (
                pre_dict['stem.0.weight'][:, :, 0]
                + pre_dict['stem.0.weight'][:, :, 1]
                + pre_dict['stem.0.weight'][:, :, 2]
            )
            pre_dict['stem.0.weight'] /= 3
        encoder_dict = self.encoder_module.state_dict()
        encoder_dict.update(pre_dict)
        self.encoder_module.load_state_dict(encoder_dict)

    # ------------------------------------------------------------------

    def _encode_memory(self, seq_img, seq_lens):
        """Returns the [B, T_out, mem_dim] tensor that both the CTC head
        and the attention decoder consume."""
        embeddings = self.encoder_module(seq_img.permute(0, 2, 1, 3, 4))
        # shape: [B, T_out, 256]
        if self.recurrent_module and self.opts.recurrent_type.lower() != 'transformers':
            x_packed = pack_padded_sequence(
                embeddings, seq_lens.cpu(), batch_first=True, enforce_sorted=False,
            )
            outputs_packed, _ = self.recurrent_module(x_packed)
            outputs_padded, _ = pad_packed_sequence(outputs_packed, batch_first=True)
            return outputs_padded                                  # [B, T_out, 2H]
        if self.recurrent_module and self.opts.recurrent_type.lower() == 'transformers':
            return self.recurrent_module(embeddings)               # [B, T_out, d_model]
        return embeddings                                          # [B, T_out, 256]

    def _ctc_logits_from_memory(self, memory):
        if self.recurrent_module is not None:
            return self.fc2(F.relu(self.fc1(memory)))
        return self.fc2(memory)

    # ------------------------------------------------------------------

    def forward(self, seq_img, seq_lens, attn_input=None):
        """
        seq_img    : [B, T, C, H, W]
        seq_lens   : LongTensor [B]   (already-downsampled lengths from collate)
        attn_input : LongTensor [B, L] of teacher-forcing tokens (with SOS prefix)
                     -- only used when self.use_joint and we want the attn branch
        Returns:
          (ctc_logits, attn_logits_or_None)
        """
        memory = self._encode_memory(seq_img, seq_lens)            # [B, T_out, mem_dim]
        ctc_logits = self._ctc_logits_from_memory(memory)          # [B, T_out, vocab_ctc]

        if self.use_joint and attn_input is not None:
            attn_logits = self._attn_forward(memory, attn_input)
            return ctc_logits, attn_logits
        return ctc_logits, None

    def _attn_forward(self, memory, attn_input):
        """Teacher-forced attention decoder forward."""
        B, L = attn_input.shape
        tgt_emb = self.token_embed(attn_input)                     # [B, L, mem_dim]
        pos_idx = torch.arange(L, device=attn_input.device)
        tgt_emb = tgt_emb + self.pos_embed(pos_idx).unsqueeze(0)

        # Both masks must share a dtype (PyTorch 2.x): use BOOL for both.
        # A True entry means "not allowed to attend".  The causal mask is
        # upper-triangular (can't see the future); the pad mask blocks PAD.
        causal_mask = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=memory.device), diagonal=1,
        )                                                          # [L, L] bool
        tgt_pad_mask = (attn_input == self.PAD_TOKEN)              # [B, L] bool

        attn_out = self.attn_decoder(
            tgt_emb, memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
        )                                                          # [B, L, mem_dim]
        return self.attn_head(attn_out)                            # [B, L, vocab_attn]

    @torch.no_grad()
    def decode_attention(self, seq_img, seq_lens, max_len=32):
        """Greedy autoregressive attention decode.

        Returns LongTensor [B, decoded_len] with tokens starting AFTER the
        initial SOS.  Tokens past EOS are set to PAD.
        """
        memory = self._encode_memory(seq_img, seq_lens)
        B = memory.size(0)
        device = memory.device

        tokens   = torch.full((B, 1), self.SOS_TOKEN, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(max_len):
            L = tokens.size(1)
            tgt_emb = self.token_embed(tokens)
            pos_idx = torch.arange(L, device=device)
            tgt_emb = tgt_emb + self.pos_embed(pos_idx).unsqueeze(0)
            causal = torch.triu(
                torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1,
            )                                                      # [L, L] bool
            attn_out = self.attn_decoder(tgt_emb, memory, tgt_mask=causal)
            logits = self.attn_head(attn_out[:, -1, :])           # [B, vocab_attn]
            next_tok = logits.argmax(-1)                          # [B]
            next_tok = torch.where(finished,
                                   torch.full_like(next_tok, self.PAD_TOKEN),
                                   next_tok)
            finished = finished | (next_tok == self.EOS_TOKEN)
            tokens = torch.cat([tokens, next_tok.unsqueeze(1)], dim=1)
            if finished.all():
                break
        return tokens[:, 1:]                                       # strip the SOS prefix


# ----------------------------------------------------------------------

if __name__ == "__main__":
    from options import AirTypingOptions
    import sys
    # Smoke test: build + forward both branches.
    sys.argv = [
        'model.py',
        '--model_type=r3d', '--num_res_layer=1', '--data_type=english',
        '--use_joint_decoder=True', '--loss_type=joint', '--img_size=112',
    ]
    opts = AirTypingOptions().parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gt = GestureTranslator(opts).to(device)
    total     = sum(p.numel() for p in gt.parameters())
    trainable = sum(p.numel() for p in gt.parameters() if p.requires_grad)
    print(f"Total params:    {total/1e6:.2f} M")
    print(f"Trainable:       {trainable/1e6:.2f} M")
    x = torch.rand(2, 64, 3, 112, 112).to(device)
    x_lens = torch.LongTensor([16, 14]).to(device)
    attn_in = torch.tensor([[gt.SOS_TOKEN, 3, 8, 15, 12, 12, 15, gt.PAD_TOKEN, gt.PAD_TOKEN, gt.PAD_TOKEN],
                            [gt.SOS_TOKEN, 21, 19, 8, 14, 7, gt.PAD_TOKEN, gt.PAD_TOKEN, gt.PAD_TOKEN, gt.PAD_TOKEN]]).to(device)
    ctc, attn = gt(x, x_lens, attn_input=attn_in)
    print(f"CTC logits  : {tuple(ctc.shape)}    (expect [2, T_out, {gt.vocab_ctc}])")
    print(f"Attn logits : {tuple(attn.shape)}   (expect [2, 10, {gt.vocab_attn}])")
    decoded = gt.decode_attention(x, x_lens, max_len=12)
    print(f"Greedy decode: {tuple(decoded.shape)}   (expect [2, <=12])")
