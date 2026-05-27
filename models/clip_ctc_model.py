"""
models/clip_ctc_model.py — WiTA SigLIP cross-modal CTC model.

Two forward modes
-----------------
forward(clips, input_lens) ............... ONLINE
    Runs the (frozen) SigLIP vision encoder per-frame, then the cross-modal
    head.  Slow — use only for inference / debugging, NOT for training.

forward_cached(feats, input_lens) ........ CACHED
    Skips the vision encoder entirely; expects pre-extracted features
    [B, T, D].  This is the training-time path.

Both return  (ctc_log_probs [B, T, V], enc_lens [B])  to match the existing
trainer / evaluator contract.

Why two paths
-------------
SigLIP-So400m is ~400M params.  Running it inside the training loop on T4 is
prohibitively slow (~25s/batch).  Since the encoder is frozen, we extract
features once (see datasets/feature_cache.py) and train the lightweight head
on cached features — each epoch becomes ~1 min instead of ~20 min.
"""

from __future__ import annotations
import logging

import torch
import torch.nn as nn

from ..configs.default import Config
from .encoders.siglip_encoder import SigLIPVisionEncoder
from .heads.clip_cross_modal_head import CLIPCrossModalHead, build_prototypes

logger = logging.getLogger(__name__)


class WiTACLIPCTCModel(nn.Module):
    """
    SigLIP cross-modal CTC recognizer.

    Parameters
    ----------
    cfg        : top-level Config (uses cfg.encoder, cfg.vocab, cfg.recurrent)
    encoder    : optional pre-built SigLIPVisionEncoder.  If None and online
                 mode is needed, one is built on first forward() call.
    prototypes : optional pre-built [V, D] prototype tensor.  If None, built
                 from cfg.encoder.siglip_* template fields.
    """

    def __init__(
        self,
        cfg:        Config,
        encoder:    SigLIPVisionEncoder | None = None,
        prototypes: torch.Tensor | None = None,
    ):
        super().__init__()
        self.cfg = cfg

        ec  = cfg.encoder
        vc  = cfg.vocab

        # ── Vision encoder (lazy) ────────────────────────────────────────
        # Encoder is large; allow None for cached-only training.
        self.encoder: SigLIPVisionEncoder | None = encoder
        self._visual_dim: int | None = (
            encoder.out_dim if encoder is not None else getattr(ec, "out_dim", None)
        )
        if self._visual_dim is None:
            raise ValueError(
                "out_dim must be known to build the head — either pass `encoder` "
                "or set cfg.encoder.out_dim to the SigLIP hidden size."
            )

        # ── Prototypes ───────────────────────────────────────────────────
        if prototypes is None:
            prototypes = build_prototypes(
                chars          = vc.chars,
                blank_idx      = vc.blank_idx,
                sep_idx        = vc.sep_idx,
                ctc_vocab_size = vc.ctc_vocab_size,
                model_name     = getattr(ec, "siglip_model_name",
                                          "google/siglip-so400m-patch14-384"),
                char_template  = getattr(ec, "siglip_char_template",
                                          "the letter {ch}"),
                blank_template = getattr(ec, "siglip_blank_template",
                                          "no character"),
                sep_template   = getattr(ec, "siglip_sep_template",
                                          "a brief pause between letters"),
            )
        if prototypes.shape != (vc.ctc_vocab_size, self._visual_dim):
            raise ValueError(
                f"prototypes shape mismatch — expected "
                f"[{vc.ctc_vocab_size}, {self._visual_dim}], got "
                f"{tuple(prototypes.shape)}"
            )

        # ── Cross-modal head ─────────────────────────────────────────────
        self.head = CLIPCrossModalHead(
            visual_dim     = self._visual_dim,
            prototypes     = prototypes,
            adapter_arch   = getattr(ec, "siglip_temporal_arch", "lstm"),
            adapter_hidden = getattr(ec, "siglip_adapter_hidden", 512),
            adapter_layers = getattr(ec, "siglip_adapter_layers", 1),
            dropout        = getattr(ec, "siglip_dropout", 0.1),
            init_tau       = getattr(ec, "siglip_init_tau", 0.07),
            learnable_tau  = getattr(ec, "siglip_learnable_tau", True),
        )

        logger.info(
            "[WiTACLIPCTCModel] visual_dim=%d vocab=%d adapter=%s "
            "online_encoder=%s",
            self._visual_dim, vc.ctc_vocab_size,
            getattr(ec, "siglip_temporal_arch", "lstm"),
            "present" if encoder is not None else "lazy",
        )

    # ------------------------------------------------------------------ #
    # Encoder management                                                 #
    # ------------------------------------------------------------------ #

    def _ensure_encoder(self) -> SigLIPVisionEncoder:
        """Build the SigLIP encoder lazily for online forward()."""
        if self.encoder is None:
            ec = self.cfg.encoder
            self.encoder = SigLIPVisionEncoder(
                model_name = getattr(ec, "siglip_model_name",
                                      "google/siglip-so400m-patch14-384"),
                image_size = ec.img_size,
            )
            # Move to the same device as the head's prototypes.
            self.encoder = self.encoder.to(self.head.prototypes.device)
        return self.encoder

    # ------------------------------------------------------------------ #
    # CTC log-prob production                                            #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        x:          torch.Tensor,
        input_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Auto-routing forward — picks online vs cached based on input ndim.

        x          : [B, T, 3, H, W] raw clips (online)  OR  [B, T, D] cached feats
        input_lens : [B] int — number of valid frames per clip
        returns    : (ctc_log_probs [B, T, V], enc_lens [B])

        This dispatch lets the existing trainer call `model(x, lens)` whether
        x is a clip tensor or pre-extracted features — no trainer changes needed.
        """
        if x.dim() == 5:
            enc = self._ensure_encoder()
            feats = enc.forward_video(x)              # [B, T, D]
            return self.forward_cached(feats, input_lens)
        if x.dim() == 3:
            return self.forward_cached(x, input_lens)
        raise ValueError(
            f"expected [B, T, 3, H, W] clips or [B, T, D] features, "
            f"got shape {tuple(x.shape)}"
        )

    def forward_cached(
        self,
        feats:      torch.Tensor,
        input_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Cached forward — feats are pre-extracted SigLIP features.

        feats      : [B, T, D]
        input_lens : [B] int — number of valid frames per clip
        returns    : (ctc_log_probs [B, T, V], enc_lens [B])
        """
        B, T, _D = feats.shape
        # Build mask only when adapter cares about it (transformer).
        mask = None
        if self.head.adapter.arch == "transformer":
            arange = torch.arange(T, device=feats.device).unsqueeze(0)   # [1, T]
            mask = arange >= input_lens.unsqueeze(1)                     # [B, T] True at pad

        log_probs = self.head(feats, mask=mask)        # [B, T, V]

        # No temporal subsampling — adapter preserves length, so enc_lens
        # equals input_lens.  Cap to T just in case.
        enc_lens = input_lens.clamp(max=T).to(torch.int32)
        return log_probs, enc_lens

    # ------------------------------------------------------------------ #
    # Greedy CTC decode  (matches evaluator contract)                    #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def decode_ctc_greedy(
        self,
        clips_or_feats: torch.Tensor,
        input_lens:     torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Greedy argmax + CTC merge.  Accepts either raw clips [B, T, 3, H, W]
        (online) or cached features [B, T, D] (cached).  Decides by ndim.

        returns : list of [T_i] int tensors of de-duplicated, blank-stripped indices.
        """
        if clips_or_feats.dim() == 5:
            log_probs, enc_lens = self.forward(clips_or_feats, input_lens)
        elif clips_or_feats.dim() == 3:
            log_probs, enc_lens = self.forward_cached(clips_or_feats, input_lens)
        else:
            raise ValueError(
                f"expected 5D clips or 3D feats, got {clips_or_feats.dim()}D"
            )

        blank = self.cfg.vocab.blank_idx
        argmax = log_probs.argmax(dim=-1)              # [B, T]

        out: list[torch.Tensor] = []
        for b in range(argmax.shape[0]):
            seq    = argmax[b, : enc_lens[b].item()].tolist()
            merged: list[int] = []
            prev: int | None = None
            for tok in seq:
                if tok != prev and tok != blank:
                    merged.append(tok)
                prev = tok
            out.append(torch.tensor(merged, dtype=torch.int32))
        return out

    # ------------------------------------------------------------------ #
    # Stub hooks for trainer compatibility                               #
    # ------------------------------------------------------------------ #

    def unfreeze_backbone(self) -> None:
        """No-op — SigLIP stays frozen.  Trainer calls this on schedule."""
        logger.info(
            "[WiTACLIPCTCModel] unfreeze_backbone called but SigLIP "
            "remains frozen (intentional for cross-modal recipe)."
        )
