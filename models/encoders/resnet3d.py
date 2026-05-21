"""
models/encoders/resnet3d.py — Self-contained VideoResNet family.

Re-implements WiTA baseline resnet3d.py with key fixes:
  • NO global `opts` object imported at module level (the original was
    a global-state anti-pattern that broke any multi-config experiment).
  • All configuration passed via constructor parameters.
  • Identical architecture to the baseline — checkpoints are compatible.
  • Supports r3d, mc3, rmc3, r2plus1d, r2d.

Reference: "A Closer Look at Spatiotemporal Convolutions for Action Recognition"
           Tran et al. 2018  (https://arxiv.org/abs/1711.11248)
"""

from __future__ import annotations
import torch
import torch.nn as nn

from configs.default import EncoderConfig


# ---------------------------------------------------------------------------
# 3-D convolution building blocks
# ---------------------------------------------------------------------------

class Conv3DSimple(nn.Conv3d):
    """Full 3-D convolution (r3d kernel: 3×3×3)."""
    def __init__(self, in_planes, out_planes, midplanes=None, stride=1, padding=1):
        super().__init__(in_planes, out_planes,
                         kernel_size=(3, 3, 3), stride=stride,
                         padding=padding, bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)


class Conv3DNoTemporal(nn.Conv3d):
    """2-D convolution embedded in 3-D (kernel: 1×3×3)."""
    def __init__(self, in_planes, out_planes, midplanes=None, stride=1, padding=1):
        super().__init__(in_planes, out_planes,
                         kernel_size=(1, 3, 3),
                         stride=(1, stride, stride),
                         padding=(0, padding, padding), bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return (1, stride, stride)


class Conv2Plus1D(nn.Sequential):
    """Factorised (2+1)-D convolution: spatial 1×3×3 then temporal 3×1×1."""
    def __init__(self, in_planes, out_planes, midplanes, stride=1, padding=1,
                 track_running: bool = True):
        super().__init__(
            nn.Conv3d(in_planes, midplanes, kernel_size=(1, 3, 3),
                      stride=(1, stride, stride), padding=(0, padding, padding), bias=False),
            nn.BatchNorm3d(midplanes, track_running_stats=track_running),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, out_planes, kernel_size=(3, 1, 1),
                      stride=(stride, 1, 1), padding=(padding, 0, 0), bias=False),
        )

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)


# ---------------------------------------------------------------------------
# BasicBlock / Bottleneck
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None,
                 track_running: bool = True):
        mid = (inplanes * planes * 27) // (inplanes * 9 + 3 * planes)
        super().__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, mid, stride),
            nn.BatchNorm3d(planes, track_running_stats=track_running),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, mid),
            nn.BatchNorm3d(planes, track_running_stats=track_running),
        )
        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(out + x)


# ---------------------------------------------------------------------------
# Stems
# ---------------------------------------------------------------------------

class BasicStem(nn.Sequential):
    def __init__(self, temporal_kernel: int = 3, track_running: bool = True):
        super().__init__(
            nn.Conv3d(3, 64, kernel_size=(temporal_kernel, 7, 7),
                      stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64, track_running_stats=track_running),
            nn.ReLU(inplace=True),
        )


class R2Plus1dStem(nn.Sequential):
    def __init__(self, track_running: bool = True):
        super().__init__(
            nn.Conv3d(3, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2),
                      padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(45, track_running_stats=track_running),
            nn.ReLU(inplace=True),
            nn.Conv3d(45, 64, kernel_size=(3, 1, 1), stride=(1, 1, 1),
                      padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(64, track_running_stats=track_running),
            nn.ReLU(inplace=True),
        )


# ---------------------------------------------------------------------------
# VideoResNet
# ---------------------------------------------------------------------------

class VideoResNet(nn.Module):
    """
    Generic 3-D ResNet for video / clip encoding.

    Parameters
    ----------
    block         : BasicBlock or Bottleneck
    conv_makers   : list of 4 conv-builder classes (one per layer)
    layers        : list of 4 ints (blocks per layer)
    stem          : stem module instance
    out_dim       : output feature dimension (fc layer)
    pooling       : 'average' | 'max'
    track_running : BatchNorm track_running_stats
    """

    def __init__(
        self,
        block,
        conv_makers: list,
        layers:      list[int],
        stem:        nn.Module,
        out_dim:     int  = 256,
        pooling:     str  = "average",
        track_running: bool = True,
    ):
        super().__init__()
        self.inplanes = 64
        self.pooling  = pooling
        self.stem     = stem

        self.layer1 = self._make_layer(block, conv_makers[0], 64,  layers[0], 1, track_running)
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], 2, track_running)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], 1, track_running)  # stride=1 (baseline)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], 2, track_running)

        self.maxpool = nn.MaxPool3d((2, 3, 3))
        self.avgpool = nn.AvgPool3d((2, 3, 3))
        self.adaptive_avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.adaptive_maxpool = nn.AdaptiveMaxPool3d((None, 1, 1))

        self.fc = nn.Linear(512 * block.expansion, out_dim)
        self._init_weights()

    def _make_layer(self, block, conv_builder, planes, n_blocks, stride, track_running):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion,
                               track_running_stats=track_running),
            )
        layers = [block(self.inplanes, planes, conv_builder, stride, downsample, track_running)]
        self.inplanes = planes * block.expansion
        for _ in range(1, n_blocks):
            layers.append(block(self.inplanes, planes, conv_builder,
                                track_running=track_running))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, C, T, H, W]  (3-D ResNet convention)
        Returns [B, T', out_dim]  where T' is temporally downsampled.
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)

        # Intermediate spatial pool (only for mc3 / rmc3 — mirrors baseline logic)
        arch = getattr(self, "_arch", "r3d")
        if arch not in ("r3d", "r2plus1d"):
            if self.pooling == "max":
                x = self.maxpool(x)
            else:
                x = self.avgpool(x)

        x = self.layer3(x)
        x = self.layer4(x)

        # Adaptive spatial pool → [B, 512, T', 1, 1]
        if self.pooling == "average":
            x = self.adaptive_avgpool(x)
        else:
            x = self.adaptive_maxpool(x)

        # Squeeze spatial dims → [B, 512, T'] → permute → [B, T', 512]
        B, C, T = x.shape[0], x.shape[1], x.shape[2]
        x = x.view(B, C, T).permute(0, 2, 1)   # [B, T', 512]
        x = self.fc(x)                           # [B, T', out_dim]
        return x


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------

def _build(arch, block, conv_makers, layers, stem_cls, stem_kwargs, cfg: EncoderConfig):
    """Internal factory."""
    stem = stem_cls(**stem_kwargs)
    model = VideoResNet(
        block=block,
        conv_makers=conv_makers,
        layers=layers,
        stem=stem,
        out_dim=cfg.out_dim,
        pooling=cfg.pooling,
        track_running=cfg.track_running_stats,
    )
    model._arch = arch   # stored for the forward-pass branch selector
    return model


def build_encoder(cfg: EncoderConfig) -> VideoResNet:
    """
    Instantiate the 3-D ResNet encoder specified by EncoderConfig.

    Optionally loads ImageNet-pretrained weights for r3d / mc3 / r2plus1d
    (cfg.pretrained=True).  The fc layer is re-initialised to cfg.out_dim
    regardless of pretrained weights.
    """
    N  = cfg.num_res_layers
    TR = cfg.track_running_stats

    arch = cfg.arch.lower()

    # Korean with 2 layers uses temporal_kernel=1 for the stem
    stem_temporal = 1 if (arch in ("r3d",) and N == 2) else 3

    stem_kwargs_basic    = dict(temporal_kernel=stem_temporal, track_running=TR)
    stem_kwargs_r2plus1d = dict(track_running=TR)

    CONFIGS = {
        "r3d": (
            BasicBlock,
            [Conv3DSimple] * 4,
            [N] * 4,
            BasicStem, stem_kwargs_basic,
        ),
        "mc3": (
            BasicBlock,
            [Conv3DSimple] * 2 + [Conv3DNoTemporal] * 2,
            [N] * 4,
            BasicStem, stem_kwargs_basic,
        ),
        "rmc3": (
            BasicBlock,
            [Conv3DNoTemporal] * 2 + [Conv3DSimple] * 2,
            [N] * 4,
            BasicStem, stem_kwargs_basic,
        ),
        "r2plus1d": (
            BasicBlock,
            [Conv2Plus1D] * 4,
            [N] * 4,
            R2Plus1dStem, stem_kwargs_r2plus1d,
        ),
        "r2d": (
            BasicBlock,
            [Conv3DNoTemporal] * 4,
            [N] * 4,
            BasicStem, stem_kwargs_basic,
        ),
    }

    if arch not in CONFIGS:
        raise ValueError(f"Unknown encoder arch '{arch}'. Choose from {list(CONFIGS)}")

    block, conv_makers, layers, stem_cls, stem_kw = CONFIGS[arch]
    model = _build(arch, block, conv_makers, layers, stem_cls, stem_kw, cfg)

    if cfg.pretrained:
        _load_pretrained(model, arch)

    return model


def _load_pretrained(model: VideoResNet, arch: str) -> None:
    """Load matching torchvision pretrained weights (fc excluded)."""
    import torchvision

    _map = {
        "r3d":      lambda: torchvision.models.video.r3d_18(pretrained=True),
        "mc3":      lambda: torchvision.models.video.mc3_18(pretrained=True),
        "r2plus1d": lambda: torchvision.models.video.r2plus1d_18(pretrained=True),
    }
    if arch not in _map:
        raise ValueError(f"Pretrained weights not available for arch='{arch}'.")

    src = _map[arch]().state_dict()
    del src["fc.weight"], src["fc.bias"]

    dst = model.state_dict()
    matching = {k: v for k, v in src.items() if k in dst and dst[k].shape == v.shape}
    dst.update(matching)
    model.load_state_dict(dst, strict=False)
