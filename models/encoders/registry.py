"""
models/encoders/registry.py — Encoder registry for WiTA v2.

Provides a lightweight, dict-based registry so new backbone families
(e.g. Video Swin Transformer, ViViT, TimeSformer) can be registered
without touching WiTAHybridModel or build_model().

Phase 1 (current)
-----------------
Only the VideoResNet family is built-in. The registry is pre-populated
with all five variants and build_encoder() delegates to it.

Phase 2 (planned)
-----------------
To add Video Swin Transformer:

    1. Implement the backbone in models/encoders/swin3d.py, giving it
       the same interface as VideoResNet:
         • __init__(cfg: EncoderConfig)
         • forward(x: Tensor[B, C, T, H, W]) → Tensor[B, T', out_dim]

    2. Register it at module import time:

        # models/encoders/swin3d.py  (bottom of file)
        from models.encoders.registry import register_encoder
        register_encoder("swin_t", VideoSwinTiny)
        register_encoder("swin_s", VideoSwinSmall)

    3. Add the new arch names to EncoderConfig.arch's Literal type hint.

    4. That's it — build_encoder() and WiTAHybridModel pick it up
       automatically.

Registry contract
-----------------
Every registered class must accept a single ``cfg: EncoderConfig``
argument and expose:
  • cfg.out_dim  — output feature dimension (set by the encoder)
  • forward(x)  — returns [B, T', out_dim]
"""
from __future__ import annotations

from typing import Callable, Type
import torch.nn as nn

from ...configs.default import EncoderConfig


# ---------------------------------------------------------------------------
# Internal registry storage
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Type[nn.Module]] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_encoder(name: str, cls: Type[nn.Module]) -> Type[nn.Module]:
    """
    Register an encoder class under *name*.

    Can be used as a decorator or called directly:

        @register_encoder("my_net")
        class MyNet(nn.Module): ...

        # or:
        register_encoder("my_net", MyNet)

    Parameters
    ----------
    name : str
        Arch string that matches EncoderConfig.arch.
    cls  : nn.Module subclass
        Constructor must accept a single EncoderConfig argument.

    Returns
    -------
    cls  — unchanged, so the decorator form works.
    """
    if name in _REGISTRY:
        raise ValueError(
            f"Encoder '{name}' is already registered. "
            f"Existing entry: {_REGISTRY[name].__module__}.{_REGISTRY[name].__qualname__}"
        )
    _REGISTRY[name] = cls
    return cls


def list_encoders() -> list[str]:
    """Return all registered encoder names (sorted)."""
    return sorted(_REGISTRY.keys())


def build_encoder(cfg: EncoderConfig) -> nn.Module:
    """
    Instantiate the encoder for *cfg.arch*.

    Raises
    ------
    KeyError if the arch name is not in the registry.
    """
    arch = cfg.arch.lower()
    if arch not in _REGISTRY:
        available = ", ".join(list_encoders()) or "(none)"
        raise KeyError(
            f"Unknown encoder arch '{arch}'. "
            f"Registered encoders: {available}"
        )
    return _REGISTRY[arch](cfg)


# ---------------------------------------------------------------------------
# Built-in Phase 1 registrations (VideoResNet family)
# ---------------------------------------------------------------------------
# Imported here so they execute at registry-import time.
# This keeps the registration site co-located with the implementation
# while still running automatically when the registry is first imported.

def _register_resnet_family() -> None:
    """
    Register all VideoResNet variants from models/encoders/resnet3d.py.

    Called once at module load. Deferred to a function so that circular
    import errors surface with a clear traceback rather than a silent
    partially-initialised module.
    """
    try:
        from models.encoders.resnet3d import build_encoder as _resnet_build
    except ImportError as exc:
        # Allow the registry to load even if resnet3d is unavailable
        # (e.g. during unit tests that stub out the encoders package).
        import warnings
        warnings.warn(
            f"Could not import resnet3d: {exc}. "
            "VideoResNet family will not be available in the registry.",
            ImportWarning,
            stacklevel=2,
        )
        return

    # Wrap the existing per-arch factory so it conforms to the
    # register_encoder contract (constructor takes a single EncoderConfig).
    class _ResNetWrapper(nn.Module):
        """Thin shim: routes EncoderConfig → resnet3d.build_encoder()."""
        def __init__(self, cfg: EncoderConfig):
            super().__init__()
            self._enc = _resnet_build(cfg)

        def forward(self, x):       # type: ignore[override]
            return self._enc(x)

    # The resnet3d module already handles arch dispatch internally via
    # build_encoder(cfg), so a single wrapper class covers all variants.
    for _arch in ("r3d", "mc3", "rmc3", "r2plus1d", "r2d"):
        # Create a uniquely-named subclass per arch so repr() is readable.
        _cls = type(f"VideoResNet_{_arch}", (_ResNetWrapper,), {})
        _REGISTRY[_arch] = _cls


_register_resnet_family()


# ---------------------------------------------------------------------------
# Phase 2 stub — Video Swin Transformer
# ---------------------------------------------------------------------------
#
# When you are ready to implement Video Swin support, create the file
# models/encoders/swin3d.py and add the following at the bottom:
#
#   from models.encoders.registry import register_encoder
#
#   @register_encoder("swin_t")
#   class VideoSwinTiny(nn.Module):
#       def __init__(self, cfg: EncoderConfig):
#           super().__init__()
#           # ... build the model ...
#
#       def forward(self, x: torch.Tensor) -> torch.Tensor:
#           # x: [B, C, T, H, W]  →  return [B, T', cfg.out_dim]
#           ...
#
# Then add "swin_t" (and any other swin variants) to the Literal type in
# configs/default.py:
#
#   arch: Literal["r3d","mc3","rmc3","r2plus1d","r2d","swin_t","swin_s"] = "r3d"
#
# Nothing else needs to change.
