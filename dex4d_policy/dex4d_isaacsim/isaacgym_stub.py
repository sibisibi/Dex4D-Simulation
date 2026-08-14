"""A refusing stub for `isaacgym`, so one PPO serves both backends.

The RL stack imports gymapi at module level but only uses it in the testing
branch, which the lab path never takes. Every attribute here raises, so a real
reach into gymapi fails loudly. Call install() before importing the PPO.
"""
from __future__ import annotations

import sys
from types import ModuleType


class _Refuse:
    """Any attribute access raises, naming what was reached for."""

    def __init__(self, path="isaacgym"):
        object.__setattr__(self, "_path", path)

    def __getattr__(self, name):
        raise RuntimeError(
            f"{self._path}.{name} was touched inside an Isaac Sim process. "
            f"isaacgym cannot be imported here.")

    def __call__(self, *a, **k):
        raise RuntimeError(f"{self._path} was called inside an Isaac Sim process.")


# Dunders must answer normally, `inspect.getmodule` probes `__file__` while
# building tracebacks and refusing those would mask unrelated errors.
_ALLOW = {"__file__": None, "__path__": [], "__all__": [], "__spec__": None,
          "__loader__": None, "__package__": "", "__doc__": None}


def _make_refuse(path):
    def _getattr(name):
        if name in _ALLOW:
            return _ALLOW[name]
        raise RuntimeError(
            f"{path}.{name} was touched inside an Isaac Sim process. "
            f"isaacgym cannot be imported here.")
    return _getattr


def install() -> None:
    """Put the refusing stub in sys.modules if isaacgym is genuinely absent."""
    existing = sys.modules.get("isaacgym")
    if existing is not None and getattr(existing, "_is_refusing_stub", False):
        return
    if existing is None:
        try:
            import isaacgym as existing  # noqa: F401
        except ImportError:
            existing = None
    # A namespace package, or anything without a usable gymapi, still needs
    # replacing. Only a working isaacgym is left alone.
    if existing is not None and getattr(existing, "__file__", None) is not None:
        try:
            from isaacgym import gymapi  # noqa: F401
            return
        except Exception:
            pass
    mod = ModuleType("isaacgym")
    # The RL stack does `from isaacgym import gymutil, gymapi` and
    # `from isaacgym.torch_utils import *`, so from-imports must resolve too.
    for name in ("gymapi", "gymutil", "gymtorch", "torch_utils"):
        sub = ModuleType(f"isaacgym.{name}")
        sub.__getattr__ = _make_refuse(f"isaacgym.{name}")
        setattr(mod, name, sub)
        sys.modules[f"isaacgym.{name}"] = sub
    mod.__getattr__ = _make_refuse("isaacgym")
    mod._is_refusing_stub = True
    sys.modules["isaacgym"] = mod
