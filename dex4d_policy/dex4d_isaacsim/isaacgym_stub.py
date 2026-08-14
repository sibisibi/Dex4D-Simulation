"""A refusing stub for `isaacgym`, so one PPO serves both backends.

Dex4D's `algorithms/rl/ppo/ppo.py:25` imports `VideoRecorder` and `MetricWriter`
from `utils/result_recorder.py`, and that module does `from isaacgym import
gymapi` at import time. isaacgym cannot be imported in an Isaac Sim process, so
without this the lab side would have to fork the 2,418 line RL stack, which is
exactly the shared thing worth keeping.

`gymapi` is only touched inside `VideoRecorder`, which the lab path never
constructs. `VideoRecorder` is built at `ppo.py:157`, which is inside the
`is_testing` branch, and the lab runs training. `MetricWriter` uses no gymapi at
all.

So every attribute of this stub raises on access. If the lab path ever does
reach into gymapi, it fails loudly at that line rather than quietly doing
something else. Install it with `install()` BEFORE importing the PPO.
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
            f"isaacgym cannot be imported here. The lab path is not supposed to "
            f"reach gymapi, so this is a real bug rather than a missing stub.")

    def __call__(self, *a, **k):
        raise RuntimeError(f"{self._path} was called inside an Isaac Sim process.")


# Dunders have to answer normally. `inspect.getmodule` probes `__file__` while
# building tracebacks and during `from x import *`, so refusing those turns any
# unrelated error into this one and hides it.
_ALLOW = {"__file__": None, "__path__": [], "__all__": [], "__spec__": None,
          "__loader__": None, "__package__": "", "__doc__": None}


def _make_refuse(path):
    def _getattr(name):
        if name in _ALLOW:
            return _ALLOW[name]
        raise RuntimeError(
            f"{path}.{name} was touched inside an Isaac Sim process. isaacgym "
            f"cannot be imported here. The lab path is not supposed to reach it, "
            f"so this is a real bug rather than a missing stub.")
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
    # A namespace package, or anything without a usable gymapi, is not a real
    # isaacgym and must still be replaced. Only a working one is left alone.
    if existing is not None and getattr(existing, "__file__", None) is not None:
        try:
            from isaacgym import gymapi  # noqa: F401
            return                        # a real one exists, leave it alone
        except Exception:
            pass
    mod = ModuleType("isaacgym")
    # utils/util.py:4 does `from isaacgym import gymutil, gymapi`, and
    # torch_jit_utils does `from isaacgym.torch_utils import *`, so the stub has
    # to satisfy a from-import of each name as well as attribute access.
    for name in ("gymapi", "gymutil", "gymtorch", "torch_utils"):
        sub = ModuleType(f"isaacgym.{name}")
        sub.__getattr__ = _make_refuse(f"isaacgym.{name}")
        setattr(mod, name, sub)
        sys.modules[f"isaacgym.{name}"] = sub
    mod.__getattr__ = _make_refuse("isaacgym")
    mod._is_refusing_stub = True
    sys.modules["isaacgym"] = mod
