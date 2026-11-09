"""rl: stage-0 representation + network utilities.

Submodules (`representations`, `networks`, `rules`) are lazy-loaded via
PEP 562 `__getattr__` so that `import rl` does not pull in Nengo-bound
modules. The Stage-0 modern env (requirements-stage0.txt) does not install
Nengo; the legacy ACTrial path runs on the legacy env (requirements.txt,
Modal) where Nengo IS installed and the lazy attribute resolution succeeds
when `net.networks.ActorCritic` etc. are first accessed.
"""

from .utils import sparsity_to_x_intercept, softmax

_LAZY_SUBMODULES = ("representations", "networks", "rules")


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        import importlib
        mod = importlib.import_module("." + name, __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_SUBMODULES))
