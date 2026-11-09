"""rl.networks: network implementations.

Each network is lazy-loaded via PEP 562 `__getattr__`. This keeps
`import rl.networks` cheap and side-effect-free on the Stage-0 modern
env (which has no Nengo installed). The Nengo-bound classes
(ActorCritic, ActorCriticLDN, LDN) only fail to import when actually
requested — same behaviour they always had, but without forcing the
package-level failure on DQN-only consumers.
"""

__all__ = ["ActorCritic", "ActorCriticLDN", "LDN", "DQN"]


def __getattr__(name):
    if name == "ActorCritic":
        from .acBasic_nengo import ActorCritic
        globals()["ActorCritic"] = ActorCritic
        return ActorCritic
    if name == "ActorCriticLDN":
        from .acLDN import ActorCriticLDN
        globals()["ActorCriticLDN"] = ActorCriticLDN
        return ActorCriticLDN
    if name == "LDN":
        from .ldn import LDN
        globals()["LDN"] = LDN
        return LDN
    if name == "DQN":
        from .dqnBasic import DQN
        globals()["DQN"] = DQN
        return DQN
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(__all__))
