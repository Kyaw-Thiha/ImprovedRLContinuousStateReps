"""rl.rules: learning-rule implementations (legacy, Nengo-bound).

All entries are lazy-loaded via PEP 562 `__getattr__`. The Stage-0 DQN
path never touches this package; the legacy ACTrial code accesses
`net.rules.TD0Center` etc. on the legacy Modal env (requirements.txt)
where Nengo is installed.
"""

__all__ = ["TD0", "TD0iG", "TDn", "TDL", "TDt", "TD0Center"]


def __getattr__(name):
    if name == "TD0":
        from .td0_nengo import TD0
        globals()["TD0"] = TD0
        return TD0
    if name == "TD0iG":
        from .td0iG import TD0iG
        globals()["TD0iG"] = TD0iG
        return TD0iG
    if name == "TDn":
        from .tdn import TDn
        globals()["TDn"] = TDn
        return TDn
    if name == "TDL":
        from .tdlambda import TDL
        globals()["TDL"] = TDL
        return TDL
    if name == "TDt":
        from .tdtheta import TDt
        globals()["TDt"] = TDt
        return TDt
    if name == "TD0Center":
        from .td0centering_nengo import TD0Center
        globals()["TD0Center"] = TD0Center
        return TD0Center
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(__all__))
