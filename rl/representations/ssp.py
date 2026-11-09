"""SSP-based state representation for the Stage-0 DQN path.

Wraps the upstream `sspspace` package (pinned in requirements-stage0.txt)
to provide a tiny adapter class with the surface that the rest of the
codebase expects from a state encoder:

    rep = SSPRep(domain_dim=4, length_scale=1.0, n_rotates=5, n_scales=5)
    phi = rep.map(state)            # 1-D ndarray of shape (ssp_dim,)

The encoding math (matrix-multiply → complex exponential → inverse FFT)
lives entirely in upstream `sspspace`. This module is just the glue:
the SSPEncoder.handle / ssp_dim / encode pipeline, plus the `.v` unwrap
required by upstream's wrapped SSP return type.
"""

import numpy as np
from sspspace import HexagonalSSPSpace


class SSPRep:
    """HexSSP state representation. Forward-only (no unmap needed for RL)."""

    def __init__(
        self,
        domain_dim: int,
        length_scale: float = 1.0,
        n_rotates: int = 5,
        n_scales: int = 5,
        kernel: str = "hypergeom",
    ):
        self._encoder = HexagonalSSPSpace(
            domain_dim=domain_dim,
            n_rotates=n_rotates,
            n_scales=n_scales,
            kernel=kernel,
            length_scale=length_scale,
        )
        self.size_out = self._encoder.ssp_dim

    def map(self, state):
        # upstream encode() returns a wrapped SSP object; .v is the ndarray.
        # .v must be reshaped, not the wrapper — see docs/handoffs/01-stage0-repo-rebuild.md §2.
        return self._encoder.encode(np.atleast_2d(state)).v.reshape(-1)

    def get_state(self, state, env=None):
        return state
