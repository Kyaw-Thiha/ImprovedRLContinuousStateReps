"""
Tile Coding Software version 3.0beta
by Rich Sutton
based on a program created by Steph Schaeffer and others
External documentation and recommendations on the use of this code is available in the
reinforcement learning textbook by Sutton and Barto, and on the web.
These need to be understood before this code is.

This software is for Python 3 or more.

This is an implementation of grid-style tile codings, based originally on
the UNH CMAC code (see http://www.ece.unh.edu/robots/cmac.htm), but by now highly changed.
Here we provide a function, "tiles", that maps floating and integer
variables to a list of tiles, and a second function "tiles-wrap" that does the same while
wrapping some floats to provided widths (the lower wrap value is always 0).

The float variables will be gridded at unit intervals, so generalization
will be by approximately 1 in each direction, and any scaling will have
to be done externally before calling tiles.

Num-tilings should be a power of 2, e.g., 16. To make the offsetting work properly, it should
also be greater than or equal to four times the number of floats.

The first argument is either an index hash table of a given size (created by (make-iht size)),
an integer "size" (range of the indices from 0), or nil (for testing, indicating that the tile
coordinates are to be returned without being converted to indices).
"""

import numpy as np

basehash = hash


class IHT:
    "Structure to handle collisions"

    def __init__(self, sizeval):
        self.size = sizeval
        self.overfullCount = 0
        self.dictionary = {}

    def __str__(self):
        "Prepares a string for printing whenever this object is printed"
        return (
            "Collision table:"
            + " size:"
            + str(self.size)
            + " overfullCount:"
            + str(self.overfullCount)
            + " dictionary:"
            + str(len(self.dictionary))
            + " items"
        )

    def count(self):
        return len(self.dictionary)

    def fullp(self):
        return len(self.dictionary) >= self.size

    def getindex(self, obj, readonly=False):
        d = self.dictionary
        if obj in d:
            return d[obj]
        elif readonly:
            return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfullCount == 0:
                print("IHT full, starting to allow collisions")
            self.overfullCount += 1
            return basehash(obj) % self.size
        else:
            d[obj] = count
            return count


def hashcoords(coordinates, m, readonly=False):
    if type(m) == IHT:
        return m.getindex(tuple(coordinates), readonly)
    if type(m) == int:
        return basehash(tuple(coordinates)) % m
    if m == None:
        return coordinates


from math import floor, log
from itertools import zip_longest


def tiles(ihtORsize, numtilings, floats, ints=[], readonly=False):
    """returns num-tilings tile indices corresponding to the floats and ints"""
    qfloats = [floor(f * numtilings) for f in floats]
    Tiles = []
    for tiling in range(numtilings):
        tilingX2 = tiling * 2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append((q + b) // numtilings)
            b += tilingX2
        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))
    return Tiles


def tileswrap(ihtORsize, numtilings, floats, wrapwidths, ints=[], readonly=False):
    """returns num-tilings tile indices corresponding to the floats and ints, wrapping some floats"""
    qfloats = [floor(f * numtilings) for f in floats]
    Tiles = []
    for tiling in range(numtilings):
        tilingX2 = tiling * 2
        coords = [tiling]
        b = tiling
        for q, width in zip_longest(qfloats, wrapwidths):
            c = (q + b % numtilings) // numtilings
            coords.append(c % width if width else c)
            b += tilingX2
        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))
    return Tiles


class TileCodingRep(object):
    """Multi-hot tile-coding representation for continuous states."""

    def __init__(
        self,
        env,
        num_tilings=8,
        tiles_per_dim=None,
        iht_size=4096,
        bounds_low=None,
        bounds_high=None,
        state_indices=None,
    ):
        self.env = env
        self.num_tilings = num_tilings
        self.iht = IHT(iht_size)
        self.size_out = iht_size
        # Exactly `num_tilings` tiles are active for any state (one per tiling) —
        # DQN uses this to switch to a sparse index-based encode/update path
        # instead of materializing a dense `size_out`-length one-hot vector.
        # See map_indices() below and rl/networks/dqnBasic.py.
        self.n_active = num_tilings

        obs_dim = len(self.env.observation_space.high)
        if state_indices is None:
            state_indices = tuple(range(obs_dim))
        self.state_indices = np.asarray(state_indices, dtype=int)

        if tiles_per_dim is None:
            tiles_per_dim = (8,) * len(self.state_indices)
        self.tiles_per_dim = np.asarray(tiles_per_dim, dtype=float)

        if bounds_low is None:
            bounds_low = np.asarray(self.env.observation_space.low, dtype=float)
        if bounds_high is None:
            bounds_high = np.asarray(self.env.observation_space.high, dtype=float)

        self.lower = np.asarray(bounds_low, dtype=float)[self.state_indices]
        self.upper = np.asarray(bounds_high, dtype=float)[self.state_indices]
        self.ranges = self.upper - self.lower
        self.result = np.zeros(self.size_out)

    def _select_state(self, state):
        state = np.asarray(state, dtype=float)
        if state.shape[0] == len(self.state_indices):
            return state
        return state[self.state_indices]

    def _active_tiles(self, state):
        state = self._select_state(state)
        clipped = np.clip(state, self.lower, self.upper)
        scaled = (clipped - self.lower) / self.ranges * self.tiles_per_dim
        return tiles(self.iht, self.num_tilings, scaled)

    def map(self, state):
        """Dense multi-hot encoding (length size_out). Kept for callers that need
        a full feature vector (e.g. the A2C/Nengo pipeline in trial_cartpole_nengo.py)."""
        active_tiles = self._active_tiles(state)
        self.result[:] = 0
        self.result[active_tiles] = 1
        return self.result.copy()

    def map_indices(self, state):
        """Sparse encoding: just the `num_tilings` active tile indices (int64).

        Equivalent information to map()'s one-hot vector, but O(num_tilings)
        instead of O(size_out) to produce and to consume — size_out can be up
        to 131072 while num_tilings is at most 32. Used by DQN (dqnBasic.py),
        which detects this method via hasattr() and switches its replay buffer
        + update() math to index-based gather/scatter ops instead of dense
        matmuls over size_out. See docs/07-decision-log.md for the profiling
        that motivated this (dense update() cost was ~0.23us/dim, so at
        size_out=131072 a single gradient update cost ~30ms vs ~1.6ms at 4096).
        """
        return np.asarray(self._active_tiles(state), dtype=np.int64)

    def get_state(self, state, env=None):
        state = self._select_state(state)
        return np.clip(state, self.lower, self.upper)
