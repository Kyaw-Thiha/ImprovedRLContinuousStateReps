import numpy as np

class OneHotRepRB(object):
    '''Create one-hot representation. I.e. the state is represented as a list of 0's and a 1. 
    This method works with RatBox.
    '''
    def __init__(self, ranges):
        ##step 1
        self.ranges = ranges
        self.factors = np.array([np.prod(ranges[x+1:]) for x in range(len(ranges))], dtype=int)
        ##step 2
        self.result = np.zeros(np.prod(ranges))
        self.size_out = len(self.result)

    def map(self, state):
        index = 0
        ##step 3
        for i,v in enumerate(state):
            index += v*self.factors[i]
        self.result[:] = 0
        ##step 4
        #print(f'checking index {index}')
        if index%1 == 0:
            index=int(index)
        self.result[index] = 1
        return self.result.copy()

    def get_state(self, state, env=None):
        discrete_state = state

        state_space_high = np.asarray([env.width, env.height, 360])
        state_space_low = np.asarray([0, 0, 0])
        
        discrete_obs_size = np.array(self.ranges)

        discrete_obs_size = np.array(discrete_obs_size, dtype='int')
        discrete_obs_win_size = (state_space_high - state_space_low)/discrete_obs_size
        discrete_state = (state - state_space_low)/discrete_obs_win_size

        # Clip to valid bin range. Without this, an exact-upper-boundary state
        # produces index == n_bins (out of range). Latent bug — never triggered
        # with the loose CartPole bounds (pole angle threshold 0.2094 < obs-space
        # upper 0.418), but a one-line guard fixes it.
        discrete_state = tuple(np.clip(discrete_state, 0, np.array(discrete_obs_size) - 1).astype(int))
        
        return discrete_state
    
class OneHotRepCP(object):
    '''Create one-hot representation. I.e. the state is represented as a list of 0's and a 1.
    This method works with CartPole v1.
    '''
    def __init__(self, ranges, bounds_low=None, bounds_high=None):
        ##step 1
        self.ranges = ranges
        self.factors = np.array([np.prod(ranges[x+1:]) for x in range(len(ranges))], dtype=int)
        ##step 2
        self.result = np.zeros(np.prod(ranges))
        self.size_out = len(self.result)
        self.bounds_low = bounds_low
        self.bounds_high = bounds_high
        # A one-hot vector has exactly one nonzero entry — DQN (dqnBasic.py)
        # detects this via n_active and switches to a sparse index-based
        # encode/update path instead of materializing the full size_out-length
        # (up to n_bins^4) one-hot vector, same mechanism as TileCodingRep.
        self.n_active = 1

    def _index(self, state):
        index = 0
        ##step 3
        for i,v in enumerate(state):
            index += v*self.factors[i]
        ##step 4
        if index%1 == 0:
            index=int(index)
        return index

    def map(self, state):
        """Dense one-hot vector (length size_out). Kept for callers that need
        a full feature vector (e.g. the A2C/Nengo pipeline)."""
        index = self._index(state)
        self.result[:] = 0
        self.result[index] = 1
        return self.result.copy()

    def map_indices(self, state):
        """Sparse encoding: the single active index (int64), see TileCodingRep
        .map_indices for the equivalent DQN-side mechanism this plugs into."""
        return np.array([self._index(state)], dtype=np.int64)

    def get_state(self, state, env=None):

        discrete_state = state

        state_space_low  = self.bounds_low  if self.bounds_low  is not None else env.observation_space.low
        state_space_high = self.bounds_high if self.bounds_high is not None else env.observation_space.high
        
        discrete_obs_size = np.array(self.ranges)
        
        discrete_obs_size = np.array(discrete_obs_size, dtype = 'int')
        discrete_obs_win_size = (state_space_high - state_space_low) / discrete_obs_size
        
        discrete_state = (state - state_space_low)/discrete_obs_win_size
        discrete_state = tuple(discrete_state.astype(int))
        
        return discrete_state