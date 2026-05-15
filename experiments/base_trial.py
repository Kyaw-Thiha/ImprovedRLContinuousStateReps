import pytry
import nengo


class BaseTrial(pytry.Trial):
    """Shared parameter definitions for all trial types (AC, DQN, etc.).

    Subclasses add algorithm-specific params and implement evaluate().
    """

    def params(self):
        ## Task Parameters
        self.param("Number of learning trials", trials=1000)
        self.param("Number of time steps per trial", steps=500)
        self.param("Number of time steps on done", n_done=1)
        self.param("Number of time steps on reset", n_reset=1)
        self.param("Task or Environment", env="CartPole-v1")
        self.param("Duration of task time step", env_dt=0.001)

        ## Output / Logging
        self.param("Create render gifs", gifs=False)
        self.param("Comments prior to running trial", pre_comment="N/A")
