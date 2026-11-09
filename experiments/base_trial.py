import pytry


class BaseTrial(pytry.Trial):
    """Shared parameter definitions for all trial types (AC, DQN, etc.).

    Subclasses add algorithm-specific params and implement evaluate().

    Nengo-free on purpose: this base class is shared by both the Stage-0
    DQN path and the legacy ACTrial path. ACTrial-specific Nengo imports
    live in trial_cartpole_nengo.py; the legacy env (requirements.txt, Python
    3.9, nengo 3.2) is what runs ACTrial on Modal. The Stage-0 path runs
    on the modern env (requirements-stage0.txt, Python 3.11+) which does
    not install Nengo at all.
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
        self.param("Log metrics to Weights & Biases", use_wandb=False)
