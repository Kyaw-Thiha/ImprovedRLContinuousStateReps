import matplotlib.pyplot as plt
import numpy as np
import pyperclip
import nengo
import sys, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from trial_cartpole import ACTrial
import rl as net

trials = 1000
learnTrials = trials
ac = ACTrial()

# Reward Centering configs
reward_center_mode = "none"  # "none", "simple", "value"
reward_center_beta = 0.001
reward_center_eta = 1.0
reward_center_init = 0.0

representation_name = "TileCoding"

data_dir_ = os.path.join(
    REPO_ROOT, "data", "reward_centering_20trials", representation_name, reward_center_mode
)
os.makedirs(data_dir_, exist_ok=True)

for i in range(20):
    pre_comment_ = f"rep={representation_name}, reward_center={reward_center_mode}, run={i}"

    metadata = ac.run(
        seed=i,
        trials=trials,
        ### environment parameters ###
        steps=500,
        env="CartPole-v1",
        n_done=1,
        gifs=False,
        ### model-specific parameters ###
        rep_="TileCoding",
        num_tilings=16,
        tiles_per_dim=(8, 8, 8, 8),
        iht_size=65536,
        tile_state_indices=(0, 1, 2, 3),
        eps=0.259453,
        lr=0.283112,
        ###
        ### common model parameters
        rule="TD0",
        dynamic_epsilon=True,
        act_dis=0.9,
        state_dis=0.99,
        learnTrials=learnTrials,
        on_policy_override=False,
        force_rho_one=True,
        ###
        ### Reward Centering
        reward_center_mode=reward_center_mode,
        reward_center_beta=reward_center_beta,
        reward_center_eta=reward_center_eta,
        reward_center_init=reward_center_init,
        ###
        ### data saving specifications
        verbose=False,
        data_dir=data_dir_,
        pre_comment=pre_comment_,
        ###
    )

    print(metadata["trial_ID"])
    print("episodes to learn: ", metadata["episodes_to_learn"])
    print("terminal reward, learning: ", metadata["terminal_reward_learning"])
    print("terminal reward: ", metadata["terminal_reward"])
    print("dimensionality: ", metadata["dimensionality"])

    post_comment = f"rep={representation_name}, reward_center={reward_center_mode}, run={i}"
    with open(os.path.join(data_dir_, "{}.txt".format(metadata["trial_ID"])), "a") as data_file:
        data_file.write("post_comment = " + post_comment)
