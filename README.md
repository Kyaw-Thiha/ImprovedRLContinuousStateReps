# Improving Reinforcement Learning with Biologically Motivated Continuous State Representations

Repository to accompany [Bartlett, Simone, Dumont, Furlong, Eliasmith, Orchard & Stewart (2022)]() "Improving Reinforcement Learning with Biologically Motivated Continuous State Representations" ICCM Paper.

## Reproducing results and figures

**Note: Make sure to import all dependencies listed in the *requirements.txt* file**

Hyperparameter optimization
* To run hyperparameter optimization for a given network architecture, do: `nnictl create --config [CONFIG_FILENAME].YML`
* For RatBox, the configuration files are located in `ratboxExperiments/nni_exps`. The network configuration for a given experiment is indicated in the file name. 
* For Cartpole, configuration files are in /cartpoleExperiments. To modify optimization for a particular state discretization in the tabular approach, modify n_bins_ parameter within the `exp_cartpole_discrete.py` script.

To reproduce results: RatBox
* `ratboxExperiments/getBestParams.ipynb` reports the best hyperparameters found for RatBox
* *exp.py files in `ratboxExperiments/run10` each run RatBox experiments with 10 random seeds and saves results
* `ratboxExperiments/exploreData.ipynb` generates various plots and results

To reproduce results: CartPole
* `cartpoleExperiments/run_trial_cartpole_repeats.py` will run a particular network architecture 10 times each initialized with a different randomly selected seed. The best-performing hyperparameters identified from our hyperparameter optimization are in the script. Uncomment the relevant lines as instructed in the script to run different network architectures.
* `cartpoleData/parse_trial_metadata.py` creates a summary of the metadata associated with each trial (network architecture, performance, etc.) and saves it at metadata-summary.csv
* `cartpoleData/merge_episodic_rwds.py` extracts reward data for all models across the 10 initial seeds and merges this into one file saved at cartpoleData/processed/all-episodic-rewards.csv (for easy plotting)
* `cartpoleExperiments/a2c_baseline_cartpole.py` runs the baseline model on CartPole and saves the results

For running the reward centering experiment with discrete representation,
```bash
python cartpoleExperiments/run_reward_centering.py
```

For running the reward centering with hex ssp representation
```bash
python cartpoleExperiments/run_hexssp_centering.py
```

After that, you can parse and merge the raw data for further analysis.
```bash
python cartpoleData/parse_reward_centering_metadata.py
```

```bash
python cartpoleData/merge_reward_centering_rwds.py
```

Then, you can generate plots for those using
```bash
python cartpoleExperiments/plotting/plot_cartpole_ablation.py \
    --metadata cartpoleData/processed/reward-centering-metadata-summary.csv \
    --rewards cartpoleData/processed/reward-centering-all-episodic-rewards.csv \
    --output-dir cartpoleExperiments/plotting/output/reward-centering \
    --prefix reward-centering
```

Reproducing manuscript figures
* `figures/generate_figure4.py` and `figures/generate_figure5.py` generates figures 4 and 5 as they appear in the paper
* `figures/generate_figure6.py` makes the comparison plot and reports the statistics used in the paper

## To Cite this Work
When citing this work please use the following: <br>
Bartlett, M., Simone, K., Dumont, N. S.-Y., Furlong, P., Eliasmith, C., Orchard, J., & Stewart, T. Improving Reinforcement Learning with Biologically Motivated Continuous State Representations. Via mathpsych.org/presentation/1221.
