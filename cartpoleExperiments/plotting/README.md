# CartPole Plotting

Reusable plotting scripts for processed CartPole ablation outputs.

## Main script

`plot_cartpole_ablation.py` expects:
- a metadata summary CSV with columns including `rep_`, `reward_center_mode`, `seed`, `terminal_reward`, and `episodes_to_learn`
- a merged episodic rewards CSV whose columns are unique `trial_ID` values matching the metadata CSV

Example:

```bash
python cartpoleExperiments/plotting/plot_cartpole_ablation.py \
  --metadata cartpoleData/processed/reward-centering-metadata-summary.csv \
  --rewards cartpoleData/processed/reward-centering-all-episodic-rewards.csv \
  --output-dir cartpoleExperiments/plotting/output/reward-centering \
  --prefix reward-centering
```

Outputs:
- learning-curve figure with a 5th-95th percentile band by default
- terminal-reward and episodes-to-learn summary panels
- interaction plot for terminal reward
- interaction plot for episodes to learn
- summary statistics table as CSV
