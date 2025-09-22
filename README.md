# Diffusion Policy Experiments

Michael's personal diffusion policy experiments/implementation/research in the Robot Locomotion Group, based off of [original diffusion policy paper](https://diffusion-policy.cs.columbia.edu/) and work done by Abhinav Agarwal and Adam Wei .

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

### Prerequisites
- Python 3.9 to 3.11
- Poetry (for dependency management)

### Local Installation:

1. **Install Poetry** (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. **Add Poetry to your PATH** (add this to your `~/.bashrc` or `~/.zshrc`):
```bash
export PATH="/home/$USER/.local/bin:$PATH"
```

3. **Install dependencies**:
```bash
poetry install
```

4. **Activate the Poetry environment**:
```bash
source $(poetry env info --path)/bin/activate
```

### Supercloud Installation:
```
module load anaconda/2023b
pip install huggingface-hub==0.25.2 --no-deps
pip install diffusers==0.11.1 --no-deps
pip install numba==0.60.0
pip install wandb
pip install einops
pip install zarr
```

### Obtaining Training Data
Obtain `sim_sim_tee_data_carbon_large.zarr` from Adam Wei. Place in `data/diffusion_experiments/planar_pushing/sim_sim_tee_data_carbon_large.zarr` folder.


## Running

### Running locally
```bash
python train.py --config-dir=config/planar_pushing --config-name=2_obs.yaml hydra.run.dir=data/outputs/planar_pushing/2_obs/
```

### Running on Supercloud:
```bash
# Interactively:
LLsub -i full
module load anaconda/2023b
wandb offline  # Supercloud compute nodes have no internet
python train.py --config-dir=config/planar_pushing --config-name=2_obs.yaml hydra.run.dir=data/outputs/planar_pushing/2_obs/

# Non-interactively:
module load anaconda/2023b
wandb offline  # Supercloud compute nodes have no internet
LLsub ./submit_training.sh -s 20 -g volta:1
```

Remember to also obtain and `desktop_to_supercloud.sh` (if running on Supercloud) the dataset files.


## Important Configurations

Probably the most important configurations are `horizon`, `n_obs_steps`, and `past_action_visible`.

`n_obs_steps` overlaps with `horizon` in this sense:
```
time axis :  0 1 | 2 3 4 … 15
             ^^^^^ n_obs_steps = 2
             ^^^^^^^^^^^^^^^ horizon T = 16
```

The trajectory occurs over the entire horizon and therefore may have past actions included as well:
```
t = 0,1     … To-1        (past two actions)      ← have already happened
t = 2 … 15                  (future actions)      ← should be predicted
```

The network will learn to re-predict those already-executed actions.

If `past_actions_visible` is true, then the first `n_obs_steps - 1` ground-truth (unnoised) steps will be passed to the model effectively as conditioning, and loss will be compared only against the `horizon - (n_obs_steps - 1)` predicted actions. (Note that we subtract `1` from `n_obs_steps` because actions happen after observations; i.e. if we have 2 observations, then we have already executed the action for the first observation, and need to predict the action for corresponding to the second observation).


Note that this implementation uses global observation conditioning (i.e. observations are not part of per-timestep token, if transformer variant is used).