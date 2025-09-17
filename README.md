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
python train.py --config-dir=config/planar_pushing --config-name=2_obs.yaml hydra.run.dir=data/outputs/planar_pushing/2_obs/

# Non-interactively:
LLsub ./submit_training.sh -s 20 -g volta:1
```

Remember to also obtain and `desktop_to_supercloud.sh` (if running on Supercloud) the dataset files.