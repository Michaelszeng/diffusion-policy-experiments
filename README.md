# Diffusion Policy Experiments

Michael's diffusion policy training repository based off of [original diffusion policy paper](https://diffusion-policy.cs.columbia.edu/) and work done by Abhinav Agarwal and Adam Wei. This is a much cleaned-up version of the original repo.

This codebase currently implements only U-Net-based architectures, and supports Robomimic, R3M, and from-scratch ResNet image encoders as well as Timm image encoders (including CLIP and DINOv3). Architectures include both traditional FiLM-conditioned U-Nets, cross-attention conditioned U-Nets from [1](https://dp-with-long-context.github.io/), and cross-attention conditioned U-Nets with the double-encoder from [2](TODO).

Additional Features:
- Multi-GPU training via HuggingFace `accelerate`
- `bf16`-mixed precision
- Multi-dataset mixing via weighted `zarr_configs` (episode caps + sampling weights)
- Long observation horizons with efficient `key_first_k` sampling (only the used obs steps are loaded)


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

2. **Add Poetry to your PATH**:
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

### SLURM Cluster Installation

```bash
python3 -m venv env --without-pip
source env/bin/activate
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python3 /tmp/get-pip.py --no-warn-script-location
pip install -r requirements.txt
```

### Secrets (Weights & Biases)

Training scripts expect a `.secrets` file with your W&B API key. Copy the template and fill in your key:

```bash
cp .secrets.template .secrets
```

Then edit `.secrets` and replace `your_wandb_api_key_here` with your key from [wandb.ai/authorize](https://wandb.ai/authorize). `.secrets` is gitignored — never commit it.


## Running

Before training policy, you must write a `.yaml` config file that defines the policy architecture, dataset path, and various training parameters. See `config/` for examples.

### Running locally:
```bash
python train.py --config-dir=config/planar_pushing --config-name=2_obs.yaml hydra.run.dir=data/outputs/planar_pushing/2_obs/ dataloader.batch_size=4 val_dataloader.batch_size=4
```

To resume training from an interrupted run: in your config file, set `training.resume: true`. Optionally (recommended), also set `logging.resume: true` and set `logging.id` to the wandb ID that you want to continue from (i.e. "thfb8nrq" for training run `data/outputs/maniskill/2_obs/wandb/offline-run-20251118_134543-thfb8nrq`).


### Running on a SLURM Cluster:

Set your cluster parameters at the top of the sbatch file, then set your config directory and name, then:

```bash
sbatch submit_training_cluster.sbatch
```



## Important Configurations

Probably the most important configurations are `horizon` (the prediction horizon) and `n_obs_steps` (the observation horizon/context length).

Note that `n_obs_steps` overlaps with `horizon` in this sense:
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

The network will learn to re-predict those already-executed actions as an auxiliary prediction objective.


## Running Inference

This repo primarily handles policy training, and assumes the evaluation environment will provide scripts for loading the policy checkpoint and querying it.

However, `policy_inference.py` exists for running inference on hardware setups; the script loads a trained checkpoint, binds a ZMQ REP socket, and, when a client sends a request, the script runs the policy and serves the outputted action predictions. Specifically, clients send an observation window (raw PNG frames for the newest timesteps, optionally cached encoder features for older ones, plus low-dim state) to the ZMQ server; the server encodes only the new frames, runs DDIM/DDPM denoising, and returns the predicted future actions along with the newly computed features so the client can keep its cache. It currently supports `DiffusionUnetTimmAttentionPolicy` and `DiffusionUnetTimmFilmPolicy`.

```bash
python policy_inference.py -i path/to/checkpoint.ckpt --ip 0.0.0.0 --port 8766 --device cuda:0
```

You will have to write the client-side yourself.




## Citation

If you use this repo in your work, please cite [Revisiting Open-Loop Execution in Robotics: Toward Reactive, Higher-Performing Policies]():

```
TODO
```