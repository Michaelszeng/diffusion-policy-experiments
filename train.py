"""
Usage:
Training:
python train.py --config-dir=config --config-name=<config_name> \
    hydra.run.dir=data/outputs/`date +"%Y.%m.%d"`/`date +"%H.%M.%S"`_<desc> \
    task.dataset.zarr_path=<data_path>
"""

import os
import sys

import hydra
import torch
from omegaconf import OmegaConf

from diffusion_policy.workspace.base_workspace import BaseWorkspace

# Set HYDRA_FULL_ERROR=1 for detailed error reporting
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["WANDB_START_METHOD"] = "thread"

# Fail fast if CUDA isn't actually usable. Without this guard, PyTorch silently
# falls back to CPU when `cuInit()` fails (e.g. wedged driver state or broken
# cgroup GPU device permissions after a slurmd restart), and training proceeds
# at ~hundreds of seconds per iteration with no obvious error. Set
# ALLOW_CPU_TRAINING=1 to bypass for intentional CPU debug runs.
if os.environ.get("ALLOW_CPU_TRAINING", "0") != "1":
    assert torch.cuda.is_available() and torch.cuda.device_count() > 0, (
        "CUDA is not available - refusing to train on CPU. "
        "If running under SLURM, the assigned node likely has a bad cgroup or "
        "driver state (check stderr for 'CUDA driver initialization failed'); "
        "requeue the job, ideally with --exclude=<bad_node>. "
        "Set ALLOW_CPU_TRAINING=1 to override."
    )

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=None,
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
