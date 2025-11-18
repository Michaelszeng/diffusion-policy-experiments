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
from omegaconf import OmegaConf

from diffusion_policy.workspace.base_workspace import BaseWorkspace

# Set HYDRA_FULL_ERROR=1 for detailed error reporting
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["WANDB_START_METHOD"] = "thread"

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
