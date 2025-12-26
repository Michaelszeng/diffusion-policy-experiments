"""
NOT WELL TESTED.

NOTE: Currently only supports single dataset.
"""

if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import copy
import os
import pathlib
import random
import shutil

import hydra
import numpy as np
import torch
import tqdm
from diffusers.training_utils import EMAModel
from omegaconf import ListConfig, OmegaConf
from torch.utils.data import DataLoader

import wandb
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainDiffusionUnetLowdimWorkspaceNoEnv(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetLowdimPolicy
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetLowdimPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseLowdimDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseLowdimDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()
        torch.save(normalizer, os.path.join(self.output_dir, "normalizer.pt"))

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self._print_dataset_diagnostics(cfg, dataset, train_dataloader, val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # Calculate total training steps based on num_epochs or total_train_steps
        if cfg.training.num_epochs is not None:
            num_training_steps = (
                len(train_dataloader) * cfg.training.num_epochs
            ) // cfg.training.gradient_accumulate_every
        else:
            num_training_steps = cfg.training.total_train_steps // cfg.training.gradient_accumulate_every

        # Calculate number of epochs if using total_train_steps
        if cfg.training.num_epochs is None:
            # Calculate epochs needed to reach total_train_steps
            steps_per_epoch = len(train_dataloader)
            num_epochs = (cfg.training.total_train_steps + steps_per_epoch - 1) // steps_per_epoch  # ceil division
            print(f"Training for {num_epochs} epochs to reach {cfg.training.total_train_steps} total steps")
        else:
            num_epochs = cfg.training.num_epochs
            print(f"Training for {num_epochs} epochs")

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=num_training_steps,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1,
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir), config=OmegaConf.to_container(cfg, resolve=True), **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint managers
        if not isinstance(cfg.checkpoint, ListConfig):  # Single checkpoint manager
            topk_managers = [
                TopKCheckpointManager(
                    save_dir=os.path.join(self.output_dir, "checkpoints"),
                    **cfg.checkpoint.topk,
                )
            ]
            save_last_ckpt = cfg.checkpoint.save_last_ckpt
            save_last_snapshot = cfg.checkpoint.save_last_snapshot
        else:  # Multiple checkpoint managers
            topk_managers = []
            save_last_ckpt = False
            save_last_snapshot = False
            for ckpt_cfg in cfg.checkpoint:
                topk_managers.append(
                    TopKCheckpointManager(
                        save_dir=os.path.join(self.output_dir, "checkpoints"),
                        **ckpt_cfg.topk,
                    )
                )
                save_last_ckpt = save_last_ckpt or ckpt_cfg.save_last_ckpt
                save_last_snapshot = save_last_snapshot or ckpt_cfg.save_last_snapshot

        # device transfer
        if cfg.training.device == "mps":
            # MPS does not support float64
            self.model = self.model.to(torch.float32)
            if self.ema_model is not None:
                self.ema_model = self.ema_model.to(torch.float32)
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(torch.float32)
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        val_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        if cfg.training.device == "mps":
                            batch = dict_apply(batch, lambda x: x.to(torch.float32))
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                        # compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) and batch_idx >= (
                            cfg.training.max_train_steps - 1
                        ):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.training.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                if cfg.training.device == "mps":
                                    batch = dict_apply(batch, lambda x: x.to(torch.float32))
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                if val_sampling_batch is None:
                                    val_sampling_batch = batch
                                loss = policy.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) and batch_idx >= (
                                    cfg.training.max_val_steps - 1
                                ):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log["val_loss"] = val_loss

                # run diffusion sampling on a validation batch
                if (self.epoch % cfg.training.sample_every) == 0 and cfg.training.log_val_mse:
                    with torch.no_grad():
                        # Get the validation batch
                        val_batch = dict_apply(
                            val_sampling_batch,
                            lambda x: x.to(device, non_blocking=True),
                        )
                        val_obs_dict = {"obs": val_batch["obs"]}
                        if cfg.policy.use_target_cond and "target" in val_batch:
                            val_obs_dict["target"] = val_batch["target"]
                        val_gt_action = val_batch["action"]

                        # Evaluate MSE when diffusing with DDPM
                        if cfg.training.eval_mse_DDPM:
                            result = policy.predict_action(val_obs_dict, use_DDIM=False)
                            pred_action = result["action_pred"]
                            mse = torch.nn.functional.mse_loss(pred_action, val_gt_action)
                            step_log["val_ddpm_mse"] = mse.item()

                        # Evaluate MSE when diffusing with DDIM
                        if cfg.training.eval_mse_DDIM:
                            result = policy.predict_action(val_obs_dict, use_DDIM=True)
                            pred_action = result["action_pred"]
                            mse = torch.nn.functional.mse_loss(pred_action, val_gt_action)
                            step_log["val_ddim_mse"] = mse.item()

                        # Free RAM
                        del val_batch
                        del val_obs_dict
                        del val_gt_action
                        del result
                        del pred_action
                        del mse

                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if save_last_ckpt:
                        self.save_checkpoint()
                    if save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace("/", "_")
                        metric_dict[new_key] = value

                    # Metric-based Top-K checkpointing
                    topk_ckpt_paths = []
                    for i, topk_manager in enumerate(topk_managers):
                        protected_ckpts = self._get_protected_paths(i, topk_managers)
                        ckpt_path = topk_manager.get_ckpt_path(metric_dict, protected_ckpts)
                        topk_ckpt_paths.append(ckpt_path)

                    for i, topk_ckpt_path in enumerate(topk_ckpt_paths):
                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)
                            break
                # last epoch => save last checkpoint
                if self.epoch == num_epochs - 1 and save_last_ckpt:
                    self.save_checkpoint()
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

    def _get_protected_paths(self, topk_manager_idx, topk_managers):
        """
        Returns the paths that should not be deleted by topk_manager
        """
        if len(topk_managers) == 1:
            return set()

        topk_manager = topk_managers[topk_manager_idx]

        protected_paths = set()
        for manager in topk_managers:
            protected_paths.update(manager.get_path_value_map().keys())

        # Remove the paths that can be deleted
        # If a ckpt is ONLY being tracked by topk_manager, it can be deleted
        for path in topk_manager.get_path_value_map().keys():
            protected = False
            for i, manager in enumerate(topk_managers):
                if i == topk_manager_idx:
                    continue
                if path in manager.get_path_value_map().keys():
                    protected = True
                    break
            if not protected:
                protected_paths.remove(path)

        return protected_paths

    def _print_dataset_diagnostics(self, cfg, dataset, train_dataloader, val_dataloader):
        """CURRENTLY ONLY SUPPORTS SINGLE DATASET."""
        print()
        print("============= Dataset Diagnostics =============")
        print(f"[Training] Number of batches: {len(train_dataloader)}")
        print(f"[Val] Number of batches: {len(val_dataloader)}")
        print()

        val_dataset = dataset.get_validation_dataset()
        if hasattr(dataset, "zarr_paths"):
            dataset_path = dataset.zarr_paths[0]
        elif hasattr(dataset, "h5_paths"):
            dataset_path = dataset.h5_paths[0]
        else:
            dataset_path = "Unknown dataset path"
        print(f"Dataset: {dataset_path}")
        print("------------------------------------------------")
        print(f"Number of training demonstrations: {np.sum(dataset.train_masks[0])}")
        print(f"Number of validation demonstrations: {np.sum(dataset.val_masks[0])}")
        print(f"Number of training samples: {len(dataset.samplers[0])}")
        print(f"Number of validation samples: {len(val_dataset)}")
        print(f"Approx. number of training batches: {len(dataset.samplers[0]) // cfg.dataloader.batch_size}")
        print(f"Approx. number of validation batches: {len(val_dataset) // cfg.val_dataloader.batch_size}")
        print()
        print("================================================")


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = TrainDiffusionUnetLowdimWorkspaceNoEnv(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
