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
from collections import defaultdict

import dill
import hydra
import numpy as np
import torch
import tqdm
from omegaconf import ListConfig, OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader

import wandb
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.policy.diffusion_unet_hybrid_image_targeted_policy import (
    DiffusionUnetHybridImageTargetedPolicy,
)
from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


class DataParallelWrapper(DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class TrainDiffusionUnetHybridWorkspaceNoEnv(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        num_GPU = torch.cuda.device_count()
        self.model = hydra.utils.instantiate(cfg.policy)
        if num_GPU > 0:
            self.model = self.model.to(torch.device("cuda"))
        else:
            self.model = self.model.to(torch.device("cpu"))

        print(f"Running on {num_GPU} GPU(s).")
        self.model = DataParallelWrapper(self.model, device_ids=range(num_GPU))

        # load pretrained model if finetuning
        if "pretrained_checkpoint" in cfg and cfg.pretrained_checkpoint is not None:
            print(f"Loading pretrained model from {cfg.pretrained_checkpoint}.")
            path = pathlib.Path(cfg.pretrained_checkpoint)
            payload = torch.load(path.open("rb"), pickle_module=dill)
            self.model.load_state_dict(payload["state_dicts"]["model"])
        else:
            print("Initializing model using default parameters.")

        self.ema_model: DiffusionUnetHybridImageTargetedPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state with variable learning rates
        self.encoder_lr_scale = getattr(cfg.training, "encoder_lr_scale", 1.0)
        self.use_variable_lr = getattr(cfg.training, "use_variable_lr", False)

        if self.use_variable_lr and hasattr(self.model, "obs_encoder"):
            # Check if this is MC3 encoder by looking for the specific policy class
            # is_mc3_model = 'r3d' in self.model.__class__.__module__

            # if is_mc3_model:
            print(
                f"Using variable learning rate: encoder LR = {cfg.optimizer.lr * self.encoder_lr_scale:.6f}, rest = {cfg.optimizer.lr:.6f}"
            )

            # Separate parameters for encoder and rest of model
            encoder_params = list(self.model.obs_encoder.parameters())
            encoder_param_ids = {id(p) for p in encoder_params}

            other_params = [p for p in self.model.parameters() if id(p) not in encoder_param_ids]

            # Create parameter groups with different learning rates
            param_groups = [
                {
                    "params": encoder_params,
                    "lr": cfg.optimizer.lr * self.encoder_lr_scale,
                },
                {"params": other_params, "lr": cfg.optimizer.lr},
            ]

            # Create optimizer with parameter groups
            import torch.optim as optim

            optimizer_class = getattr(optim, cfg.optimizer._target_.split(".")[-1])  # Extract class name

            optimizer_kwargs = dict(cfg.optimizer)
            optimizer_kwargs.pop("_target_")
            optimizer_kwargs.pop("lr")  # Remove base lr since we're setting per group

            self.optimizer = optimizer_class(param_groups, **optimizer_kwargs)
        else:
            print("Variable LR requested but not MC3 model - using standard optimizer")
            self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

        # configure mixed precision training
        self.use_amp = getattr(cfg.training, "use_amp", False)
        self.scaler = GradScaler() if self.use_amp else None
        print(f"Mixed precision training: {'enabled' if self.use_amp else 'disabled'}")

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
                # self.epoch is loaded with the last completed epoch
                # the current epoch is the next epoch (hence += 1)
                self.epoch += 1

        # configure dataset and save normalizer
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()
        torch.save(normalizer, os.path.join(self.output_dir, "normalizer.pt"))

        # configure validation datasets
        self.num_datasets = dataset.get_num_datasets()
        self.sample_probabilities = dataset.get_sample_probabilities()
        val_dataloaders = []
        for i in range(self.num_datasets):
            val_dataset = dataset.get_validation_dataset(i)
            val_dataloaders.append(DataLoader(val_dataset, **cfg.val_dataloader))
        self._print_dataset_diagnostics(cfg, dataset, train_dataloader, val_dataloaders)
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # If total_train_steps is set, use it to determine and override num_epochs
        training_steps = getattr(cfg.training, "total_train_steps", None)
        if training_steps is not None:
            assert cfg.training.gradient_accumulate_every == 1, (
                "Gradient accumulation not supported with total_train_steps"
            )
            single_epoch_steps = len(train_dataloader)
            num_epochs = int(training_steps // single_epoch_steps)
            if training_steps % single_epoch_steps != 0:
                num_epochs += 1
            print(f"Training for {num_epochs} epochs to achieve {training_steps} steps.")
            cfg.training.num_epochs = num_epochs

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs)
            // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1,
        )

        if getattr(cfg.policy, "rescale_encoder_gradients", False):
            hooks_by_param = defaultdict(list)
            for name, p in self.model.obs_encoder.named_parameters():
                # getattr will return {} if no hooks were registered
                hooks = getattr(p, "_backward_hooks", {})
                if hooks:
                    hooks_by_param[name] = list(hooks.items())

            # Print a report
            for name, hook_list in hooks_by_param.items():
                print(f"Param {name} has {len(hook_list)} hook(s):")
                for hook_id, fn in hook_list:
                    print(f"  id={hook_id}, fn={fn}")

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            settings=wandb.Settings(
                start_method="thread",
                _disable_stats=True,  # Don't log stuff like CPU, GPU, system memory, etc. just to reduce data amount
            ),
            **cfg.logging,
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        assert cfg.training.checkpoint_every % cfg.training.val_every == 0
        if not isinstance(cfg.checkpoint, ListConfig):  # Single checkpoint manager
            # configure single checkpoint manager
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
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        val_sampling_batches = [None] * self.num_datasets

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
        resume_epoch = self.epoch
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(resume_epoch, cfg.training.num_epochs):
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
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        # reorder channels from HWC to CHW to match robomimic's vision encoder
                        for key in dataset.rgb_keys:
                            batch["obs"][key] = torch.moveaxis(batch["obs"][key], -1, 2) / 255.0
                        # batch_size = batch["action"].shape[0]
                        # print(f"Outside batch size: {batch_size}")

                        # construct noisy trajectory
                        trajectory = self.model.normalizer["action"].normalize(batch["action"])
                        noisy_trajectory, timesteps, noise = self.model.noise_trajectory(trajectory)

                        # Forward pass with optional mixed precision
                        if self.use_amp:
                            with autocast(dtype=torch.float16):
                                pred = self.model(batch, noisy_trajectory, timesteps)
                                raw_loss = self.model.compute_loss(trajectory, noise, pred)
                                loss = raw_loss / cfg.training.gradient_accumulate_every
                        else:
                            pred = self.model(batch, noisy_trajectory, timesteps)
                            raw_loss = self.model.compute_loss(trajectory, noise, pred)
                            loss = raw_loss / cfg.training.gradient_accumulate_every

                        # Backward pass with optional mixed precision
                        if self.use_amp:
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            if self.use_amp:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                                self.optimizer.zero_grad()
                            else:
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
                    # End of batch
                # End of epoch

                # At the end of each epoch, replace train_loss with epoch average
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
                        val_loss_per_dataset = []
                        for dataset_idx in range(self.num_datasets):
                            val_dataloader = val_dataloaders[dataset_idx]
                            val_losses = list()
                            with tqdm.tqdm(
                                val_dataloader,
                                desc=f"Dataset {dataset_idx} validation, epoch {self.epoch}",
                                leave=False,
                                mininterval=cfg.training.tqdm_interval_sec,
                            ) as tepoch:
                                for batch_idx, batch in enumerate(tepoch):
                                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                    # reorder channels from HWC to CHW to match robomimic's vision encoder
                                    for key in dataset.rgb_keys:
                                        batch["obs"][key] = torch.moveaxis(batch["obs"][key], -1, 2) / 255.0
                                    if val_sampling_batches[dataset_idx] is None:
                                        val_sampling_batches[dataset_idx] = batch

                                    # construct normalized noisy trajectory for inference
                                    trajectory = self.model.normalizer["action"].normalize(batch["action"])
                                    noisy_trajectory, timesteps, noise = self.model.noise_trajectory(trajectory)

                                    # generate inpainting mask
                                    condition_mask = self.model.mask_generator(trajectory.shape)
                                    loss_mask = ~condition_mask

                                    # forward pass and compute loss
                                    pred = self.model(batch, noisy_trajectory, timesteps, condition_mask)
                                    loss = self.model.compute_loss(trajectory, noise, pred, loss_mask)
                                    val_losses.append(loss)

                                    if (cfg.training.max_val_steps is not None) and batch_idx >= (
                                        cfg.training.max_val_steps - 1
                                    ):
                                        break
                            # End of validation loss computation loop

                            if len(val_losses) > 0:
                                val_loss = torch.mean(torch.tensor(val_losses)).item()
                                val_loss_per_dataset.append(val_loss)
                                # log epoch average validation loss
                                step_log[f"val_loss_{dataset_idx}"] = val_loss
                        # End val_dataloader loop

                        # Compute overall_val_loss
                        overall_val_loss = 0
                        for i in range(self.num_datasets):
                            overall_val_loss += self.sample_probabilities[i] * val_loss_per_dataset[i]
                        step_log["val_loss"] = overall_val_loss

                # run diffusion sampling on a _single_ validation batch from each dataset
                if (self.epoch % cfg.training.sample_every) == 0 and cfg.training.log_val_mse:
                    with torch.no_grad():
                        val_ddpm_action_mses = []
                        val_ddim_action_mses = []
                        for dataset_idx in range(self.num_datasets):
                            # Get the validation batch for this dataset
                            val_sampling_batch = val_sampling_batches[dataset_idx]
                            val_batch = dict_apply(
                                val_sampling_batch,
                                lambda x: x.to(device, non_blocking=True),
                            )
                            val_obs_dict = {key: val_batch[key] for key in val_batch.keys() if key != "action"}
                            val_gt_action = val_sampling_batch["action"]

                            # Evaluate MSE when diffusing with DDPM
                            if cfg.training.eval_mse_DDPM:
                                result = policy.predict_action(val_obs_dict, use_DDIM=False)
                                pred_action = result["action_pred"]
                                mse = torch.nn.functional.mse_loss(pred_action, val_gt_action)
                                step_log[f"val_ddpm_mse_{dataset_idx}"] = mse.item()
                                val_ddpm_action_mses.append(mse.item())

                            # Evaluate MSE when diffusing with DDPM
                            if cfg.training.eval_mse_DDIM:
                                result = policy.predict_action(val_obs_dict, use_DDIM=True)
                                pred_action = result["action_pred"]
                                mse = torch.nn.functional.mse_loss(pred_action, val_gt_action)
                                step_log[f"val_ddim_mse_{dataset_idx}"] = mse.item()
                                val_ddim_action_mses.append(mse.item())

                        # Compute weighted val action MSEs
                        if cfg.training.eval_mse_DDPM:
                            val_ = 0
                            for i in range(self.num_datasets):
                                val_ += self.sample_probabilities[i] * val_ddpm_action_mses[i]
                            step_log["val_ddpm_mse"] = val_
                        if cfg.training.eval_mse_DDIM:
                            val_ = 0
                            for i in range(self.num_datasets):
                                val_ += self.sample_probabilities[i] * val_ddim_action_mses[i]
                            step_log["val_ddim_mse"] = val_

                        del val_batch
                        del val_obs_dict
                        del val_gt_action
                        del result
                        del pred_action
                        del mse

                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # "latest.ckpt"
                    if save_last_ckpt:
                        self.save_checkpoint()

                    # Optional full state snapshot
                    if save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace("/", "_")
                        metric_dict[new_key] = value

                    # Metricbased Top-K checkpointing
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
                if self.epoch == cfg.training.num_epochs - 1:
                    self.save_checkpoint()
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

    def _print_dataset_diagnostics(self, cfg, dataset, train_dataloader, val_dataloaders):
        print()
        print("============= Dataset Diagnostics =============")
        print(f"Number of datasets: {self.num_datasets}")
        print(f"Sample probabilities: {self.sample_probabilities}")
        print(f"[Training] Number of batches: {len(train_dataloader)}")
        for i in range(self.num_datasets):
            print(f"[Val {i}] Number of batches: {len(val_dataloaders[i])}")
        print()

        for i in range(self.num_datasets):
            val_dataset = dataset.get_validation_dataset(i)
            # Support both zarr_paths and h5_paths for different dataset types
            if hasattr(dataset, "zarr_paths"):
                dataset_path = dataset.zarr_paths[i]
            elif hasattr(dataset, "h5_paths"):
                dataset_path = dataset.h5_paths[i]
            else:
                dataset_path = "Unknown dataset path"
            print(f"Dataset {i}: {dataset_path}")
            print("------------------------------------------------")
            print(f"Number of training demonstrations: {np.sum(dataset.train_masks[i])}")
            print(f"Number of validation demonstrations: {np.sum(dataset.val_masks[i])}")
            print(f"Number of training samples: {len(dataset.samplers[i])}")
            print(f"Number of validation samples: {len(val_dataset)}")
            print(f"Approx. number of training batches: {len(dataset.samplers[i]) // cfg.dataloader.batch_size}")
            print(f"Approx. number of validation batches: {len(val_dataset) // cfg.val_dataloader.batch_size}")
            print(f"Sample probability: {self.sample_probabilities[i]}")
            print()
        print("================================================")

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


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = TrainDiffusionUnetHybridWorkspaceNoEnv(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
