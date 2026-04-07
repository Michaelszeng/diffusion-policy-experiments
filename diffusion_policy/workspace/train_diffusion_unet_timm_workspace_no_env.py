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
import torch.optim as optim
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from omegaconf import ListConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainDiffusionUnetTimmWorkspaceNoEnv(BaseWorkspace):
    """
    Training workspace for policies that expose a simple compute_loss(batch)
    interface (DiffusionUnetTimmAttentionPolicy, DiffusionUnetTimmFilmPolicy).

    Differences from TrainDiffusionUnetHybridWorkspaceNoEnv:
    - Training/validation use policy.compute_loss(batch) directly.
    """

    include_keys = ["global_step", "epoch", "topk_managers"]  # Attributes to save as keys in ckpt

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        self.topk_managers = None

        # Set random seeds for reproducibility
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Instantiate model (device placement and DDP wrapping handled in run() via accelerator.prepare)
        self.model = hydra.utils.instantiate(cfg.policy)

        # Load pretrained weights if finetuning from a checkpoint
        if "pretrained_checkpoint" in cfg and cfg.pretrained_checkpoint is not None:
            print(f"Loading pretrained model from {cfg.pretrained_checkpoint}.")
            path = pathlib.Path(cfg.pretrained_checkpoint)
            payload = torch.load(path.open("rb"), pickle_module=dill)
            self.model.load_state_dict(payload["state_dicts"]["model"])
        else:
            print("Initializing model using default parameters.")

        # Configure EMA model (a persistent copy of weights updated via exponential moving average)
        self.ema_model = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # Support optional per-encoder learning rate scaling
        self.encoder_lr_scale = getattr(cfg.training, "encoder_lr_scale", None)

        if self.encoder_lr_scale is not None and self.encoder_lr_scale != 1.0 and hasattr(self.model, "obs_encoder"):
            print(
                f"Using scaled encoder LR: encoder LR = {cfg.optimizer.lr * self.encoder_lr_scale:.6f}, "
                f"rest = {cfg.optimizer.lr:.6f}"
            )
            encoder_params = list(self.model.obs_encoder.parameters())
            encoder_param_ids = {id(p) for p in encoder_params}
            other_params = [p for p in self.model.parameters() if id(p) not in encoder_param_ids]
            optimizer_class = getattr(optim, cfg.optimizer._target_.split(".")[-1])
            optimizer_kwargs = dict(cfg.optimizer)
            optimizer_kwargs.pop("_target_")
            optimizer_kwargs.pop("lr")  # Remove base lr since we're setting per group
            self.optimizer = optimizer_class(
                [{"params": encoder_params, "lr": cfg.optimizer.lr * self.encoder_lr_scale},
                 {"params": other_params, "lr": cfg.optimizer.lr}],
                **optimizer_kwargs,
            )
        else:
            self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())

        # Configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # Determine mixed precision from config ("no" / "fp16" / "bf16")
        mixed_precision = getattr(cfg.training, "mixed_precision", None) or "no"

        # Accelerator replaces: DataParallel, GradScaler, autocast, and wandb.init.
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        accelerator = Accelerator(
            log_with="wandb",
            mixed_precision=mixed_precision,
            kwargs_handlers=[ddp_kwargs],
        )

        # Init W&B logging via accelerator (only logs on main process)
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        project_name = wandb_cfg.pop("project")
        wandb_cfg["dir"] = str(self.output_dir)
        wandb_cfg["settings"] = {"_disable_stats": True}
        accelerator.init_trackers(
            project_name=project_name,
            config={**OmegaConf.to_container(cfg, resolve=True), "output_dir": self.output_dir},
            init_kwargs={"wandb": wandb_cfg},
        )

        # Resume training from latest checkpoint if requested
        if cfg.training.resume:
            if getattr(cfg.training, "checkpoint_path", None) is not None:
                ckpt_path = pathlib.Path(cfg.training.checkpoint_path)
                if ckpt_path.is_file():
                    accelerator.print(f"Resuming from user-specified checkpoint {ckpt_path}")
                    self.load_checkpoint(path=ckpt_path)
                    self.epoch += 1
                else:
                    accelerator.print(f"ATTENTION: Resume requested but checkpoint {ckpt_path} not found. Starting from scratch.")
            else:
                latest_ckpt_path = self.get_checkpoint_path()
                if latest_ckpt_path.is_file():
                    accelerator.print(f"Resuming from checkpoint {latest_ckpt_path}")
                    self.load_checkpoint(path=latest_ckpt_path)
                    # self.epoch is loaded with the last completed epoch;
                    # the current epoch is the next one
                    self.epoch += 1
                else:
                    accelerator.print(
                        f"ATTENTION: Resume requested but no checkpoint supplied by user and latest checkpoint "
                        f"{latest_ckpt_path} not found. Starting from scratch."
                    )

        # Configure dataset
        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)

        # Create training dataloader.
        # For multi-GPU: use DistributedSampler so each process sees a disjoint shard.
        # For single-GPU: use the regular dataloader config directly.
        dataloader_cfg = OmegaConf.to_container(cfg.dataloader, resolve=True)
        if accelerator.num_processes > 1:
            shuffle = dataloader_cfg.pop("shuffle", True)
            drop_last = dataloader_cfg.pop("drop_last", True)
            train_sampler = DistributedSampler(
                dataset,
                num_replicas=accelerator.num_processes,
                rank=accelerator.process_index,
                shuffle=shuffle,
                seed=cfg.training.seed,
            )
            train_dataloader = DataLoader(dataset, sampler=train_sampler, drop_last=drop_last, **dataloader_cfg)
        else:
            train_sampler = None
            train_dataloader = DataLoader(dataset, **cfg.dataloader)

        # Compute normalizer on the main process and save to disk, then all processes load it.
        normalizer_path = os.path.join(self.output_dir, "normalizer.pt")
        if accelerator.is_main_process:
            normalizer = dataset.get_normalizer()
            torch.save(normalizer, normalizer_path)
        accelerator.wait_for_everyone()
        normalizer = torch.load(normalizer_path)

        # Configure validation datasets (not distributed — all processes see full val data)
        self.num_datasets = dataset.get_num_datasets()
        self.sample_probabilities = dataset.get_sample_probabilities()
        val_dataloaders = []
        for i in range(self.num_datasets):
            val_dataloaders.append(DataLoader(dataset.get_validation_dataset(i), **cfg.val_dataloader))
        if accelerator.is_main_process:
            self._print_dataset_diagnostics(cfg, dataset, train_dataloader, val_dataloaders)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # If total_train_steps is set, use it to determine and override num_epochs
        if getattr(cfg.training, "total_train_steps", None) is not None:
            assert cfg.training.gradient_accumulate_every == 1
            single_epoch_steps = len(train_dataloader)
            num_epochs = int(cfg.training.total_train_steps // single_epoch_steps)
            if cfg.training.total_train_steps % single_epoch_steps != 0:
                num_epochs += 1
            accelerator.print(f"Training for {num_epochs} epochs to achieve {cfg.training.total_train_steps} steps.")
            cfg.training.num_epochs = num_epochs

        # Configure LR scheduler
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

        # Configure EMA updater (the algorithm that updates self.ema_model each step)
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # Prepare model and optimizer with accelerate.
        # This wraps the model in DDP (multi-GPU) and applies mixed precision to forward().
        self.model, self.optimizer = accelerator.prepare(self.model, self.optimizer)
        # Keep a reference to the unwrapped model for attribute access
        unwrapped_model = accelerator.unwrap_model(self.model)
        device = accelerator.device

        # EMA model is not prepared by accelerate; move it to device manually
        if self.ema_model is not None:
            self.ema_model.to(device)

        # Optionally compile the diffusion UNet (the hot inner loop).
        # NOTE: currently untested
        if getattr(cfg.training, "use_torch_compile", False):
            compile_mode = getattr(cfg.training, "torch_compile_mode", "default")
            unwrapped_model.model = torch.compile(unwrapped_model.model, mode=compile_mode)
            if self.ema_model is not None:
                self.ema_model.model = torch.compile(self.ema_model.model, mode=compile_mode)
            accelerator.print(f"torch.compile enabled on ConditionalUnet1D model only (mode={compile_mode}).")

        # Propagate mixed_precision to the unwrapped model and EMA model so that
        # predict_action's internal torch.autocast works correctly.
        # (compute_loss is not forward(), so DDP autocast does NOT apply there automatically)
        unwrapped_model.mixed_precision = mixed_precision
        if self.ema_model is not None:
            self.ema_model.mixed_precision = mixed_precision

        accelerator.print(f"Running on {accelerator.num_processes} GPU(s), mixed_precision={mixed_precision}.")

        # Configure checkpoint managers
        assert cfg.training.checkpoint_every % cfg.training.val_every == 0
        save_last_ckpt = cfg.checkpoint.save_last_ckpt
        save_last_snapshot = cfg.checkpoint.save_last_snapshot

        # Initialize new topk managers if starting from scratch or loading an old checkpoint without them
        if self.topk_managers is None:
            if not isinstance(cfg.checkpoint.topk, ListConfig):  # Single checkpoint manager
                self.topk_managers = [
                    TopKCheckpointManager(
                        save_dir=os.path.join(self.output_dir, "checkpoints"),
                        **cfg.checkpoint.topk,
                    )
                ]
            else:  # Multiple checkpoint managers
                self.topk_managers = []
                for topk_cfg in cfg.checkpoint.topk:
                    self.topk_managers.append(
                        TopKCheckpointManager(
                            save_dir=os.path.join(self.output_dir, "checkpoints"),
                            **topk_cfg,
                        )
                    )
        # Update loaded topk managers with new config (e.g. if k changed, or if output_dir changed)
        else:
            if not isinstance(cfg.checkpoint.topk, ListConfig):
                if len(self.topk_managers) > 0:
                    self.topk_managers[0].k = cfg.checkpoint.topk.k
                    self.topk_managers[0].format_str = cfg.checkpoint.topk.format_str
                    self.topk_managers[0].mode = cfg.checkpoint.topk.mode
                    self.topk_managers[0].monitor_key = cfg.checkpoint.topk.monitor_key
                    self.topk_managers[0].save_dir = os.path.join(self.output_dir, "checkpoints")
            else:
                for i, topk_cfg in enumerate(cfg.checkpoint.topk):
                    if i < len(self.topk_managers):
                        self.topk_managers[i].k = topk_cfg.k
                        self.topk_managers[i].format_str = topk_cfg.format_str
                        self.topk_managers[i].mode = topk_cfg.mode
                        self.topk_managers[i].monitor_key = topk_cfg.monitor_key
                        self.topk_managers[i].save_dir = os.path.join(self.output_dir, "checkpoints")
                    else:
                        self.topk_managers.append(
                            TopKCheckpointManager(
                                save_dir=os.path.join(self.output_dir, "checkpoints"),
                                **topk_cfg,
                            )
                        )

        # Save one validation batch per dataset to use for MSE evaluation each epoch
        val_sampling_batches = [None] * self.num_datasets

        if cfg.training.debug:  # Debug mode: just run for a couple epochs/steps
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # ── Training Loop ────────────────────────────────────────────────────
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        resume_epoch = self.epoch
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(resume_epoch, cfg.training.num_epochs):
                # Set epoch on DistributedSampler so each epoch gets a different shuffle
                if train_sampler is not None:
                    train_sampler.set_epoch(local_epoch_idx)

                step_log = dict()
                train_losses = []
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    disable=not accelerator.is_main_process,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # Device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        # Reorder channels from HWC to CHW and normalize to [0, 1]
                        for key in dataset.rgb_keys:
                            batch["obs"][key] = torch.moveaxis(batch["obs"][key], -1, 2) / 255.0

                        # Forward pass — DDP wrapper applies autocast from mixed_precision
                        raw_loss = self.model(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every

                        # Backward pass — accelerator handles gradient scaling (fp16 GradScaler)
                        accelerator.backward(loss)

                        # Step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            grad_clip = getattr(cfg.training, "gradient_clip_val", None)
                            if grad_clip is not None:
                                accelerator.clip_grad_norm_(self.model.parameters(), grad_clip)
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # Update EMA model
                        if cfg.training.use_ema:
                            ema.step(unwrapped_model)

                        # Logging
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
                            # Log of last batch is combined with validation metrics below
                            accelerator.log(step_log, step=self.global_step)
                            if accelerator.is_main_process:
                                json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) and batch_idx >= (
                            cfg.training.max_train_steps - 1
                        ):
                            break

                # At the end of each epoch, replace train_loss with epoch average
                step_log["train_loss"] = np.mean(train_losses)

                # ── Validation ───────────────────────────────────────────────────
                # Use EMA model for evaluation if available, otherwise use unwrapped training model
                eval_policy = self.ema_model if cfg.training.use_ema else unwrapped_model
                eval_policy.eval()

                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_loss_per_dataset = []
                        for dataset_idx in range(self.num_datasets):
                            val_losses = []
                            with tqdm.tqdm(
                                val_dataloaders[dataset_idx],
                                desc=f"Dataset {dataset_idx} validation, epoch {self.epoch}",
                                leave=False,
                                disable=not accelerator.is_main_process,
                                mininterval=cfg.training.tqdm_interval_sec,
                            ) as tepoch:
                                for batch_idx, batch in enumerate(tepoch):
                                    # Device transfer
                                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                    # Reorder channels from HWC to CHW to match robomimic's vision encoder
                                    for key in dataset.rgb_keys:
                                        batch["obs"][key] = torch.moveaxis(batch["obs"][key], -1, 2) / 255.0
                                    if val_sampling_batches[dataset_idx] is None:
                                        val_sampling_batches[dataset_idx] = batch

                                    # Forward pass with explicit autocast to match training precision
                                    # since eval_policy is not a DDP wrapper, autocast is not automatic
                                    with torch.autocast(
                                        device_type=device.type,
                                        dtype=torch.bfloat16 if mixed_precision == "bf16" else torch.float16,
                                        enabled=mixed_precision != "no",
                                    ):
                                        loss = eval_policy.compute_loss(batch)
                                    val_losses.append(loss.item())

                                    if (cfg.training.max_val_steps is not None) and batch_idx >= (
                                        cfg.training.max_val_steps - 1
                                    ):
                                        break

                            if val_losses:
                                val_loss = np.mean(val_losses)
                                val_loss_per_dataset.append(val_loss)
                                step_log[f"val_loss_{dataset_idx}"] = val_loss

                        # Compute weighted aggregate validation loss across datasets
                        overall_val_loss = sum(
                            self.sample_probabilities[i] * val_loss_per_dataset[i]
                            for i in range(self.num_datasets)
                        )
                        step_log["val_loss"] = overall_val_loss

                # ── DDPM/DDIM clean action MSE validation ────────────────────────
                if (self.epoch % cfg.training.sample_every) == 0 and cfg.training.log_val_mse:
                    with torch.no_grad():
                        val_ddpm_action_mses = []
                        val_ddim_action_mses = []
                        for dataset_idx in range(self.num_datasets):
                            # Device transfer
                            val_batch = dict_apply(
                                val_sampling_batches[dataset_idx],
                                lambda x: x.to(device, non_blocking=True),
                            )
                            val_gt_action = val_batch["action"]

                            if cfg.training.eval_mse_DDPM:
                                result = eval_policy.predict_action(val_batch["obs"], use_DDIM=False)
                                mse = torch.nn.functional.mse_loss(result["action_pred"], val_gt_action)
                                step_log[f"val_ddpm_mse_{dataset_idx}"] = mse.item()
                                val_ddpm_action_mses.append(mse.item())

                            if cfg.training.eval_mse_DDIM:
                                result = eval_policy.predict_action(val_batch["obs"], use_DDIM=True)
                                mse = torch.nn.functional.mse_loss(result["action_pred"], val_gt_action)
                                step_log[f"val_ddim_mse_{dataset_idx}"] = mse.item()
                                val_ddim_action_mses.append(mse.item())

                        if cfg.training.eval_mse_DDPM and val_ddpm_action_mses:
                            step_log["val_ddpm_mse"] = sum(
                                self.sample_probabilities[i] * val_ddpm_action_mses[i]
                                for i in range(self.num_datasets)
                            )
                        if cfg.training.eval_mse_DDIM and val_ddim_action_mses:
                            step_log["val_ddim_mse"] = sum(
                                self.sample_probabilities[i] * val_ddim_action_mses[i]
                                for i in range(self.num_datasets)
                            )

                # ── Checkpointing (main process only) ────────────────────────────
                if (self.epoch % cfg.training.checkpoint_every) == 0 and accelerator.is_main_process:
                    # Pass the unwrapped model so save_checkpoint serializes clean weights
                    ckpt_kwargs = {"state_dict_overrides": {"model": unwrapped_model}}

                    if save_last_ckpt:
                        self.save_checkpoint(**ckpt_kwargs)
                    if save_last_snapshot:
                        self.save_snapshot()

                    # Sanitize metric names (replace "/" with "_" for compatibility)
                    metric_dict = {k.replace("/", "_"): v for k, v in step_log.items()}
                    topk_ckpt_paths = []
                    for i, topk_manager in enumerate(self.topk_managers):
                        protected = self._get_protected_paths(i, self.topk_managers)
                        topk_ckpt_paths.append(topk_manager.get_ckpt_path(metric_dict, protected))

                    # Metric-based Top-K checkpointing
                    for topk_ckpt_path in topk_ckpt_paths:
                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path, **ckpt_kwargs)
                            break

                # Save checkpoint at end of last epoch (main process only)
                if self.epoch == cfg.training.num_epochs - 1 and save_last_ckpt and accelerator.is_main_process:
                    self.save_checkpoint(state_dict_overrides={"model": unwrapped_model})

                # Sync all processes after checkpointing before moving to next epoch
                accelerator.wait_for_everyone()

                eval_policy.train()

                accelerator.log(step_log, step=self.global_step)
                if accelerator.is_main_process:
                    json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        self.join_saving_thread()  # Wait for any in-flight checkpoint save to complete
        accelerator.end_training()

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
            print(f"Sample probability: {self.sample_probabilities[i]}")
            print()
        print("================================================")


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = TrainDiffusionUnetTimmWorkspaceNoEnv(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
