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
from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


class DataParallelWrapper(DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class TrainDiffusionUnetTimmWorkspaceNoEnv(BaseWorkspace):
    """
    Training workspace for policies that expose a simple compute_loss(batch)
    interface (DiffusionUnetTimmAttentionPolicy, DiffusionUnetTimmFilmPolicy).

    Differences from TrainDiffusionUnetHybridWorkspaceNoEnv:
    - Training/validation use policy.compute_loss(batch) directly.
    - MSE evaluation calls policy.predict_action(batch["obs"]).
    - No DDIM evaluation path (set eval_mse_DDIM: false in config).
    """

    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        num_GPU = torch.cuda.device_count()
        self.model = hydra.utils.instantiate(cfg.policy)
        device = torch.device("cuda" if num_GPU > 0 else "cpu")
        self.model = self.model.to(device)
        print(f"Running on {num_GPU} GPU(s).")
        self.model = DataParallelWrapper(self.model, device_ids=range(num_GPU))

        if "pretrained_checkpoint" in cfg and cfg.pretrained_checkpoint is not None:
            print(f"Loading pretrained model from {cfg.pretrained_checkpoint}.")
            path = pathlib.Path(cfg.pretrained_checkpoint)
            payload = torch.load(path.open("rb"), pickle_module=dill)
            self.model.load_state_dict(payload["state_dicts"]["model"])
        else:
            print("Initializing model using default parameters.")

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
            import torch.optim as optim
            optimizer_class = getattr(optim, cfg.optimizer._target_.split(".")[-1])
            optimizer_kwargs = dict(cfg.optimizer)
            optimizer_kwargs.pop("_target_")
            optimizer_kwargs.pop("lr")
            self.optimizer = optimizer_class(
                [{"params": encoder_params, "lr": cfg.optimizer.lr * self.encoder_lr_scale},
                 {"params": other_params, "lr": cfg.optimizer.lr}],
                **optimizer_kwargs,
            )
        else:
            self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())

        self.global_step = 0
        self.epoch = 0

        self.use_amp = getattr(cfg.training, "use_amp", False)
        self.scaler = GradScaler() if self.use_amp else None
        print(f"Mixed precision training: {'enabled' if self.use_amp else 'disabled'}")

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
                self.epoch += 1

        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()
        torch.save(normalizer, os.path.join(self.output_dir, "normalizer.pt"))

        self.num_datasets = dataset.get_num_datasets()
        self.sample_probabilities = dataset.get_sample_probabilities()
        val_dataloaders = []
        for i in range(self.num_datasets):
            val_dataloaders.append(DataLoader(dataset.get_validation_dataset(i), **cfg.val_dataloader))
        self._print_dataset_diagnostics(cfg, dataset, train_dataloader, val_dataloaders)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        if getattr(cfg.training, "total_train_steps", None) is not None:
            assert cfg.training.gradient_accumulate_every == 1
            single_epoch_steps = len(train_dataloader)
            num_epochs = int(cfg.training.total_train_steps // single_epoch_steps)
            if cfg.training.total_train_steps % single_epoch_steps != 0:
                num_epochs += 1
            print(f"Training for {num_epochs} epochs to achieve {cfg.training.total_train_steps} steps.")
            cfg.training.num_epochs = num_epochs

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs)
            // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            settings=wandb.Settings(start_method="thread", _disable_stats=True),
            **cfg.logging,
        )
        wandb.config.update({"output_dir": self.output_dir})

        assert cfg.training.checkpoint_every % cfg.training.val_every == 0
        if not isinstance(cfg.checkpoint, ListConfig):
            topk_managers = [
                TopKCheckpointManager(
                    save_dir=os.path.join(self.output_dir, "checkpoints"),
                    **cfg.checkpoint.topk,
                )
            ]
            save_last_ckpt = cfg.checkpoint.save_last_ckpt
            save_last_snapshot = cfg.checkpoint.save_last_snapshot
        else:
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

        val_sampling_batches = [None] * self.num_datasets

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        log_path = os.path.join(self.output_dir, "logs.json.txt")
        resume_epoch = self.epoch
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(resume_epoch, cfg.training.num_epochs):
                step_log = dict()

                # ── Training ────────────────────────────────────────────────────
                train_losses = []
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        for key in dataset.rgb_keys:
                            batch["obs"][key] = torch.moveaxis(batch["obs"][key], -1, 2) / 255.0

                        if self.use_amp:
                            with autocast(dtype=torch.float16):
                                raw_loss = self.model.compute_loss(batch)
                                loss = raw_loss / cfg.training.gradient_accumulate_every
                        else:
                            raw_loss = self.model.compute_loss(batch)
                            loss = raw_loss / cfg.training.gradient_accumulate_every

                        if self.use_amp:
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()

                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            if self.use_amp:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                                self.optimizer.zero_grad()
                            else:
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                            lr_scheduler.step()

                        if cfg.training.use_ema:
                            ema.step(self.model)

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
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) and batch_idx >= (
                            cfg.training.max_train_steps - 1
                        ):
                            break

                step_log["train_loss"] = np.mean(train_losses)

                # ── Validation ───────────────────────────────────────────────────
                policy = self.ema_model if cfg.training.use_ema else self.model
                policy.eval()

                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_loss_per_dataset = []
                        for dataset_idx in range(self.num_datasets):
                            val_losses = []
                            with tqdm.tqdm(
                                val_dataloaders[dataset_idx],
                                desc=f"Dataset {dataset_idx} validation, epoch {self.epoch}",
                                leave=False,
                                mininterval=cfg.training.tqdm_interval_sec,
                            ) as tepoch:
                                for batch_idx, batch in enumerate(tepoch):
                                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                    for key in dataset.rgb_keys:
                                        batch["obs"][key] = torch.moveaxis(batch["obs"][key], -1, 2) / 255.0
                                    if val_sampling_batches[dataset_idx] is None:
                                        val_sampling_batches[dataset_idx] = batch

                                    val_losses.append(self.model.compute_loss(batch).item())

                                    if (cfg.training.max_val_steps is not None) and batch_idx >= (
                                        cfg.training.max_val_steps - 1
                                    ):
                                        break

                            if val_losses:
                                val_loss = np.mean(val_losses)
                                val_loss_per_dataset.append(val_loss)
                                step_log[f"val_loss_{dataset_idx}"] = val_loss

                        overall_val_loss = sum(
                            self.sample_probabilities[i] * val_loss_per_dataset[i]
                            for i in range(self.num_datasets)
                        )
                        step_log["val_loss"] = overall_val_loss

                # ── Diffusion MSE on a single validation batch ───────────────────
                if (self.epoch % cfg.training.sample_every) == 0 and cfg.training.log_val_mse:
                    with torch.no_grad():
                        val_ddpm_action_mses = []
                        for dataset_idx in range(self.num_datasets):
                            val_batch = dict_apply(
                                val_sampling_batches[dataset_idx],
                                lambda x: x.to(device, non_blocking=True),
                            )
                            val_gt_action = val_batch["action"]

                            if cfg.training.eval_mse_DDPM:
                                result = policy.predict_action(val_batch["obs"])
                                mse = torch.nn.functional.mse_loss(result["action_pred"], val_gt_action)
                                step_log[f"val_ddpm_mse_{dataset_idx}"] = mse.item()
                                val_ddpm_action_mses.append(mse.item())

                        if cfg.training.eval_mse_DDPM and val_ddpm_action_mses:
                            step_log["val_ddpm_mse"] = sum(
                                self.sample_probabilities[i] * val_ddpm_action_mses[i]
                                for i in range(self.num_datasets)
                            )

                # ── Checkpointing ────────────────────────────────────────────────
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    if save_last_ckpt:
                        self.save_checkpoint()
                    if save_last_snapshot:
                        self.save_snapshot()

                    metric_dict = {k.replace("/", "_"): v for k, v in step_log.items()}
                    topk_ckpt_paths = []
                    for i, topk_manager in enumerate(topk_managers):
                        protected = self._get_protected_paths(i, topk_managers)
                        topk_ckpt_paths.append(topk_manager.get_ckpt_path(metric_dict, protected))

                    for topk_ckpt_path in topk_ckpt_paths:
                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)
                            break

                if self.epoch == cfg.training.num_epochs - 1 and save_last_ckpt:
                    self.save_checkpoint()

                policy.train()
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

    def _get_protected_paths(self, topk_manager_idx, topk_managers):
        if len(topk_managers) == 1:
            return set()
        topk_manager = topk_managers[topk_manager_idx]
        protected_paths = set()
        for manager in topk_managers:
            protected_paths.update(manager.get_path_value_map().keys())
        for path in topk_manager.get_path_value_map().keys():
            protected = any(
                i != topk_manager_idx and path in topk_managers[i].get_path_value_map()
                for i in range(len(topk_managers))
            )
            if not protected:
                protected_paths.discard(path)
        return protected_paths


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
