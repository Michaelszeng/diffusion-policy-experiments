import copy
import pathlib
import threading
from typing import Optional

import dill
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


class BaseWorkspace:
    include_keys = tuple()
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir

    def run(self):
        """
        Create any resource shouldn't be serialized as local variables
        """
        pass

    def save_checkpoint(
        self,
        path=None,
        tag="latest",
        exclude_keys=None,
        include_keys=None,
        use_thread=True,
        state_dict_overrides=None,
    ):
        """
        Checkpoint format:

        The checkpoint saved to disk is a single dictionary ("payload") with
        the following top-level keys:

        - "cfg": The Hydra/OmegaConf configuration object used to construct
          this workspace (type: OmegaConf). This enables reconstructing the
          workspace via `create_from_checkpoint`.

        - "state_dicts": A mapping from attribute name (str) on this workspace
          instance to a state_dict object. Any attribute that has both
          `state_dict()` and `load_state_dict()` methods (e.g., torch.nn.Module,
          torch.optim.Optimizer, schedulers, samplers) is included here unless
          its name is listed in `exclude_keys`.
            - key: attribute name on `self` (str)
            - value: result of `attr.state_dict()` (potentially moved to CPU if
              `use_thread=True` to avoid holding GPU tensors while saving)

        - "pickles": A mapping from attribute name (str) to a raw `dill` bytes
          blob for objects that do not expose state_dict/load_state_dict but are
          explicitly requested to be saved. By default this includes any names
          listed in `self.include_keys` and also "_output_dir". You can further
          control this with the `include_keys` argument when calling
          `save_checkpoint`.
            - key: attribute name on `self` (str)
            - value: `dill.dumps(attr)` (bytes)

        Selection rules
        - Excluded: Any attribute name present in `exclude_keys` is skipped even
          if it has a state_dict.
        - Included as pickle: Any attribute name present in `include_keys`
          (plus "_output_dir" by default) is serialized into the "pickles" map
          using `dill`.
        - Overridden: If `state_dict_overrides` is provided, it is a dict mapping
          attribute names to substitute objects whose `.state_dict()` will be
          called instead of the real attribute's. Useful for saving an unwrapped
          model in place of a DDP-wrapped one (e.g. after `accelerator.prepare`).

        Loading
        - `load_payload` restores entries by calling `load_state_dict` for all
          entries in "state_dicts" (except those excluded at load time) and by
          `dill.loads`-ing the entries in "pickles" for the requested keys.
        - `create_from_checkpoint` constructs a new workspace from `payload["cfg"]`
          and then delegates to `load_payload`.

        Returns absolute filesystem path to the saved checkpoint file.
        """
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath("checkpoints", f"{tag}.ckpt")
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ("_output_dir",)

        # Join any in-flight save thread before starting a new one to avoid races
        if self._saving_thread is not None:
            self._saving_thread.join()

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {"cfg": self.cfg, "state_dicts": dict(), "pickles": dict()}

        overrides = state_dict_overrides or {}
        for key, value in self.__dict__.items():
            # Allow callers to substitute a different object for state_dict extraction
            # (e.g. an unwrapped model in place of a DDP-wrapped one)
            value_for_sd = overrides.get(key, value)
            if hasattr(value_for_sd, "state_dict") and hasattr(value_for_sd, "load_state_dict"):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload["state_dicts"][key] = _copy_to_cpu(value_for_sd.state_dict())
                    else:
                        payload["state_dicts"][key] = value_for_sd.state_dict()
            elif key in include_keys:
                payload["pickles"][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda: torch.save(payload, path.open("wb"), pickle_module=dill)
            )
            self._saving_thread.start()
        else:
            torch.save(payload, path.open("wb"), pickle_module=dill)
        return str(path.absolute())

    def get_checkpoint_path(self, tag="latest"):
        return pathlib.Path(self.output_dir).joinpath("checkpoints", f"{tag}.ckpt")

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload["pickles"].keys()

        for key, value in payload["state_dicts"].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload["pickles"]:
                self.__dict__[key] = dill.loads(payload["pickles"][key])

    def load_checkpoint(self, path=None, tag="latest", exclude_keys=None, include_keys=None, **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open("rb"), pickle_module=dill, **kwargs)
        self.load_payload(payload, exclude_keys=exclude_keys, include_keys=include_keys)
        return payload

    @classmethod
    def create_from_checkpoint(cls, path, exclude_keys=None, include_keys=None, **kwargs):
        """
        Create a workspace from a checkpoint.
        """
        payload = torch.load(open(path, "rb"), pickle_module=dill)
        instance = cls(payload["cfg"])
        instance.load_payload(
            payload=payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs,
        )
        return instance

    def save_snapshot(self, tag="latest"):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath("snapshots", f"{tag}.pkl")
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open("wb"), pickle_module=dill)
        return str(path.absolute())

    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, "rb"), pickle_module=dill)

    def join_saving_thread(self):
        """Wait for any in-flight background checkpoint save to complete."""
        if self._saving_thread is not None:
            self._saving_thread.join()

    def _get_protected_paths(self, topk_manager_idx, topk_managers):
        """
        Returns the set of checkpoint paths that the given manager must not delete.

        When multiple TopKCheckpointManagers track different metrics (e.g. val_loss and
        success_rate), the same checkpoint file may appear in more than one manager's
        top-K list. Before a manager evicts a checkpoint to make room for a new one,
        we must ensure it does not delete a file that another manager still needs.
        """
        if len(topk_managers) == 1:
            return set()

        topk_manager = topk_managers[topk_manager_idx]

        # Start with the union of every checkpoint tracked by any manager
        protected_paths = set()
        for manager in topk_managers:
            protected_paths.update(manager.get_path_value_map().keys())

        # Un-protect any checkpoint that this manager owns but no other manager tracks —
        # those are safe for this manager to delete on its own.
        for path in topk_manager.get_path_value_map().keys():
            tracked_by_other = any(
                path in manager.get_path_value_map()
                for i, manager in enumerate(topk_managers)
                if i != topk_manager_idx
            )
            if not tracked_by_other:
                protected_paths.remove(path)

        return protected_paths


def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu")
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)
