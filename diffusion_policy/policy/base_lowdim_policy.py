from typing import Dict

import torch

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.common.normalizer import LinearNormalizer


class BaseLowdimPolicy(ModuleAttrMixin):
    # ========= inference  ============
    # also as self.device and self.dtype for inference device transfer
    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            obs: B,To,Do
        return:
            action: B,Ta,Da
        To = 3
        Ta = 4
        T = 6
        |o|o|o|
        | | |a|a|a|a|
        |o|o|
        | |a|a|a|a|a|
        | | | | |a|a|
        """
        raise NotImplementedError()

    def reset(self):
        """Reset state for stateful policies"""
        pass

    # ========== training ===========
    def set_normalizer(self, normalizer: LinearNormalizer):
        """No standard training interface except setting normalizer"""
        raise NotImplementedError()
