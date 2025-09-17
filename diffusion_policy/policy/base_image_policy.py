from typing import Dict

import torch

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.common.normalizer import LinearNormalizer


class BaseImagePolicy(ModuleAttrMixin):
    """
    Base class for image-based policies.

    init accepts keyword argument shape_meta, see config/task/*_image.yaml.
    shape_meta is a dictionary describing dimensions of observations and actions.
    """

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            str: B,To,*  (* accommodates different image sizes/formats)
        return: B,Ta,Da
        """
        raise NotImplementedError()

    def reset(self):
        """Reset state for stateful policies"""
        pass

    # ========== training ===========
    def set_normalizer(self, normalizer: LinearNormalizer):
        """No standard training interface except setting normalizer"""
        raise NotImplementedError()
