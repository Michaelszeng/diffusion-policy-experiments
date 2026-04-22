# Compatibility shim: r3m_config_util was moved to diffusion_policy.model.vision.
# Saved checkpoint configs still reference this old path, so re-export from there.
from diffusion_policy.model.vision.r3m_config_util import (  # noqa: F401
    R3MObsEncoder,
    ResNetObsEncoder,
)
