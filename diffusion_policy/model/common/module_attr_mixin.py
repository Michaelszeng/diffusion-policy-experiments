import torch.nn as nn


class ModuleAttrMixin(nn.Module):
    """
    Base class for classes that inherit from nn.Module and need to access the
    device and dtype of the module.
    """

    def __init__(self):
        super().__init__()
        self._dummy_variable = (
            nn.Parameter()
        )  # ensure module always has at least 1 parameter to query

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
