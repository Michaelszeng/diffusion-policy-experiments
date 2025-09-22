import torch

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin


class DummyMaskGenerator(ModuleAttrMixin):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, shape):
        device = self.device
        mask = torch.ones(size=shape, dtype=torch.bool, device=device)
        return mask


class LowdimMaskGenerator(ModuleAttrMixin):
    def __init__(
        self,
        action_dim,
        obs_dim,
        # obs mask setup
        max_n_obs_steps=2,
        fix_obs_steps=True,
        # action mask
        action_visible=False,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim  # = 0 using global observation conditioning
        self.max_n_obs_steps = max_n_obs_steps
        self.fix_obs_steps = fix_obs_steps
        self.action_visible = action_visible

    @torch.no_grad()
    def forward(self, shape, seed=None):
        """
        Generate a boolean mask for trajectory conditioning during diffusion training.

        Args:
            shape: (B, T, D) where B=batch_size, T=horizon, D=action_dim+obs_dim
            seed: Optional random seed for reproducible masking

        Returns:
            mask: Boolean tensor (B, T, D) where True = "known/given", False = "predict"
        """
        device = self.device
        B, T, D = shape
        assert D == (self.action_dim + self.obs_dim)

        rng = torch.Generator(device=device)
        if seed is not None:
            rng = rng.manual_seed(seed)

        # Create dimension type masks
        # Determine which dimensions in the trajectory are actions vs observations
        dim_mask = torch.zeros(size=shape, dtype=torch.bool, device=device)  # (B, T, D) all False
        is_action_dim = dim_mask.clone()
        is_action_dim[..., : self.action_dim] = True  # First 13 dims are actions -> True
        # Result: is_action_dim = (B, T, self.action_dim) all True

        is_obs_dim = ~is_action_dim  # Flip: True for obs dims, False for action dims
        # Result: is_obs_dim = (B, T, self.action_dim) all False

        # Generate observation timestep mask
        # Determine how many initial timesteps should have "known" observations
        if self.fix_obs_steps:
            # Always use the maximum number of observation steps
            obs_steps = torch.full((B,), fill_value=self.max_n_obs_steps, device=device)  # (B,)
            # Result: obs_steps = [self.max_n_obs_steps, self.max_n_obs_steps, ...] (one per batch item)
        else:
            # Randomly vary the number of observation steps (1 to max_n_obs_steps)
            obs_steps = torch.randint(
                low=1,
                high=self.max_n_obs_steps + 1,  # max_n_obs_steps + 1 for exclusive upper bound
                size=(B,),
                generator=rng,
                device=device,
            )

        # Create timestep indices for masking
        steps = torch.arange(0, T, device=device).reshape(1, T).expand(B, T)  # (B, T)
        # Result: steps = [[0,1,2,3,...,T], [0,1,2,3,...,T], ...] for each batch

        # Mark timesteps that should have "known" observations
        obs_mask = (steps.T < obs_steps).T.reshape(B, T, 1).expand(B, T, D)  # (B, T, D)
        # (steps.T < obs_steps).T creates: [[True,True,False,False,...], ...] (first self.max_n_obs_steps timesteps
        # True, 2 in this example)
        # After reshape/expand: (B, T, D) where first self.max_n_obs_steps timesteps are True across all dimensions

        # Apply to observation dimensions only
        obs_mask = obs_mask & is_obs_dim
        # Result: obs_mask = all False (since is_obs_dim is all False, since obs_dim=0, since we are using global
        # observation conditioning)

        # Generate action timestep mask
        # Determine if past actions should be visible during training
        if self.action_visible:
            # Calculate how many action timesteps should be "known"
            # One less than observation steps (since we see obs then predict action)
            action_steps = torch.maximum(
                obs_steps - 1,  # obs_steps - 1 = [1, 1, 1, ...]
                torch.tensor(0, dtype=obs_steps.dtype, device=obs_steps.device),
            )  # Ensure >= 0
            # Result: action_steps = [1, 1, 1, ...]

            # Create action timestep mask (same pattern as obs_mask)
            action_mask = (steps.T < action_steps).T.reshape(B, T, 1).expand(B, T, D)
            # (steps.T < action_steps).T creates: [[True,True,False,False,...], ...] (first action_steps timesteps
            # True, 1 in this example)
            # After reshape/expand: (B, T, D) where first action_steps timesteps are True across all dimensions

            # Apply to action dimensions only
            action_mask = action_mask & is_action_dim

        # Combine all masks
        mask = obs_mask  # Start with observation mask (all False)
        if self.action_visible:
            mask = mask | action_mask  # Union of obs and action masks

        return mask


def test():
    self = LowdimMaskGenerator(2, 20, max_n_obs_steps=3, action_visible=True)
