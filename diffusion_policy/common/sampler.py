"""
Dataset samplers (sample individual training samples from a episodic replay buffer).
"""

from typing import Optional

import numba
import numpy as np

from diffusion_policy.common.replay_buffer import ReplayBuffer


@numba.jit(nopython=True)
def create_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    episode_mask: np.ndarray,
    pad_before: int = 0,
    pad_after: int = 0,
    downsample_steps: int = 1,
    debug: bool = True,
) -> np.ndarray:
    """
    Generate array of indices for each sample in the dataset.

    Output format: [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
     - buffer_start_idx: index of the first frame in the ReplayBuffer to sample
     - buffer_end_idx: index of the last frame in the ReplayBuffer to sample + 1
     - sample_start_idx: index of the first real frame in the sample (before that is padding)
     - sample_end_idx: index of the last real frame in the sample (after that is padding) + 1
    """
    episode_mask.shape == episode_ends.shape
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)
    N = max(downsample_steps, 1)

    indices = list()
    # Iterate over each episode
    for i in range(len(episode_ends)):
        # Skip episode if it is not in the train mask
        if not episode_mask[i]:
            continue

        # Get the start and end indices of the current episode
        start_idx = 0 if i == 0 else episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        # Allow up to pad_before sample positions to fall before the episode start
        min_window_start = start_idx - pad_before * N
        # Require at least (sequence_length - pad_after) in-bounds sample positions
        max_window_start = end_idx - (sequence_length - pad_after - 1) * N - 1

        # Iterate over each absolute buffer index in the current episode
        for idx in range(min_window_start, max_window_start + 1):
            # Calculate idx (in the window) of the first valid frame in the sequence.
            # If the sequence starts before the episode, we skip those out-of-bounds frames.
            if idx < start_idx:
                sample_start_idx = (start_idx - idx + N - 1) // N
            else:
                sample_start_idx = 0

            # Calculate idx (in the window) of the first frame that falls past the end of the episode.
            remaining = end_idx - idx
            if remaining <= 0:  # Entire window falls outside episode boundary
                sample_end_idx = 0
            else:
                sample_end_idx = (remaining + N - 1) // N
                if sample_end_idx > sequence_length:
                    sample_end_idx = sequence_length

            buffer_start_idx = idx + sample_start_idx * N
            buffer_end_idx = idx + (sample_end_idx - 1) * N + 1

            if debug:
                assert sample_start_idx >= 0
                assert sample_end_idx > sample_start_idx
                assert buffer_start_idx >= start_idx
                assert buffer_end_idx <= end_idx

            indices.append([buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx])

    indices = np.array(indices)
    return indices


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask, max_n, seed=0):
    # subsample training data
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask


class ImprovedDatasetSampler:
    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        sequence_length: int,
        shape_meta: dict,
        pad_before: int = 0,
        pad_after: int = 0,
        keys=None,
        key_first_k=dict(),
        episode_mask: Optional[np.ndarray] = None,
        downsample_steps: int = 1,
    ):
        """
        key_first_k: dict str: int
            Only take first k data from these keys (to improve performance)

        downsample_steps: int
            Stride between sample positions in the replay buffer.
            With N = downsample_steps, sample position j (0-indexed) reads buffer index idx + j*N. 
            Example: downsample_steps = 3 to sample 10 Hz windows from a 30 Hz buffer.

        Padding behavior:
            Observation padding (episode start) is automatic: when a window extends
            before the start of an episode, obs keys are pre-padded by repeating the
            first real frame. Non-obs keys (e.g. action) are left as zeros since
            there is no meaningful prior value to repeat.
            Post-padding (episode end) repeats the last real frame for all keys,
            equivalent to SequenceSampler's action_padding=True behavior.
        """
        assert sequence_length >= 1
        assert downsample_steps >= 1

        if keys is None:
            keys = list(replay_buffer.keys())

        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        if np.any(episode_mask):
            indices = create_indices(
                episode_ends,
                sequence_length=sequence_length,
                pad_before=pad_before,
                pad_after=pad_after,
                episode_mask=episode_mask,
                downsample_steps=downsample_steps,
            )
        else:
            indices = np.zeros((0, 4), dtype=np.int64)

        # (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        # With stride N>1, buffer_end_idx - buffer_start_idx is the buffer *span*
        # (not the frame count); the slice [buffer_start_idx:buffer_end_idx:N]
        # yields sample_end_idx - sample_start_idx frames.
        self.indices = indices
        self.keys = list(keys)  # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k
        self.downsample_steps = downsample_steps

        # ImprovedDatasetSampler-specific initialization
        self.shape_meta = shape_meta
        self.obs_dict_keys = shape_meta["obs"].keys()
        self.rgb_keys = [key for key in self.obs_dict_keys if shape_meta["obs"][key]["type"] == "rgb"]
        self.data_dict_keys = [key for key in self.keys if key not in self.obs_dict_keys]

    def __len__(self):
        return len(self.indices)

    def sample_data(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        N = self.downsample_steps
        n_sampled = sample_end_idx - sample_start_idx
        datagram = dict()
        datagram["obs"] = dict()
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            # performance optimization, avoid small allocation if possible
            if key not in self.key_first_k:
                sample = input_arr[buffer_start_idx:buffer_end_idx:N]
            else:
                # performance optimization, only load used obs steps
                k_data = min(self.key_first_k[key], n_sampled)
                # fill value with NaN to catch bugs
                # the non-loaded region should never be used
                sample = np.full(
                    (n_sampled,) + input_arr.shape[1:],
                    fill_value=np.nan,
                    dtype=input_arr.dtype,
                )
                try:
                    sample[:k_data] = input_arr[
                        buffer_start_idx : buffer_start_idx + (k_data - 1) * N + 1 : N
                    ]
                except Exception:
                    import pdb
                    pdb.set_trace()
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                # Padding is needed: sample is shorter than sequence_length because
                # this window extends before the episode start (sample_start_idx > 0)
                # or past the episode end (sample_end_idx < sequence_length).
                data = np.zeros(
                    shape=(self.sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype,
                )
                if sample_start_idx > 0 and key in self.obs_dict_keys:
                    # Pre-pad: repeat the first real frame into the leading slots.
                    # Only done for obs keys — action/other keys are left as zeros
                    # since there is no meaningful "prior" value to repeat.
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < self.sequence_length:
                    # Post-pad: repeat the last real frame into the trailing slots.
                    # Applied to all keys so the policy sees a plausible final state.
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            # Route sampled data into the output datagram.
            # RGB obs keys are kept as uint8 (compressed zarr stays on disk until
            # sliced here); all other obs keys are cast to float32.
            if key == "state":
                datagram["obs"]["agent_pos"] = data.astype(np.float32)
            elif key in self.rgb_keys:
                datagram["obs"][key] = data.astype(np.uint8)
            elif key in self.obs_dict_keys:
                datagram["obs"][key] = data.astype(np.float32)
            elif key == "target":
                datagram["target"] = data[0].astype(np.float32)
            else:
                datagram[key] = data.astype(np.float32)
        return datagram
