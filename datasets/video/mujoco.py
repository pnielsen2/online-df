from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
from omegaconf import DictConfig

from .base_video import BaseAdvancedVideoDataset, SPLIT


class MujocoDataset(BaseAdvancedVideoDataset):
    """
    A dataset for loading MuJoCo trajectories.
    Each "frame" of the video is a concatenation of the state, action, and reward.
    """

    def __init__(
        self,
        cfg: DictConfig,
        split: SPLIT = "training",
        current_epoch: Optional[int] = None,
    ):
        super().__init__(cfg, split, current_epoch)
        self.env_name = cfg.env_name

    def _should_download(self) -> bool:
        return not (self.save_dir / f"{self.env_name}_all_trajectories.npz").exists()

    def download_dataset(self) -> None:
        print(f"MuJoCo data not found at {self.save_dir / self.env_name}_all_trajectories.npz")
        print("Please run the `scripts/generate_mujoco_data.py` script first.")
        raise FileNotFoundError

    def build_metadata(self, split: SPLIT) -> None:
        """
        Builds metadata from the generated .npz file.
        """
        if split != "training":
            return

        npz_path = self.save_dir / f"{self.env_name}_all_trajectories.npz"
        if not npz_path.exists():
            self.download_dataset()

        data = np.load(npz_path, allow_pickle=True)
        trajectories = data['trajectories']

        video_pts = [torch.arange(len(traj['states'])) for traj in trajectories]
        video_paths = [f"{i}.traj" for i in range(len(trajectories))]

        metadata = {
            "video_paths": video_paths,
            "video_pts": video_pts,
            "trajectories": trajectories,
        }
        torch.save(metadata, self.metadata_dir / f"{split}.pt")

    def load_video(
        self,
        video_metadata: Dict[str, Any],
        start_frame: int,
        end_frame: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Loads and concatenates states, actions, and rewards into a single tensor.
        This concatenated tensor is treated as our "video".
        """
        traj_idx = int(video_metadata["video_paths"].split('.')[0])
        trajectory = self.metadata['trajectories'][traj_idx]

        states = trajectory['states'][start_frame:end_frame]
        actions = trajectory['actions'][start_frame:end_frame]
        rewards = trajectory['rewards'][start_frame:end_frame]

        # Ensure reward is the correct shape (T, 1) for concatenation
        if rewards.ndim == 1:
            rewards = rewards[:, None]

        # Concatenate (state, action, reward) for each time step
        concatenated_frames = np.concatenate([states, actions, rewards], axis=-1)

        # Reshape to be like a video (T, C, H, W) where H and W are 1
        return torch.from_numpy(concatenated_frames).float().unsqueeze(-1).unsqueeze(-1)

    def load_cond(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> None:
        """
        No separate conditioning is needed, so we return None.
        """
        return None