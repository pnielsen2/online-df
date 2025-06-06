from typing import Any, Dict, Optional
import torch
from omegaconf import DictConfig
import minari
from .base_video import (
    BaseAdvancedVideoDataset,
    SPLIT,
)


class AtariDataset(BaseAdvancedVideoDataset):
    """
    An advanced video dataset for Atari gameplay from the Minari dataset.
    """

    def download_dataset(self):
        """
        Downloads the Atari dataset from the Minari library.
        """
        minari.download_dataset("frostbite-v0")

    def __init__(
        self,
        cfg: DictConfig,
        split: SPLIT = "training",
        current_epoch: Optional[int] = None,
    ):
        if split == "validation":
            split = "test"
        super().__init__(cfg, split, current_epoch)

    def load_cond(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        """
        Loads the actions as conditions.
        """
        dataset = minari.load_dataset("frostbite-v0")
        actions = []
        for episode_data in dataset.iterate_episodes():
            actions.extend(episode_data.actions)
        return torch.tensor(actions[start_frame:end_frame])