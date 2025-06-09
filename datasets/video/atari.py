from typing import Any, Dict, List, Optional
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
import minari
import numpy as np
from .base_video import (
    BaseAdvancedVideoDataset,
    SPLIT,
)

class AtariDataset(BaseAdvancedVideoDataset):
    """
    An advanced video dataset for Atari gameplay from the Minari dataset.
    """
    def __init__(
        self,
        cfg: DictConfig,
        split: SPLIT = "training",
        current_epoch: Optional[int] = None,
    ):
        self.dataset_name = "atari/frostbite/expert-v0"
        self._episodes = None
        # The base class init will call all necessary setup methods
        super().__init__(cfg, "training", current_epoch)

    @property
    def episodes(self) -> List:
        """Lazy-loads the episodes from the Minari dataset."""
        if self._episodes is None:
            self.minari_dataset = minari.load_dataset(self.dataset_name)
            self._episodes = list(self.minari_dataset.iterate_episodes())
        return self._episodes

    def download_dataset(self) -> None:
        """Implementation of the abstract method to download the dataset."""
        minari.download_dataset(self.dataset_name)

    def build_metadata(self, split: SPLIT) -> None:
        # This dataset doesn't use file-based metadata that needs building.
        # We override this method to do nothing.
        pass

    def load_metadata(self) -> List[Dict[str, Any]]:
        # Create metadata on the fly from the loaded episodes.
        return [{"length": len(ep.observations)} for ep in self.episodes]

    def video_length(self, video_metadata: Dict[str, Any]) -> int:
        return video_metadata["length"]

    def load_video(
        self,
        video_metadata: Dict[str, Any],
        start_frame: int,
        end_frame: Optional[int] = None,
    ) -> torch.Tensor:
        video_idx = self.metadata.index(video_metadata)
        episode = self.episodes[video_idx]
        video = torch.from_numpy(np.array(episode.observations[start_frame:end_frame]))
        video = video.permute(0, 3, 1, 2).float() / 255.0
        return video

    def load_cond(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        video_idx = self.metadata.index(video_metadata)
        episode = self.episodes[video_idx]
        actions = episode.actions[start_frame:end_frame]
        return F.one_hot(torch.from_numpy(actions), num_classes=18).float()