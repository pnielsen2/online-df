from datasets.video import (
    MinecraftAdvancedVideoDataset,
    Kinetics600AdvancedVideoDataset,
    RealEstate10KAdvancedVideoDataset,
    RealEstate10KMiniAdvancedVideoDataset,
    RealEstate10KOODAdvancedVideoDataset,
    AtariDataset,
    MujocoDataset,
)
from algorithms.dfot import DFoTVideo, DFoTVideoPose
from .base_exp import BaseLightningExperiment
from .data_modules.utils import _data_module_cls
import logging


class VideoGenerationExperiment(BaseLightningExperiment):

    """
    A video generation experiment
    """

    compatible_algorithms = dict(
        dfot_video=DFoTVideo,
        dfot_video_pose=DFoTVideoPose,
        sd_video=DFoTVideo,
        sd_video_3d=DFoTVideoPose,
        dfot_video_mujoco=DFoTVideo,  # Add this line to register our new algorithm
    )

    compatible_datasets = dict(
        # video datasets
        minecraft=MinecraftAdvancedVideoDataset,
        realestate10k=RealEstate10KAdvancedVideoDataset,
        realestate10k_ood=RealEstate10KOODAdvancedVideoDataset,
        realestate10k_mini=RealEstate10KMiniAdvancedVideoDataset,
        kinetics_600=Kinetics600AdvancedVideoDataset,
        atari=AtariDataset,
        mujoco=MujocoDataset,
    )

    data_module_cls = _data_module_cls