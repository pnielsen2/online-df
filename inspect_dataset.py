import minari
import numpy as np
import gymnasium as gym

# --- This script loads a local Minari dataset and prints its episode length statistics ---

try:
    dataset_name = "atari/frostbite/expert-v0"
    print(f"Loading dataset: {dataset_name} to inspect episode lengths...")

    # Load the dataset from local storage
    dataset = minari.load_dataset(dataset_name)

    episode_lengths = []
    # Iterate through all the episodes in the dataset
    for episode_data in dataset.iterate_episodes():
        # The length of an episode is the number of observations (or actions)
        episode_lengths.append(len(episode_data.observations))

    print("\n" + "="*40)
    print(f"Statistics for: {dataset_name}")
    print("="*40)
    print(f"Number of episodes found: {len(episode_lengths)}")
    if episode_lengths:
        print(f"Max episode length: {np.max(episode_lengths)}")
        print(f"Min episode length: {np.min(episode_lengths)}")
        print(f"Average episode length: {np.mean(episode_lengths):.2f}")
        # This is the number the training script uses to filter videos
        required_length = 1 + (2 - 1) * 2 # Calculated with max_frames=2, frame_skip=2
        print(f"\nFor reference, with max_frames=2, the required length is {required_length} frames.")
    else:
        print("No episodes found in the dataset.")
    print("="*40)

except Exception as e:
    print(f"\nAn error occurred: {e}")