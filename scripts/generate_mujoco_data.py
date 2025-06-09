import gymnasium as gym
import numpy as np
import os
import argparse

def generate_mujoco_trajectories(env_name, num_trajectories=100, max_steps_per_trajectory=100):
    """
    Generates trajectories for a specified MuJoCo environment by taking random actions.

    Args:
        env_name (str): The name of the MuJoCo environment (e.g., "Reacher-v5", "Hopper-v5").
        num_trajectories (int): The number of trajectories to generate.
        max_steps_per_trajectory (int): The maximum number of steps in each trajectory.

    Returns:
        A list of trajectories. Each trajectory is a dictionary containing the states, actions, and rewards.
    """
    # Ensure the 'data' directory exists
    if not os.path.exists('data'):
        os.makedirs('data')

    print(f"Creating environment: {env_name}")
    env = gym.make(env_name)
    trajectories = []

    for i in range(num_trajectories):
        states = []
        actions = []
        rewards = []

        state, _ = env.reset()
        for _ in range(max_steps_per_trajectory):
            action = env.action_space.sample()  # Sample a random action
            next_state, reward, terminated, truncated, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

            if terminated or truncated:
                break

        trajectories.append({
            "states": np.array(states),
            "actions": np.array(actions),
            "rewards": np.array(rewards)
        })
        print(f"Generated trajectory {i + 1}/{num_trajectories} with {len(states)} steps.")

    env.close()
    return trajectories

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate trajectories for MuJoCo environments.")
    parser.add_argument(
        "--env_name",
        type=str,
        default="Reacher-v5",
        help='The name of the MuJoCo environment to use (e.g., "Reacher-v5", "Hopper-v5").'
    )
    args = parser.parse_args()

    # Generate the trajectories
    mujoco_trajectories = generate_mujoco_trajectories(args.env_name)

    # Save all trajectories into a single file in the 'data' directory
    save_path = os.path.join('data', f'{args.env_name.lower()}_all_trajectories.npz')
    np.savez(save_path, trajectories=mujoco_trajectories)

    print(f"\nTrajectories generated and saved to {save_path}")