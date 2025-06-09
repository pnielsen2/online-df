import minari
import gymnasium as gym
import pandas as pd
from tqdm import tqdm

# --- This script lists all remote MuJoCo datasets with detailed info (METADATA-ONLY) ---

print("Fetching remote dataset list from Farama servers...")
try:
    # This is the correct and efficient way to get the metadata for all remote datasets
    remote_datasets = minari.list_remote_datasets()

    # Filter for just the MuJoCo datasets
    mujoco_datasets = {name: data for name, data in remote_datasets.items() if name.startswith('mujoco/')}

    if not mujoco_datasets:
        print("No MuJoCo datasets found on the remote server.")
        exit()

    print(f"Found {len(mujoco_datasets)} MuJoCo datasets. Inspecting metadata...")

    table_data = []
    # Iterate through the metadata we already fetched
    for name, data in tqdm(mujoco_datasets.items(), desc="Processing metadata"):
        try:
            # Recreate a dummy environment from the env_spec in the metadata
            # This does NOT download the large data files
            env = gym.make(data['env_spec'])

            # Get observation and action space details
            obs_space = str(env.observation_space.shape)
            action_space = str(env.action_space.shape)

            # Get FPS from the environment's metadata
            fps = 'N/A'
            if hasattr(env, 'dt') and env.dt is not None:
                fps = int(1 / env.dt)
            elif hasattr(env, 'metadata') and 'render_fps' in env.metadata:
                fps = env.metadata['render_fps']

            table_data.append({
                "Name": name,
                "Obs Space Shape": obs_space,
                "Action Space Shape": action_space,
                "Obs per Second (FPS)": fps,
                "Total Steps": f"{data['total_steps']:,}",
                "Size": data.get('dataset_size', 'N/A'),
            })

            # Close the dummy environment to free resources
            env.close()

        except Exception as e:
            print(f"    -> Could not inspect {name}. Error: {e}")

    # Display the data in a table using pandas
    if table_data:
        pd.set_option('display.max_rows', None)
        df = pd.DataFrame(table_data)
        print("\n" + "="*120)
        print("Available MuJoCo Datasets")
        print("="*120)
        print(df.to_string())

except Exception as e:
    print(f"\nAn error occurred while fetching the dataset list: {e}")