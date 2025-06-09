import hydra
from omegaconf import OmegaConf, DictConfig
import traceback

def debug_experiment(experiment_name: str, overrides: list = []):
    """
    Initializes Hydra and composes the configuration for a given experiment,
    then prints the result or the error.
    """
    print("=" * 80)
    print(f"DEBUG: Composing config for 'experiment={experiment_name}'")
    if overrides:
        print(f"DEBUG: With command-line overrides: {overrides}")
    print("=" * 80)

    try:
        # Use hydra.initialize to set up the config path
        with hydra.initialize(config_path="configurations", version_base="1.3"):
            # Use hydra.compose to build the config object
            cfg: DictConfig = hydra.compose(
                config_name="config",
                overrides=[f"experiment={experiment_name}"] + overrides
            )

            print(f"\n--- SUCCESS: Resolved Config for experiment='{experiment_name}' ---")
            # We print the resolved config to see the final values
            print(OmegaConf.to_yaml(cfg, resolve=True))
            print("-" * 80)

            # Specifically check the values of 'name' and 'exp_name'
            print(f"Value of cfg.name: {cfg.get('name')}")
            print(f"Value of cfg.exp_name: {cfg.get('exp_name')}")
            print("-" * 80)

    except Exception as e:
        print(f"\n--- FAILED to compose config for experiment='{experiment_name}' ---")
        print(f"ERROR: {e}\n")
        print("--- FULL TRACEBACK ---")
        traceback.print_exc()
        print("-" * 80)


if __name__ == "__main__":
    # Before running, please ensure the file from my last response exists at:
    # 'configurations/experiment/mujoco_video_generation.yaml'

    # Test 1: The failing case. This will help us see the error again.
    debug_experiment("mujoco_video_generation")

    # Test 2: The default working case. We'll use this as a reference.
    debug_experiment("video_generation")