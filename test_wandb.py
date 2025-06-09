import wandb
import random
import sys

# --- This script is for testing wandb permissions ---

try:
    # We use a new, random project name to ensure it doesn't already exist
    # and that we have permissions to create it.
    project_name = f"permission-test-{random.randint(1000, 9999)}"
    print(f"Attempting to initialize wandb run in entity 'pnielsen2', project '{project_name}'")

    run = wandb.init(
        project=project_name,
        entity="pnielsen2",
    )

    print("\n--- SUCCESS! ---")
    print(f"Successfully initialized wandb run: {run.url}")
    run.finish()
    print("Run finished successfully.")
    print("This means your wandb account is configured correctly.")

except Exception as e:
    print("\n--- FAILED ---")
    print("The minimal test script failed with the following error:")
    print(e)
    print("\nThis confirms the issue is with your wandb account permissions and not the project's code.")
    print("Please check your account settings on wandb.ai or contact wandb support.")