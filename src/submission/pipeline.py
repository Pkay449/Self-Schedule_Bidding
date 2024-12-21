import subprocess
import sys

def run_script(script_name):
    """
    Runs the training, offline generation and evaluation scripts for project.

    Parameters:
    script_name (str): Name of the Python script to run.
    """
    try:
        print(f"Running {script_name}...")
        result = subprocess.run(
            [sys.executable, script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        print(f"Output from {script_name}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:\n{e.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    scripts = [
        "train_optPolicy_tqdm.py",
        "generate_offline_evaluate.py",
    ]

    for script in scripts:
        run_script(script)

    print("Pipeline completed successfully.")
