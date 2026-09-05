import subprocess
import os


def run_pipeline():
   
    os.environ['PYTHONPATH'] = '.'

    
    commands = [
        'python src/preprocess.py',
        'python src/train.py',
        'python src/predict.py'
    ]

    for cmd in commands:
        print(f"\n{'=' * 50}")
        print(f"Running: {cmd}")
        print('=' * 50)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        if result.returncode != 0:
            print(f"Command failed with code {result.returncode}")
            break


if __name__ == "__main__":
    run_pipeline()
