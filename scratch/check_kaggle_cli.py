import os
import subprocess
import json

# Check kaggle CLI options for forum / discussion
result = subprocess.run(["kaggle", "--help"], capture_output=True, text=True)
print("Kaggle CLI subcommands:")
print(result.stdout)
