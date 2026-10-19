import subprocess
import json

def run_cmd(cmd):
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return res.stdout

print("--- Listing top kernels for playground-series-s6e9 ---")
out_kernels = run_cmd("kaggle kernels list -c playground-series-s6e9 --sort-by scoreDescending --page-size 20")
print(out_kernels)

print("--- Listing most voted kernels for playground-series-s6e9 ---")
out_voted = run_cmd("kaggle kernels list -c playground-series-s6e9 --sort-by voteCount --page-size 20")
print(out_voted)
