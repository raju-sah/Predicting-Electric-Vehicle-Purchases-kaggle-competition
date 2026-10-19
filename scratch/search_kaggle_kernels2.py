import subprocess

def run_cmd(cmd):
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return res.stdout

print("--- Listing with search 's6e9' ---")
print(run_cmd("kaggle kernels list -s 's6e9' --sort-by voteCount --page-size 20"))

print("--- Listing with search 'electric vehicle' ---")
print(run_cmd("kaggle kernels list -s 'electric vehicle' --sort-by voteCount --page-size 15"))
