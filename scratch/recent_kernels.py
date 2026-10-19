import subprocess

def run_cmd(cmd):
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return res.stdout

print("--- Listing recent s6e9 kernels ---")
print(run_cmd("kaggle kernels list -s 's6e9' --sort-by dateRun --page-size 30"))
