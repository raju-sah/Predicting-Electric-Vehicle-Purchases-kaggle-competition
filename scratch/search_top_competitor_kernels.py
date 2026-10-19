import subprocess

def run_cmd(cmd):
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return res.stdout

print("--- Searching for Naveenkumar kernels ---")
print(run_cmd("kaggle kernels list -s 'naveenkumar' --page-size 10"))

print("--- Searching for Ravi Ramakrishnan kernels ---")
print(run_cmd("kaggle kernels list -s 'ravi' -s 's6e9' --page-size 10"))

print("--- Searching for Sergio kernels ---")
print(run_cmd("kaggle kernels list -s 'sergio' -s 's6e9' --page-size 10"))
