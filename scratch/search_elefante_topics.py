import subprocess

def run_cmd(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout

print("--- Searching kernels by user 'Paul Bryan Elefante' / 'elefante' ---")
print(run_cmd("kaggle kernels list -s 'elefante'"))

print("--- Searching kernels by user 'cstdy' ---")
print(run_cmd("kaggle kernels list -s 'cstdy'"))

print("--- Searching kernels by user 'optimistix' ---")
print(run_cmd("kaggle kernels list -s 'optimistix'"))

print("--- Searching kernels with '0.9465' in title ---")
print(run_cmd("kaggle kernels list -s '0.9465'"))
