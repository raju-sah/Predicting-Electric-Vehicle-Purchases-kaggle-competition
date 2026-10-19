import subprocess

def run_cmd(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout

print("--- Elefante kernels ---")
print(run_cmd("kaggle kernels list -s 'elefante'"))

print("--- Nina kernels ---")
print(run_cmd("kaggle kernels list -s 'nina' -s 's6e9'"))

print("--- Ravi kernels ---")
print(run_cmd("kaggle kernels list -s 'ravi' -s 's6e9'"))

print("--- Hayashi (haya / yhay81) kernels ---")
print(run_cmd("kaggle kernels list -s 'yhay81'"))
