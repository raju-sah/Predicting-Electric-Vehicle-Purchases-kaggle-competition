import subprocess

cmd = "cmp scratch/kernels_pulled/taeyangg4_094650_out/submission.csv subs/v26_taeyang_lexsort_master.csv"
res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print("CMP output:", res.stdout, res.stderr, res.returncode)

# Check sha256 of both files
h1 = subprocess.run("sha256sum scratch/kernels_pulled/taeyangg4_094650_out/submission.csv", shell=True, capture_output=True, text=True).stdout
h2 = subprocess.run("sha256sum subs/v26_taeyang_lexsort_master.csv", shell=True, capture_output=True, text=True).stdout
print("Hash 1:", h1)
print("Hash 2:", h2)
