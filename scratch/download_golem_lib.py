import subprocess

cmd = ["kaggle", "datasets", "download", "-d", "dariushafshar/s6e9-golem-oof-library", "-p", "scratch/golem_lib/", "--unzip"]
res = subprocess.run(cmd, capture_output=True, text=True)
print(res.stdout, res.stderr)
