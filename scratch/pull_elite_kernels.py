import subprocess

kernels = [
    "beraterolelk/ev-purchase-grandmaster-eda-ensemble",
    "megayak/the-1000-leak-a-from-scratch-0-946-pipeline",
    "dariushafshar/s6e9-0-938-phenomenon-0-946-generator"
]

for k in kernels:
    user, slug = k.split("/")
    dest = f"scratch/{user}/"
    cmd = ["kaggle", "kernels", "pull", "-p", dest, k]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(k, res.stdout, res.stderr)
