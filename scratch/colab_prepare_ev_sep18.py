from pathlib import Path


data_dir = Path("/content/ev_s6e9")
data_dir.mkdir(parents=True, exist_ok=True)
print(data_dir)
