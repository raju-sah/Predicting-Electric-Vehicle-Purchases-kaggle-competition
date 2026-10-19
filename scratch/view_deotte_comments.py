import subprocess, json

res = subprocess.run(["kaggle", "competitions", "topics", "show", "738991", "--format", "json"], capture_output=True, text=True)
data = json.loads(res.stdout)
for c in data.get("comments", []):
    if "Georgy" in c.get("authorName", "") or "Botaylaamin" in c.get("authorName", ""):
        print(f"=== {c.get('authorName')} ===")
        print(c.get("content"))
