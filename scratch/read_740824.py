import subprocess, json

res = subprocess.run(["kaggle", "competitions", "topics", "show", "740824", "--format", "json"], capture_output=True, text=True)
data = json.loads(res.stdout)
print("TITLE:", data["topic"]["title"])
print("BODY:\n", data["topic"].get("body", "NO BODY FIELD"))
print("KEYS:", list(data["topic"].keys()))
if "content" in data["topic"]:
    print("CONTENT:\n", data["topic"]["content"])
