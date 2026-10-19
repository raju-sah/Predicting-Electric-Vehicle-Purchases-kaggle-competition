import json
import subprocess

res = subprocess.run("kaggle competitions topics show playground-series-s6e9 740769 --format json", shell=True, capture_output=True, text=True)
s = res.stdout.strip()
start = s.find('{')
end = s.rfind('}')
data = json.loads(s[start:end+1])
print("Title:", data["topic"]["title"])
print("Post date:", data["topic"]["postDate"])
for c in data.get("comments", []):
    print(f"\n--- {c.get('authorName')} ({c.get('postDate')}) ---")
    print(c.get('content') or c.get('body') or '')
