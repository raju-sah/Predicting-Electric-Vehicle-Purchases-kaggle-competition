import json
import subprocess
import sys

topic_ids = [740769, 741447, 741755, 741117, 740049, 739910]

for tid in topic_ids:
    print(f"\n==================== TOPIC {tid} ====================")
    cmd = f"kaggle competitions topics show playground-series-s6e9 {tid} --format json"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode == 0:
        try:
            # might have warning at end, grab json
            s = res.stdout.strip()
            # find first { or [
            start = s.find('{')
            end = s.rfind('}')
            if start != -1 and end != -1:
                data = json.loads(s[start:end+1])
                print(f"Title: {data.get('title')}")
                print(f"Author: {data.get('author') or data.get('authorName')}")
                content = data.get('content') or data.get('body') or data.get('message') or ""
                print(f"Body:\n{content[:1500]}\n")
            else:
                print("Could not find json braces:", s[:300])
        except Exception as e:
            print(f"Parse error: {e}, raw: {res.stdout[:300]}")
    else:
        print("Command failed:", res.stderr)

