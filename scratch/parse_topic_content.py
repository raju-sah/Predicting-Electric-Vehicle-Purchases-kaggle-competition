import subprocess
import json

topic_ids = [740769, 741447, 740049, 741755, 741456, 739498]

for tid in topic_ids:
    print(f"\n=======================================================")
    print(f"TOPIC {tid}")
    print(f"=======================================================")
    res = subprocess.run(f"kaggle competitions topics show playground-series-s6e9 {tid} --format json", shell=True, capture_output=True, text=True)
    if res.returncode == 0:
        try:
            s = res.stdout.strip()
            start = s.find('{')
            end = s.rfind('}')
            data = json.loads(s[start:end+1])
            top = data.get("topic", {})
            print(f"Title: {top.get('title')}")
            print(f"Author: {top.get('authorName')}")
            print(f"Date: {top.get('postDate')}")
            print(f"Topic content:\n{top.get('content') or top.get('body') or top.get('message') or ''}\n")
            comments = data.get("comments", [])
            print(f"--- Comments ({len(comments)}) ---")
            for c in comments:
                print(f"\n[{c.get('authorName')} @ {c.get('postDate')}]:")
                c_text = c.get('content') or c.get('body') or c.get('message') or ''
                print(c_text[:1000])
        except Exception as e:
            print("Error parsing:", e)
    else:
        print("Failed to run cmd:", res.stderr)

