import json
import subprocess
import re

def parse_kaggle_json(out):
    # Find JSON array [...]
    match = re.search(r'(\[\s*\{.*\}\s*\])', out, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    # Or try loading line by line / up to last ]
    idx = out.rfind(']')
    if idx != -1:
        start = out.find('[')
        return json.loads(out[start:idx+1])
    return json.loads(out)

for sort in ['new', 'hot', 'recent']:
    res = subprocess.run(f"kaggle competitions topics list playground-series-s6e9 -s {sort} --format json", shell=True, capture_output=True, text=True)
    try:
        items = parse_kaggle_json(res.stdout)
        print(f"=== Sort: {sort} ({len(items)} items) ===")
        for t in items:
            print(f"[{t.get('id')}] {t.get('postDate', '')[:19]} | V:{t.get('votes', 0):2d} | C:{t.get('commentCount', 0):2d} | {t.get('title')}")
    except Exception as e:
        print(f"Error for {sort}: {e}")

