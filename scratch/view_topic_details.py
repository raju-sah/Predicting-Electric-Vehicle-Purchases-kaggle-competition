import subprocess
import json

def get_topic(topic_id):
    res = subprocess.run(["kaggle", "competitions", "topic-messages", "list", "-t", str(topic_id)], capture_output=True, text=True)
    if res.returncode == 0 and res.stdout.strip():
        print(f"--- TOPIC {topic_id} MESSAGES ---")
        print(res.stdout)
    else:
        # try topic show with json or csv
        res2 = subprocess.run(["kaggle", "competitions", "topics", "show", str(topic_id), "--format", "json"], capture_output=True, text=True)
        print(f"--- TOPIC {topic_id} JSON ---")
        print(res2.stdout[:3000])

for tid in [740769, 740775, 740824]:
    get_topic(tid)
