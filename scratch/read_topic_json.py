import subprocess
import json

topics = [740775, 740441, 740159, 739280, 740824, 738991]
for t in topics:
    cmd = ["kaggle", "competitions", "topic-messages", "playground-series-s6e9", str(t), "--format", "json"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    try:
        data = json.loads(res.stdout)
        print(f"\n==================== TOPIC {t} ====================")
        for msg in data:
            print(f"Author: {msg.get('authorName')}, Votes: {msg.get('votes')}")
            print(f"Content:\n{msg.get('content')[:1500]}\n---")
    except Exception as e:
        print(f"Error parsing topic {t}: {e}")
