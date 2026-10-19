import subprocess

topics = [740775, 740441, 740159, 739280, 740824]
for t in topics:
    print(f"\n==================== TOPIC {t} ====================")
    cmd = ["kaggle", "competitions", "topic-messages", "playground-series-s6e9", str(t)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout[:2500])
