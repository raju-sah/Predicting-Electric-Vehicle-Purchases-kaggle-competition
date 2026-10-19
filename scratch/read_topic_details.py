import subprocess

topics = [740775, 740441, 740159, 739280]
for t in topics:
    print(f"\n==================== TOPIC {t} ====================")
    cmd = ["kaggle", "competitions", "topic-messages", "-t", str(t)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout[:3000])
