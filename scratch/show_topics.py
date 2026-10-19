import subprocess

def show_topic(tid):
    res = subprocess.run(["kaggle", "competitions", "topics", "show", "-c", "playground-series-s6e9", "-t", str(tid)], capture_output=True, text=True)
    return res.stdout

print("=== TOPIC 740775 ===")
print(show_topic(740775)[:2500])

print("\n=== TOPIC 740769 ===")
print(show_topic(740769)[:2500])
