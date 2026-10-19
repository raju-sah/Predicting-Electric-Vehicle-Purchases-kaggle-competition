import os
import json

path = os.path.expanduser("~/.gemini/antigravity-cli/settings.json")
if os.path.exists(path):
    with open(path) as f:
        data = json.load(f)
    print("Keys in settings.json:", list(data.keys()))
    print("Full settings:", data)
else:
    print("Does not exist")
