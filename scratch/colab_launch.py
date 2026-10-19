import time
print("launching", flush=True)
import subprocess
subprocess.Popen("nohup python /content/train_realmlp.py > /content/train_realmlp.log 2>&1 &", shell=True)
time.sleep(2)
print("detached OK", flush=True)
