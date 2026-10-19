import subprocess, sys
cmd = [sys.executable, "-u", "/tmp/xgb_local.py"]
with open("scratch/xgb_local.log", "w") as logf:
    subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True,
                     cwd="/home/raju/AI-ML Projects/Kaggle Competation/Predicting Electric Vehicle Purchases")
print("detached launcher done")
