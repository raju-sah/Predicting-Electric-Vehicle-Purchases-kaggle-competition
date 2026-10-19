# Run the training script as a background thread INSIDE the persistent kernel
# (process children get killed on websocket close; threads survive).
# runpy globals shadow builtins, so injecting print into the script globals
# redirects its logging without touching the kernel's own print.
import threading, runpy, traceback

def make_runner(script, logfile, done_key):
    def run_bg():
        logf = open(logfile, "w")
        def log(*a, **k):
            end = k.get("end", "\n")
            logf.write(" ".join(str(x) for x in a) + end)
            logf.flush()
        g = {"__name__": "__main__", "print": log}
        try:
            runpy.run_path(script, run_name="__main__")
            logf.write("\nDONE_MARKER\n")
            done_key["done"] = True
        except Exception:
            logf.write("\nEXC:\n" + traceback.format_exc() + "\n")
        finally:
            logf.flush()
            logf.close()
    return run_bg

if "BG_JOBS" not in globals():
    BG_JOBS = {}
done_key = {}
target = make_runner("/content/train_realmlp.py", "/content/train_realmlp.log", done_key)
t = threading.Thread(target=target, daemon=True)
t.start()
BG_JOBS["realmlp"] = {"thread": t, "done": done_key, "log": "/content/train_realmlp.log"}
print("thread started:", t.name, "alive:", t.is_alive())
