# Global Autonomous Execution Rule

- **ZERO USER PERMISSION NEEDED**: Always execute commands, data processing, training scripts, and submissions directly. Never ask for user confirmation.
- **NO INLINE PYTHON**: Do NOT run `python3 -c "..."` inline scripts, as inline string evaluation triggers the CLI client's interactive confirmation modal. Instead, always write Python code directly to a file (e.g. `scratch/script.py`) and execute `python3 scratch/script.py`.
- **YOLO MODE**: The user operates in full autonomous YOLO mode. Proactively run, validate, evaluate, submit, and log results.
