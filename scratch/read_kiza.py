with open("scratch/kernels/kiza123123_s6e9-generator-forensics-5-submission-pipeline/s6e9-generator-forensics-5-submission-pipeline.py") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if any(term in line.lower() for term in ["deotte", "recipe", "base_margin", "init_score", "formula", "catboost"]):
        print(f"Line {i+1}: {line.strip()}")
