with open("scratch/kernels/kiza123123_s6e9-generator-forensics-5-submission-pipeline/s6e9-generator-forensics-5-submission-pipeline.py") as f:
    lines = f.readlines()
for i in range(110, 150):
    print(f"{i+1}: {lines[i]}", end="")
