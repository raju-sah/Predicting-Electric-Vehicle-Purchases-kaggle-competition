with open("scratch/kernels_pulled/taeyangg4_094650_out/submission.csv") as f1, open("subs/v26_taeyang_lexsort_master.csv") as f2:
    for i in range(10):
        l1 = f1.readline().strip()
        l2 = f2.readline().strip()
        print(f"Line {i+1}:")
        print(f"  f1: {l1}")
        print(f"  f2: {l2}")
