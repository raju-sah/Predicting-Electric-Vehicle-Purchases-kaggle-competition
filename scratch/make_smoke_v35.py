import re
src = open("train_v35_zoom_max.py").read()
src = src.replace('train = pd.read_csv("train.csv")',
                  'train = pd.read_csv("train.csv").sample(frac=0.15, random_state=7).reset_index(drop=True)')
src = src.replace('test = pd.read_csv("test.csv")',
                  'test = pd.read_csv("test.csv").sample(frac=0.06, random_state=7).reset_index(drop=True)')
src = src.replace('np.save("oof_v35_zoom_max.npy", oof2)', 'pass  # smoke: no save')
src = src.replace('np.save("oof_v35_zoom_max_pass1.npy", oof1)', 'pass')
src = src.replace('np.save("test_v35_zoom_max.npy", test2)', 'pass')
src = src.replace('np.save("test_v35_zoom_max_pass1.npy", test1)', 'pass')
src = src.replace("split2 = pd.qcut", "import sys; sys.exit(0)\nsplit2 = pd.qcut") if "--pass1only" in __import__("sys").argv else src
open("scratch/smoke_v35.py", "w").write(src)
print("written scratch/smoke_v35.py")
