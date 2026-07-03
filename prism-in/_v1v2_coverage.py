import glob, os
from collections import Counter

v1 = sorted(glob.glob("trigger_results_in_morning_2*.json"))
v2 = sorted(glob.glob("trigger_results_v2/trigger_results_in_morning_2*.json"))

def ed(f):
    b = os.path.basename(f).replace("trigger_results_in_morning_", "").replace(".json", "")
    return b

def em(f):
    return ed(f)[:6]

v1d = [ed(f) for f in v1]
v2d = [ed(f) for f in v2]
print(f"V1: {len(v1)} files, {v1d[0]} to {v1d[-1]}")
print(f"V2: {len(v2)} files, {v2d[0]} to {v2d[-1]}")

v1m = Counter(em(f) for f in v1)
v2m = Counter(em(f) for f in v2)
ams = sorted(set(list(v1m) + list(v2m)))
print()
print(f"{'Month':<8} {'V1':>4} {'V2':>4}")
print("-" * 20)
for m in ams:
    print(f"{m:<8} {v1m.get(m, 0):>4} {v2m.get(m, 0):>4}")

v1s = set(v1d)
v2s = set(v2d)
print(f"\nV1-only dates: {len(v1s - v2s)}")
print(f"V2-only dates: {len(v2s - v1s)}")
print(f"Common dates: {len(v1s & v2s)}")
