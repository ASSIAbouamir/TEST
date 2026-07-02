import sys
import os
import json

# Add the directory to python path
sys.path.insert(0, r"c:\Users\hp assia\Desktop\Automatisation")

from server import compute_dynamic_stats

stats = compute_dynamic_stats()
print("CURRENT DYNAMIC STATS:")
print(json.dumps(stats, indent=2, ensure_ascii=False))
