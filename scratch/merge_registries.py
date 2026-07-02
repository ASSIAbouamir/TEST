import json
from pathlib import Path

reg_path = Path("data/legal_updates_registry.json")
bak_path = Path("data/legal_updates_registry.json.bak")

seen_ids = set()

if reg_path.exists():
    try:
        data = json.loads(reg_path.read_text(encoding="utf-8"))
        seen_ids.update(data.get("seen_ids", []))
    except Exception as e:
        print(f"Error loading current: {e}")

if bak_path.exists():
    try:
        data = json.loads(bak_path.read_text(encoding="utf-8"))
        seen_ids.update(data.get("seen_ids", []))
    except Exception as e:
        print(f"Error loading bak: {e}")

# Save merged
merged_data = {
    "seen_ids": sorted(list(seen_ids)),
    "last_updated": json.loads(reg_path.read_text(encoding="utf-8")).get("last_updated") if reg_path.exists() else ""
}

reg_path.write_text(json.dumps(merged_data, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Merged registry. Total seen ids: {len(seen_ids)}")

# Remove backup
if bak_path.exists():
    bak_path.unlink()
    print("Deleted backup file.")
