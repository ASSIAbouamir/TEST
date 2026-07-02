import json

with open("scratch_adala_next_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

page_props = data.get("props", {}).get("pageProps", {})
print("pageProps keys:", list(page_props.keys()))
for k, v in page_props.items():
    if isinstance(v, list):
        print(f"Key '{k}' is a list of length {len(v)}")
    elif isinstance(v, dict):
        print(f"Key '{k}' is a dict with keys {list(v.keys())}")
    else:
        print(f"Key '{k}' is type {type(v)}: {v}")
