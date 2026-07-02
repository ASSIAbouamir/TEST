import json

with open("scratch_adala_next_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

resources = data.get("props", {}).get("pageProps", {}).get("resources", [])
print(f"Number of resources: {len(resources)}")
if resources:
    print("First resource sample keys:", list(resources[0].keys()))
    print("First resource sample details:")
    # Print it formatted
    print(json.dumps(resources[0], ensure_ascii=False, indent=2))
    
    print("\nListing all resources titles and links:")
    for idx, r in enumerate(resources[:10]):
        title = r.get("title_fr") or r.get("title_ar") or r.get("title")
        doc_path = r.get("document_path")
        pub_date = r.get("publish_date") or r.get("created_at")
        print(f"{idx+1}. Title: {title} | Path: {doc_path} | Date: {pub_date}")
else:
    print("No resources found in list.")
