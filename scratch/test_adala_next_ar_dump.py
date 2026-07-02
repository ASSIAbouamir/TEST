import requests
from bs4 import BeautifulSoup
import json
import sys

def test_adala_next_data_ar():
    url = "https://adala.justice.gov.ma/ar/projects-of-laws"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        r = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(r.text, 'html.parser')
        next_data = soup.find('script', id='__NEXT_DATA__')
        if next_data:
            data = json.loads(next_data.string)
            page_props = data.get("props", {}).get("pageProps", {})
            resources = page_props.get("resources", [])
            print(f"Number of resources: {len(resources)}")
            
            with open("scratch_adala_ar_details.json", "w", encoding="utf-8") as f:
                json.dump(resources, f, ensure_ascii=False, indent=2)
            print("Successfully dumped resources to scratch_adala_ar_details.json")
            
            for idx, item in enumerate(resources):
                print(f"Resource {idx+1}:")
                # print keys and basic info that is safe (or encode print in utf-8)
                title = item.get("title") or item.get("title_ar") or item.get("title_fr") or "No title"
                doc_path = item.get("document_path")
                # print via utf-8 stdout
                sys.stdout.buffer.write(f"  Title: {title}\n  Path: {doc_path}\n".encode('utf-8'))
        else:
            print("No __NEXT_DATA__ tag found.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_adala_next_data_ar()
