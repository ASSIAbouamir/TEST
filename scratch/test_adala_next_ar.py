import requests
from bs4 import BeautifulSoup
import json

def test_adala_next_data_ar():
    url = "https://adala.justice.gov.ma/ar/projects-of-laws"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        r = requests.get(url, headers=headers, timeout=15)
        print(f"Status: {r.status_code}")
        soup = BeautifulSoup(r.text, 'html.parser')
        next_data = soup.find('script', id='__NEXT_DATA__')
        if next_data:
            print("Found __NEXT_DATA__ in AR!")
            data = json.loads(next_data.string)
            page_props = data.get("props", {}).get("pageProps", {})
            print("pageProps keys in AR:", list(page_props.keys()))
            resources = page_props.get("resources", [])
            print(f"Number of resources in AR: {len(resources)}")
            if resources:
                print("First resource details in AR:")
                print(json.dumps(resources[0], ensure_ascii=False, indent=2))
                
                # Let's see some fields
                for idx, r in enumerate(resources[:10]):
                    title = r.get("title_fr") or r.get("title_ar") or r.get("title")
                    doc_path = r.get("document_path")
                    created_at = r.get("created_at") or r.get("publish_date")
                    print(f"{idx+1}. Title: {title} | Path: {doc_path} | Date: {created_at}")
            else:
                print("No resources in AR list.")
        else:
            print("No __NEXT_DATA__ tag found in AR.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_adala_next_data_ar()
