import requests
from bs4 import BeautifulSoup
import json

def test_adala_next_data():
    url = "https://adala.justice.gov.ma/fr/projects-of-laws"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        r = requests.get(url, headers=headers, timeout=15)
        print(f"Status: {r.status_code}")
        soup = BeautifulSoup(r.text, 'html.parser')
        next_data = soup.find('script', id='__NEXT_DATA__')
        if next_data:
            print("Found __NEXT_DATA__!")
            data = json.loads(next_data.string)
            # Save data to check it
            with open("scratch_adala_next_data.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print("Saved to scratch_adala_next_data.json")
            
            # Let's inspect the keys
            print(f"Keys: {list(data.keys())}")
            if 'props' in data:
                print(f"Props keys: {list(data['props'].keys())}")
                if 'pageProps' in data['props']:
                    print(f"pageProps keys: {list(data['props']['pageProps'].keys())}")
        else:
            print("No __NEXT_DATA__ tag found.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_adala_next_data()
