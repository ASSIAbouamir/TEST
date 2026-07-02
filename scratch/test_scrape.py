import requests
from bs4 import BeautifulSoup

def test_adala_fr():
    url = "https://adala.justice.gov.ma/fr/projects-of-laws"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        r = requests.get(url, headers=headers, timeout=15)
        print(f"Adala FR Status: {r.status_code}")
        soup = BeautifulSoup(r.text, 'html.parser')
        # Print some info
        print(f"Title: {soup.title.text if soup.title else 'No title'}")
        
        # Let's search for any link containing /api/uploads or pdf
        links = soup.find_all('a', href=True)
        pdf_links = [l for l in links if 'pdf' in l['href'].lower() or 'upload' in l['href'].lower()]
        print(f"Found {len(pdf_links)} pdf/upload links:")
        for l in pdf_links[:5]:
            print(f"  Text: {l.get_text(strip=True)} | Href: {l['href']}")
            
        # Let's check headers/h5 to see if text is in HTML
        h5s = soup.find_all('h5')
        print(f"Found {len(h5s)} h5 elements:")
        for h in h5s[:5]:
            print(f"  {h.get_text(strip=True)}")
    except Exception as e:
        print(f"Adala FR Error: {e}")

def test_adala_ar():
    url = "https://adala.justice.gov.ma/ar/projects-of-laws"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        r = requests.get(url, headers=headers, timeout=15)
        print(f"Adala AR Status: {r.status_code}")
        soup = BeautifulSoup(r.text, 'html.parser')
        print(f"Title: {soup.title.text if soup.title else 'No title'}")
        
        links = soup.find_all('a', href=True)
        pdf_links = [l for l in links if 'pdf' in l['href'].lower() or 'upload' in l['href'].lower()]
        print(f"Found {len(pdf_links)} pdf/upload links:")
        for l in pdf_links[:5]:
            print(f"  Text: {l.get_text(strip=True)} | Href: {l['href']}")
            
        h5s = soup.find_all('h5')
        print(f"Found {len(h5s)} h5 elements:")
        for h in h5s[:5]:
            print(f"  {h.get_text(strip=True)}")
    except Exception as e:
        print(f"Adala AR Error: {e}")

def test_legifrance():
    # Légifrance has RSS feeds for the Journal Officiel
    # e.g., https://www.legifrance.gouv.fr/copieRss?idRss=1 (Lois et décrets)
    # Let's test checking that RSS feed
    url = "https://www.legifrance.gouv.fr/copieRss?idRss=1"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        r = requests.get(url, headers=headers, timeout=15)
        print(f"Legifrance RSS Status: {r.status_code}")
        soup = BeautifulSoup(r.text, 'xml')
        items = soup.find_all('item')
        print(f"Found {len(items)} items in Legifrance RSS:")
        for item in items[:5]:
            title = item.find('title').get_text(strip=True) if item.find('title') else 'No title'
            link = item.find('link').get_text(strip=True) if item.find('link') else 'No link'
            print(f"  Title: {title[:80]} | Link: {link}")
    except Exception as e:
        print(f"Legifrance Error: {e}")

if __name__ == "__main__":
    print("--- Testing Adala FR ---")
    test_adala_fr()
    print("\n--- Testing Adala AR ---")
    test_adala_ar()
    print("\n--- Testing Legifrance ---")
    test_legifrance()
