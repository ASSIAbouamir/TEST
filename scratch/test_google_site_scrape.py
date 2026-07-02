import requests
from bs4 import BeautifulSoup

def test_google_news_site(site, query=""):
    # Google News search query format: site:example.com query
    # URL encode query
    from urllib.parse import quote_plus
    q = f"site:{site} {query}".strip()
    url = f"https://news.google.com/rss/search?hl=fr&gl=FR&ceid=FR:fr&when=7d&q={quote_plus(q)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        r = requests.get(url, headers=headers, timeout=15)
        print(f"Google News RSS for {q} Status: {r.status_code}")
        soup = BeautifulSoup(r.text, 'xml')
        items = soup.find_all('item')
        print(f"Found {len(items)} items:")
        for item in items[:5]:
            title = item.find('title').get_text(strip=True) if item.find('title') else 'No title'
            link = item.find('link').get_text(strip=True) if item.find('link') else 'No link'
            pub_date = item.find('pubDate').get_text(strip=True) if item.find('pubDate') else 'No date'
            print(f"  Title: {title[:80]}")
            print(f"  Link: {link}")
            print(f"  Date: {pub_date}")
            print("-" * 20)
    except Exception as e:
        print(f"Error for {site}: {e}")

if __name__ == "__main__":
    print("--- Testing site:legifrance.gouv.fr ---")
    test_google_news_site("legifrance.gouv.fr")
    print("\n--- Testing site:adala.justice.gov.ma ---")
    test_google_news_site("adala.justice.gov.ma")
