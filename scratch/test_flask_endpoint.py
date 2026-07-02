import requests
import json

url = "http://127.0.0.1:5000/api/query"
payload = {
    "query": "Est-ce que le rejet d'hydrocarbures au Bénin entraîne une peine de prison ?",
    "country": "Bénin",
    "agent": "Recursive Retrieval Agent"
}

headers = {
    "Content-Type": "application/json"
}

try:
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    print("STATUS CODE:", response.status_code)
    res_data = response.json()
    print("\nANSWER:")
    print(res_data.get("answer"))
    print("\nSOURCES:")
    for src in res_data.get("sources", []):
        print(f"- ID: {src['id']} | Title: {src['title']} | Country: {src['country']}")
        print(f"  Snippet: {src['text'][:150]}...")
    print("\nRESOLVED DOC:")
    print(f"Theme: {res_data.get('resolved_theme')} | Country: {res_data.get('resolved_country')}")
except Exception as e:
    print("ERROR:", e)
